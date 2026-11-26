"""Home's picture AI on the Raspberry Pi (a Home Assistant add-on).

The TVs on the home network send it the picture their ad watch is looking at (320x180, JPEG) and get back how much
it's an ad and what the AI saw (for how much it looks like the channel's program, which each TV keeps). The model is
CLIP ViT-B/32's picture tower (OpenAI, MIT; Xenova's ONNX copy at a fixed version), downloaded once from Hugging Face
into the add-on's own storage and used only if its SHA-256 is right. Since 1.1.0 a TV can also ask about one spot of
the picture, where it has found the channel's own logo: Apple's MobileCLIP2-S0 picture tower looks at that spot,
zoomed, on its own core at the same time, and says what it saw there (for how much it looks like the logo's spot
usually does, which the TV keeps). Since 1.2.0 a TV can ask it to read the letters of a logo it has found (the
logo's spot, a few pictures put together so what moves behind it fades): PaddleOCR's PP-OCRv4 models (Apache-2.0,
RapidOCR's ONNX copies, in its package at a fixed version, each model checked by its SHA-256) find its lines of letters
and read them, so the TV can tell the channel's own logo (it spells the channel's name) from other marks that don't
move. Since 1.4.0 it can read a whole picture too (`side`: the letters looked for with the picture's longer side up to
1280 points instead of a logo's 256), so a TV can ask what a picture says, like a film's end credits. Since 1.3.0 it
remembers the ads the TVs have seen: once a TV's ad screen has covered a break, the TV sends
the AI's fingerprints of the ads in it (512 numbers each, never a picture), each is kept once, and every answer says
how like the nearest remembered ad the picture is, so an ad seen before is known on any channel and either TV.
Nothing leaves the home network: the pictures are looked at and forgotten, never kept or logged; the ads' fingerprints
are kept in the app's own storage. Announced on the network (mDNS, _homeai._tcp) so the TVs find it.

    POST /look[?spot=x0,y0,x1,y1]   a JPEG  ->  {"model", "score", "embedding" (512 float32, base64), "ad", "like",
                                                "sure", "known", "ms"}, and for a spot {"spot_model", "spot", "spot_like",
                                                "spot_ms"}
    POST /ads                       {"model", "ads": [[scale, 512 int8 (base64)], ...]}  ->  {"new", "again", "ads"}
    POST /read[?side=N]             a PNG or JPEG (a logo's spot; with side, a whole picture looked at with its longer
                                                side N, 256 to 1280)  ->  {"read_model", "lines" [{"text", "score"}],
                                                "read_ms", "side"}
    GET  /status                            ->  {"model", "version", "ready", "note", "looks", "ms", "up", "spot_model",
                                                "spot_ready", "spot_note", "spot_looks", "spot_ms", "read_model",
                                                "read_ready", "read_note", "reads", "read_ms", "read_side_max", "ads",
                                                "ads_note", "bench"}
    POST /bench                             ->  starts the bench (1.5.0): bigger models timed here, one at a time
    GET  /bench                             ->  {"state", "note", "at", "memory_mb", "models": [{"id", "what", "size_mb",
                                                "ms_1", "ms_3", "per_second", "load_s", "memory_mb", "note"}, ...]}

Since 1.5.0 it can time bigger models on this Pi, for the ad AI's next model (the bench, started from the home
network): each is downloaded (a fixed version, checked by its SHA-256), timed on 1 core and on 3 while no TV is
asking, and deleted again; the models the TVs use now are timed too, to compare.
"""
import base64
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
import shutil
import socket
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

VERSION = '1.5.1'
PORT = 8765
MODEL_ID = 'clip-vit-b32'
MODEL_URL = ('https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/'
             '6b7c523703ae26e39be4b03774966f8571c7f523/onnx/vision_model.onnx')
MODEL_SIZE = 351_685_709
MODEL_SHA = 'fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40'
# Over AD: an ad; under LIKE: not the program the channel has been showing (set on his two recordings). Until a TV
# knows a channel's look, the score alone has to clear AD by SURE either way (between: not sure).
AD, LIKE, SURE = -0.0253, 0.70, 0.01
SIZE = 224
MAX_PIXELS = 4_000_000  # a picture bigger than 2560x1440 is turned away
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/data' if os.path.isdir('/data') else HERE
MODEL = os.path.join(DATA, 'clip-vit-b32-vision.onnx')
THREADS = int(os.environ.get('AI_THREADS', '3'))
# The logo's spot (1.1.0): Apple's MobileCLIP2-S0 picture tower (Apple's research license, apple-amlr; plhery's ONNX
# copy at a fixed version), on the processor's fourth core. Its spot's likeness to the logo's usual look, at or over
# SPOT_LIKE: the logo's there (set on his two recordings, where the TV found the logo).
SPOT_ID = 'mobileclip2-s0'
SPOT_URL = ('https://huggingface.co/plhery/mobileclip2-onnx/resolve/'
            'ba95759a5bdbaca53e9111e2550a76ec09c8fd9e/onnx/s0/vision_model.onnx')
SPOT_SIZE = 45_555_784
SPOT_SHA = '13d20ebfa8a8f63890eb2727fe4dc63009ff970f43e0f7d9d2ed999659f70c8a'
SPOT_LIKE = 0.78
SPOT_IN = 256
SPOT_MODEL = os.path.join(DATA, 'mobileclip2-s0-vision.onnx')
SPOT_THREADS = int(os.environ.get('SPOT_THREADS', '1'))
# Reading a logo (1.2.0): PaddleOCR's PP-OCRv4 detection and recognition models (Apache-2.0), RapidOCR's ONNX copies,
# from its package (rapidocr-onnxruntime 1.4.4, installed without the rest of it); each checked before it's used.
READ_ID = 'ppocr-v4'
READ_FILES = {'det': ('ch_PP-OCRv4_det_infer.onnx', 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'),
              'rec': ('ch_PP-OCRv4_rec_infer.onnx', '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b')}
READ_PIXELS = 2_600_000  # a logo's spot: up to about 1600x1600
READ_SIDE = 256          # the spot is looked for letters with its longer side this long
READ_SIDE_MAX = 1280     # a whole picture (?side=): up to this long (a TV's 960x540 picture, its credits' letters)
READ_LINE_H = 48         # each line is read this high
# The ads the TVs have seen (1.3.0): each kept once (a fingerprint ADS_SAME or more like a kept one is that one, seen
# again), with no limit but a ceiling it would take decades of watching to reach (ADS_CEILING: then the ones seen
# longest ago go, down to 99% of it); a TV sends at most ADS_BATCH at a time (a break's worth is about a hundred). On his recordings the
# same ad came back at 0.95 or more, and no picture of the program came within 0.93 of an ad. From ADS_INDEX_FROM ads
# on they're sorted into ADS_GROUPS groups by how they look (an index, made again when they've grown sixteenfold), so
# each picture is compared only with the ads in the ADS_PROBE groups nearest it: as quick with a million as with a
# thousand. New ones go on the end of a log (ads.log), folded into the main file (ads.npz) when the log passes
# ADS_LOG_MAX or the app starts, so a break's ads never rewrite the whole memory.
ADS_FILE = os.path.join(DATA, 'ads.npz')
ADS_LOG = os.path.join(DATA, 'ads.log')
ADS_SAME = 0.95
ADS_CEILING = 2_000_000
ADS_BATCH = 600
ADS_INDEX_FROM = 8_192
ADS_GROUPS = 1024
ADS_PROBE = 8
ADS_LOG_MAX = 32 << 20
ADS_HEAD = b'HPA-ADS1'
ADS_REC = 520  # a logged ad: its 512 numbers a byte each, its scale (float32), the day it was seen (int32)

direction = np.fromfile(os.path.join(HERE, 'direction.bin'), dtype='<f4')
state = {'ready': False, 'note': 'starting', 'looks': 0, 'ms': 0.0, 'since': time.time(),
         'spot_ready': False, 'spot_note': 'waiting for the main model', 'spot_looks': 0, 'spot_ms': 0.0,
         'read_ready': False, 'read_note': 'waiting for the other models', 'reads': 0, 'read_ms': 0.0, 'last_look': 0.0}
session = None
spot_session = None
reader = None
read_lock = threading.Lock()
run_lock = threading.Lock()
spot_lock = threading.Lock()
spot_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ---- (1.5.0) The bench: how fast bigger models run on this Pi, for the ad AI's next model ----
HF = 'https://huggingface.co/'
MC2 = HF + 'plhery/mobileclip2-onnx/resolve/ba95759a5bdbaca53e9111e2550a76ec09c8fd9e/onnx/'
SG2B = HF + 'onnx-community/siglip2-base-patch16-256-ONNX/resolve/d1114256522a37ffa257a0a58017348ab0058db2/onnx/'
SG2L = HF + 'onnx-community/siglip2-large-patch16-256-ONNX/resolve/c03fb63ddd5fe363212cbd08110f92a35250db6a/onnx/'
# (id, what, address (None: a model the TVs use now), size, SHA-256, picture side): the smallest first.
BENCH = [
    ('clip-vit-b32', "CLIP ViT-B/32 (the AI's now)", None, 351_685_709, None, 224),
    ('mobileclip2-s0', "MobileCLIP2-S0 (the logo's spot now)", None, 45_555_784, None, 256),
    ('siglip2-b16-256-int8', 'SigLIP 2 B/16, 256 (8-bit)', SG2B + 'vision_model_int8.onnx', 94_737_653,
     '8243a496e524bf8fa19830ff42a40a4bae2e97acc2e288d4ae97601edf954977', 256),
    ('mobileclip2-s2', 'MobileCLIP2-S2', MC2 + 's2/vision_model.onnx', 143_044_797,
     'a841f72c5a5085748bbe271a1d5718aba877822a15cba865bdbd0d37036b849e', 256),
    ('siglip2-l16-256-int8', 'SigLIP 2 L/16, 256 (8-bit)', SG2L + 'vision_model_int8.onnx', 319_618_816,
     '861d6b698f7390f9d5282a91b06d745ec86fc30f13ff638da84583acc83ed2ba', 256),
    ('mobileclip2-b', 'MobileCLIP2-B', MC2 + 'b/vision_model.onnx', 345_545_489,
     '2f3873e400fd1adc5298b72d29e81bb69337e58601cf2853c43bc1211631eecd', 224),
    ('siglip2-b16-256', 'SigLIP 2 B/16, 256', SG2B + 'vision_model.onnx', 371_992_072,
     'f5cb16728a704703f05516ded628397e11dbca4de2eb5db04b0c0bcee988aa7a', 256),
    ('mobileclip2-l14', 'MobileCLIP2-L/14', MC2 + 'l14/vision_model.onnx', 1_216_159_663,
     'd48d4392f5805f59896009ea5460c7fa29f59722edd0d7d26c054b0980020998', 224),
    ('siglip2-l16-256', 'SigLIP 2 L/16, 256', SG2L + 'vision_model.onnx', 1_264_345_246,
     'a68762da4fab7335558d4713c77e42d8abe23dcd48d38b5dbe1c00765c07ca56', 256),
]
BENCH_RUNS = 8          # timed runs for each number (after one to warm up)
BENCH_BUDGET = 150      # seconds at most for each number (at least 2 runs)
BENCH_QUIET = 20        # no TV asking for this long before a timing starts
BENCH_WAIT = 1800       # waits this long at most for that, then times anyway (and says so)
BENCH_ROOM = 1 << 30    # storage left free besides the model
# (1.5.1) One TV at a time (two are seldom on at once): what counts is how many pictures a second the Pi gets through,
# not how soon one answer comes (the look ahead gives a head start of seconds, from what's shown to the newest picture
# downloaded), so each model is timed on all the cores one picture at a time, with four pictures at once (a core each)
# and with four in one batch; and 8-bit copies made here (its matrix math only; or all of it, set on the TVs' recorded
# pictures), with how like the original they see.
NORM = {'clip': (MEAN, STD), 'unit': (np.zeros(3, np.float32), np.ones(3, np.float32)),
        'half': (np.full(3, 0.5, np.float32), np.full(3, 0.5, np.float32))}
# (id, what, source, size, SHA-256, picture side, mean and spread, make): source None (a model the TVs use now), an
# address, or the id of a model above to make an 8-bit copy of (make: 'dynamic' or 'static').
BENCH_ONE = [
    ('clip-vit-b32', "CLIP ViT-B/32 (the AI's now)", None, 351_685_709, None, 224, 'clip', None),
    ('mobileclip2-s0', "MobileCLIP2-S0 (the logo's spot now)", None, 45_555_784, None, 256, 'unit', None),
    ('mobileclip2-s0-q8s', 'MobileCLIP2-S0, 8-bit (all of it)', 'mobileclip2-s0', 0, None, 256, 'unit', 'static'),
    ('mobileclip2-s2', 'MobileCLIP2-S2', MC2 + 's2/vision_model.onnx', 143_044_797,
     'a841f72c5a5085748bbe271a1d5718aba877822a15cba865bdbd0d37036b849e', 256, 'unit', None),
    ('mobileclip2-s2-q8d', 'MobileCLIP2-S2, 8-bit (its matrix math)', 'mobileclip2-s2', 0, None, 256, 'unit', 'dynamic'),
    ('mobileclip2-s2-q8s', 'MobileCLIP2-S2, 8-bit (all of it)', 'mobileclip2-s2', 0, None, 256, 'unit', 'static'),
    ('mobileclip2-b', 'MobileCLIP2-B', MC2 + 'b/vision_model.onnx', 345_545_489,
     '2f3873e400fd1adc5298b72d29e81bb69337e58601cf2853c43bc1211631eecd', 224, 'unit', None),
    ('mobileclip2-b-q8d', 'MobileCLIP2-B, 8-bit (its matrix math)', 'mobileclip2-b', 0, None, 224, 'unit', 'dynamic'),
    ('mobileclip2-b-q8s', 'MobileCLIP2-B, 8-bit (all of it)', 'mobileclip2-b', 0, None, 224, 'unit', 'static'),
    ('siglip2-b16-256-int8', 'SigLIP 2 B/16, 256 (8-bit)', SG2B + 'vision_model_int8.onnx', 94_737_653,
     '8243a496e524bf8fa19830ff42a40a4bae2e97acc2e288d4ae97601edf954977', 256, 'half', None),
]
THREADS_ALL = os.cpu_count() or 4  # all of the Pi's cores (4)
BENCH_PAR_S = 20       # about this long for the pictures-at-once timing
BENCH_CALIBRATE = 48   # recorded pictures an 8-bit copy is set on (the next 16 check it)
ONE_TV = 2.0           # pictures a second one TV sends at most (every picture while a break may start or is on)
MEDIA = 'http://127.0.0.1:8766'  # Home media on this Pi (its recordings)
bench = {'state': 'not run yet', 'note': '', 'at': None, 'memory_mb': None, 'models': []}
bench_lock = threading.Lock()


class AdMemory:
    """The ads the TVs have seen (1.3.0): the AI's fingerprints, a byte a number with each one's scale (of length 1 as
    kept), how many times each was seen and the day it last was, and the group it's in; kept in the app's storage
    (ads.npz and ads.log), with the model they came from (another model's are of no use: the memory starts over)."""

    def __init__(self, path, log):
        self.path, self.logp = path, log
        self.lock = threading.RLock()
        self.work = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # the index and the tidying, one at a time
        self.busy = False
        self.note = 'none yet'
        self._clear()

    def _clear(self):
        self.n = 0
        self._alloc(1024)
        self.cent = None    # the groups' centers (ADS_GROUPS x 512, length 1), once there's an index, and what all the
        self.mu = None      # ads have in common (taken away before grouping: else a few groups would take nearly all)
        self.order = None   # the ads indexed, by group, and where each group starts in it
        self.starts = None
        self.extra = None   # each group's ads since the index was made
        self.indexed = 0

    def _alloc(self, cap):
        self.q = np.zeros((cap, 512), np.int8)
        self.s = np.zeros(cap, np.float32)
        self.count = np.zeros(cap, np.int32)
        self.day = np.zeros(cap, np.int32)
        self.cell = np.full(cap, -1, np.int32)

    def _grow(self):
        n, cap = self.n, len(self.q) * 2
        q, s, count, day, cell = self.q, self.s, self.count, self.day, self.cell
        self._alloc(cap)
        self.q[:n], self.s[:n], self.count[:n], self.day[:n], self.cell[:n] = q[:n], s[:n], count[:n], day[:n], cell[:n]

    @staticmethod
    def quant(v):
        """A fingerprint a byte a number, and the scale that makes it length 1 again."""
        q = np.clip(np.round(v / (max(float(np.abs(v).max()), 1e-9) / 127.0)), -127, 127).astype(np.int8)
        return q, 1.0 / max(float(np.linalg.norm(q.astype(np.float32))), 1e-9)

    def _add(self, q, sc, day, count=1):
        if self.n == len(self.q):
            self._grow()
        i = self.n
        self.q[i], self.s[i], self.day[i], self.count[i] = q, sc, day, count
        self.cell[i] = -1
        if self.cent is not None:
            c = int(np.argmax(self.cent @ self._apart(q.astype(np.float32) * sc)))
            self.cell[i] = c
            self.extra[c].append(i)
        self.n += 1
        return i

    def _apart(self, x):
        """Fingerprints with what all the ads have in common taken away (length 1 again): what tells them apart."""
        y = x - self.mu
        return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)

    def _best(self, e):
        """The remembered ad most like a fingerprint: how like (-1 with none) and which."""
        if self.n == 0:
            return -1.0, -1
        if self.cent is None:  # not many yet: all of them
            best, bi = -1.0, -1
            for a in range(0, self.n, 16384):
                b = min(self.n, a + 16384)
                sims = (self.q[a:b].astype(np.float32) @ e) * self.s[a:b]
                j = int(np.argmax(sims))
                if sims[j] > best:
                    best, bi = float(sims[j]), a + j
            return best, bi
        top = np.argpartition(-(self.cent @ self._apart(e)), ADS_PROBE)[:ADS_PROBE]
        parts = [self.order[self.starts[c]:self.starts[c + 1]] for c in top]
        parts += [np.array(self.extra[c], np.int64) for c in top if self.extra[c]]
        idx = np.concatenate(parts)
        if len(idx) == 0:
            return -1.0, -1
        sims = (self.q[idx].astype(np.float32) @ e) * self.s[idx]
        j = int(np.argmax(sims))
        return float(sims[j]), int(idx[j])

    def known(self, e):
        """How like the nearest remembered ad a fingerprint is (0 with none)."""
        with self.lock:
            return max(0.0, self._best(e)[0])

    def learn(self, vs, today):
        """A break's ads from a TV: each new one kept (on the end of the log), each one seen before counted again."""
        new = again = 0
        recs = []
        with self.lock:
            first = self.n
            for v in vs:
                sim, i = self._best(v)
                if i >= 0 and sim >= ADS_SAME:
                    self.count[i] += 1
                    self.day[i] = today
                    again += 1
                    continue
                q, sc = self.quant(v)
                self._add(q, sc, today)
                recs.append(q.tobytes() + np.array([sc], '<f4').tobytes() + np.array([today], '<i4').tobytes())
                new += 1
            if recs:
                self._log(recs, first)
            self.note = '%d ads' % self.n
            self._next()
            return new, again, self.n

    def _log(self, recs, first):
        with open(self.logp, 'ab') as f:
            if f.tell() == 0:
                f.write(b'%s %s %d\n' % (ADS_HEAD, MODEL_ID.encode(), first))
            f.write(b''.join(recs))

    def _replay(self, path):
        """A log's ads after the main file's (a log left from a tidy that didn't finish is read first)."""
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            return 0
        nl = data.find(b'\n')
        head = data[:nl].split() if nl > 0 else []
        if len(head) != 3 or head[0] != ADS_HEAD or head[1] != MODEL_ID.encode():
            return 0
        row, added = int(head[2]), 0
        for a in range(nl + 1, len(data) - ADS_REC + 1, ADS_REC):  # a record cut short at the end is left out
            if row == self.n:
                r = data[a:a + ADS_REC]
                self._add(np.frombuffer(r[:512], np.int8), float(np.frombuffer(r[512:516], '<f4')[0]),
                          int(np.frombuffer(r[516:520], '<i4')[0]))
                added += 1
            elif row > self.n:
                break
            row += 1
        return added

    def load(self):
        with self.lock:
            logged = 0
            try:
                try:
                    with np.load(self.path, allow_pickle=False) as z:
                        if str(z['model']) != MODEL_ID:
                            raise LookupError('another model')
                        n = len(z['q'])
                        self._alloc(max(1024, 2 * n))
                        self.q[:n], self.s[:n], self.count[:n], self.day[:n] = z['q'], z['scale'], z['count'], z['day']
                        self.n = n
                        if len(z['cent']):
                            self.cell[:n] = z['cell']
                            self.mu = z['mu'].astype(np.float32)
                            self._index_from(z['cent'].astype(np.float32), int(z['indexed']))
                except FileNotFoundError:
                    pass
                logged = self._replay(self.logp + '.old') + self._replay(self.logp)
                self.note = '%d ads' % self.n
            except LookupError:
                self._clear()
                self._drop_logs()
                self.note = 'started over (its ads were seen by another model)'
            except Exception as e:
                self._clear()
                self._drop_logs()
                self.note = "couldn't be read (%s): started over" % type(e).__name__
            say('the ads remembered: ' + self.note + (' (%d from the log)' % logged if logged else ''))
            if logged or os.path.exists(self.logp + '.old'):
                self.busy = True
                self.work.submit(self._job, self._tidy)
            self._next()

    def _drop_logs(self):
        for p in (self.logp, self.logp + '.old'):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    def _index_from(self, cent, indexed):
        """The index from the groups' centers, for the ads already in their groups (their cells)."""
        n = self.n
        self.cent = cent
        cells = self.cell[:n]
        self.order = np.argsort(cells, kind='stable').astype(np.int64)
        self.starts = np.searchsorted(cells[self.order], np.arange(len(cent) + 1))
        self.extra = [[] for _ in range(len(cent))]
        self.indexed = indexed
        for i in range(n):  # any not in a group yet (none, as kept)
            if cells[i] < 0:
                c = int(np.argmax(cent @ self._apart(self.q[i].astype(np.float32) * self.s[i])))
                self.cell[i] = c
                self.extra[c].append(i)

    def _next(self):
        """The index made (again) or the log folded in, when it's time (in the background, one at a time)."""
        if self.busy:
            return
        if self.n >= ADS_INDEX_FROM and (self.cent is None or self.n >= 16 * self.indexed):
            self.busy = True
            self.work.submit(self._job, self._index)
        elif self.n >= ADS_CEILING or (os.path.exists(self.logp) and os.path.getsize(self.logp) > ADS_LOG_MAX):
            self.busy = True
            self.work.submit(self._job, self._tidy)

    def _job(self, f):
        try:
            f()
        except Exception as e:
            say("the ads' memory: %s didn't finish (%s)" % (f.__name__.strip('_'), type(e).__name__))
        finally:
            with self.lock:
                self.busy = False
                self._next()

    def _index(self):
        """The ads sorted into groups by how they look (k-means on up to 16 ads a group, then every ad to its nearest)."""
        with self.lock:
            n0, q, s = self.n, self.q, self.s  # the first n0 don't change while this runs (only a tidy changes them)
        t0 = time.time()
        k = ADS_GROUPS
        rng = np.random.default_rng(n0)
        pick = np.sort(rng.choice(n0, size=min(n0, 16 * k), replace=False))
        x = q[pick].astype(np.float32) * s[pick, None]
        mu = x.mean(axis=0)
        x = x - mu
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        cent = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(6):
            a = self._nearest(x, cent)
            counts = np.bincount(a, minlength=k)
            order = np.argsort(a, kind='stable')
            full = counts > 0
            sums = np.zeros_like(cent)
            sums[full] = np.add.reduceat(x[order], np.concatenate([[0], np.cumsum(counts)[:-1]])[full], axis=0)
            sums[~full] = x[rng.choice(len(x), size=int((~full).sum()))]
            cent = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-9)
        cells = np.empty(n0, np.int32)
        for a0 in range(0, n0, 8192):
            b0 = min(n0, a0 + 8192)
            y = q[a0:b0].astype(np.float32) * s[a0:b0, None] - mu
            cells[a0:b0] = self._nearest(y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9), cent)
        with self.lock:
            self.cell[:n0] = cells
            self.cell[n0:self.n] = -1
            self.mu = mu.astype(np.float32)
            self._index_from(cent.astype(np.float32), n0)
        say("the ads' memory: indexed %d ads in %d groups (%.0f s)" % (n0, k, time.time() - t0))
        self._tidy()

    @staticmethod
    def _nearest(x, cent):
        out = np.empty(len(x), np.int32)
        for a in range(0, len(x), 4096):
            out[a:a + 4096] = np.argmax(x[a:a + 4096] @ cent.T, axis=1)
        return out

    def _tidy(self):
        """The log folded into the main file: what's remembered now written whole (from a copy, while it goes on
        answering), then the log that was folded in let go. At the ceiling, the ones seen longest ago go first (down to
        99% of it)."""
        with self.lock:
            if self.n >= ADS_CEILING:  # down to 99% of it, the ones seen longest ago (of those, the fewest times) gone
                n = self.n
                keep = np.sort(np.lexsort((self.count[:n], self.day[:n]))[n - int(ADS_CEILING * 0.99):])
                kept = [a[keep] for a in (self.q, self.s, self.count, self.day, self.cell)]
                cent, mu, indexed = self.cent, self.mu, self.indexed
                self._clear()
                self._alloc(max(1024, 2 * len(keep)))
                self.n = len(keep)
                self.q[:self.n], self.s[:self.n], self.count[:self.n], self.day[:self.n], self.cell[:self.n] = kept
                if cent is not None:
                    self.mu = mu
                    self._index_from(cent, indexed)  # the groups stay as they were
                self._drop_logs()  # its rows are numbered anew: the file below has them all
            n = self.n
            snap = {'model': np.array(MODEL_ID), 'q': self.q[:n].copy(), 'scale': self.s[:n].copy(),
                    'count': self.count[:n].copy(), 'day': self.day[:n].copy(), 'cell': self.cell[:n].copy(),
                    'cent': self.cent.copy() if self.cent is not None else np.zeros((0, 512), np.float32),
                    'mu': self.mu.copy() if self.mu is not None else np.zeros(512, np.float32),
                    'indexed': np.array(self.indexed)}
            if os.path.exists(self.logp):
                os.replace(self.logp, self.logp + '.old')  # new ones go to a fresh log meanwhile
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as f:
            np.savez(f, **snap)
        os.replace(tmp, self.path)
        try:
            os.remove(self.logp + '.old')
        except FileNotFoundError:
            pass


ads = AdMemory(ADS_FILE, ADS_LOG)


def learn_ads(body):
    """POST /ads: {"model", "ads": [[scale, 512 int8 (base64)], ...]}."""
    j = json.loads(body)
    if j.get('model') != MODEL_ID:
        raise LookupError('another model')
    vs = []
    for item in (j.get('ads') or [])[:ADS_BATCH]:
        scale, b64 = float(item[0]), item[1]
        q = np.frombuffer(base64.b64decode(b64), np.int8)
        if len(q) != 512 or not np.isfinite(scale) or scale <= 0:
            raise ValueError('not a fingerprint')
        v = q.astype(np.float32) * scale
        n = np.linalg.norm(v)
        if n > 0:
            vs.append(v / n)
    new, again, total = ads.learn(vs, int(time.time() // 86400))
    return {'new': new, 'again': again, 'ads': total}


def say(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), s, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch(path, url, size, sha, key='note', what='the model'):
    """A model in place and checked (a half-download goes on where it stopped); tries until it's there."""
    wait = 30
    while not (os.path.isfile(path) and os.path.getsize(path) == size):
        part = path + '.part'
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        state[key] = 'downloading %s (%d of %d MB)' % (what, have // 1_000_000, size // 1_000_000)
        say(state[key])
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'home-picture-ai', **({'Range': 'bytes=%d-' % have} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, 'ab' if have and r.status == 206 else 'wb') as out:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            if os.path.getsize(part) != size:
                raise IOError('ended at %d MB' % (os.path.getsize(part) // 1_000_000))
            if sha256(part) != sha:
                os.remove(part)
                raise IOError("its check didn't match (deleted)")
            os.replace(part, path)
            say('%s downloaded and checked' % what)
        except Exception as e:
            state[key] = "%s didn't download (%s); trying again in %d s" % (what, type(e).__name__, wait)
            say(state[key])
            time.sleep(wait)
            wait = min(wait * 2, 1800)


def load():
    global session
    t0 = time.time()
    o = ort.SessionOptions()
    o.intra_op_num_threads = THREADS
    o.inter_op_num_threads = 1
    o.add_session_config_entry('session.intra_op.allow_spinning', '0')  # no busy-waiting between pictures
    try:
        session = ort.InferenceSession(MODEL, o, providers=['CPUExecutionProvider'])
    except Exception:
        if sha256(MODEL) != MODEL_SHA:  # damaged on disk: fetched again
            os.remove(MODEL)
            say('the model was damaged (deleted, fetched again)')
            return False
        raise
    session.run(None, {'pixel_values': np.zeros((1, 3, SIZE, SIZE), np.float32)})
    state['ready'] = True
    state['note'] = 'ready (loaded in %.1f s, %d threads)' % (time.time() - t0, THREADS)
    say(state['note'])
    return True


def load_spot():
    """Apple's model for the logo's spot, on its own core; true once it's ready."""
    global spot_session
    t0 = time.time()
    o = ort.SessionOptions()
    o.intra_op_num_threads = SPOT_THREADS
    o.inter_op_num_threads = 1
    o.add_session_config_entry('session.intra_op.allow_spinning', '0')
    try:
        s = ort.InferenceSession(SPOT_MODEL, o, providers=['CPUExecutionProvider'])
    except Exception:
        if sha256(SPOT_MODEL) != SPOT_SHA:
            os.remove(SPOT_MODEL)
            say("the logo's model was damaged (deleted, fetched again)")
            return False
        raise
    s.run(None, {'pixel_values': np.zeros((1, 3, SPOT_IN, SPOT_IN), np.float32)})
    spot_session = s
    state['spot_ready'] = True
    state['spot_note'] = 'ready (loaded in %.1f s, %d thread)' % (time.time() - t0, SPOT_THREADS)
    say("the logo's model: " + state['spot_note'])
    return True


class Reader:
    """A logo's lines of letters: the detection model's map of where letters are, split into lines by the rows and
    columns where there are none, and each line read by the recognition model (its best path)."""

    def __init__(self, det_path, rec_path):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.det = ort.InferenceSession(det_path, o, providers=['CPUExecutionProvider'])
        self.rec = ort.InferenceSession(rec_path, o, providers=['CPUExecutionProvider'])
        chars = self.rec.get_modelmeta().custom_metadata_map['character'].splitlines()
        self.chars = [''] + chars + [' ']

    def read(self, im, side=READ_SIDE):
        rgb = im.convert('RGB')
        w, h = rgb.size
        k = side / max(w, h)
        dw, dh = max(32, int(round(w * k / 32)) * 32), max(32, int(round(h * k / 32)) * 32)
        x = (np.asarray(rgb.resize((dw, dh), Image.BICUBIC), np.float32) / 255.0 - 0.5) / 0.5
        prob = self.det.run(None, {'x': x.transpose(2, 0, 1)[None]})[0][0, 0]
        sx, sy = w / dw, h / dh
        out = []
        for x0, y0, x1, y1 in letter_lines(prob):
            line = rgb.crop((int(x0 * sx), int(y0 * sy), int(np.ceil(x1 * sx)), int(np.ceil(y1 * sy))))
            text, score = self.read_line(line)
            if text.strip():
                out.append({'text': text.strip(), 'score': round(score, 3)})
        return out

    def read_line(self, line):
        w, h = line.size
        nw = max(16, min(1280, int(round(READ_LINE_H * w / max(1, h)))))
        x = (np.asarray(line.resize((nw, READ_LINE_H), Image.BICUBIC), np.float32) / 255.0 - 0.5) / 0.5
        p = self.rec.run(None, {'x': x.transpose(2, 0, 1)[None]})[0][0]
        best, conf = p.argmax(1), p.max(1)
        text, scores, last = [], [], 0
        for i, c in zip(best, conf):
            if i != last and i != 0:
                text.append(self.chars[i] if i < len(self.chars) else '')
                scores.append(float(c))
            last = i
        return ''.join(text), (sum(scores) / len(scores) if scores else 0.0)


def letter_lines(prob, thresh=0.3, keep=0.5, unclip=1.6):
    """The detection map's lines (x0, y0, x1, y1): rows with letters split where there are none, each band's columns
    split where a gap is wider than the band is high; a line's average must be at least `keep`; widened back (the
    model draws them shrunk)."""
    on = prob > thresh
    H, W = on.shape
    boxes = []
    for r0, r1 in stretches(on.any(1), 1):
        band = on[r0:r1]
        for c0, c1 in stretches(band.any(0), max(2, int((r1 - r0) * 1.2))):
            rows = np.where(on[r0:r1, c0:c1].any(1))[0]
            y0, y1 = r0 + rows[0], r0 + rows[-1] + 1
            area = on[y0:y1, c0:c1]
            if min(c1 - c0, y1 - y0) < 2 or float(prob[y0:y1, c0:c1][area].mean()) < keep:
                continue
            d = (c1 - c0) * (y1 - y0) * unclip / (2 * ((c1 - c0) + (y1 - y0)))
            boxes.append((max(0, c0 - d), max(0, y0 - d), min(W, c1 + d), min(H, y1 + d)))
    return boxes


def stretches(mask, gap):
    """Where a row of true/false is true, joined over gaps shorter than `gap`."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((s, p + 1))
            s = i
        p = i
    out.append((s, p + 1))
    return out


def load_read():
    """The reading models, from RapidOCR's package, each checked; true once they're ready."""
    global reader
    t0 = time.time()
    spec = importlib.util.find_spec('rapidocr_onnxruntime')
    if spec is None or not spec.submodule_search_locations:
        state['read_note'] = "the reading models' package isn't here"
        say(state['read_note'])
        return False
    where = os.path.join(list(spec.submodule_search_locations)[0], 'models')
    paths = {}
    for k, (name, sha) in READ_FILES.items():
        p = os.path.join(where, name)
        if not os.path.isfile(p) or sha256(p) != sha:
            state['read_note'] = "the reading model %s isn't the one expected (its check didn't match)" % name
            say(state['read_note'])
            return False
        paths[k] = p
    r = Reader(paths['det'], paths['rec'])
    r.read(Image.new('RGB', (64, 32)))
    reader = r
    state['read_ready'] = True
    state['read_note'] = 'ready (loaded in %.1f s, 1 thread)' % (time.time() - t0)
    say('reading logos: ' + state['read_note'])
    return True


def read_side(query):
    """The longer side a picture is looked at with: a logo's (256), or ?side=N for a whole picture (256 to 1280, in steps
    of 32)."""
    for kv in query.split('&'):
        if kv.startswith('side='):
            try:
                n = int(kv[5:])
            except ValueError:
                raise ValueError('side')
            return max(READ_SIDE, min(READ_SIDE_MAX, int(round(n / 32)) * 32))
    return READ_SIDE


def read_logo(data, side=READ_SIDE):
    im = Image.open(io.BytesIO(data))
    if im.width * im.height > READ_PIXELS or min(im.width, im.height) < 8:
        raise ValueError('size')
    t0 = time.time()
    with read_lock:
        lines = reader.read(im, side)
    ms = (time.time() - t0) * 1000
    state['reads'] += 1
    state['read_ms'] = ms if state['reads'] == 1 else state['read_ms'] * 0.9 + ms * 0.1
    return {'read_model': READ_ID, 'lines': lines, 'read_ms': round(ms), 'side': side}


def axis(n_in):
    s = np.clip((np.arange(SIZE, dtype=np.float32) + 0.5) * (n_in / SIZE) - 0.5, 0, n_in - 1)
    i0 = np.floor(s).astype(int)
    return i0, np.minimum(i0 + 1, n_in - 1), (s - i0).astype(np.float32)


def prepare(rgb):
    """As the TV's Clip.prepare: bilinear to 224x224, each colour by CLIP's mean and spread; [1, 3, 224, 224]."""
    h, w, _ = rgb.shape
    y0, y1, fy = axis(h)
    x0, x1, fx = axis(w)
    a = rgb.astype(np.float32)
    top = a[y0][:, x0] * (1 - fx)[None, :, None] + a[y0][:, x1] * fx[None, :, None]
    bot = a[y1][:, x0] * (1 - fx)[None, :, None] + a[y1][:, x1] * fx[None, :, None]
    v = (top * (1 - fy)[:, None, None] + bot * fy[:, None, None]) / 255.0
    return ((v - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def spot_box(q):
    """The spot asked about ("x0,y0,x1,y1" in the picture's pixels), or None if there's none or it isn't one."""
    if not q:
        return None
    try:
        b = [int(v) for v in q.split(',')]
    except ValueError:
        return None
    return b if len(b) == 4 else None


def spot_look(rgb, box):
    """The spot, zoomed to fit 256x256 in its own proportions (black around it), to Apple's model."""
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1 - x0 >= 8 and y1 - y0 >= 8):
        return {}
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    k = SPOT_IN / max(x1 - x0, y1 - y0)
    cw, ch = max(1, round((x1 - x0) * k)), max(1, round((y1 - y0) * k))
    a = np.zeros((SPOT_IN, SPOT_IN, 3), np.uint8)
    a[(SPOT_IN - ch) // 2:(SPOT_IN - ch) // 2 + ch, (SPOT_IN - cw) // 2:(SPOT_IN - cw) // 2 + cw] = np.asarray(crop.resize((cw, ch), Image.BILINEAR))
    x = (a.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    t0 = time.time()
    with spot_lock:
        e = spot_session.run(['image_embeds'], {'pixel_values': x})[0][0]
    ms = (time.time() - t0) * 1000
    e = e / np.linalg.norm(e)
    state['spot_looks'] += 1
    state['spot_ms'] = ms if state['spot_looks'] == 1 else state['spot_ms'] * 0.95 + ms * 0.05
    return {'spot_model': SPOT_ID, 'spot_like': SPOT_LIKE, 'spot_ms': round(ms),
            'spot': base64.b64encode(e.astype('<f4').tobytes()).decode('ascii')}


def look(jpeg, box=None):
    im = Image.open(io.BytesIO(jpeg))
    if im.width * im.height > MAX_PIXELS:
        raise ValueError('too big')
    rgb = np.asarray(im.convert('RGB'))
    x = prepare(rgb)
    spot = spot_pool.submit(spot_look, rgb, box) if box and state['spot_ready'] else None  # at the same time, its own core
    t0 = time.time()
    with run_lock:
        e = session.run(['image_embeds'], {'pixel_values': x})[0][0]
    ms = (time.time() - t0) * 1000
    e = e / np.linalg.norm(e)
    state['looks'] += 1
    state['last_look'] = time.time()
    state['ms'] = ms if state['looks'] == 1 else state['ms'] * 0.95 + ms * 0.05
    out = {'model': MODEL_ID, 'score': float(e @ direction), 'ad': AD, 'like': LIKE, 'sure': SURE, 'ms': round(ms),
           'known': round(ads.known(e), 4), 'embedding': base64.b64encode(e.astype('<f4').tobytes()).decode('ascii')}
    if spot is not None:
        try:
            out.update(spot.result(timeout=5))
        except Exception:
            pass  # the main answer still counts
    return out


def meminfo(key):
    """A /proc/meminfo line's value in bytes (MemTotal, MemAvailable), or None."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith(key + ':'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def rss():
    """This app's memory in use now, in bytes."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def bench_get(path, url, size, sha):
    """A bench model downloaded and checked (a few tries); raises if it can't be."""
    for attempt in range(3):
        part = path + '.part'
        try:
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            if have > size:
                os.remove(part)
                have = 0
            req = urllib.request.Request(url, headers={'User-Agent': 'home-picture-ai', **({'Range': 'bytes=%d-' % have} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, 'ab' if have and r.status == 206 else 'wb') as out:
                shutil.copyfileobj(r, out, 1 << 20)
            if os.path.getsize(part) != size:
                raise IOError('it ended at %d MB' % (os.path.getsize(part) // 1_000_000))
            if sha256(part) != sha:
                os.remove(part)
                raise IOError("its check didn't match")
            os.replace(part, path)
            return
        except Exception as e:
            if attempt == 2:
                if os.path.isfile(part):
                    os.remove(part)
                raise IOError("didn't download (%s)" % (e if isinstance(e, IOError) else type(e).__name__))
            time.sleep(20)


def bench_quiet(r):
    """Waits till no TV has asked for a while (the bench never slows the TVs' answers); true if it was quiet."""
    if bench.get('busy'):  # (1.5.1) asked to time even while a TV asks (its times then a little high, and said so)
        return time.time() - state['last_look'] >= BENCH_QUIET
    t0 = time.time()
    while time.time() - state['last_look'] < BENCH_QUIET:
        if time.time() - t0 > BENCH_WAIT:
            return False
        r['now'] = 'waiting for the TVs to stop asking'
        time.sleep(2)
    return True


def bench_session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.add_session_config_entry('session.intra_op.allow_spinning', '0')
    return ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])


def bench_time(path, side, threads, r):
    """One number: the middle of the timed runs (ms) on this many cores, its load time (s) and memory (MB), and whether
    a TV asked meanwhile."""
    m0 = rss()
    t0 = time.time()
    s = bench_session(path, threads)
    load_s = time.time() - t0
    name = s.get_inputs()[0].name
    x = np.random.default_rng(1).random((1, 3, side, side), dtype=np.float32)
    s.run(None, {name: x})  # warms up
    mem = max(0, rss() - m0)
    times, start = [], time.time()
    looks = state['looks']
    while len(times) < BENCH_RUNS and (len(times) < 2 or time.time() - start < BENCH_BUDGET):
        r['now'] = 'timing on %d core%s (%d of %d)' % (threads, '' if threads == 1 else 's', len(times) + 1, BENCH_RUNS)
        t1 = time.time()
        s.run(None, {name: x})
        times.append((time.time() - t1) * 1000)
    busy = state['looks'] != looks
    del s
    times.sort()
    return round(times[len(times) // 2]), round(load_s, 1), round(mem / 1e6), busy


def bench_parallel(path, side, n, ms_1, r):
    """(1.5.1) Pictures a second with n pictures at once, one core each (the look ahead's head start lets pictures wait
    their turn), and whether a TV asked meanwhile."""
    r['now'] = 'timing %d pictures at once, one core each' % n
    sessions = [bench_session(path, 1) for _ in range(n)]
    name = sessions[0].get_inputs()[0].name
    xs = [np.random.default_rng(i).random((1, 3, side, side), dtype=np.float32) for i in range(n)]
    for k, s in enumerate(sessions):
        s.run(None, {name: xs[k]})
    rounds = max(2, min(20, round(BENCH_PAR_S * 1000 / max(1, ms_1))))
    looks = state['looks']

    def work(k):
        for _ in range(rounds):
            sessions[k].run(None, {name: xs[k]})

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(work, range(n)))
    wall = time.time() - t0
    del sessions
    return round(n * rounds / wall, 2), state['looks'] != looks


def bench_batch(path, side, n, threads, r):
    """(1.5.1) Pictures a second with n pictures in one batch on this many cores, or None if it takes one at a time."""
    s = bench_session(path, threads)
    inp = s.get_inputs()[0]
    if isinstance(inp.shape[0], int) and inp.shape[0] != n:
        return None, False
    x = np.random.default_rng(2).random((n, 3, side, side), dtype=np.float32)
    try:
        s.run(None, {inp.name: x})
    except Exception:
        return None, False
    times, start = [], time.time()
    looks = state['looks']
    while len(times) < 5 and (len(times) < 2 or time.time() - start < BENCH_BUDGET / 2):
        r['now'] = 'timing %d pictures in a batch on %d cores (%d of 5)' % (n, threads, len(times) + 1)
        t1 = time.time()
        s.run(None, {inp.name: x})
        times.append(time.time() - t1)
    times.sort()
    return round(n / times[len(times) // 2], 2), state['looks'] != looks


def rec_pictures(limit=200):
    """(1.5.1) The TVs' recorded pictures (what they send for a look, 320x180 JPEGs), newest minutes first, from Home
    media on this Pi (its recordings of his channels): for setting an 8-bit copy and checking it against the original."""
    out = []
    try:
        with urllib.request.urlopen(MEDIA + '/rec', timeout=10) as r:
            names = list(reversed(json.loads(r.read()).get('last') or []))
    except Exception:
        return out
    for n in names:
        try:
            with urllib.request.urlopen(MEDIA + '/rec/' + urllib.parse.quote(n), timeout=60) as r:
                data = r.read()
        except Exception:
            continue
        out.extend(rec_jpegs(data))
        if len(out) >= limit:
            break
    return out[:limit]


def rec_jpegs(data):
    """The 'F' pictures in a recording (Home's RecFile: a kind byte, the time, then the kind's own fields)."""
    out, i, n = [], 0, len(data)
    while i + 9 <= n:
        k = data[i]
        i += 9  # the kind and the time
        if k == 0x46:  # F: length, JPEG
            ln = int.from_bytes(data[i:i + 4], 'big')
            out.append(data[i + 4:i + 4 + ln])
            i += 4 + ln
        elif k == 0x47:  # G: width, height, length, JPEG (a full-size picture)
            ln = int.from_bytes(data[i + 4:i + 8], 'big')
            i += 8 + ln
        elif k == 0x41:  # A: the sound's level
            i += 2
        elif k == 0x53:  # S: peak, width, flatness, bands
            i += 6 + data[i + 5] if i + 6 <= n else n
        elif k == 0x45:  # E: a note
            i += 2 + int.from_bytes(data[i:i + 2], 'big')
        else:
            break
    return [j for j in out if j[:2] == b'\xff\xd8']


def bench_input(jpeg, side, norm):
    """A recorded picture as a model's input: squeezed to side x side, by its mean and spread."""
    mean, std = NORM[norm]
    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB').resize((side, side), Image.BILINEAR), np.float32) / 255.0
    return ((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def embed_name(s):
    names = [o.name for o in s.get_outputs()]
    for want in ('image_embeds', 'pooler_output'):
        if want in names:
            return want
    return names[0]


def make_q8(src, dst, mode, side, norm, pics, r):
    """(1.5.1) An 8-bit copy of a model, made here: 'dynamic' (its matrix math, nothing to set) or 'static' (all of it,
    set on recorded pictures)."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)  # the quantizer's advice, not for this log
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    if mode == 'dynamic':
        r['now'] = 'making its 8-bit copy'
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'])
        return
    if len(pics) < 16:
        raise IOError('too few recorded pictures to set it on (%d)' % len(pics))

    class Reader(CalibrationDataReader):
        def __init__(self, name):
            self.left = [bench_input(j, side, norm) for j in pics[:BENCH_CALIBRATE]]
            self.name = name

        def get_next(self):
            r['now'] = 'setting its 8-bit copy on recorded pictures (%d left)' % len(self.left)
            return {self.name: self.left.pop()} if self.left else None

    name = bench_session(src, 1).get_inputs()[0].name
    quantize_static(src, dst, Reader(name), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)


def fidelity(src, dst, side, norm, pics, r):
    """(1.5.1) How like the original an 8-bit copy sees recorded pictures (it never set on): the mean and the least
    cosine of their embeddings."""
    held = pics[BENCH_CALIBRATE:BENCH_CALIBRATE + 16] or pics[:16]
    if not held:
        return None, None
    a, b = bench_session(src, THREADS_ALL), bench_session(dst, THREADS_ALL)
    na, nb = a.get_inputs()[0].name, b.get_inputs()[0].name
    oa, ob = embed_name(a), embed_name(b)
    cos = []
    for k, j in enumerate(held):
        r['now'] = 'checking its 8-bit copy against the original (%d of %d)' % (k + 1, len(held))
        x = bench_input(j, side, norm)
        ea = a.run([oa], {na: x})[0].reshape(-1)
        eb = b.run([ob], {nb: x})[0].reshape(-1)
        cos.append(float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9)))
    return round(float(np.mean(cos)), 4), round(float(np.min(cos)), 4)


def bench_run(which='one'):
    """Every bench model in turn: downloaded (or made), timed, deleted (the TVs' own models only timed)."""
    try:
        bench_models(which)
    except Exception as e:
        bench['note'] = 'stopped (%s)' % type(e).__name__
        say('the bench: ' + bench['note'])
    finally:
        if bench['state'] == 'running':
            bench['state'] = 'stopped'


def bench_view():
    """The bench as it is now, for an answer (copied while it may be changing)."""
    for _ in range(10):
        try:
            return json.loads(json.dumps(bench))
        except RuntimeError:
            time.sleep(0.05)
    return {'state': bench['state'], 'note': 'busy: ask again'}


def bench_models(which='one'):
    with bench_lock:
        models = BENCH_ONE if which == 'one' else [m + ('unit', None) for m in BENCH]
        bench.update({'state': 'running', 'note': '', 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': [], 'set': which,
                      'memory_mb': round((meminfo('MemTotal') or 0) / 1e6), 'cores': os.cpu_count()})
        say('the bench (%s): started (%d models)' % (which, len(models)))
        made, pics = {}, None
        paths = {}
        for mid, what, src, size, sha, side, norm, make in models:
            r = {'id': mid, 'what': what, 'size_mb': round(size / 1e6), 'note': '', 'now': 'waiting'}
            bench['models'].append(r)
            path = MODEL if mid == MODEL_ID else SPOT_MODEL if mid == SPOT_ID else os.path.join(DATA, 'bench-%s.onnx' % mid)
            try:
                if make:
                    base = paths.get(src)
                    if not base or not os.path.isfile(base):
                        r['note'] = "not timed: the original wasn't here"
                        continue
                    if pics is None:
                        r['now'] = 'reading recorded pictures'
                        pics = rec_pictures()
                        bench['pictures'] = len(pics)
                    t0 = time.time()
                    make_q8(base, path, make, side, norm, pics, r)
                    made[mid] = path
                    r['make_s'] = round(time.time() - t0)
                    r['size_mb'] = round(os.path.getsize(path) / 1e6)
                    r['fidelity'], r['fidelity_least'] = fidelity(base, path, side, norm, pics, r)
                elif src:
                    free = shutil.disk_usage(DATA).free
                    if free < size + BENCH_ROOM:
                        r['note'] = 'not timed: no room for it (%d MB free)' % (free // 1_000_000)
                        continue
                    r['now'] = 'downloading'
                    t0 = time.time()
                    bench_get(path, src, size, sha)
                    r['download_s'] = round(time.time() - t0)
                    made[mid] = path
                elif not os.path.isfile(path):
                    r['note'] = 'not timed: not downloaded yet'
                    continue
                paths[mid] = path
                avail = meminfo('MemAvailable')
                fsize = os.path.getsize(path)
                if avail is not None and avail < fsize * (1.6 if which == 'all' else 5) + (300 << 20):
                    r['note'] = 'not timed: too little memory free (%d MB)' % (avail // 1_000_000)
                    continue
                busy = False
                for threads in ((1, THREADS) if which == 'all' else (1, THREADS_ALL)):
                    quiet = bench_quiet(r)
                    ms, load_s, mem, asked = bench_time(path, side, threads, r)
                    r['ms_%d' % threads] = ms
                    r['load_s'] = load_s
                    r['memory_mb'] = max(r.get('memory_mb', 0), mem)
                    busy = busy or asked or not quiet
                if which == 'all':
                    r['per_second'] = round(1000 / max(1, r['ms_%d' % THREADS]), 1)
                else:
                    modes = {'one at a time on %d cores' % THREADS_ALL: round(1000 / max(1, r['ms_%d' % THREADS_ALL]), 2)}
                    quiet = bench_quiet(r)
                    ps, asked = bench_parallel(path, side, THREADS_ALL, r['ms_1'], r)
                    busy = busy or asked or not quiet
                    r['at_once_per_second'] = ps
                    modes['%d at once, a core each' % THREADS_ALL] = ps
                    quiet = bench_quiet(r)
                    bs, asked = bench_batch(path, side, THREADS_ALL, THREADS_ALL, r)
                    busy = busy or asked or not quiet
                    if bs is not None:
                        r['batch_per_second'] = bs
                        modes['%d in a batch on %d cores' % (THREADS_ALL, THREADS_ALL)] = bs
                    best = max(modes, key=modes.get)
                    r['per_second'], r['best'] = modes[best], best
                    r['every_picture'] = r['per_second'] >= ONE_TV
                if busy:
                    r['note'] = 'a TV asked while it was timed (its times may be a little high)'
                say('the bench: %s, %d ms on 1 core, %.2f pictures a second%s' % (
                    what, r['ms_1'], r['per_second'], (' (%s)' % r['best']) if 'best' in r else ''))
            except Exception as e:
                r['note'] = 'not timed: %s' % (e if isinstance(e, IOError) else type(e).__name__)
                say('the bench: %s %s' % (what, r['note']))
            finally:
                r.pop('now', None)
        for path in made.values():  # only for the bench: not kept
            for f in (path, path + '.part'):
                if os.path.isfile(f):
                    os.remove(f)
        bench['state'] = 'done'
        timed = [m for m in bench['models'] if 'per_second' in m]
        bench['note'] = '%d of %d timed' % (len(timed), len(models))
        say('the bench: done (%s)' % bench['note'])


def home_network(addr):
    try:
        ip = ipaddress.ip_address(addr)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'home-picture-ai'

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        if self.path.split('?')[0] == '/status':
            return self.reply(200, {'model': MODEL_ID, 'version': VERSION, 'ready': state['ready'], 'note': state['note'], 'looks': state['looks'],
                                    'ms': round(state['ms']), 'up': round(time.time() - state['since']), 'spot_model': SPOT_ID,
                                    'spot_ready': state['spot_ready'], 'spot_note': state['spot_note'], 'spot_looks': state['spot_looks'],
                                    'spot_ms': round(state['spot_ms']), 'read_model': READ_ID, 'read_ready': state['read_ready'],
                                    'read_note': state['read_note'], 'reads': state['reads'], 'read_ms': round(state['read_ms']),
                                    'read_side_max': READ_SIDE_MAX, 'ads': ads.n, 'ads_note': ads.note,
                                    'bench': bench['state'] + (' (%s)' % bench['note'] if bench['note'] else '')})
        if self.path.split('?')[0] == '/bench':
            return self.reply(200, bench_view())
        self.reply(404, {'note': 'GET /status, GET /bench, POST /look, POST /read, POST /ads or POST /bench'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        path, _, query = self.path.partition('?')
        if path == '/bench':
            if bench['state'] == 'running':
                return self.reply(409, {'note': 'the bench is already running', 'bench': bench_view()})
            if not state['ready']:
                return self.reply(503, {'note': 'not until the main model is ready'})
            which = 'all' if 'set=all' in query.split('&') else 'one'
            bench['state'] = 'running'
            bench['busy'] = 'busy=1' in query.split('&')
            threading.Thread(target=bench_run, args=(which,), daemon=True).start()
            return self.reply(202, {'note': 'the bench (%s) started: GET /bench says how far it is' % which})
        if path == '/ads':
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > 1_000_000:
                return self.reply(400, {'note': "a break's ads, up to 1 MB"})
            body = self.rfile.read(n)
            try:
                return self.reply(200, learn_ads(body))
            except LookupError:
                return self.reply(409, {'note': 'these were seen by another model (this one is %s)' % MODEL_ID})
            except Exception as e:
                return self.reply(400, {'note': 'not ads it could keep (%s)' % type(e).__name__})
        if path == '/read':
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > 4_000_000:
                return self.reply(400, {'note': 'a PNG or JPEG, up to 4 MB'})
            body = self.rfile.read(n)
            if not state['read_ready']:
                return self.reply(503, {'note': state['read_note']})
            try:
                side = read_side(query)
            except ValueError:
                return self.reply(400, {'note': 'side: a number of points, 256 to 1280'})
            try:
                return self.reply(200, read_logo(body, side))
            except Exception as e:
                return self.reply(400, {'note': 'not a picture it could read (%s)' % type(e).__name__})
        if path != '/look':
            return self.reply(404, {'note': 'POST /look, POST /read or POST /ads'})
        box = None
        for kv in query.split('&'):
            if kv.startswith('spot='):
                box = spot_box(kv[5:])
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > 4_000_000:
            return self.reply(400, {'note': 'a JPEG, up to 4 MB'})
        body = self.rfile.read(n)
        if not state['ready']:
            return self.reply(503, {'note': state['note']})
        try:
            return self.reply(200, look(body, box))
        except Exception as e:
            return self.reply(400, {'note': 'not a picture it could read (%s)' % type(e).__name__})

    def log_message(self, *args):
        pass  # no log of who asked what


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.0.1', 9))  # no packet is sent; it picks the home network's address
        return s.getsockname()[0]
    finally:
        s.close()


def announce():
    """On the home network as _homeai._tcp (mDNS), so the TVs find it without being told where; again if its address
    moves. Only on the home network's side (not Home Assistant's own inside networks)."""
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        say('no mDNS library: the TVs will need its address')
        return
    zc, info, ip = None, None, None
    host = socket.gethostname().split('.')[0] or 'home-picture-ai'
    while True:
        try:
            now = lan_ip()
            if now != ip:
                if zc:
                    zc.unregister_all_services()
                    zc.close()
                    zc = None
                ip = now
                zc = Zeroconf(interfaces=[ip])
                info = ServiceInfo('_homeai._tcp.local.', 'Home picture AI._homeai._tcp.local.',
                                   addresses=[socket.inet_aton(ip)], port=PORT,
                                   properties={'model': MODEL_ID, 'version': VERSION}, server=host + '.local.')
                zc.register_service(info, allow_name_change=True)
                say('announced on the network at %s:%d' % (ip, PORT))
        except Exception as e:
            say('announcing failed (%s); trying again in a minute' % type(e).__name__)
            ip = None
        time.sleep(60)


def main():
    srv = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    say('Home picture AI %s, answering on port %d' % (VERSION, PORT))
    ads.load()
    threading.Thread(target=announce, daemon=True).start()
    wait = 30
    while True:  # never gives up: a problem is said in the log and tried again, later each time
        try:
            fetch(MODEL, MODEL_URL, MODEL_SIZE, MODEL_SHA)
            if load():
                break
        except Exception as e:
            state['note'] = "the model didn't load (%s); trying again in %d s" % (type(e).__name__, wait)
            say(state['note'])
            time.sleep(wait)
            wait = min(wait * 2, 1800)
    wait = 30
    while True:  # the logo's model after the main one (the TVs need that first); the main one answers meanwhile
        try:
            fetch(SPOT_MODEL, SPOT_URL, SPOT_SIZE, SPOT_SHA, 'spot_note', "the logo's model")
            if load_spot():
                break
        except Exception as e:
            state['spot_note'] = "the logo's model didn't load (%s); trying again in %d s" % (type(e).__name__, wait)
            say(state['spot_note'])
            time.sleep(wait)
            wait = min(wait * 2, 1800)
    try:  # reading logos (in the app itself, nothing to download)
        load_read()
    except Exception as e:
        state['read_note'] = "the reading models didn't load (%s)" % type(e).__name__
        say(state['read_note'])
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
