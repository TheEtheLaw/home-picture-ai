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
are kept in the app's own storage. Since 1.9.0 each channel has its own lesson: what the AI saw of a channel's pictures
(its numbers, never the pictures) is kept 21 days, and each night a lean is learned from the channel's own real breaks
and tested on a day it didn't learn from before it's used (see DOCS.md). Announced on the network (mDNS, _homeai._tcp)
so the TVs find it.

    POST /look[?spot=x0,y0,x1,y1]   a JPEG  ->  {"model", "score", "embedding" (512 float32, base64), "ad", "like",
                                                "sure", "known", "ms"}, and for a spot {"spot_model", "spot", "spot_like",
                                                "spot_ms"}
    POST /ads                       {"model", "ads": [[scale, 512 int8 (base64)], ...]}  ->  {"new", "again", "ads"}
    POST /ads/forget                {"model", "shows": [[scale, 512 int8 (base64)], ...], "logo": "goes"}  ->  {"forgot",
                                                "ads"} (1.7.0: the show's own pictures; remembered ads like them let go;
                                                1.7.2: only with "logo": "goes", a logo the TV saw go away in breaks)
    POST /read[?side=N][&signs=1]   a PNG or JPEG (a logo's spot; with side, a whole picture looked at with its longer
                                                side N, 256 to 1280)  ->  {"read_model", "lines" [{"text", "score",
                                                "box"}], "read_ms", "side"}, and with signs=1 (1.6.0) "signs" (an ad's
                                                words: "url", "phone", "price", "words", "small", "ad", "why")
    GET  /status                            ->  {"model", "version", "ready", "note", "looks", "ms", "up", "spot_model",
                                                "spot_ready", "spot_note", "spot_looks", "spot_ms", "read_model",
                                                "read_ready", "read_note", "reads", "read_ms", "read_side_max", "ads",
                                                "ads_note", "bench"}
    POST /bench[?set=all|read]              ->  starts the bench (1.5.0): bigger models timed here, one at a time;
                                                set=read (1.6.0): the screen's words on the recordings
    GET  /bench                             ->  {"state", "note", "at", "memory_mb", "models": [{"id", "what", "size_mb",
                                                "ms_1", "ms_3", "per_second", "load_s", "memory_mb", "note"}, ...]}
    POST /look?ch=NAME&t=MS[&replay=1]      (1.9.0) the same, the channel's lesson in it ("lesson": {"score", "line",
                                                "from", "held"}) while one is in use; what the AI saw kept by channel
    POST /lessons/event[?replay=1]          {"channel", "kind": "run"|"screen"|"show", ...}  ->  {"kept"}
    GET  /lessons                           ->  each channel's lesson: learned, tested, used or not, why; what's kept
    POST /lessons/learn[?before=MS][&channels=A,B]  ->  learns now

Since 1.5.0 it can time bigger models on this Pi, for the ad AI's next model (the bench, started from the home
network): each is downloaded (a fixed version, checked by its SHA-256), timed on 1 core and on 3 while no TV is
asking, and deleted again; the models the TVs use now are timed too, to compare. Since 1.6.0 it can say what a whole
picture's lines say of an ad (web addresses, phone numbers, prices, fine print), and the bench can read the pictures of
his recordings (on Home media, on this Pi) to see how often each is there in the breaks and in the programs.
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
import re
import shutil
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

VERSION = '1.9.0'
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
# (1.7.0) The show's own pictures (a TV's, with the channel's logo on) let go of the remembered ads ADS_SAME or more like
# them, up to ADS_FORGET_EACH a picture: a break that wasn't one (the AI taking a show for ads) taught the memory the show.
ADS_FORGET_EACH = 32
# (1.7.1) Paused: a channel can keep its logo on through its ads (on FS1 the logo in the bottom bar stays), so a TV's
# "show pictures" with the logo on can be real ads. (1.7.2) On again, only for a TV that says its pictures' logo is one
# it has seen go away in breaks ("logo": "goes", Home 2.08: its being on is the show's); others let nothing go.
ADS_FORGET_ON = True

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
        self.dead = 0       # (1.7.0) ones let go (their scale 0: never alike again), and whether that's written yet
        self.dirty = False
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
            self.note = '%d ads' % (self.n - self.dead)
            self._next()
            return new, again, self.n - self.dead

    def forget(self, vs):
        """(1.7.0) The show's own pictures from a TV (the channel's logo on): every remembered ad ADS_SAME or more like
        one is let go (its scale 0, so nothing is alike it again), written whole soon after. How many went, how many
        are left."""
        gone = 0
        with self.lock:
            for v in vs:
                for _ in range(ADS_FORGET_EACH):
                    sim, i = self._best(v)
                    if i < 0 or sim < ADS_SAME:
                        break
                    self.s[i] = 0.0
                    gone += 1
            if gone:
                self.dead += gone
                self.dirty = True
                self.note = '%d ads' % (self.n - self.dead)
                self._next()
            return gone, self.n - self.dead

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
                self.dead = int((self.s[:self.n] == 0).sum())
                self.note = '%d ads' % (self.n - self.dead)
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
        elif self.dirty or self.n >= ADS_CEILING or (os.path.exists(self.logp) and os.path.getsize(self.logp) > ADS_LOG_MAX):
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
        live = np.flatnonzero(s[:n0] > 0)  # (1.7.0) not the ones let go
        pick = np.sort(rng.choice(live, size=min(len(live), 16 * k), replace=False))
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
            self.dirty = False  # what's let go is in this copy
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


def fingerprints(j, key):
    """[[scale, 512 int8 (base64)], ...] under key, each of length 1."""
    vs = []
    for item in (j.get(key) or [])[:ADS_BATCH]:
        scale, b64 = float(item[0]), item[1]
        q = np.frombuffer(base64.b64decode(b64), np.int8)
        if len(q) != 512 or not np.isfinite(scale) or scale <= 0:
            raise ValueError('not a fingerprint')
        v = q.astype(np.float32) * scale
        n = np.linalg.norm(v)
        if n > 0:
            vs.append(v / n)
    return vs


def learn_ads(body):
    """POST /ads: {"model", "ads": [[scale, 512 int8 (base64)], ...]}."""
    j = json.loads(body)
    if j.get('model') != MODEL_ID:
        raise LookupError('another model')
    new, again, total = ads.learn(fingerprints(j, 'ads'), int(time.time() // 86400))
    return {'new': new, 'again': again, 'ads': total}


def forget_ads(body):
    """(1.7.0) POST /ads/forget: {"model", "shows": [[scale, 512 int8 (base64)], ...]}: the show's own pictures."""
    j = json.loads(body)
    if j.get('model') != MODEL_ID:
        raise LookupError('another model')
    shows = fingerprints(j, 'shows')
    if not ADS_FORGET_ON:
        return {'forgot': 0, 'ads': ads.n - ads.dead, 'note': 'letting go is paused for now (1.7.1)'}
    if j.get('logo') != 'goes':
        return {'forgot': 0, 'ads': ads.n - ads.dead,
                'note': 'only pictures whose logo the TV has seen go away in breaks let go of ads (a TV before Home 2.08)'}
    gone, left = ads.forget(shows)
    if gone:
        say('the ads remembered: %d let go, the show\'s own pictures from a TV (%d left)' % (gone, left))
    return {'forgot': gone, 'ads': left}


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

    def __init__(self, det_path, rec_path, threads=1):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
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
            box = (int(x0 * sx), int(y0 * sy), min(w, int(np.ceil(x1 * sx))), min(h, int(np.ceil(y1 * sy))))
            text, score = self.read_line(rgb.crop(box))
            if text.strip():  # (1.6.0) where the line is, in parts of the picture's width and height
                out.append({'text': text.strip(), 'score': round(score, 3),
                            'box': [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]})
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


def read_paths():
    """The reading models' files, from RapidOCR's package, each checked by its SHA-256; raises why if one can't be used."""
    spec = importlib.util.find_spec('rapidocr_onnxruntime')
    if spec is None or not spec.submodule_search_locations:
        raise LookupError("the reading models' package isn't here")
    where = os.path.join(list(spec.submodule_search_locations)[0], 'models')
    paths = {}
    for k, (name, sha) in READ_FILES.items():
        p = os.path.join(where, name)
        if not os.path.isfile(p) or sha256(p) != sha:
            raise LookupError("the reading model %s isn't the one expected (its check didn't match)" % name)
        paths[k] = p
    return paths


def load_read():
    """The reading models, from RapidOCR's package, each checked; true once they're ready."""
    global reader
    t0 = time.time()
    try:
        paths = read_paths()
    except LookupError as e:
        state['read_note'] = str(e)
        say(state['read_note'])
        return False
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


# ---- (1.6.0) An ad's words: what a whole picture's lines say that ads say and programs seldom do ----
# Read from a whole picture (POST /read?side=N&signs=1, and the bench on the recordings): web addresses, phone numbers,
# prices, and fine print (the small lines at the bottom: terms, offers, a drug's or a car's small print). Each is
# counted and kept (its words) so a TV can weigh them; `ad` is whether any of the strong ones is there.
SIGN_SCORE = 0.70        # a line read less surely than this isn't used
SIGN_SMALL = 0.036       # a line's box under this much of the picture's height is small print (a 1080 picture: 39 points)
SIGN_SMALL_CHARS = 12    # small print: a line of at least this many letters
SIGN_KEPT = 4            # the signs' words kept for each kind, in an answer
TLD = ('com|net|org|gov|edu|tv|co|io|us|info|biz|app|shop|store|health|care|law|bank|insurance|news|fm|ly|me|ca|travel'
       '|online|site|website|club|fun|live|life|today|tech|ai|auto|cars|deals|homes')
URL_RE = re.compile(r'(?<![a-z0-9@.-])(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9-]{1,40}(?:\.[a-z0-9-]{2,40})?\.(?:' + TLD +
                    r')(?![a-z0-9-])(?:/[a-z0-9/_-]{1,40})?')
PHONE_RE = re.compile(r'(?<![0-9])(?:1[\s.-]?)?\(?8(?:00|33|44|55|66|77|88)\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}(?![0-9])'
                      r'|(?<![0-9])(?:1[\s.-]?8(?:00|33|44|55|66|77|88)[\s.-]|8(?:00|33|44|55|66|77|88)-)[a-z][a-z0-9-]{2,}'
                      r'|(?<![0-9])\(?[2-9][0-9]{2}\)?[\s.-][0-9]{3}[\s.-][0-9]{4}(?![0-9])'
                      r'|\btext\s+[a-z0-9]{2,12}\s+to\s+[0-9]{5,6}\b')
PRICE_RE = re.compile(r'\$\s?[0-9][0-9,]*(?:\.[0-9]{2})?(?:\s?(?:/|per\s|a\s)(?:mo|month|yr|year|day|week)\b)?'
                      r'|\b[0-9]+\.[0-9]{2}\s?/\s?mo\b|\b[0-9]+(?:\.[0-9]+)?\s?%\s?(?:apr|apy|off)\b|\bmsrp\b|\bapr\b')
FINE_RE = re.compile(r'terms (?:and|&) conditions|terms apply|restrictions apply|conditions apply|exclusions apply'
                     r'|see (?:store|dealer|site|website|details|terms|offer|full)|for (?:full )?details|offer (?:valid|ends|expires|good|available)'
                     r'|limited time|while supplies last|subject to|not available in all|eligib|qualified|credit approval'
                     r'|participating|results (?:may )?vary|individual results|professional driver|closed course|do not attempt'
                     r'|dramati[sz]ation|simulated|actor portrayal|paid (?:actor|endorsement|spokesperson)|compensated'
                     r'|ask your doctor|side effects|prescription|available on the app store|google play|member fdic'
                     r'|equal housing|nmls|underwritten by|all rights reserved|free shipping|order now|call now|shop now'
                     r'|learn more at|visit us at|apply now|sign up at|download the app|data rates may apply')
OCR_FIX = [(re.compile(r'\s*[.,]\s*(c[o0]m|net|org|gov|edu)(?![a-z])'), r'.\1'), (re.compile(r'\.c0m'), '.com'),
           (re.compile(r'w\s?w\s?w\s?[.,]\s?'), 'www.')]


def sign_text(text):
    """A line's text as the signs look at it: small letters, the reader's usual slips around web addresses mended."""
    t = ' '.join(text.lower().split())
    for rx, to in OCR_FIX:
        t = rx.sub(to, t)
    return t


def ad_signs(lines):
    """(1.6.0) What a whole picture's lines say of an ad: web addresses, phone numbers, prices, fine print's words, small
    lines; `ad` if any web address, phone number or price is there, or fine print's words (two, or one with small lines)."""
    out = {'url': [], 'phone': [], 'price': [], 'words': [], 'small': 0, 'lines': 0}
    seen = set()
    for ln in lines:
        if float(ln.get('score', 0)) < SIGN_SCORE:
            continue
        t = sign_text(str(ln.get('text', '')))
        if not t:
            continue
        out['lines'] += 1
        box = ln.get('box')
        small = bool(box) and len(box) == 4 and (box[3] - box[1]) < SIGN_SMALL and len(t) >= SIGN_SMALL_CHARS
        if small:
            out['small'] += 1
        for kind, rx in (('url', URL_RE), ('phone', PHONE_RE), ('price', PRICE_RE), ('words', FINE_RE)):
            for m in rx.finditer(t):
                w = m.group(0).strip()
                if (kind, w) not in seen:
                    seen.add((kind, w))
                    out[kind].append(w)
    out['ad'] = bool(out['url'] or out['phone'] or out['price'] or (out['words'] and out['small']) or len(out['words']) >= 2)
    out['why'] = ', '.join(k for k in ('url', 'phone', 'price', 'words') if out[k]) + (', small print' if out['small'] else '')
    for k in ('url', 'phone', 'price', 'words'):
        out[k] = out[k][:SIGN_KEPT]
    return out


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


def look(jpeg, box=None, ch=None, t=None, replay=False):
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
    key = lesson_key(ch)
    if key:  # (1.9.0) the channel's: what the AI saw kept for its lesson (not a replay's), and its lesson's word
        if t is not None and not replay:
            try:
                looklog.add(key, t, e)
            except Exception as ex:
                state['lesson_note'] = "couldn't keep what it saw (%s)" % type(ex).__name__
        la = lesson_answer(key, e)
        if la is not None:
            out['lesson'] = la
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
    names = list(reversed(rec_names()))
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
    out = [data[a:a + ln] for _, _, (a, ln) in rec_records(data, 'F')]
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
    """Every bench model in turn: downloaded (or made), timed, deleted (the TVs' own models only timed); or (1.6.0,
    set=read) the screen's words on the recordings."""
    try:
        if which == 'read':
            bench_read()
        elif isinstance(which, dict):
            score_run(which)
        else:
            bench_models(which)
    except Exception as e:
        bench['note'] = 'stopped (%s)' % (e if isinstance(e, (IOError, RuntimeError)) else type(e).__name__)
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


# ---- (1.6.0) The bench on the recordings: what the screen's words say in the breaks and in the programs ----
READ_BENCH_N = 120         # pictures read from the breaks, and as many from the programs (at most)
READ_BENCH_THREADS = 2     # the bench's reader: 2 of the Pi's cores (the TVs keep the others)
READ_BENCH_SIDE = 1280     # read as a TV would read a whole picture
READ_BENCH_TIMED = 6       # pictures timed on the TVs' own reader (1 core), at 960 and at 1280
BREAK_INSIDE = 2_000       # a break's picture: this far inside its screen at least (ms)
PROGRAM_CLEAR = 30_000     # a program's picture: this far from any break's screen at least
TEST_SCREEN = 10_000       # the PC's test screen is up this long
BACK_CLEAR = 120_000       # a break screen taken down with Back: nothing near it counted as either
SCREEN_RE = re.compile(r'^(.*?): the break screen (on|off): (.*)$')
MOMENT_RE = re.compile(r'\((?:at its moment, ([0-9.]+) s ahead of the screen|its moment passed ([0-9.]+) s ago)\)$')
PLANNED_RE = re.compile(r'planned for a picture ([0-9.]+) s ahead of the screen')
EDGE_ON_RE = re.compile(r'; on at (the [^,;]*?), ([0-9.]+) s before its first ad picture')
EDGE_OFF_RE = re.compile(r'; off at (the [^;(]*?)(?: \(|$|;)')


def rec_names():
    """(1.6.0) Every recording on Home media, oldest first (its list since 1.0.8; an older one says its newest five)."""
    try:
        with urllib.request.urlopen(MEDIA + '/rec?all=1', timeout=30) as r:
            return [f['name'] for f in json.loads(r.read()).get('files') or []]
    except urllib.error.HTTPError:
        pass
    except Exception:
        return []
    try:
        with urllib.request.urlopen(MEDIA + '/rec', timeout=10) as r:
            return list(json.loads(r.read()).get('last') or [])
    except Exception:
        return []


def rec_get(name):
    with urllib.request.urlopen(MEDIA + '/rec/' + urllib.parse.quote(name), timeout=120) as r:
        return r.read()


def rec_records(data, kinds='FGE'):
    """(1.6.0) A recording's records in order (Home's RecFile: a kind byte, the time, the kind's own fields), the kinds
    asked for: ('F', t, (start, length)) a picture the TV looked at, ('G', t, (start, length, width, height)) a full-size
    one (where its JPEG is in the data), ('E', t, text) a note. Stops at anything it doesn't know."""
    i, n = 0, len(data)
    while i + 9 <= n:
        k = data[i]
        t = int.from_bytes(data[i + 1:i + 9], 'big', signed=True)
        i += 9
        if k == 0x46:  # F: length, JPEG
            ln = int.from_bytes(data[i:i + 4], 'big')
            if 'F' in kinds:
                yield 'F', t, (i + 4, ln)
            i += 4 + ln
        elif k == 0x47:  # G: width, height, length, JPEG
            ln = int.from_bytes(data[i + 4:i + 8], 'big')
            if 'G' in kinds:
                yield 'G', t, (i + 8, ln, int.from_bytes(data[i:i + 2], 'big'), int.from_bytes(data[i + 2:i + 4], 'big'))
            i += 8 + ln
        elif k == 0x41:  # A: the sound's level
            i += 2
        elif k == 0x53:  # S: peak, width, flatness, bands
            i += 6 + data[i + 5] if i + 6 <= n else n
        elif k == 0x45:  # E: a note
            ln = int.from_bytes(data[i:i + 2], 'big')
            if 'E' in kinds:
                yield 'E', t, data[i + 2:i + 2 + ln].decode('utf-8', 'replace')
            i += 2 + ln
        else:
            break


def screen_moment(t, rest):
    """When a break screen note's change was seen: its moment ahead of (or passed before) the note, else at once."""
    m = MOMENT_RE.search(rest)
    if not m:
        return t
    return t + int(float(m.group(1)) * 1000) if m.group(1) else t - int(float(m.group(2)) * 1000)


def edge_kind(rest, on):
    """(2.05's notes) Where a break screen's change was put: to a cut (by the logo, the sound, the only one, an end card),
    by a silence or black picture, or at the AI's picture as before."""
    m = (EDGE_ON_RE if on else EDGE_OFF_RE).search(rest)
    if m:
        w = m.group(1)
        return ('logo' if 'logo' in w else 'sound' if 'sound' in w else 'only cut' if 'only cut' in w
                else 'end card' if 'end card' in w else 'cut'), (float(m.group(2)) if on else None)
    low = rest.lower()
    if 'silen' in low or 'black' in low:
        return 'silence or black', None
    return 'picture', None


def rec_breaks(notes):
    """The breaks in the recordings' notes [(time, text)], in order: the screens' spans [(on, off, channel)], the spans
    to leave out (the PC's tests, Back), where each change was put since 2.05, and the channels' changes."""
    spans, skip, edges = [], [], {'on': {}, 'off': {}, 'before_s': []}
    on_at, on_ch, cur = None, None, None
    for t, text in notes:
        if text.startswith('channel '):  # each minute's recording starts with it
            ch = text[len('channel '):]
            if ch != cur and on_at is not None:  # another channel with a break on: that break ends there
                spans.append((on_at, t, on_ch))
                on_at = None
            cur = ch
            continue
        if 'the break screen taken down (Back)' in text:
            skip.append((t - BACK_CLEAR, t + BACK_CLEAR))
            continue
        m = SCREEN_RE.match(text)
        if not m:
            if ': left after ' in text and on_at is not None:
                spans.append((on_at, t, on_ch))
                on_at = None
            continue
        ch, which, rest = m.group(1), m.group(2), m.group(3)
        if rest.startswith('a test from the PC'):
            if 'not now' not in rest:
                p = PLANNED_RE.search(rest)
                at = t + (int(float(p.group(1)) * 1000) if p else 0)
                skip.append((at - BREAK_INSIDE, at + TEST_SCREEN + BREAK_INSIDE))
            continue
        at = screen_moment(t, rest)
        kind, before = edge_kind(rest, which == 'on')
        edges[which][kind] = edges[which].get(kind, 0) + 1
        if before is not None and len(edges['before_s']) < 200:
            edges['before_s'].append(round(before, 2))
        if which == 'on':
            if on_at is None:
                on_at, on_ch = at, ch
        elif on_at is not None:
            spans.append((on_at, max(at, on_at), on_ch))
            on_at = None
    return spans, skip, edges, on_at


def pick_even(items, n):
    """N of the items, spread evenly over them (all of them if there are no more)."""
    if len(items) <= n:
        return list(items)
    return [items[int(i * len(items) / n)] for i in range(n)]


HEIGHTS = (0.02, 0.025, 0.03, 0.036, 0.045, 0.06)  # the long lines' heights counted between these (parts of the picture)


def read_bench_add(stats, label, signs, ms, lines=()):
    s = stats[label]
    s['read'] += 1
    s['ms'].append(ms)
    for ln in lines:  # how high its long lines are (small print's line, set from these)
        b = ln.get('box')
        if b and float(ln.get('score', 0)) >= SIGN_SCORE and len(sign_text(str(ln.get('text', '')))) >= SIGN_SMALL_CHARS:
            k = sum(1 for x in HEIGHTS if b[3] - b[1] >= x)
            s['heights'][k] += 1
    if signs['lines']:
        s['with_words'] += 1
    for k in ('url', 'phone', 'price', 'words'):
        if signs[k]:
            s[k] += 1
            for w in signs[k]:
                ex = s['examples'].setdefault(k, [])
                if w not in ex and len(ex) < 25:
                    ex.append(w)
    if signs['small']:
        s['small'] += 1
    if signs['ad']:
        s['ad'] += 1


def read_bench_view(stats):
    out = {}
    for label, s in stats.items():
        n = max(1, s['read'])
        ms = sorted(s['ms'])
        out[label] = {'read': s['read'], 'share': {k: round(s[k] / n, 3) for k in ('ad', 'url', 'phone', 'price', 'words', 'small', 'with_words')},
                      'long_lines_by_height': dict(zip(['under 2%'] + ['%g%% up' % (x * 100) for x in HEIGHTS], s['heights'])),
                      'ms': {'mean': round(sum(ms) / len(ms)) if ms else None, 'p90': ms[int(len(ms) * 0.9)] if ms else None},
                      'examples': s['examples']}
    return out


def bench_read():
    """(1.6.0) The screen's words on his recordings: pictures from the breaks (where the TV's break screen was up) and
    from the programs (well away from any) read as a TV reads a whole picture, and how often each ad sign is there in
    each; where 2.05's break screens were put (to a cut by the logo, the sound, the only cut; by a silence or black; at
    the AI's picture); the TVs' own reading timed. Numbers and the signs' words only: no picture is kept."""
    with bench_lock:
        bench.update({'state': 'running', 'note': '', 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': [], 'set': 'read',
                      'memory_mb': round((meminfo('MemTotal') or 0) / 1e6), 'cores': os.cpu_count()})
        r = bench['read'] = {'now': 'listing the recordings'}
        say('the bench (read): started')
        if not state['read_ready']:
            raise RuntimeError('the reading models are not ready')
        names = rec_names()
        r.update({'recordings': len(names), 'now': 'reading the notes'})
        notes, fulls, size = [], [], 0
        for i, name in enumerate(names):
            r['now'] = 'reading the notes (%d of %d recordings)' % (i + 1, len(names))
            try:
                data = rec_get(name)
            except Exception:
                continue
            size += len(data)
            for k, t, v in rec_records(data, 'GE'):
                if k == 'E':
                    notes.append((t, v))
                else:
                    fulls.append((t, name))
            del data
        notes.sort(key=lambda x: x[0])
        spans, skip, edges, open_on = rec_breaks(notes)
        fulls.sort()
        in_break, in_program = [], []
        for t, name in fulls:
            if any(a <= t <= b for a, b in skip):
                continue
            if any(on + BREAK_INSIDE <= t <= off - BREAK_INSIDE for on, off, _ in spans):
                in_break.append((t, name))
            elif all(t < on - PROGRAM_CLEAR or t > off + PROGRAM_CLEAR for on, off, _ in spans) and \
                    (open_on is None or t < open_on - PROGRAM_CLEAR):
                in_program.append((t, name))
        r.update({'bytes': size, 'full_pictures': len(fulls), 'breaks': len(spans),
                  'break_minutes': round(sum(off - on for on, off, _ in spans) / 60000, 1),
                  'hours': round((fulls[-1][0] - fulls[0][0]) / 3600000, 1) if fulls else 0,
                  'edges': edges, 'candidates': {'break': len(in_break), 'program': len(in_program)}})
        chosen = [(t, name, 'break') for t, name in pick_even(in_break, READ_BENCH_N)] + \
                 [(t, name, 'program') for t, name in pick_even(in_program, READ_BENCH_N)]
        by_file = {}
        for t, name, label in chosen:
            by_file.setdefault(name, {})[t] = label
        paths = read_paths()
        bench_reader = Reader(paths['det'], paths['rec'], READ_BENCH_THREADS)
        stats = {lb: {'read': 0, 'ms': [], 'with_words': 0, 'url': 0, 'phone': 0, 'price': 0, 'words': 0, 'small': 0,
                      'ad': 0, 'examples': {}, 'heights': [0] * (len(HEIGHTS) + 1)} for lb in ('break', 'program')}
        program_ads, timed_jpegs, done = [], [], 0
        for name in sorted(by_file):
            want = by_file[name]
            try:
                data = rec_get(name)
            except Exception:
                continue
            for k, t, (start, ln, w, h) in rec_records(data, 'G'):
                label = want.get(t)
                if label is None:
                    continue
                jpg = bytes(data[start:start + ln])
                done += 1
                r['now'] = 'reading picture %d of %d' % (done, len(chosen))
                try:
                    t0 = time.time()
                    lines = bench_reader.read(Image.open(io.BytesIO(jpg)), READ_BENCH_SIDE)
                    ms = round((time.time() - t0) * 1000)
                except Exception:
                    continue
                signs = ad_signs(lines)
                read_bench_add(stats, label, signs, ms, lines)
                if label == 'program' and signs['ad'] and len(program_ads) < 30:
                    program_ads.append({'at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t / 1000)), 'recording': name,
                                        'why': signs['why']})
                if len(timed_jpegs) < READ_BENCH_TIMED and label == 'break':
                    timed_jpegs.append(jpg)
                r['result'] = read_bench_view(stats)
            del data
        r['program_with_ad_signs'] = program_ads
        r['now'] = "timing the TVs' own reading (1 core)"
        live, asked = {}, False
        for side in (960, 1280):
            ms = []
            for jpg in timed_jpegs:
                asked = asked or time.time() - state['last_look'] < BENCH_QUIET
                with read_lock:
                    t0 = time.time()
                    reader.read(Image.open(io.BytesIO(jpg)), side)
                    ms.append(round((time.time() - t0) * 1000))
            live[str(side)] = round(sum(ms) / len(ms)) if ms else None
        r['tv_reader_ms'] = live
        r['tv_asking_meanwhile'] = asked or time.time() - state['last_look'] < BENCH_QUIET
        r['result'] = read_bench_view(stats)
        r['now'] = 'done'
        bench['state'] = 'done'
        say('the bench (read): done, %d pictures read' % done)


# ---- (1.8.0) Scoring: how well each model tells his channels' ads from their programs, on his recordings ----
# Each model's own text half says what an ad is and what a program is (the same words for every model); its picture
# half looks at pictures of his recordings whose answers come with the bench (a channel's afternoon game: the breaks the
# TV's replays found real, its program well away from them; another channel's morning: the screens he answered). How
# well each parts them: its lean (by those words), its likeness to the program's look (as a TV learns it), both, and
# what it could learn (a lean set on one part of a channel's answers, tried on the rest and on another channel).
# Numbers only: no picture is kept or sent anywhere, and the models downloaded for it are deleted after.
VOCAB_URL = 'https://raw.githubusercontent.com/mlfoundations/open_clip/v2.24.0/src/open_clip/bpe_simple_vocab_16e6.txt.gz'
VOCAB_SIZE, VOCAB_SHA = 1_356_917, '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
VOCAB = os.path.join(DATA, 'bpe_simple_vocab_16e6.txt.gz')
XEN = HF + 'Xenova/clip-vit-base-patch32/resolve/6b7c523703ae26e39be4b03774966f8571c7f523/onnx/'
# (id, what, picture half (None: on this Pi already), its size and SHA-256, side, mean and spread, text half, its size
# and SHA-256): the ones the bench found fast enough for every picture of one TV (2 a second) on this Pi.
SCORE_MODELS = [
    (MODEL_ID, "CLIP ViT-B/32 (the AI's now)", None, 0, None, 224, 'clip',
     XEN + 'text_model.onnx', 254_058_553, '3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc'),
    (SPOT_ID, "MobileCLIP2-S0 (the logo's spot now)", None, 0, None, 256, 'unit',
     MC2 + 's0/text_model.onnx', 254_053_669, 'df590d47744f2ee9f3ccb67c4414d17419568c05bca0c4d166f2faeedf8b92f3'),
    ('mobileclip2-s2', 'MobileCLIP2-S2', MC2 + 's2/vision_model.onnx', 143_044_797,
     'a841f72c5a5085748bbe271a1d5718aba877822a15cba865bdbd0d37036b849e', 256, 'unit',
     MC2 + 's2/text_model.onnx', 254_053_669, '622f10372bca71b5017f2efc5f8c2886610a2592b636de8984d717f03213f031'),
    ('mobileclip2-b', 'MobileCLIP2-B', MC2 + 'b/vision_model.onnx', 345_545_489,
     '2f3873e400fd1adc5298b72d29e81bb69337e58601cf2853c43bc1211631eecd', 224, 'unit',
     MC2 + 'b/text_model.onnx', 254_342_238, 'f5cbb08e1eb6a5da690554a43cd37304835c0412c547ddb4ae86f0f8759686e0'),
]
# What an ad is, and what a program is: the same words for every model.
SCORE_ADS = [
    'a TV commercial', 'a television advertisement', 'a commercial break on TV', 'a car commercial',
    "an ad for a phone app with the app's logo", 'a product shown with its price', 'an ad with a promo code and a website',
    "a brand's logo on a plain background at the end of a commercial", 'an insurance commercial',
    'a fast food commercial', 'a medicine commercial with fine print', 'a movie trailer', 'a promo for a TV show',
    'a sports betting app commercial']
SCORE_PROGRAMS = [
    'a live basketball game on TV', 'a live football game on TV', 'a sports broadcast with the score on screen',
    'a sports studio show with hosts at a desk', 'a TV news broadcast with an anchor', 'a talk show',
    'a scene from a TV drama', 'a scene from a movie', 'a documentary', 'an interview on TV', 'a morning show with hosts',
    'a live game shown beside an ad', "a sponsor's graphic during a sports broadcast",
    "a sports channel's logo and ticker over the game", 'a sitcom scene']
SCORE_FP = 0.02     # the share of program pictures that may be called ads, for "ads caught" (the show protected first)
SCORE_LEARN = 0.3   # the first part of a set's program pictures (in time) that makes its program's look; the rest tests it
SCORE_MOST = 3000   # pictures at most in all (the rest left out evenly)
SCORE_MEMORY = 3    # a model's file this many times over (and 300 MB) free in memory before it's loaded


def bytes_to_unicode():
    """CLIP's byte-to-letter table (OpenAI's, MIT): every byte a printable letter, for the word pieces."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ClipTokens:
    """CLIP's word pieces (OpenAI's tokenizer, MIT; MobileCLIP2's too), for plain English words: 77 numbers, the start,
    the pieces, the end, then 0s."""
    PAT = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+""")

    def __init__(self, vocab_gz):
        import gzip
        merges = gzip.open(vocab_gz).read().decode('utf-8').split('\n')
        merges = [tuple(m.split()) for m in merges[1:49152 - 256 - 2 + 1]]
        self.byte_encoder = bytes_to_unicode()
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + '</w>' for v in vocab]
        vocab += [''.join(m) for m in merges]
        vocab += ['<|startoftext|>', '<|endoftext|>']
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.sot, self.eot = self.encoder['<|startoftext|>'], self.encoder['<|endoftext|>']

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        return word

    def ids(self, text, n=77):
        text = re.sub(r'\s+', ' ', text).strip().lower()
        v = [self.sot]
        for tok in self.PAT.findall(text):
            v += [self.encoder[t] for t in self.bpe(''.join(self.byte_encoder[b] for b in tok.encode('utf-8')))]
        v = v[:n - 1] + [self.eot]
        return v + [0] * (n - len(v))


def roomy(path):
    """Raises if too little memory is free to load this model."""
    avail = meminfo('MemAvailable')
    if avail is not None and avail < os.path.getsize(path) * SCORE_MEMORY + (300 << 20):
        raise IOError('too little memory free (%d MB)' % (avail // 1_000_000))


def score_words(path, toks):
    """A model's lean, from its own text half: the ads' words' average less the programs', length 1."""
    roomy(path)
    s = bench_session(path, THREADS_ALL)
    ids = np.array([toks.ids(t) for t in SCORE_ADS + SCORE_PROGRAMS], np.int64)
    ins = s.get_inputs()
    feeds = {}
    for i in ins:
        dt = np.int32 if 'int32' in i.type else np.int64
        mask = 'mask' in i.name.lower() and len(ins) > 1
        feeds[i.name] = ((ids != 0) if mask else ids).astype(dt)
    outs = s.get_outputs()
    name = next((o.name for o in outs if o.name == 'text_embeds'), None) or \
        next((o.name for o in outs if len(o.shape) == 2), outs[0].name)
    # One description at a time (a text half may take only one).
    out = np.concatenate([s.run([name], {k: v[i:i + 1] for k, v in feeds.items()})[0] for i in range(len(ids))]).astype(np.float64)
    del s
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    d = out[:len(SCORE_ADS)].mean(0) - out[len(SCORE_ADS):].mean(0)
    return (d / np.linalg.norm(d)).astype(np.float32)


def auc(pos, neg):
    """How often an ad's picture scores over a program's (0.5 a coin's toss, 1 every time); ties count half."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return None
    a = np.concatenate([pos, neg])
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (first + (counts + 1) / 2.0)[inv]  # each value's average rank (1 the lowest)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def caught(pos, neg, fp=SCORE_FP):
    """The share of ads over the line that lets at most fp of the programs over it, and the line."""
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    line = float(np.quantile(np.asarray(neg, np.float64), 1 - fp))
    return float(np.mean(np.asarray(pos) > line)), line


def probe_fit(x, y):
    """What a lean learned from answers could be: a straight line through the pictures (ridge regression, ads and
    programs weighed alike), its middle and its direction."""
    x = x.astype(np.float64)
    w = np.where(y == 1, 0.5 / max(1, int((y == 1).sum())), 0.5 / max(1, int((y == 0).sum()))) * len(y)
    mid = (w[:, None] * x).sum(0) / w.sum()
    xc = x - mid
    a = (xc * w[:, None]).T @ xc
    a += np.eye(x.shape[1]) * (0.1 * np.trace(a) / x.shape[1] + 1e-9)
    return mid, np.linalg.solve(a, (xc * w[:, None]).T @ np.where(y == 1, 1.0, -1.0))


def score_pictures(sets, r):
    """Every set's pictures with their answers, from Home media: [(set, 1 an ad or 0 the program, time, JPEG, span)],
    one each `every` ms at most for each answer; SCORE_MOST in all at most, the rest left out evenly."""
    names = rec_names()
    out = []
    for si, st in enumerate(sets):
        key = re.sub(r'[^A-Z0-9]', '', str(st.get('channel', '')).upper())
        lo, hi = str(st.get('from', '')), str(st.get('to', '~'))
        files = [n for n in names if len(n) > 20 and n.endswith('.rec') and key and n[:len(lo)] >= lo and n[:len(hi)] <= hi
                 and key in re.sub(r'[^A-Z0-9]', '', n[16:-4].upper())]
        every = max(1, int(st.get('every', 4000)))
        spans = [(1, int(a), int(b)) for a, b in st.get('ads', [])] + [(0, int(a), int(b)) for a, b in st.get('programs', [])]
        used = set()  # one picture each `every` ms of each span (a minute sent twice: once)
        for i, n in enumerate(files):
            r['now'] = 'reading %s (%d of %d minutes)' % (st.get('name') or key, i + 1, len(files))
            try:
                data = rec_get(n)
            except Exception:
                continue
            for _, t, (a, ln) in rec_records(data, 'F'):
                k = next((k for k, (_, x, y) in enumerate(spans) if x <= t <= y), None)
                if k is None or (k, t // every) in used:
                    continue
                j = bytes(data[a:a + ln])
                if j[:2] != b'\xff\xd8':
                    continue
                used.add((k, t // every))
                out.append((si, spans[k][0], t, j, k))
            del data
        st['_files'] = len(files)
    if len(out) > SCORE_MOST:
        out = pick_even(out, SCORE_MOST)
    return out


def score_set(st, emb, score, lab, ts, span, ref):
    """One set's numbers for one model (ref: the set its line and learned lean are taken from, with them)."""
    res = {'name': st.get('name', ''), 'auc': auc(score[lab == 1], score[lab == 0])}
    res['caught'], res['line'] = caught(score[lab == 1], score[lab == 0])
    prog = np.flatnonzero(lab == 0)
    if st.get('look') and len(prog) >= 20 and (lab == 1).any():
        # Its likeness to the program's look, as a TV learns it: made from the first part of the program, tried after.
        cut = np.sort(ts[prog])[int(len(prog) * SCORE_LEARN)]
        lookv = emb[prog[ts[prog] < cut]].mean(0)
        lookv /= np.linalg.norm(lookv)
        test = ts >= cut
        like, tl, tsc = emb[test] @ lookv, lab[test], score[test]
        res['look_auc'] = auc(-like[tl == 1], -like[tl == 0])
        res['look_caught'], _ = caught(-like[tl == 1], -like[tl == 0])
        z = (tsc - tsc.mean()) / (tsc.std() + 1e-9) - (like - like.mean()) / (like.std() + 1e-9)
        res['both_auc'] = auc(z[tl == 1], z[tl == 0])
        res['both_caught'], _ = caught(z[tl == 1], z[tl == 0])
    if st.get('probe'):
        # What it could learn: a lean set on the first half of the set's answers (in time), tried on the second.
        half = np.median(ts)
        a, b = ts < half, ts >= half
        if len(set(lab[a])) == 2 and len(set(lab[b])) == 2:
            mid, d = probe_fit(emb[a], lab[a])
            p = (emb[b] - mid) @ d
            res['probe_auc'] = auc(p[lab[b] == 1], p[lab[b] == 0])
            res['probe_caught'], _ = caught(p[lab[b] == 1], p[lab[b] == 0])
    if ref is not None:
        # Another channel by the first one's line, and by what was learned there: each answered span's share over it.
        line, mid, d = ref
        if line is not None:
            res['spans'] = [[('ad' if lab[span == k][0] == 1 else 'show'), round(float(np.mean(score[span == k] > line)), 3),
                             int((span == k).sum())] for k in sorted(set(span.tolist()))]
        if d is not None:
            p = (emb - mid) @ d
            res['probe_from_auc'] = auc(p[lab == 1], p[lab == 0])
    for k in list(res):
        if isinstance(res[k], float):
            res[k] = round(res[k], 4)
    return res


def score_run(spec):
    """(1.8.0) Each scoring model in turn: its text half's lean, its picture half on the answered pictures, and how well
    they part the ads from the programs, set by set."""
    with bench_lock:
        sets = list(spec.get('sets') or [])
        bench.update({'state': 'running', 'note': '', 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': [], 'set': 'score',
                      'memory_mb': round((meminfo('MemTotal') or 0) / 1e6), 'cores': os.cpu_count()})
        r = bench['score'] = {'now': 'getting the words', 'sets': [], 'models': []}
        say('the bench (score): started (%d sets)' % len(sets))
        if not (os.path.isfile(VOCAB) and os.path.getsize(VOCAB) == VOCAB_SIZE and sha256(VOCAB) == VOCAB_SHA):
            bench_get(VOCAB, VOCAB_URL, VOCAB_SIZE, VOCAB_SHA)
        toks = ClipTokens(VOCAB)
        pics = score_pictures(sets, r)
        tz = int(spec.get('tz_minutes', 0))
        clock = lambda t: time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t / 1000 + tz * 60))
        for si, st in enumerate(sets):
            ts = [p[2] for p in pics if p[0] == si]
            r['sets'].append({'name': st.get('name', ''), 'files': st.get('_files', 0),
                              'ads': sum(1 for p in pics if p[0] == si and p[1] == 1),
                              'programs': sum(1 for p in pics if p[0] == si and p[1] == 0),
                              'first': clock(min(ts)) if ts else None, 'last': clock(max(ts)) if ts else None})
        if not pics:
            bench['note'] = 'no answered pictures in the recordings'
            raise RuntimeError(bench['note'])
        lab_all = np.array([p[1] for p in pics])
        ts_all = np.array([p[2] for p in pics], np.int64)
        set_all = np.array([p[0] for p in pics])
        span_all = np.array([p[4] for p in pics])
        for mid, what, src, size, sha, side, norm, turl, tsize, tsha in SCORE_MODELS:
            m = {'id': mid, 'what': what, 'note': '', 'sets': []}
            r['models'].append(m)
            got = []
            try:
                vpath = MODEL if mid == MODEL_ID else SPOT_MODEL if mid == SPOT_ID else os.path.join(DATA, 'score-%s-vision.onnx' % mid)
                tpath = os.path.join(DATA, 'score-%s-text.onnx' % mid)
                free = shutil.disk_usage(DATA).free
                if free < tsize + size + BENCH_ROOM:
                    raise IOError('no room for it (%d MB free)' % (free // 1_000_000))
                if not src and not os.path.isfile(vpath):
                    raise IOError('its picture half is not downloaded yet')
                r['now'] = '%s: downloading its text half' % what
                got.append(tpath)
                bench_get(tpath, turl, tsize, tsha)
                d = score_words(tpath, toks)
                os.remove(tpath)
                if src:
                    r['now'] = '%s: downloading its picture half' % what
                    got.append(vpath)
                    bench_get(vpath, src, size, sha)
                roomy(vpath)
                s = bench_session(vpath, THREADS_ALL)
                inp = s.get_inputs()[0].name
                out_name = embed_name(s)
                emb, run_s = [], 0.0
                for k, p in enumerate(pics):
                    if k % 25 == 0:
                        r['now'] = '%s: %d of %d pictures' % (what, k, len(pics))
                    while time.time() - state['last_look'] < BENCH_QUIET and not bench.get('busy'):
                        r['now'] = '%s: waiting for the TVs to stop asking (%d of %d pictures)' % (what, k, len(pics))
                        time.sleep(2)
                    x = bench_input(p[3], side, norm)
                    t0 = time.time()
                    e = s.run([out_name], {inp: x})[0][0].astype(np.float32)
                    run_s += time.time() - t0
                    emb.append(e / np.linalg.norm(e))
                del s
                emb = np.array(emb, np.float32)
                m['ms'] = round(run_s * 1000 / len(pics))
                score = emb @ d
                if mid == MODEL_ID and len(direction) == emb.shape[1]:
                    # The TVs' lean now (direction.bin) beside it: how like the two are, and how the TVs' own does.
                    m['words_like_now'] = round(float(d @ direction), 4)
                    live = emb @ direction
                ref = None
                for si, st in enumerate(sets):
                    sel = set_all == si
                    if not sel.any():
                        m['sets'].append({'name': st.get('name', ''), 'note': 'no pictures'})
                        continue
                    lab, ts, span = lab_all[sel], ts_all[sel], span_all[sel]
                    use = None
                    if st.get('line_from') is not None and ref is not None and int(st['line_from']) == ref[0]:
                        use = ref[1:]
                    res = score_set(st, emb[sel], score[sel], lab, ts, span, use)
                    if mid == MODEL_ID and len(direction) == emb.shape[1]:
                        lv = live[sel]
                        a = auc(lv[lab == 1], lv[lab == 0])
                        res['now_auc'] = None if a is None else round(a, 4)
                        res['now_over_ad'] = {'ads': round(float(np.mean(lv[lab == 1] > AD)), 3) if (lab == 1).any() else None,
                                              'programs': round(float(np.mean(lv[lab == 0] > AD)), 3) if (lab == 0).any() else None}
                    m['sets'].append(res)
                    if ref is None and st.get('probe') and len(set(lab.tolist())) == 2:
                        pm, pd = probe_fit(emb[sel], lab)
                        ref = (si, res.get('line'), pm, pd)
                m['dim'] = int(emb.shape[1])
                m['note'] = 'scored %d pictures (%d ms each on %d cores)' % (len(pics), m['ms'], THREADS_ALL)
                say('the bench (score): %s: %s' % (what, json.dumps(m['sets'])))
            except Exception as e:
                m['note'] = "couldn't be scored (%s)" % (e if isinstance(e, (IOError, RuntimeError)) else type(e).__name__)
                say('the bench (score): %s: %s' % (what, m['note']))
            finally:
                for p in got:
                    for f in (p, p + '.part'):
                        if os.path.isfile(f) and os.path.basename(f).startswith('score-'):
                            os.remove(f)
        r['now'] = 'done'
        bench['state'] = 'done'
        bench['note'] = '%d of %d scored' % (sum(1 for m in r['models'] if 'ms' in m), len(SCORE_MODELS))
        say('the bench (score): done (%s)' % bench['note'])


# ---- (1.9.0) Each channel's own lesson: what its breaks look like, learned from its own confirmed breaks ----
# The AI's lean is the same for every channel; but a channel's ads and its programs each look their own way, and a lean
# learned from a channel's own breaks parts them far better (on FS1's afternoon, one learned from its first half caught
# 98% of the second half's ad pictures with 2% of the game's taken for ads). So the TVs say which channel each picture
# is of and when (POST /look?ch=&t=), and how each run of the AI's ads ended (a real break: a silent or black cut at its
# start, ads seen before, an ad's words; or not), when the screen was up and which show was on (POST /lessons/event).
# What the AI saw of each picture (its 512 numbers, never the picture) is kept by channel for LESSON_DAYS days. Each
# night (once the TVs have been quiet a while), each channel's lesson is learned again: ad pictures from its real
# breaks, program pictures well away from any run of the AI's ads or any screen. Before a lesson is used it's tested
# on the newest day it didn't learn from: it must part that day's breaks from its program, and the screen it would put
# up must cover the real breaks and stay off the program (a show it can't keep off is held, and goes by the TV's rules).
# A channel whose own lesson hasn't passed may borrow another channel's, if that one passes this channel's own test
# (the same channel from another provider, a sister channel). A lesson that passed goes with every answer for its
# channel; the TV gives the AI its freedom there.
LESSONS = os.path.join(DATA, 'lessons')
LESSON_DAYS = 21            # the AI's numbers and the TVs' events kept this many days
LESSON_DAY_SHIFT = 10       # a day ends at 10:00 UTC (5 AM in New York, 2 AM in Los Angeles)
LESSON_EVERY = 4000         # ms between two pictures of an answer, to learn from
LESSON_TRIM = 10_000        # a real break's first and last 10 s left out (where the AI's run began and ended)
LESSON_AWAY = 40_000        # the program: at least this far from any run of the AI's ads and any screen
LESSON_NEAR = 20_000        # a screen this near a real break is that break's
LESSON_GAP = 10_000         # pictures further apart than this aren't one stretch of watching
LESSON_RUN_START, LESSON_RUN_END = 3, 2  # as the TV: 3 ad words in a row start a run, 2 of the program end it
LESSON_FP = 0.02            # its line: 2% of the program's pictures over it
# To pass: it parts the test day's breaks from its program, its screen covers the real breaks, and it puts the screen
# up outside them at most PASS_FALSE seconds an hour; a show watched at least HOLD_LEAST minutes with more than
# HOLD_FALSE seconds an hour of it is held (the TV's rules there).
PASS_AUC, PASS_CAUGHT, PASS_COVER, PASS_FALSE = 0.97, 0.85, 0.8, 30
HOLD_FALSE, HOLD_LEAST = 60, 15
LEAST_BREAKS, LEAST_PROGRAM = 2, 300        # a test: real breaks, and program pictures (LESSON_EVERY apart: 20 min)
LEAST_TRAIN_BREAKS, LEAST_TRAIN_PROGRAM = 3, 450  # to learn from (30 min of program)
LESSON_AGAIN = 20 * 3600    # learned again after this long, once the TVs are quiet
LESSON_QUIET = 1800         # quiet: no TV has asked for this long
LOOK_HEAD = b'HPAL1'
lessons = {'state': 'not learned yet', 'at': None, 'model': None, 'channels': {}}
lesson_state = {'use': {}}  # the lessons the answers use: channel -> (middle, direction, line, from, held)
lesson_lock = threading.Lock()   # learning one at a time
lesson_write = threading.Lock()  # the files being added to


def lesson_key(ch):
    """A channel's key: its name's letters and digits, in capitals (as the recordings' names have it)."""
    return re.sub(r'[^A-Z0-9]', '', str(ch or '').upper())[:64]


def lesson_day(t):
    """The day a picture's time is in (ms since 1970), as 'YYYYMMDD'."""
    return time.strftime('%Y%m%d', time.gmtime(t / 1000 - LESSON_DAY_SHIFT * 3600))


def looks_dir(key=None):
    d = os.path.join(LESSONS, 'looks', re.sub(r'[^a-z0-9.-]', '', MODEL_ID.lower()))
    return d if key is None else os.path.join(d, key)


def events_dir(key=None):
    d = os.path.join(LESSONS, 'events')
    return d if key is None else os.path.join(d, key)


class LookLog:
    """What the AI saw of each channel's pictures (by the picture's own time): a file a day for each channel, each
    picture 12 bytes and a byte a number. Only the AI's numbers: never a picture."""

    def __init__(self):
        self.files = {}

    def add(self, key, t, e):
        scale = float(np.abs(e).max()) / 127.0 or 1e-9
        q = np.clip(np.round(e / scale), -127, 127).astype(np.int8)
        rec = struct.pack('>qf', int(t), scale) + q.tobytes()
        day = lesson_day(t)
        with lesson_write:
            f = self.files.get((key, day))
            if f is None:
                while len(self.files) >= 4:  # (two TVs, each about to start a new day)
                    self.files.pop(next(iter(self.files))).close()
                d = looks_dir(key)
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, day + '.bin')
                new = not os.path.isfile(p) or os.path.getsize(p) < len(LOOK_HEAD) + 2
                f = open(p, 'ab')
                if new:
                    f.seek(0)
                    f.truncate()
                    f.write(LOOK_HEAD + struct.pack('>h', len(e)))
                self.files[(key, day)] = f
            f.write(rec)
            f.flush()

    def close(self):
        with lesson_write:
            for f in self.files.values():
                f.close()
            self.files.clear()

    @staticmethod
    def days(key):
        d = looks_dir(key)
        return sorted(n[:8] for n in os.listdir(d) if n.endswith('.bin') and len(n) == 12) if os.path.isdir(d) else []

    @staticmethod
    def read(key, day):
        """A day's pictures of a channel: their times (ms) and what the AI saw (length 1), in time order."""
        p = os.path.join(looks_dir(key), day + '.bin')
        try:
            with open(p, 'rb') as f:
                data = f.read()
        except OSError:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        if data[:len(LOOK_HEAD)] != LOOK_HEAD:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        dim = struct.unpack('>h', data[len(LOOK_HEAD):len(LOOK_HEAD) + 2])[0]
        body = data[len(LOOK_HEAD) + 2:]
        rl = 12 + dim
        n = len(body) // rl
        a = np.frombuffer(body[:n * rl], np.uint8).reshape(n, rl)
        t = a[:, :8].copy().view('>i8').reshape(n).astype(np.int64)
        scale = a[:, 8:12].copy().view('>f4').reshape(n).astype(np.float32)
        e = a[:, 12:].copy().view(np.int8).astype(np.float32) * scale[:, None]
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        order = np.argsort(t, kind='stable')
        return t[order], e[order]


looklog = LookLog()


def lesson_event(body, replay=False):
    """(1.9.0) A TV's word on its channel: a run of the AI's ads that ended (real or not), a screen that came down, the
    show on now. A replay's are left out (its pictures are the recordings', already counted)."""
    j = json.loads(body)
    key = lesson_key(j.get('channel'))
    kind = j.get('kind')
    if not key or kind not in ('run', 'screen', 'show'):
        raise ValueError('not an event')
    ev = {'kind': kind}
    if kind == 'show':
        ev['at'] = int(j['at'])
        ev['name'] = str(j.get('name') or '')[:120]
        when = ev['at']
    else:
        ev['from'], ev['to'] = int(j['from']), int(j['to'])
        if not 0 <= ev['to'] - ev['from'] <= 3 * 3600 * 1000:
            raise ValueError('not a span')
        if kind == 'run':
            ev['real'] = bool(j.get('real'))
            ev['why'] = str(j.get('why') or '')[:120]
        when = ev['from']
    if replay:
        return {'kept': False, 'note': "a replay's: not kept"}
    d = events_dir(key)
    with lesson_write:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, lesson_day(when) + '.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(ev) + '\n')
    return {'kept': True}


def lesson_events(key, days):
    """A channel's events on these days: its runs [(from, to, real)], screens [(from, to)], shows [(at, name)]."""
    runs, screens, shows = [], [], []
    for day in days:
        try:
            with open(os.path.join(events_dir(key), day + '.jsonl'), encoding='utf-8') as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for ln in lines:
            try:
                ev = json.loads(ln)
            except ValueError:
                continue
            if ev.get('kind') == 'run':
                runs.append((int(ev['from']), int(ev['to']), bool(ev.get('real'))))
            elif ev.get('kind') == 'screen':
                screens.append((int(ev['from']), int(ev['to'])))
            elif ev.get('kind') == 'show':
                shows.append((int(ev['at']), str(ev.get('name') or '')))
    return sorted(runs), sorted(screens), sorted(shows)


def in_spans(t, spans):
    """For each time, whether it's inside any of the spans [(from, to)]."""
    out = np.zeros(len(t), bool)
    for a, b in spans:
        out |= (t >= a) & (t <= b)
    return out


def lesson_labels(t, runs, screens):
    """Each picture's answer: 1 an ad (inside a real break, its edges left out), 0 the program (well away from any run
    and any screen), -1 neither."""
    ad = in_spans(t, [(a + LESSON_TRIM, b - LESSON_TRIM) for a, b, real in runs if real and b - a > 2 * LESSON_TRIM])
    near = in_spans(t, [(a - LESSON_AWAY, b + LESSON_AWAY) for a, b, _ in runs] + [(a - LESSON_AWAY, b + LESSON_AWAY) for a, b in screens])
    lab = np.full(len(t), -1, np.int8)
    lab[~near] = 0
    lab[ad] = 1
    return lab


def lesson_sample(t, lab):
    """One picture of each answer each LESSON_EVERY ms: their places."""
    keep, seen = [], set()
    for i in np.flatnonzero(lab >= 0):
        k = (int(lab[i]), int(t[i]) // LESSON_EVERY)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return np.array(keep, np.int64)


def lesson_screen(t, words):
    """The screen a lesson would put up on a day's pictures (in time order): its runs [(from, to)] of ad words (3 in a
    row start one, 2 of the program end it; a gap in the watching ends it too)."""
    out, row_ad, row_prog, start, last = [], 0, 0, None, None
    on = False
    prev = None
    t = [int(x) for x in t]
    for k in range(len(t)):
        if prev is not None and t[k] - prev > LESSON_GAP:
            if on:
                out.append((start, last))
            on, row_ad, row_prog = False, 0, 0
        prev = t[k]
        if words[k]:
            row_ad += 1
            row_prog = 0
            if not on and row_ad == 1:
                first = t[k]
            if not on and row_ad >= LESSON_RUN_START:
                on, start = True, first
            last = t[k] if on else last
        else:
            row_prog += 1
            row_ad = 0
            if on and row_prog >= LESSON_RUN_END:
                out.append((start, last))
                on = False
    if on:
        out.append((start, last))
    return out


def overlap(a, b, spans):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in spans)


def watch_ms(t):
    """How long the pictures were being watched (their gaps up to LESSON_GAP)."""
    if len(t) < 2:
        return 0
    g = np.diff(t)
    return int(g[g <= LESSON_GAP].sum())


def lesson_test(t, e, runs, shows, mid, d, line):
    """A lesson on a day it didn't learn from: its screen's cover of the real breaks, and its time up outside them
    (s an hour), for the day and for each show."""
    s = (e - mid) @ d
    screen = lesson_screen(t, s > line)
    real = [(a, b) for a, b, r in runs if r]
    near = [(a - LESSON_NEAR, b + LESSON_NEAR) for a, b in real]
    covered, real_ms = int(sum(overlap(a, b, screen) for a, b in real)), int(sum(b - a for a, b in real))
    cover = covered / max(1, real_ms)
    # Each screen's time outside the real breaks, by the show it was on.
    names = [n for _, n in shows]
    starts = np.array([a for a, _ in shows], np.int64)

    def show_at(x):
        k = int(np.searchsorted(starts, x, side='right')) - 1 if len(starts) else -1
        return names[k] if k >= 0 else ''
    false_by, watch_by = {}, {}
    for a, b in screen:
        out = (b - a) - overlap(a, b, near)
        if out > 0:
            false_by[show_at(a)] = false_by.get(show_at(a), 0) + out
    if len(t) > 1:
        g = np.diff(t)
        for k in np.flatnonzero(g <= LESSON_GAP):
            n = show_at(int(t[k]))
            watch_by[n] = watch_by.get(n, 0) + int(g[k])
    # A show it can't keep the screen off is held (the TV's rules there); the rest is what's tested.
    held = sorted(n for n in watch_by if n and watch_by[n] >= HOLD_LEAST * 60_000
                  and false_by.get(n, 0) * 3600 / watch_by[n] > HOLD_FALSE)
    watched = max(1, sum(v for n, v in watch_by.items() if n not in held))
    false_s = float(sum(v for n, v in false_by.items() if n not in held)) / 1000
    per_h = false_s * 3600_000 / watched
    return {'cover': round(cover, 3), 'false_s': round(false_s, 1), 'false_per_hour': round(per_h, 1),
            'watched_min': round(watched / 60000), 'screens': len(screen), 'held': held, 'covered_ms': covered,
            'real_ms': real_ms, 'watched_ms': int(watched), 'held_ms': int(sum(v for n, v in watch_by.items() if n in held)),
            'held_false_s': round(float(sum(v for n, v in false_by.items() if n in held)) / 1000, 1)}


def lesson_check(X, y, mid, d, line):
    """How well a lesson parts answered pictures: its AUC and the share of ads over its line."""
    s = (X - mid) @ d
    a = auc(s[y == 1], s[y == 0])
    return (None if a is None else round(a, 4)), (round(float(np.mean(s[y == 1] > line)), 4) if (y == 1).any() else None)


def lesson_passes(r):
    return bool((r.get('auc') or 0) >= PASS_AUC and (r.get('caught') or 0) >= PASS_CAUGHT and
                (r.get('cover') or 0) >= PASS_COVER and r.get('false_per_hour', 1e9) <= PASS_FALSE)


def lesson_fit(X, y):
    mid, d = probe_fit(X, y)
    s = (X - mid) @ d
    line = float(np.quantile(s[y == 0], 1 - LESSON_FP))
    return mid.astype(np.float32), d.astype(np.float32), line


def lesson_channel(key, before=None):
    """A channel's data: each day's answered pictures (one each LESSON_EVERY ms of an answer), and its events."""
    days = LookLog.days(key)
    if before is not None:
        days = [x for x in days if x <= lesson_day(before)]
    runs, screens, shows = lesson_events(key, days)
    if before is not None:
        runs = [r for r in runs if r[1] < before]
        screens = [s for s in screens if s[1] < before]
    per_day = []
    for day in days:
        t, e = LookLog.read(key, day)
        if before is not None:
            keep = t < before
            t, e = t[keep], e[keep]
        if not len(t):
            continue
        lab = lesson_labels(t, runs, screens)
        pick = lesson_sample(t, lab)
        breaks = sum(1 for a, b, r in runs if r and lesson_day(a) == day)
        per_day.append({'day': day, 'X': e[pick], 'y': lab[pick].astype(np.int64), 't': t[pick], 'breaks': breaks,
                        'program': int((lab[pick] == 0).sum())})
    return per_day, runs, screens, shows


def lesson_split(test, real, tb, tp):
    """Where to split a day (its later part the test), so the later part has LEAST_BREAKS real breaks and
    LEAST_PROGRAM program pictures and the earlier one, with the other days, enough to learn from; or None."""
    ts = np.asarray(test['t'])
    if not len(ts):
        return None
    for q in (0.6, 0.5, 0.7, 0.4, 0.75):
        s = int(np.quantile(ts, q))
        late_b = sum(1 for a, b, _ in real if a >= s)
        early_b = sum(1 for a, b, _ in real if b < s)
        late_p = int(((ts >= s) & (np.asarray(test['y']) == 0)).sum())
        early_p = int(((ts < s) & (np.asarray(test['y']) == 0)).sum())
        if late_b >= LEAST_BREAKS and late_p >= LEAST_PROGRAM and tb + early_b >= LEAST_TRAIN_BREAKS and tp + early_p >= LEAST_TRAIN_PROGRAM:
            return s
    return None


def lesson_learn_one(key, before=None):
    """A channel's own lesson: learned from all but its newest day with enough answers, tested on that day; if it
    passes, learned again from all of them."""
    per_day, runs, screens, shows = lesson_channel(key, before)
    out = {'channel': key, 'days': len(per_day), 'breaks': sum(p['breaks'] for p in per_day),
           'program_min': round(sum(p['program'] for p in per_day) * LESSON_EVERY / 60000), 'passed': False}
    test = next((p for p in reversed(per_day) if p['breaks'] >= LEAST_BREAKS and p['program'] >= LEAST_PROGRAM), None)
    if test is None:
        out['note'] = 'learning: not a day with %d real breaks and %d min of program yet' % (LEAST_BREAKS, LEAST_PROGRAM * LESSON_EVERY // 60000)
        return out, None
    train = [p for p in per_day if p is not test]
    tb, tp = sum(p['breaks'] for p in train), sum(p['program'] for p in train)
    split = None
    if tb < LEAST_TRAIN_BREAKS or tp < LEAST_TRAIN_PROGRAM:
        # Not enough besides its newest day: that day split, its later part the test (a channel's first long day).
        split = lesson_split(test, [x for x in runs if x[2] and lesson_day(x[0]) == test['day']], tb, tp)
        if split is None:
            out['note'] = 'learning: %d real breaks and %d min of program besides its newest day (needs %d and %d)' % (
                tb, tp * LESSON_EVERY // 60000, LEAST_TRAIN_BREAKS, LEAST_TRAIN_PROGRAM * LESSON_EVERY // 60000)
            return out, None
    early = [] if split is None else [i for i in range(len(test['t'])) if test['t'][i] < split]
    late = slice(None) if split is None else np.array([i for i in range(len(test['t'])) if test['t'][i] >= split], np.int64)
    X = np.concatenate([p['X'] for p in train] + ([test['X'][early]] if early else []))
    y = np.concatenate([p['y'] for p in train] + ([test['y'][early]] if early else []))
    mid, d, line = lesson_fit(X, y)
    r = {'test_day': test['day']}
    if split is not None:
        r['test_from'] = split
    r['auc'], r['caught'] = lesson_check(test['X'][late], test['y'][late], mid, d, line)
    t, e = LookLog.read(key, test['day'])
    keep = (t < before if before is not None else np.ones(len(t), bool)) & (t >= (split or 0))
    t, e = t[keep], e[keep]
    r.update(lesson_test(t, e, [x for x in runs if lesson_day(x[0]) == test['day'] and x[0] >= (split or 0)], shows, mid, d, line))
    out['test'] = r
    out['passed'] = lesson_passes(r)
    out['held'] = r['held']
    if not out['passed']:
        out['note'] = 'not used: its test on %s fell short' % test['day']
        return out, None
    X = np.concatenate([p['X'] for p in per_day])
    y = np.concatenate([p['y'] for p in per_day])
    mid, d, line = lesson_fit(X, y)
    out['note'] = 'used: its own'
    return out, (mid, d, line)


def lesson_borrow(key, own, before=None):
    """A channel without its own lesson borrows another's that passes this channel's own test (all its answered
    days, its newest three's screens): the best of them."""
    per_day, runs, screens, shows = lesson_channel(key, before)
    if sum(p['breaks'] for p in per_day) < LEAST_BREAKS or sum(p['program'] for p in per_day) < LEAST_PROGRAM:
        return None
    X = np.concatenate([p['X'] for p in per_day])
    y = np.concatenate([p['y'] for p in per_day])
    best = None
    for other, (mid, d, line) in own.items():
        if other == key:
            continue
        r = {}
        r['auc'], r['caught'] = lesson_check(X, y, mid, d, line)
        tests = []
        for p in per_day[-3:]:
            t, e = LookLog.read(key, p['day'])
            if before is not None:
                keep = t < before
                t, e = t[keep], e[keep]
            tests.append(lesson_test(t, e, [x for x in runs if lesson_day(x[0]) == p['day']], shows, mid, d, line))
        r['cover'] = round(sum(x['covered_ms'] for x in tests) / max(1, sum(x['real_ms'] for x in tests)), 3)
        r['false_per_hour'] = round(sum(x['false_s'] for x in tests) * 3600_000 / max(1, sum(x['watched_ms'] for x in tests)), 1)
        r['held'] = sorted({h for x in tests for h in x['held']})
        r['watched_ms'] = sum(x['watched_ms'] for x in tests)
        r['held_ms'] = sum(x['held_ms'] for x in tests)
        # A borrowed lesson must fit this channel: it may hold (leave to the TV's rules) at most a quarter of its
        # watching; the best is the one that holds least, then puts the screen up least outside real breaks.
        if not lesson_passes(r) or r['held_ms'] > 0.25 * max(1, r['watched_ms'] + r['held_ms']):
            continue
        rank = (r['held_ms'], r['false_per_hour'], -(r['caught'] or 0), -(r['auc'] or 0), other)
        if best is None or rank < best[2]:
            best = (other, r, rank)
    return None if best is None else best[:2]


def lesson_force(key, before=None):
    """(A test's) A channel's lesson learned from all its answered pictures (before a time), used without a test of
    its own: for replaying what came after, with what came before."""
    per_day, runs, screens, shows = lesson_channel(key, before)
    if not per_day or sum(p['breaks'] for p in per_day) < 1:
        return None
    X = np.concatenate([p['X'] for p in per_day])
    y = np.concatenate([p['y'] for p in per_day])
    if not (y == 1).any() or not (y == 0).any():
        return None
    return lesson_fit(X, y)


def learn_lessons(before=None, only=None, force=False):
    """(1.9.0) Every channel's lesson learned again (one learning at a time); the answers use the new ones at once."""
    if not lesson_lock.acquire(blocking=False):
        return False
    try:
        lessons['state'] = 'learning'
        say('the lessons: learning')
        lesson_prune()
        keys = sorted(os.listdir(looks_dir())) if os.path.isdir(looks_dir()) else []
        if only:
            keys = [k for k in keys if k in only]
        chans, own, use = {}, {}, {}
        for key in keys:
            try:
                out, got = lesson_learn_one(key, before)
            except Exception as e:
                out, got = {'channel': key, 'passed': False, 'note': "couldn't learn (%s)" % type(e).__name__}, None
            if got is None and force:  # (a test's: POST /lessons/learn?force=1) used without passing a test of its own
                got = lesson_force(key, before)
                if got is not None:
                    out['note'] = 'used: its own, forced (a test: learned from all of it, not tested)'
                    out['forced'] = True
            chans[key] = out
            if got is not None:
                own[key] = got
                use[key] = got + ('own', out.get('held') or [])
        for key in keys:
            if key in own:
                continue
            try:
                b = lesson_borrow(key, own, before)
            except Exception as e:
                b = None
                chans[key]['note'] = chans[key].get('note', '') + " (borrowing couldn't be tried: %s)" % type(e).__name__
            if b is not None:
                other, r = b
                chans[key].update({'borrowed': other, 'borrow_test': r, 'held': r['held'],
                                   'note': "used: %s's, which passed this channel's own test" % other})
                use[key] = own[other] + (other, r['held'])
        lessons.update({'state': 'learned', 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': MODEL_ID, 'channels': chans,
                        'before': before})
        lesson_state['use'] = use
        lesson_save(use)
        say('the lessons: learned (%d channels, %d used: %s)' % (len(chans), len(use), ', '.join(
            '%s%s' % (k, '' if v[3] == 'own' else ' (by %s)' % v[3]) for k, v in sorted(use.items())) or 'none yet'))
        return True
    finally:
        lesson_lock.release()


def lesson_save(use):
    """The lessons in use, kept (back on a restart; another main model's aren't)."""
    data = {'model': MODEL_ID, 'summary': lessons, 'use': {k: {
        'mid': base64.b64encode(v[0].astype('<f4').tobytes()).decode('ascii'),
        'dir': base64.b64encode(v[1].astype('<f4').tobytes()).decode('ascii'),
        'line': v[2], 'from': v[3], 'held': v[4]} for k, v in use.items()}}
    os.makedirs(LESSONS, exist_ok=True)
    tmp = os.path.join(LESSONS, 'lessons.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp, os.path.join(LESSONS, 'lessons.json'))


def lesson_load():
    try:
        with open(os.path.join(LESSONS, 'lessons.json'), encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    if data.get('model') != MODEL_ID:
        return
    lessons.update(data.get('summary') or {})
    lesson_state['use'] = {k: (np.frombuffer(base64.b64decode(v['mid']), '<f4').copy(), np.frombuffer(base64.b64decode(v['dir']), '<f4').copy(),
                               float(v['line']), v['from'], list(v.get('held') or [])) for k, v in (data.get('use') or {}).items()}


def lesson_prune():
    """The AI's numbers and the events kept LESSON_DAYS days (another main model's too), then deleted."""
    oldest = lesson_day(time.time() * 1000 - LESSON_DAYS * 86400_000)
    for root in (os.path.join(LESSONS, 'looks'), events_dir()):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            for n in filenames:
                if n[:8].isdigit() and n[:8] < oldest:
                    os.remove(os.path.join(dirpath, n))
            if dirpath != root and not os.listdir(dirpath):
                os.rmdir(dirpath)


def lesson_answer(key, e):
    """A channel's lesson on this picture, for its answer (or None while it has none in use)."""
    u = lesson_state['use'].get(key)
    if u is None or len(u[0]) != len(e):
        return None
    mid, d, line, frm, held = u
    return {'score': round(float((e - mid) @ d), 4), 'line': round(line, 4), 'from': frm, 'held': held}


def lessons_view():
    v = json.loads(json.dumps(lessons))
    v['used'] = sorted(lesson_state['use'])
    # What's kept for the lessons: each channel's days of the AI's numbers, its pictures today, its words today.
    today = lesson_day(time.time() * 1000)
    kept = {}
    try:
        for key in sorted(os.listdir(looks_dir())) if os.path.isdir(looks_dir()) else []:
            days = LookLog.days(key)
            p = os.path.join(looks_dir(key), today + '.bin')
            n = 0
            if os.path.isfile(p):
                with open(p, 'rb') as f:
                    head = f.read(len(LOOK_HEAD) + 2)
                if head[:len(LOOK_HEAD)] == LOOK_HEAD:
                    n = (os.path.getsize(p) - len(head)) // (12 + struct.unpack('>h', head[len(LOOK_HEAD):])[0])
            kept[key] = {'days': len(days), 'pictures_today': n}
        for key in sorted(os.listdir(events_dir())) if os.path.isdir(events_dir()) else []:
            p = os.path.join(events_dir(key), today + '.jsonl')
            counts = {}
            if os.path.isfile(p):
                with open(p, encoding='utf-8') as f:
                    for ln in f:
                        k = (json.loads(ln) or {}).get('kind', '?')
                        counts[k] = counts.get(k, 0) + 1
            kept.setdefault(key, {'days': 0, 'pictures_today': 0})['words_today'] = counts
    except (OSError, ValueError):
        pass
    v['kept'] = kept
    v['today'] = today
    return v


def lessons_loop():
    """Learns again each night: LESSON_AGAIN after the last, once no TV has asked for LESSON_QUIET."""
    while True:
        time.sleep(600)
        try:
            last = os.path.getmtime(os.path.join(LESSONS, 'lessons.json')) if os.path.isfile(os.path.join(LESSONS, 'lessons.json')) else 0
            if time.time() - last >= LESSON_AGAIN and time.time() - state['last_look'] >= LESSON_QUIET and state['ready']:
                learn_lessons()
        except Exception as e:
            lessons['state'] = "couldn't learn (%s)" % type(e).__name__
            say('the lessons: ' + lessons['state'])


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
                                    'bench': bench['state'] + (' (%s)' % bench['note'] if bench['note'] else ''),
                                    'lessons': '%s (%d in use)' % (lessons['state'], len(lesson_state['use']))})
        if self.path.split('?')[0] == '/bench':
            return self.reply(200, bench_view())
        if self.path.split('?')[0] == '/lessons':  # (1.9.0) each channel's lesson: learned, tested, used or not, why
            return self.reply(200, lessons_view())
        self.reply(404, {'note': 'GET /status, GET /bench, POST /look, POST /read, POST /ads, POST /ads/forget or POST /bench'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        path, _, query = self.path.partition('?')
        if path == '/bench':
            if bench['state'] == 'running':
                return self.reply(409, {'note': 'the bench is already running', 'bench': bench_view()})
            if not state['ready']:
                return self.reply(503, {'note': 'not until the main model is ready'})
            asked = query.split('&')
            which = 'all' if 'set=all' in asked else 'read' if 'set=read' in asked else 'one'
            if 'set=score' in asked:  # (1.8.0) the pictures' answers come with it
                n = int(self.headers.get('Content-Length') or 0)
                if n <= 0 or n > 1_000_000:
                    return self.reply(400, {'note': 'the sets to score, as JSON, up to 1 MB'})
                try:
                    which = json.loads(self.rfile.read(n))
                    if not isinstance(which, dict) or not which.get('sets'):
                        raise ValueError('no sets')
                except Exception as e:
                    return self.reply(400, {'note': 'not sets it could use (%s)' % type(e).__name__})
            if which == 'read' and not state['read_ready']:
                return self.reply(503, {'note': 'not until the reading models are ready'})
            bench['state'] = 'running'
            bench['busy'] = 'busy=1' in query.split('&')
            threading.Thread(target=bench_run, args=(which,), daemon=True).start()
            return self.reply(202, {'note': 'the bench (%s) started: GET /bench says how far it is' % ('score' if isinstance(which, dict) else which)})
        if path == '/ads/forget':
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > 1_000_000:
                return self.reply(400, {'note': "the show's pictures, up to 1 MB"})
            body = self.rfile.read(n)
            try:
                return self.reply(200, forget_ads(body))
            except LookupError:
                return self.reply(409, {'note': 'these were seen by another model (this one is %s)' % MODEL_ID})
            except Exception as e:
                return self.reply(400, {'note': 'not pictures it could use (%s)' % type(e).__name__})
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
                answer = read_logo(body, side)
            except Exception as e:
                return self.reply(400, {'note': 'not a picture it could read (%s)' % type(e).__name__})
            if 'signs=1' in query.split('&'):  # (1.6.0) what its lines say of an ad
                answer['signs'] = ad_signs(answer['lines'])
            return self.reply(200, answer)
        if path == '/lessons/event':  # (1.9.0) a TV's word on its channel: a run of the AI's ads, a screen, the show
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > 10_000:
                return self.reply(400, {'note': 'an event, as JSON'})
            try:
                return self.reply(200, lesson_event(self.rfile.read(n), 'replay=1' in query.split('&')))
            except Exception as e:
                return self.reply(400, {'note': 'not an event it could use (%s)' % type(e).__name__})
        if path == '/lessons/learn':  # (1.9.0) learned now (a test can give a time to learn from what came before)
            if not state['ready']:
                return self.reply(503, {'note': 'not until the main model is ready'})
            before, only, force = None, None, False
            for kv in query.split('&'):
                if kv.startswith('before=') and kv[7:].isdigit():
                    before = int(kv[7:])
                elif kv.startswith('channels='):
                    only = {lesson_key(c) for c in urllib.parse.unquote(kv[9:]).split(',') if lesson_key(c)}
                elif kv == 'force=1':  # a test's: a channel's lesson used without passing a test of its own
                    force = True
            if lesson_lock.locked():
                return self.reply(409, {'note': 'already learning'})
            threading.Thread(target=learn_lessons, args=(before, only, force), daemon=True).start()
            return self.reply(202, {'note': 'learning: GET /lessons says when it is done'})
        if path != '/look':
            return self.reply(404, {'note': 'POST /look, POST /read, POST /ads or POST /lessons/event'})
        box, ch, t, replay = None, None, None, False
        for kv in query.split('&'):
            if kv.startswith('spot='):
                box = spot_box(kv[5:])
            elif kv.startswith('ch='):  # (1.9.0) the channel and the picture's own time (ms), for its lesson
                ch = urllib.parse.unquote(kv[3:])
            elif kv.startswith('t=') and kv[2:].isdigit():
                t = int(kv[2:])
            elif kv == 'replay=1':
                replay = True
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > 4_000_000:
            return self.reply(400, {'note': 'a JPEG, up to 4 MB'})
        body = self.rfile.read(n)
        if not state['ready']:
            return self.reply(503, {'note': state['note']})
        try:
            return self.reply(200, look(body, box, ch, t, replay))
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
    lesson_load()  # (1.9.0) the channels' lessons in use (this main model's)
    threading.Thread(target=lessons_loop, daemon=True).start()
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
