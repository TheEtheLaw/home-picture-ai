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
                                                "ads_note"}
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
import socket
import threading
import time
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

VERSION = '1.4.0'
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
         'read_ready': False, 'read_note': 'waiting for the other models', 'reads': 0, 'read_ms': 0.0}
session = None
spot_session = None
reader = None
read_lock = threading.Lock()
run_lock = threading.Lock()
spot_lock = threading.Lock()
spot_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


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
    state['ms'] = ms if state['looks'] == 1 else state['ms'] * 0.95 + ms * 0.05
    out = {'model': MODEL_ID, 'score': float(e @ direction), 'ad': AD, 'like': LIKE, 'sure': SURE, 'ms': round(ms),
           'known': round(ads.known(e), 4), 'embedding': base64.b64encode(e.astype('<f4').tobytes()).decode('ascii')}
    if spot is not None:
        try:
            out.update(spot.result(timeout=5))
        except Exception:
            pass  # the main answer still counts
    return out


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
                                    'read_side_max': READ_SIDE_MAX, 'ads': ads.n, 'ads_note': ads.note})
        self.reply(404, {'note': 'GET /status, POST /look, POST /read or POST /ads'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        path, _, query = self.path.partition('?')
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
