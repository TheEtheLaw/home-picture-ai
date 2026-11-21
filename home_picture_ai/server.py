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
move. Nothing leaves the home network: the pictures are looked at and forgotten, never kept or logged. Announced on
the network (mDNS, _homeai._tcp) so the TVs find it.

    POST /look[?spot=x0,y0,x1,y1]   a JPEG  ->  {"model", "score", "embedding" (512 float32, base64), "ad", "like",
                                                "sure", "ms"}, and for a spot {"spot_model", "spot", "spot_like", "spot_ms"}
    POST /read                      a PNG or JPEG (a logo's spot)  ->  {"read_model", "lines" [{"text", "score"}],
                                                "read_ms"}
    GET  /status                            ->  {"model", "version", "ready", "note", "looks", "ms", "up", "spot_model",
                                                "spot_ready", "spot_note", "spot_looks", "spot_ms", "read_model",
                                                "read_ready", "read_note", "reads", "read_ms"}
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

VERSION = '1.2.0'
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
READ_LINE_H = 48         # each line is read this high

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

    def read(self, im):
        rgb = im.convert('RGB')
        w, h = rgb.size
        k = READ_SIDE / max(w, h)
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


def read_logo(data):
    im = Image.open(io.BytesIO(data))
    if im.width * im.height > READ_PIXELS or min(im.width, im.height) < 8:
        raise ValueError('size')
    t0 = time.time()
    with read_lock:
        lines = reader.read(im)
    ms = (time.time() - t0) * 1000
    state['reads'] += 1
    state['read_ms'] = ms if state['reads'] == 1 else state['read_ms'] * 0.9 + ms * 0.1
    return {'read_model': READ_ID, 'lines': lines, 'read_ms': round(ms)}


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
           'embedding': base64.b64encode(e.astype('<f4').tobytes()).decode('ascii')}
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
                                    'read_note': state['read_note'], 'reads': state['reads'], 'read_ms': round(state['read_ms'])})
        self.reply(404, {'note': 'GET /status, POST /look or POST /read'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        path, _, query = self.path.partition('?')
        if path == '/read':
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > 4_000_000:
                return self.reply(400, {'note': 'a PNG or JPEG, up to 4 MB'})
            body = self.rfile.read(n)
            if not state['read_ready']:
                return self.reply(503, {'note': state['read_note']})
            try:
                return self.reply(200, read_logo(body))
            except Exception as e:
                return self.reply(400, {'note': 'not a picture it could read (%s)' % type(e).__name__})
        if path != '/look':
            return self.reply(404, {'note': 'POST /look or POST /read'})
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
