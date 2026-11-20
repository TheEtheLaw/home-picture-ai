"""Home's picture AI on the Raspberry Pi (a Home Assistant add-on).

The TVs on the home network send it the picture their ad watch is looking at (320x180, JPEG) and get back how much
it's an ad and what the AI saw (for how much it looks like the channel's program, which each TV keeps). The model is
CLIP ViT-B/32's picture tower (OpenAI, MIT; Xenova's ONNX copy at a fixed version), downloaded once from Hugging Face
into the add-on's own storage and used only if its SHA-256 is right. Since 1.1.0 a TV can also ask about one spot of
the picture, where it has found the channel's own logo: Apple's MobileCLIP2-S0 picture tower looks at that spot,
zoomed, on its own core at the same time, and says what it saw there (for how much it looks like the logo's spot
usually does, which the TV keeps). Nothing leaves the home network: the pictures are looked at and forgotten, never
kept or logged. Announced on the network (mDNS, _homeai._tcp) so the TVs find it.

    POST /look[?spot=x0,y0,x1,y1]   a JPEG  ->  {"model", "score", "embedding" (512 float32, base64), "ad", "like",
                                                "sure", "ms"}, and for a spot {"spot_model", "spot", "spot_like", "spot_ms"}
    GET  /status                            ->  {"model", "version", "ready", "note", "looks", "ms", "up", "spot_model",
                                                "spot_ready", "spot_note", "spot_looks", "spot_ms"}
"""
import base64
import concurrent.futures
import hashlib
import http.server
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

VERSION = '1.1.0'
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

direction = np.fromfile(os.path.join(HERE, 'direction.bin'), dtype='<f4')
state = {'ready': False, 'note': 'starting', 'looks': 0, 'ms': 0.0, 'since': time.time(),
         'spot_ready': False, 'spot_note': 'waiting for the main model', 'spot_looks': 0, 'spot_ms': 0.0}
session = None
spot_session = None
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
                                    'spot_ms': round(state['spot_ms'])})
        self.reply(404, {'note': 'GET /status or POST /look'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        path, _, query = self.path.partition('?')
        if path != '/look':
            return self.reply(404, {'note': 'POST /look'})
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
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
