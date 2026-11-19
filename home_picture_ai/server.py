"""Home's picture AI on the Raspberry Pi (a Home Assistant add-on).

The TVs on the home network send it the picture their ad watch is looking at (320x180, JPEG) and get back how much
it's an ad and what the AI saw (for how much it looks like the channel's program, which each TV keeps). The model is
CLIP ViT-B/32's picture tower (OpenAI, MIT; Xenova's ONNX copy at a fixed version), downloaded once from Hugging Face
into the add-on's own storage and used only if its SHA-256 is right. Nothing leaves the home network: the pictures
are looked at and forgotten, never kept or logged. Announced on the network (mDNS, _homeai._tcp) so the TVs find it.

    POST /look   a JPEG  ->  {"model", "score", "embedding" (512 float32, base64), "ad", "like", "ms"}
    GET  /status         ->  {"model", "version", "ready", "note", "looks", "ms", "up"}
"""
import base64
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

VERSION = '1.0.0'
PORT = 8765
MODEL_ID = 'clip-vit-b32'
MODEL_URL = ('https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/'
             '6b7c523703ae26e39be4b03774966f8571c7f523/onnx/vision_model.onnx')
MODEL_SIZE = 351_685_709
MODEL_SHA = 'fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40'
# Over AD: an ad; under LIKE: not the program the channel has been showing (set on his two recordings).
AD, LIKE = -0.022, 0.70
SIZE = 224
MAX_PIXELS = 4_000_000  # a picture bigger than 2560x1440 is turned away
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/data' if os.path.isdir('/data') else HERE
MODEL = os.path.join(DATA, 'clip-vit-b32-vision.onnx')
THREADS = int(os.environ.get('AI_THREADS', '3'))

direction = np.fromfile(os.path.join(HERE, 'direction.bin'), dtype='<f4')
state = {'ready': False, 'note': 'starting', 'looks': 0, 'ms': 0.0, 'since': time.time()}
session = None
run_lock = threading.Lock()


def say(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), s, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch():
    """The model in place and checked (a half-download goes on where it stopped); tries until it's there."""
    wait = 30
    while not (os.path.isfile(MODEL) and os.path.getsize(MODEL) == MODEL_SIZE):
        part = MODEL + '.part'
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > MODEL_SIZE:
            os.remove(part)
            have = 0
        state['note'] = 'downloading the model (%d of %d MB)' % (have // 1_000_000, MODEL_SIZE // 1_000_000)
        say(state['note'])
        try:
            req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'home-picture-ai', **({'Range': 'bytes=%d-' % have} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, 'ab' if have and r.status == 206 else 'wb') as out:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            if os.path.getsize(part) != MODEL_SIZE:
                raise IOError('ended at %d MB' % (os.path.getsize(part) // 1_000_000))
            if sha256(part) != MODEL_SHA:
                os.remove(part)
                raise IOError("its check didn't match (deleted)")
            os.replace(part, MODEL)
            say('the model downloaded and checked')
        except Exception as e:
            state['note'] = "the model didn't download (%s); trying again in %d s" % (type(e).__name__, wait)
            say(state['note'])
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


def look(jpeg):
    im = Image.open(io.BytesIO(jpeg))
    if im.width * im.height > MAX_PIXELS:
        raise ValueError('too big')
    rgb = np.asarray(im.convert('RGB'))
    x = prepare(rgb)
    t0 = time.time()
    with run_lock:
        e = session.run(['image_embeds'], {'pixel_values': x})[0][0]
    ms = (time.time() - t0) * 1000
    e = e / np.linalg.norm(e)
    state['looks'] += 1
    state['ms'] = ms if state['looks'] == 1 else state['ms'] * 0.95 + ms * 0.05
    return {'model': MODEL_ID, 'score': float(e @ direction), 'ad': AD, 'like': LIKE, 'ms': round(ms),
            'embedding': base64.b64encode(e.astype('<f4').tobytes()).decode('ascii')}


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
                                    'ms': round(state['ms']), 'up': round(time.time() - state['since'])})
        self.reply(404, {'note': 'GET /status or POST /look'})

    def do_POST(self):
        if not home_network(self.client_address[0]):
            return self.reply(403, {'note': 'home network only'})
        if self.path.split('?')[0] != '/look':
            return self.reply(404, {'note': 'POST /look'})
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > 4_000_000:
            return self.reply(400, {'note': 'a JPEG, up to 4 MB'})
        body = self.rfile.read(n)
        if not state['ready']:
            return self.reply(503, {'note': state['note']})
        try:
            return self.reply(200, look(body))
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
            fetch()
            if load():
                break
        except Exception as e:
            state['note'] = "the model didn't load (%s); trying again in %d s" % (type(e).__name__, wait)
            say(state['note'])
            time.sleep(wait)
            wait = min(wait * 2, 1800)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
