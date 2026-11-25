#!/usr/bin/env python3
"""
Home media: a Home Assistant app (add-on) for the Raspberry Pi. Home, on the TVs, asks it for the file it's about to
play; it downloads the whole file onto its own USB drive over one steady connection, and plays it to the TV as it
comes in, so every jump is instant once that part is in. A jump to a part that isn't in yet moves the download there
first; then it fills in the rest. Files go once watched, after a few days unused, or when the drive runs low.
Announced on the home network (mDNS, _homemedia._tcp, port 8766), so the TVs find it.

The drive is the app's own: labelled HOMEMEDIA (ext4), mounted inside this app only, never Home Assistant's storage.
With no drive, nothing is written anywhere: the TVs play from the source as before. A new drive is formatted once,
when the app's option says so. Links are kept in memory only: never logged, never written down.
"""
import base64
import hashlib
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

VERSION = '1.0.1'
PORT = 8766
LABEL = 'HOMEMEDIA'
TEST_ROOT = os.environ.get('HOME_MEDIA_TEST_ROOT')  # (tests off the Pi) a folder instead of the drive
MOUNT = TEST_ROOT or '/mnt/homemedia'
DIR = os.path.join(MOUNT, 'home-media')
DATA = os.environ.get('HOME_MEDIA_DATA', '/data')
OPTIONS = os.path.join(DATA, 'options.json')
CHUNK = 1 << 20            # the drive keeps track of what's in a mebibyte at a time
BLOCK = 256 << 10          # read and written in quarter mebibytes
AHEAD = 32 << 20           # a part wanted this close ahead of the download comes soon enough: no jump
KEEP_FREE = int(os.environ.get('HOME_MEDIA_KEEP_FREE', 50 << 30))  # kept free on the drive (oldest unused go first)
MIN_DRIVE = 100 << 30      # never formats anything smaller (a stray USB stick)
WAIT = 60                  # a TV waits this long for a part before its connection ends (it asks again)
RUNNING = 2                # files downloading at once (the one playing, and the next episode's)


def say(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def size_text(b):
    if b >= 1 << 30:
        return '%.1f GB' % (b / float(1 << 30))
    return '%.0f MB' % (b / float(1 << 20))


def options():
    try:
        with open(OPTIONS) as f:
            return json.load(f)
    except Exception:
        return {}


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    tmp = '%s.%d.tmp' % (path, threading.get_ident())
    with open(tmp, 'w') as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def private(ip):
    """Only the home network's own devices are answered."""
    if ip.startswith('::ffff:'):
        ip = ip[7:]
    if ip.startswith(('10.', '192.168.', '127.', '169.254.')) or ip == '::1' or ip.startswith(('fc', 'fd', 'fe80')):
        return True
    m = re.match(r'^172\.(\d+)\.', ip)
    return bool(m) and 16 <= int(m.group(1)) <= 31


# ---- The drive ----

def run(cmd, timeout=600):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.decode('utf-8', 'replace').strip()


class Drive:
    """The app's own drive: found, formatted once if the option says so, mounted inside this app, watched."""

    def __init__(self):
        self.ok = bool(TEST_ROOT)
        self.why = 'a test folder' if TEST_ROOT else 'not looked at yet'
        self.model = ''
        self.lock = threading.Lock()
        self.formatted = set()  # drives formatted since the app started (its settings file is rewritten only on a start)

    def disks(self):
        """The Pi's drives, from the kernel. Each part's filesystem, label and name are read off the drive itself: the
        app has no copy of the Pi's device database, so lsblk alone doesn't know them."""
        code, out = run(['lsblk', '-J', '-b', '-o', 'NAME,PATH,TYPE,SIZE,TRAN,MODEL,SERIAL'], 60)
        if code != 0:
            raise RuntimeError('lsblk: ' + out[:200])
        disks = json.loads(out).get('blockdevices', [])
        for d in disks:
            if not self.skipped(d):
                for p in self.parts(d):
                    p.update(self.probe(p.get('path')))
        return disks

    @staticmethod
    def probe(path):
        found = {'fstype': '', 'label': '', 'partname': ''}
        if path:
            code, out = run(['blkid', '-p', '-o', 'export', path], 60)
            names = {'TYPE': 'fstype', 'LABEL': 'label', 'PART_ENTRY_NAME': 'partname'}
            for line in out.splitlines():
                key, _, value = line.partition('=')
                if key in names:
                    found[names[key]] = value.replace('\\ ', ' ')
        return found

    @staticmethod
    def parts(d):
        return [d] + [c for c in d.get('children') or []]

    @staticmethod
    def skipped(d):
        """The Pi's own kinds of storage: the SD card, memory, loop devices, an NVMe drive."""
        return (d.get('name') or '').startswith(('mmcblk', 'zram', 'loop', 'ram', 'nvme'))

    @staticmethod
    def system(d):
        """Home Assistant's own disk (its hassos parts, also when it starts from a USB drive), the SD card, and the
        like: never touched."""
        if Drive.skipped(d):
            return True
        return any((p.get(k) or '').lower().startswith('hassos') for p in Drive.parts(d) for k in ('label', 'partname'))

    @staticmethod
    def in_use(path):
        """Why the drive can't be had for itself, or None: the kernel refuses this when any part of it is in use on the
        Pi (as Home Assistant's own disk always is)."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_EXCL)
        except OSError as e:
            return (e.strerror or type(e).__name__).lower()
        os.close(fd)
        return None

    @staticmethod
    def what(d):
        return '%s, %s' % ((d.get('model') or d.get('name') or '?').strip(), size_text(int(d.get('size') or 0)))

    @staticmethod
    def key(d):
        return '%s %s %s' % (d.get('path'), d.get('size'), (d.get('serial') or '').strip())

    def look(self):
        """Finds and mounts the drive (formatting a new one once, if the option says so). True if it's ready."""
        if TEST_ROOT:
            os.makedirs(DIR, exist_ok=True)
            return True
        with self.lock:
            if self.mounted():
                self.ok = True
                return True
            try:
                disks = self.disks()
            except Exception as e:
                return self.fail('the drives couldn\'t be listed (%s)' % e)
            ours = None
            for d in disks:
                for p in self.parts(d):
                    if p.get('label') == LABEL and p.get('fstype') == 'ext4' and not self.system(d):
                        ours = (d, p)
            if ours is None:
                usb = [d for d in disks if d.get('type') == 'disk' and (d.get('tran') or '') == 'usb' and not self.system(d)]
                if not usb:
                    return self.fail('no drive: plug it into the hub (on one of the Pi\'s blue USB 3 ports)')
                if len(usb) > 1:
                    return self.fail('more than one drive (%s): only the media drive should be plugged in'
                                     % ', '.join(self.what(d) for d in usb))
                d = usb[0]
                if self.key(d) in self.formatted:
                    return self.fail('the drive was formatted, but it doesn\'t read as this app\'s (%s)' % self.what(d))
                if not options().get('format_new_drive', False):
                    return self.fail('a drive is there (%s) but it isn\'t formatted for this app: turn on '
                                     '"Format a new drive" in the app\'s settings' % self.what(d))
                if int(d.get('size') or 0) < MIN_DRIVE:
                    return self.fail('the drive is too small for media (%s)' % self.what(d))
                busy = self.in_use(d.get('path'))
                if busy:
                    return self.fail('the drive (%s) is in use on the Pi (%s): it isn\'t formatted' % (self.what(d), busy))
                if not self.format(d):
                    return False
                ours = (d, {'path': d.get('path'), 'label': LABEL, 'fstype': 'ext4'})
            d, p = ours
            if p is d and d.get('children'):
                run(['blockdev', '--rereadpt', d.get('path')], 60)  # parts the kernel still lists from before
            os.makedirs(MOUNT, exist_ok=True)
            code, out = run(['mount', '-t', 'ext4', '-o', 'noatime', p.get('path'), MOUNT], 120)
            if code != 0:
                return self.fail('the drive couldn\'t be mounted (%s)' % ' '.join(out.split())[:200])
            os.makedirs(DIR, exist_ok=True)
            self.model = self.what(d)
            self.ok = True
            self.why = 'ready'
            st = os.statvfs(MOUNT)
            say('the drive is ready: %s, %s free' % (self.model, size_text(st.f_bavail * st.f_frsize)))
            return True

    def format(self, d):
        """Formats the whole drive for this app (ext4, HOMEMEDIA): once, and only when the app's setting says so."""
        path = d.get('path')
        say('formatting the new drive (%s) for this app: everything on it goes' % self.what(d))
        code, out = run(['wipefs', '-a', '-f', path], 120)
        if code != 0:
            return self.fail('the drive couldn\'t be cleared (%s)' % out[:200])
        run(['blockdev', '--rereadpt', path], 60)  # the kernel forgets the parts that were on it
        code, out = run(['mkfs.ext4', '-F', '-L', LABEL, '-m', '0', '-T', 'largefile4',
                         '-E', 'lazy_itable_init=1,lazy_journal_init=1', path], 1800)
        if code != 0:
            return self.fail('the drive couldn\'t be formatted (%s)' % out[-200:])
        self.formatted.add(self.key(d))
        set_option('format_new_drive', False)
        say('the drive is formatted (ext4, labelled %s)' % LABEL)
        return True

    def mounted(self):
        try:
            with open('/proc/mounts') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 1 and parts[1] == MOUNT:
                        os.statvfs(MOUNT)
                        return True
        except Exception:
            pass
        return False

    def fail(self, why):
        if why != self.why:
            say(why)
        self.ok = False
        self.why = why
        return False

    def free(self):
        try:
            st = os.statvfs(MOUNT)
            return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize
        except Exception:
            return 0, 0

    def check(self):
        """Still there and answering (a drive unplugged gives errors): if not, let go of it and look again."""
        if TEST_ROOT:
            return self.look()
        if self.ok:
            try:
                probe = os.path.join(DIR, '.alive')
                with open(probe, 'w') as f:
                    f.write(str(time.time()))
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as e:
                say('the drive stopped answering (%s): letting go of it' % type(e).__name__)
                run(['umount', '-l', MOUNT], 60)
                self.ok = False
                self.why = 'the drive stopped answering'
        return self.look()


def set_option(name, value):
    """The app's own setting changed (the format switch goes back off once used)."""
    token = os.environ.get('SUPERVISOR_TOKEN')
    if not token:
        return
    opts = options()
    opts[name] = value
    try:
        req = urllib.request.Request('http://supervisor/addons/self/options', data=json.dumps({'options': opts}).encode(),
                                     headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        say('the setting couldn\'t be changed back (%s)' % type(e).__name__)


drive = Drive()


# ---- The files ----

class Gone(Exception):
    pass


class Item:
    """One file: what's in (a mebibyte at a time), its download, and the TVs reading it."""

    def __init__(self, iid, meta):
        self.id = iid
        self.key = meta.get('key', '')
        self.name = meta.get('name', '')
        self.ext = meta.get('ext', 'mkv')
        self.size = int(meta.get('size', 0))
        self.done = bytearray((self.size + CHUNK - 1) // CHUNK) if self.size else bytearray()
        for a, b in meta.get('have', []):
            for c in range(a, min(b, len(self.done))):
                self.done[c] = 1
        self.count = sum(self.done)
        self.created = meta.get('created', time.time())
        self.last_used = meta.get('last_used', time.time())
        self.watched = meta.get('watched', False)
        self.prefetch = meta.get('prefetch', False)
        self.ranges = meta.get('ranges', True)
        self.url = None             # in memory only
        self.headers = {}
        self.state = 'complete' if self.complete() else 'waiting for its link'
        self.rate = 0.0
        self.want = -1
        self.readers = 0
        self.cond = threading.Condition()
        self.thread = None
        self.saved_at = 0
        self.save_lock = threading.Lock()
        self.gone = False

    def data(self):
        return os.path.join(DIR, self.id + '.data')

    def meta_path(self):
        return os.path.join(DIR, self.id + '.json')

    def complete(self):
        return self.size > 0 and self.count == len(self.done)

    def have_bytes(self):
        if not self.size:
            return 0
        n = self.count * CHUNK
        if self.done and self.done[-1]:
            n -= len(self.done) * CHUNK - self.size
        return max(0, n)

    def ranges_held(self):
        out, start = [], None
        for i, v in enumerate(self.done):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append([start, i])
                start = None
        if start is not None:
            out.append([start, len(self.done)])
        return out

    def save(self, force=False):
        if self.gone or (not force and time.time() - self.saved_at < 5):
            return
        self.saved_at = time.time()
        try:
            with self.save_lock:
                save_json(self.meta_path(), {'key': self.key, 'name': self.name, 'ext': self.ext, 'size': self.size,
                                         'have': self.ranges_held(), 'created': self.created, 'last_used': self.last_used,
                                             'watched': self.watched, 'prefetch': self.prefetch, 'ranges': self.ranges})
        except Exception as e:
            say('%s: couldn\'t note what\'s in (%s)' % (self.name, type(e).__name__))

    def summary(self):
        return {'id': self.id, 'name': self.name, 'size': self.size, 'have': self.have_bytes(), 'complete': self.complete(),
                'state': self.state, 'rate': int(self.rate), 'prefetch': self.prefetch, 'watched': self.watched,
                'last_used': int(self.last_used), 'play': '/play/' + self.id}

    # -- what to download next --

    def next_pos(self, at=None):
        """Where the download goes on from: the part a TV is waiting for, else (going on from {@code at}) the next part
        that isn't in, else the first one that isn't; None once it's all in."""
        if not self.size:
            return 0
        if self.want >= 0:
            c = self.want // CHUNK
            if c < len(self.done) and not self.done[c]:
                return c * CHUNK
        start = 0 if at is None else at // CHUNK
        for c in range(start, len(self.done)):
            if not self.done[c]:
                return c * CHUNK
        for c in range(0, min(start, len(self.done))):
            if not self.done[c]:
                return c * CHUNK
        return None

    def jump_wanted(self, off):
        """A TV is waiting for a part this connection won't reach soon: start over there."""
        w = self.want
        if w < 0 or not self.size:
            return False
        c = w // CHUNK
        if c >= len(self.done) or self.done[c]:
            return False
        return not (off <= w < off + AHEAD)

    # -- the download (one connection at a time) --

    def download(self):
        backoff = 5
        at = None
        while not self.gone:
            with self.cond:
                pos = self.next_pos(at)
                if pos is None:
                    self.state = 'complete'
                    self.rate = 0.0
                    self.save(True)
                    self.cond.notify_all()
                    say('%s: all in (%s)' % (self.name, size_text(self.size)))
                    return
                if not self.url:
                    self.state = 'waiting for its link'
                    self.cond.wait(30)
                    continue
                url, headers = self.url, dict(self.headers)
            if not drive.ok:
                self.state = 'no drive'
                time.sleep(10)
                continue
            if self.prefetch and others_busy(self):
                self.state = 'waiting (the one playing first)'
                time.sleep(5)
                continue
            self.state = 'downloading'
            try:
                at = self.fetch(url, headers, pos)
                backoff = 5
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404, 410):
                    with self.cond:
                        if self.url == url:
                            self.url = None  # the link ran out: the TV sends a fresh one when it asks again
                    say('%s: its link no longer works (%d): waiting for a new one' % (self.name, e.code))
                else:
                    say('%s: the source said %d; again in %d s' % (self.name, e.code, backoff))
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
            except Gone:
                return
            except Exception as e:
                say('%s: the connection broke (%s); again in %d s' % (self.name, type(e).__name__, backoff))
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def fetch(self, url, headers, pos):
        """One connection from {@code pos}, written in as it comes; returns where it stopped."""
        h = dict(headers)
        h.setdefault('User-Agent', 'HomeMedia/' + VERSION)
        if pos > 0 or not self.size:
            h['Range'] = 'bytes=%d-' % pos
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30)
        try:
            start, total = pos, None
            cr = resp.headers.get('Content-Range')
            if resp.status == 206 and cr:
                m = re.match(r'bytes (\d+)-(\d+)/(\d+|\*)', cr)
                if m:
                    start = int(m.group(1))
                    if m.group(3) != '*':
                        total = int(m.group(3))
            else:
                start = 0  # the source sent the whole file: no jumps with it
                cl = resp.headers.get('Content-Length')
                total = int(cl) if cl and cl.isdigit() else None
                if pos > 0 and self.ranges:
                    self.ranges = False
                    say('%s: its source can\'t jump: it comes in from the start' % self.name)
            if not self.size:
                if not total:
                    raise RuntimeError('no size')
                self.begin(total)
            elif total and total != self.size:
                raise RuntimeError('the file changed size')
            fd = os.open(self.data(), os.O_RDWR)
            try:
                off = start
                t0, got0 = time.time(), 0
                while not self.gone:
                    c = off // CHUNK
                    if c >= len(self.done):
                        break
                    if self.done[c] and self.ranges:
                        return off  # this part is in already: go on from the next that isn't
                    if self.ranges and self.jump_wanted(off):
                        say('%s: a jump to %s: the download goes there first' % (self.name, size_text(self.want)))
                        return None
                    end = min((c + 1) * CHUNK, self.size)
                    while off < end:
                        b = resp.read(min(BLOCK, end - off))
                        if not b:
                            raise RuntimeError('the source stopped sending')
                        if not self.done[c]:
                            os.pwrite(fd, b, off)
                        off += len(b)
                        got0 += len(b)
                    with self.cond:
                        if not self.done[c]:
                            self.done[c] = 1
                            self.count += 1
                        self.cond.notify_all()
                    dt = time.time() - t0
                    if dt >= 2:
                        r = got0 / dt
                        self.rate = r if self.rate == 0 else self.rate * 0.7 + r * 0.3
                        t0, got0 = time.time(), 0
                    self.save()
                if self.gone:
                    raise Gone()
                return off
            finally:
                os.close(fd)
                self.save(True)
        finally:
            resp.close()

    def begin(self, total):
        with self.cond:
            self.size = total
            self.done = bytearray((total + CHUNK - 1) // CHUNK)
            self.count = 0
            with open(self.data(), 'wb') as f:
                f.truncate(total)  # sparse: the drive fills as parts come in
            self.cond.notify_all()
        say('%s: %s, downloading' % (self.name, size_text(total)))
        self.save(True)

    # -- the TVs reading it --

    def wait_for(self, off):
        """Until the part holding {@code off} is in (a TV waiting: the download goes there first if it must)."""
        deadline = time.time() + WAIT
        with self.cond:
            while True:
                if self.gone:
                    raise Gone()
                c = off // CHUNK
                if self.size and c < len(self.done) and self.done[c]:
                    if self.want == off:
                        self.want = -1
                    return
                self.want = off
                self.cond.notify_all()
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError()
                self.cond.wait(min(1.0, left))

    def wait_size(self):
        deadline = time.time() + WAIT
        with self.cond:
            while not self.size:
                if self.gone:
                    raise Gone()
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError()
                self.want = 0
                self.cond.wait(min(1.0, left))

    def start(self):
        if self.thread and self.thread.is_alive():
            with self.cond:
                self.cond.notify_all()
            return
        if self.complete():
            self.state = 'complete'
            return
        self.thread = threading.Thread(target=self.download, name='dl-' + self.id, daemon=True)
        self.thread.start()

    def drop(self):
        with self.cond:
            self.gone = True
            self.cond.notify_all()
        for p in (self.data(), self.meta_path()):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except Exception as e:
                say('%s: couldn\'t remove a file (%s)' % (self.name, type(e).__name__))


items = {}
items_lock = threading.Lock()


def others_busy(me):
    """Something being played is still coming in (the next episode waits for it)."""
    with items_lock:
        return any(i is not me and not i.prefetch and not i.gone and i.state == 'downloading' for i in items.values())


def item_id(key):
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]


def load_items():
    with items_lock:
        if items or not os.path.isdir(DIR):
            return
        for f in os.listdir(DIR):
            if not f.endswith('.json'):
                continue
            iid = f[:-5]
            meta = load_json(os.path.join(DIR, f), None)
            if meta is None or not os.path.exists(os.path.join(DIR, iid + '.data')):
                continue
            items[iid] = Item(iid, meta)
        if items:
            say('%d files on the drive' % len(items))


def get(body):
    """A TV asks for a file (POST /get): it starts or goes on coming in; a fresh link replaces an old one."""
    key = str(body.get('key') or '').strip()
    url = str(body.get('url') or '').strip()
    if not key or not url.lower().startswith(('http://', 'https://')):
        raise ValueError('key and link needed')
    iid = item_id(key)
    with items_lock:
        it = items.get(iid)
        if it is None:
            name = str(body.get('name') or '')[:120] or iid
            ext = re.sub(r'[^a-z0-9]', '', str(body.get('ext') or 'mkv').lower())[:5] or 'mkv'
            it = Item(iid, {'key': key, 'name': name, 'ext': ext, 'prefetch': bool(body.get('prefetch'))})
            items[iid] = it
            say('%s: asked for%s' % (name, ' (ahead of time)' if it.prefetch else ''))
        elif not body.get('prefetch'):
            it.prefetch = False  # it's playing now
    with it.cond:
        it.url = url
        hdrs = body.get('headers') or {}
        it.headers = {str(k): str(v) for k, v in hdrs.items() if str(k).lower() in ('user-agent', 'referer', 'cookie', 'authorization')}
        it.last_used = time.time()
        it.watched = False
        it.cond.notify_all()
    it.start()
    it.save(True)
    return it.summary()


def done(body):
    """He watched it (POST /done): it goes, once no TV is reading it."""
    iid = str(body.get('id') or '') or (item_id(str(body.get('key'))) if body.get('key') else '')
    with items_lock:
        it = items.get(iid)
    if not it:
        return {'ok': False}
    it.watched = True
    it.save(True)
    return {'ok': True}


def tidy():
    """Watched files go; one not watched (the next episode waiting for the weekend, one paused partway) after a week
    and a day unused (the setting: a Saturday's is still there the next Saturday); the oldest idle ones when the drive
    runs low."""
    keep_days = float(options().get('keep_days', 8) or 8)
    now = time.time()
    with items_lock:
        all_items = list(items.values())
    for it in all_items:
        if it.readers > 0:
            continue
        why = None
        if it.watched and now - it.last_used > 60:
            why = 'watched'
        elif now - it.last_used > keep_days * 86400:
            why = 'unused for %g days' % keep_days
        if why:
            forget(it, why)
    free, total = drive.free()
    if total and free < KEEP_FREE:
        # Never one being played or coming in: only what's idle, oldest first.
        idle = [i for i in all_items if not i.gone and i.readers == 0 and i.state != 'downloading']
        for it in sorted(idle, key=lambda i: i.last_used):
            forget(it, 'the drive is getting full')
            free, total = drive.free()
            if free >= KEEP_FREE:
                break
    # Leftovers with no file of their own.
    try:
        known = set()
        with items_lock:
            for iid in items:
                known.add(iid + '.data')
                known.add(iid + '.json')
        for f in os.listdir(DIR):
            if f.startswith('.') or f in known:
                continue
            if f.endswith(('.data', '.json', '.tmp')) and time.time() - os.path.getmtime(os.path.join(DIR, f)) > 600:
                os.remove(os.path.join(DIR, f))
    except Exception:
        pass


def forget(it, why):
    with items_lock:
        if items.get(it.id) is it:
            del items[it.id]
    it.drop()
    say('%s: removed (%s)' % (it.name, why))


# ---- On the network ----

CTYPES = {'mkv': 'video/x-matroska', 'mp4': 'video/mp4', 'm4v': 'video/mp4', 'webm': 'video/webm', 'avi': 'video/x-msvideo',
          'ts': 'video/mp2t', 'm2ts': 'video/mp2t', 'mov': 'video/quicktime'}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'HomeMedia/' + VERSION

    def log_message(self, fmt, *args):
        pass  # requests aren't logged (their paths are ids, but nothing a TV sends is kept)

    def allowed(self):
        if private(self.client_address[0]):
            return True
        self.send_error(403)
        return False

    def answer(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self.allowed():
            return
        if self.path == '/status':
            free, total = drive.free()
            with items_lock:
                files = [i.summary() for i in items.values()]
            return self.answer(200, {'version': VERSION, 'drive': {'ready': drive.ok, 'why': drive.why, 'model': drive.model,
                                                                 'free': free, 'size': total}, 'files': files})
        if self.path.startswith('/play/'):
            return self.play(head=False)
        self.answer(404, {'error': 'not here'})

    def do_HEAD(self):
        if not self.allowed():
            return
        if self.path.startswith('/play/'):
            return self.play(head=True)
        self.send_response(404)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        if not self.allowed():
            return
        try:
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(n) or b'{}') if n else {}
        except Exception:
            return self.answer(400, {'error': 'not JSON'})
        try:
            if self.path == '/get':
                if not drive.ok:
                    return self.answer(503, {'error': 'no drive', 'why': drive.why})
                return self.answer(200, get(body))
            if self.path == '/done':
                return self.answer(200, done(body))
        except ValueError as e:
            return self.answer(400, {'error': str(e)})
        self.answer(404, {'error': 'not here'})

    def play(self, head):
        iid = self.path[len('/play/'):].split('?')[0].split('/')[0]
        with items_lock:
            it = items.get(iid)
        if it is None or it.gone:
            return self.answer(404, {'error': 'not here'})
        it.last_used = time.time()
        it.readers += 1
        try:
            try:
                it.wait_size()
            except (TimeoutError, Gone):
                return self.answer(504, {'error': 'the file hasn\'t started coming in'})
            size = it.size
            start, end = 0, size - 1
            rng = self.headers.get('Range')
            partial = False
            if rng:
                m = re.match(r'bytes=(\d*)-(\d*)', rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        if m.group(2):
                            end = min(int(m.group(2)), size - 1)
                    else:
                        start = max(0, size - int(m.group(2)))
                    partial = True
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header('Content-Range', 'bytes */%d' % size)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', CTYPES.get(it.ext, 'application/octet-stream'))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if partial:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
            self.end_headers()
            if head:
                return
            fd = os.open(it.data(), os.O_RDONLY)
            try:
                off = start
                while off <= end:
                    it.wait_for(off)
                    c_end = min(((off // CHUNK) + 1) * CHUNK, end + 1)
                    while off < c_end:
                        b = os.pread(fd, min(BLOCK, c_end - off), off)
                        if not b:
                            raise IOError('short read')
                        self.wfile.write(b)
                        off += len(b)
                    it.last_used = time.time()
            finally:
                os.close(fd)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, Gone):
            pass
        except Exception as e:
            say('%s: playing stopped (%s)' % (it.name, type(e).__name__))
        finally:
            it.readers -= 1
            if it.want >= 0 and it.readers == 0:
                it.want = -1


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 9))  # no packet is sent: it only picks the way out
        return s.getsockname()[0]
    finally:
        s.close()


def announce():
    """On the home network as _homemedia._tcp (mDNS), so the TVs find it; again if its address changes."""
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        say('no mDNS library: the TVs will need its address')
        return
    zc, ip = None, None
    host = socket.gethostname().split('.')[0] or 'home-media'
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
                info = ServiceInfo('_homemedia._tcp.local.', 'Home media._homemedia._tcp.local.',
                                   addresses=[socket.inet_aton(ip)], port=PORT,
                                   properties={'version': VERSION}, server=host + '.local.')
                zc.register_service(info, allow_name_change=True)
                say('announced on the network at %s:%d' % (ip, PORT))
        except Exception as e:
            say('announcing failed (%s); trying again in a minute' % type(e).__name__)
            ip = None
        time.sleep(60)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A device hanging up partway (the watchdog does, after its check) isn't news; anything else, said briefly."""
        e = sys.exc_info()[1]
        if not isinstance(e, (ConnectionError, TimeoutError)):
            say('a request failed (%s)' % type(e).__name__)


def main():
    srv = Server(('0.0.0.0', PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    say('Home media %s, answering on port %d' % (VERSION, PORT))
    if not TEST_ROOT:
        threading.Thread(target=announce, daemon=True).start()
    last_tidy = 0
    while True:  # never gives up: a problem is said in the log and looked at again
        try:
            if drive.check():
                load_items()
                if time.time() - last_tidy > 600:
                    last_tidy = time.time()
                    tidy()
        except Exception as e:
            say('looking after the drive failed (%s); again in a minute' % type(e).__name__)
        time.sleep(30 if drive.ok else 60)


if __name__ == '__main__':
    main()
