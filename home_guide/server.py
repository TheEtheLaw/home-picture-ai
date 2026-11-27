"""Home guide: a TV guide for the channels the TVs at home ask for, made each day on the Pi from the TV listing sites
the iptv-org EPG grabber reads (its code kept up to date here every day), for the channels no better guide covers.

The TVs say which of their channels need a guide (POST /channels: each channel's key, name and provider; names only,
never an address). Each is matched by its name to a channel of the listing sites the grabber reads honestly (no
made-up browser, no way around a site's checks): its words, its feed (East or West), its country. A match must be
clear (well ahead of the next best), or there is none: no guide is better than another channel's. The grabber reads
the matched channels' next two days twice a day, and the TVs take the shows (GET /programs). GET /matches lists what
each channel was matched to and how sure, GET /status how it stands.
"""
import calendar
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET

VERSION = '1.0.1'
PORT = int(os.environ.get('HOME_GUIDE_PORT', 8767))
DATA = os.environ.get('HOME_GUIDE_DATA', '/data')
EPG = os.environ.get('HOME_GUIDE_EPG', os.path.join(DATA, 'epg'))
REPO = 'https://github.com/iptv-org/epg.git'
# The listing sites read: their grabber's ways are honest (no made-up browser, no way around a site's checks).
SITES = ('tvpassport.com', 'tvinsider.com', 'tv24.co.uk', 'virgintvgo.virginmedia.com', 'plex.tv', 'sky.com')
SITE_ORDER = {s: i for i, s in enumerate(SITES)}
# Each site's channels by country, where a site covers several (sky.com: its UK and Irish ones).
SITE_COUNTRIES = {'tvpassport.com': ('us', 'ca'), 'tvinsider.com': ('us',), 'tv24.co.uk': ('uk',),
                  'virgintvgo.virginmedia.com': ('uk',), 'plex.tv': ('us', 'uk'), 'sky.com': ('uk',)}
DAYS = 2
GRAB_EVERY = 12 * 3600       # s: the shows read again this often
UPDATE_EVERY = 24 * 3600     # s: the grabber's code and its sites' channels brought up to date this often
SETTLE = 120                 # s: a new list of channels from a TV is matched and read this long after it came
MATCH_LEAST = 0.8            # a match this good at least (1.0.1: 0.74 took "Sportsnet LA Lakers" for SportsNet LA),
MATCH_AHEAD = 0.06           # and this far ahead of the next best of another name
KEEP_BACK, KEEP_AHEAD = 2 * 3600_000, 48 * 3600_000   # ms of shows kept around now
DESC_MOST = 300
TEST = os.environ.get('HOME_GUIDE_TEST') == '1'
GRAB_CMD = os.environ.get('HOME_GUIDE_GRAB')  # a test's stand-in for the grabber (a command line), else npm's

lock = threading.Lock()
work = threading.Event()     # something to do now (a TV's new list)
state = {'state': 'starting', 'note': '', 'updated_at': 0, 'grabbed_at': 0, 'grab_s': 0, 'changed_at': 0,
         'grab_note': '', 'index': 0}
wanted = {}      # TV → {'at': s, 'channels': [{'key', 'name', 'source'}]}
matches = {}     # key → {'name', 'site', 'site_id', 'site_name', 'lang', 'score'} or {'name', 'none': why}
guide = {}       # key → [[start, stop, title, desc], ...]
index = []       # the listing sites' channels: (site, site_id, lang, name, tokens, feed, country)


def say(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def path(name):
    return os.path.join(DATA, name)


def load_json(name, default):
    try:
        with open(path(name), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(name, obj):
    tmp = path(name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path(name))


def private(ip):
    """Only the home network's own devices are answered."""
    if ip.startswith('::ffff:'):
        ip = ip[7:]
    if ip.startswith(('10.', '192.168.', '127.', '169.254.')) or ip == '::1' or ip.startswith(('fc', 'fd', 'fe80')):
        return True
    m = re.match(r'^172\.(\d+)\.', ip)
    return bool(m) and 16 <= int(m.group(1)) <= 31


# ---- Names: a channel's words, its feed and its country ----

COUNTRY_WORDS = {'us': 'us', 'usa': 'us', 'uk': 'uk', 'gb': 'uk', 'ca': 'ca', 'can': 'ca', 'ie': 'uk'}
QUALITY = {'hd', 'fhd', 'uhd', '4k', 'sd', 'hevc', 'h264', 'h265', '1080', '1080p', '720p', '50fps', '60fps', 'raw',
           'backup', 'vip', 'hq', 'feed', 'the', 'dv', 'sdr', 'hdr'}
FEEDS = {'west': 'west', 'pacific': 'west', 'east': 'east', 'eastern': 'east'}
# Names written two ways: one form for both.
SAME = [(r'\bfs1\b', 'fox sports 1'), (r'\bfs2\b', 'fox sports 2'), (r'\bfox sports one\b', 'fox sports 1'),
        (r'\bnbcsn\b', 'nbc sports network'), (r'\bcnbc\b', 'cnbc'), (r'\bespnews\b', 'espnews'),
        (r'\btnt sports\b', 'tnt sports'), (r'\bbt sport\b', 'tnt sports'), (r'\bsky sports main event\b', 'sky sports main event'),
        (r'\bsky sports pl\b', 'sky sports premier league'), (r'\bsportsnet la\b', 'spectrum sportsnet la'),
        (r'\bspectrum sports net\b', 'spectrum sportsnet'), (r'\bbally sports\b', 'fanduel sports network'),
        (r'\band\b', 'and')]


def plain(s):
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(ch for ch in s if not unicodedata.combining(ch)).lower()


def variants(name):
    """The ways to read one of his channels' names: (its words, its feed, its country), a word before it read as its
    country or not ("US: ESPN" is ESPN in the US; "USA USA Network [West]" is USA Network in the US; "USA Network West"
    may be either, so both are tried)."""
    s = plain(name)
    m = re.match(r'^\s*([a-z]{2,3})\s*([:|\-])\s*(.+)$', s)
    if m and m.group(1) in COUNTRY_WORDS:
        return [words(m.group(3), COUNTRY_WORDS[m.group(1)]) + (0.0,)]
    m = re.match(r'^\s*([a-z]{2,3})\s+(.+)$', s)
    out = [words(s) + (0.0,)]
    if m and m.group(1) in COUNTRY_WORDS and len(m.group(2)) >= 2:
        w = words(m.group(2), COUNTRY_WORDS[m.group(1)])
        if set(w[0]) - SOFT:  # not "USA Network" read as "Network" in the US
            out.append(w + (0.05,))  # the other reading a little less
    return out


def words(name, country=''):
    """A channel's name as (its words, its feed: 'east', 'west' or '', its country: 'us', 'uk', 'ca' or '')."""
    s = plain(name).replace('&', ' and ').replace('+', ' plus ')
    m = re.search(r'\(\s*(usa|us|uk|can|ca)\s*\)', s)
    if m:
        country = country or COUNTRY_WORDS[m.group(1)]
        s = s[:m.start()] + ' ' + s[m.end():]
    for a, b in SAME:
        s = re.sub(a, b, s)
    toks = re.findall(r'[a-z0-9]+', s)
    toks = [t for i, t in enumerate(toks) if not (i > 0 and t == 'us')]  # "National Geographic US": its country, not its name
    feed = ''
    out = []
    for t in toks:
        if t in FEEDS:
            feed = feed or FEEDS[t]
            continue
        if t in QUALITY:
            continue
        out.append(t)
    return out, feed, country


SOFT = {'channel', 'network', 'tv', 'television'}   # words a channel's name has or hasn't ("History", "History Channel")

# (1.0.1) A local station: its call sign, in brackets or first ("FOX 11 (KTTV) Los Angeles", "KTLA Los Angeles"; "KABC-TV2"
# its second channel; not a word like "WEST" in "Fox News - WEST"), the network it's named for and its place.
_SIGN = r'([KW][A-Z]{2,3})(?:-?(?:DT|TV|LD|CD|CA|LP))?(\d{1,2})?'
CALL = re.compile(r'[(\[]\s*' + _SIGN + r'\s*[)\]]|^\s*(?:[A-Za-z]{2,3}\s*[:|]\s*)?' + _SIGN + r'\s+\S')
NETWORKS = {'abc', 'nbc', 'cbs', 'fox', 'cw', 'pbs', 'mnt', 'mytv', 'telemundo', 'univision', 'uni', 'unimas', 'ion'}


def calls(name):
    """A local station's call signs in a name, each with its channel (0: its main one; 2 in "KABC-TV2", 4 in "KTTV4")."""
    out = []
    for m in CALL.finditer(name or ''):
        sign, sub = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        if sign.lower() not in FEEDS:
            out.append((sign.lower(), int(sub) if sub else 0))
    return out


def place(toks, call_words=()):
    """The words of a name that say where (its city and state): not its network, call sign, number or a letter."""
    return {t for t in toks if t not in NETWORKS and t not in call_words and not t.isdigit() and len(t) > 1 and t not in SOFT}


def same_place(a, b):
    """Two names' places agree: either says none, or they share a word ("Bay Area, CA" and "San Jose, CA")."""
    return not a or not b or bool(a & b)


def score(mine, theirs):
    """How alike two channels' words are (0 to 1): what they share against what either has that the other hasn't;
    the same letters run together count as the same ("ESPN 2", "ESPN2"); without a word like "channel", a little less."""
    a, b = set(mine), set(theirs)
    if not a or not b:
        return 0.0
    if mine == theirs or ''.join(mine) == ''.join(theirs):
        return 1.0
    s = len(a & b) / len(a | b)
    sa, sb = a - SOFT, b - SOFT
    if sa and sb and (sa != a or sb != b):
        if sa == sb or ''.join(t for t in mine if t not in SOFT) == ''.join(t for t in theirs if t not in SOFT):
            s = max(s, 0.9)
        else:
            s = max(s, 0.9 * len(sa & sb) / len(sa | sb))
    return s


# ---- The listing sites' channels (from the grabber's sites folder) ----

def site_country(site, site_id, filename):
    if site == 'sky.com':
        return {'GB': 'uk', 'IE': 'uk'}.get(site_id.split('#')[0], '')
    if site == 'plex.tv':
        m = re.search(r'plex\.tv_([a-z]+)\.channels\.xml$', filename)
        return {'us': 'us', 'uk': 'uk'}.get(m.group(1) if m else '', '')
    if site == 'tvpassport.com':
        return 'ca' if re.search(r'\(can\)|\bcan\b', site_id) else 'us'
    return SITE_COUNTRIES.get(site, ('',))[0]


def build_index():
    """Every channel the read sites list (their channels.xml files), with its words, feed and country."""
    out, seen = [], set()
    root = os.path.join(EPG, 'sites')
    for site in SITES:
        d = os.path.join(root, site)
        try:
            files = sorted(f for f in os.listdir(d) if f.endswith('.channels.xml'))
        except OSError:
            continue
        if site == 'plex.tv':
            files = [f for f in files if f in ('plex.tv_us.channels.xml', 'plex.tv_uk.channels.xml')]
        for fn in files:
            try:
                tree = ET.parse(os.path.join(d, fn))
            except (OSError, ET.ParseError):
                continue
            for el in tree.getroot().iter('channel'):
                sid, name, lang = el.get('site_id') or '', (el.text or '').strip(), el.get('lang') or ''
                if not sid or not name or lang not in ('en', ''):
                    continue
                country = site_country(site, sid, fn)
                if not country:
                    continue
                k = (site, sid)
                if k in seen:
                    continue
                seen.add(k)
                toks, feed, _ = words(name)
                if toks:
                    cs = calls(name)
                    out.append((site, sid, lang or 'en', name, toks, feed, country, cs, place(toks, {c for c, _ in cs})))
    return out


def match_local(name, idx):
    """(1.0.1) One of his channels that's a local station, to the sites' channel of that station: by its call sign (the
    same channel of it, and not another place: a station's call sign is its own), else by the network and its place
    alone ("NBC Los Angeles", "ABC [New York]": the one station of that network there; not with a number, which the sites
    don't say: "CBS 9 Los Angeles" is KCAL, not CBS's KCBS). None if it isn't one, or no station is clearly it."""
    vs = [v for v in variants(name) if v[0] and v[2] in ('', 'us')]  # its words without a country before it ("US: ")
    if not vs:
        return None
    toks = vs[-1][0]
    mine = calls(name)
    if mine:
        here = place(toks, {c for c, _ in mine})
        found = [e for e in idx if e[6] == 'us' and any(c in e[7] for c in mine) and same_place(here, e[8])]
    else:
        net, rest = toks[0], toks[1:]
        if net not in NETWORKS or not rest or any(t.isdigit() for t in rest) or set(rest) & NETWORKS:
            return None
        found = [e for e in idx if e[6] == 'us' and e[4][0] == net and any(sub == 0 for _, sub in e[7])
                 and set(rest) <= e[8]]
        if len({c for e in found for c, sub in e[7] if sub == 0}) != 1:
            return None  # none there, or two of that network's stations: not clear
    if not found:
        return None
    found.sort(key=lambda e: (-score(toks, e[4]), SITE_ORDER.get(e[0], 9), e[3]))
    e = found[0]
    return {'site': e[0], 'site_id': e[1], 'lang': e[2], 'site_name': e[3], 'score': 1.0 if mine else 0.95,
            'by': 'its call sign' if mine else 'its network and place'}


def match_one(name, idx):
    """A channel of his to the read sites' channel it is: the best that's clearly the same, or (None, why)."""
    vs = [v for v in variants(name) if v[0]]
    if not vs:
        return None, 'no words in its name'
    local = match_local(name, idx)
    if local:
        return local, ''
    best = {}
    for toks, feed, country, less in vs:
        for site, sid, lang, sname, stoks, sfeed, scountry, _, _ in idx:
            if country and scountry != country:
                continue
            s = score(toks, stoks) - less
            if s < MATCH_LEAST - 0.15:
                continue
            if feed and sfeed and feed != sfeed:
                s -= 0.3       # another feed: hours apart
            elif feed and not sfeed:
                s -= 0.04 if feed == 'east' else 0.15   # his West channel, one with no feed said (the East's, mostly)
            elif not feed and sfeed == 'west':
                s -= 0.1       # his says no feed: the East's, as most are
            if not country and scountry != 'us':
                s -= 0.08      # his says no country: the TVs are in the US, so a US channel before another country's
            k = (site, sid)
            if k not in best or s > best[k][0]:
                best[k] = (s, SITE_ORDER.get(site, 9), site, sid, lang, sname)
    best = list(best.values())
    if not best:
        return None, 'no channel like it'
    best.sort(key=lambda x: (-x[0], x[1], x[5]))
    top = best[0]
    if top[0] < MATCH_LEAST:
        return None, 'the closest (%s on %s) not close enough (%.2f)' % (top[5], top[2], top[0])
    # The next best of another name (the same channel on another site, or written another way, isn't a rival).
    def same(n):  # its letters, and its feed (none said: the East's): "ESPN U" and "ESPNU" are one channel, and
        w = words(n)  # "Golf Channel USA" is "Golf Channel"
        return ''.join(t for t in w[0] if t not in ('usa', 'us')), w[1] or 'east'
    tw = same(top[5])
    rival = next((b for b in best[1:] if same(b[5]) != tw), None)
    if rival is not None and top[0] - rival[0] < MATCH_AHEAD:
        return None, 'not clear: %s (%.2f) or %s (%.2f)' % (top[5], top[0], rival[5], rival[0])
    return {'site': top[2], 'site_id': top[3], 'lang': top[4], 'site_name': top[5], 'score': round(top[0], 3)}, ''


def all_wanted():
    """The channels every TV asked for, once each (by key)."""
    out = {}
    with lock:
        for tv in wanted.values():
            for ch in tv.get('channels', []):
                if ch.get('key') and ch['key'] not in out:
                    out[ch['key']] = ch
    return out


def rematch():
    global index
    if not index:
        index = build_index()
        state['index'] = len(index)
    want = all_wanted()
    out = {}
    for key, ch in want.items():
        m, why = match_one(ch.get('name', ''), index)
        out[key] = dict(m, name=ch.get('name', '')) if m else {'name': ch.get('name', ''), 'none': why}
    with lock:
        matches.clear()
        matches.update(out)
    save_json('matches.json', out)
    n = sum(1 for m in out.values() if 'site' in m)
    say('matched %d of %d channels to the listing sites' % (n, len(out)))
    return out


# ---- The grabber: its code kept up to date, and run ----

def run(cmd, cwd=None, timeout=1800, env=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or '')[-4000:], (r.stderr or '')[-4000:]
    except subprocess.TimeoutExpired:
        return -1, '', 'timed out'
    except OSError as e:
        return -1, '', type(e).__name__


def node_env():
    e = dict(os.environ)
    e['NODE_ENV'] = 'production'
    e.pop('HOME_GUIDE_TEST', None)
    return e


def update():
    """The grabber's code (iptv-org's epg) to its latest, its packages if they changed, and its channel data."""
    if TEST:
        state['updated_at'] = time.time()
        return True
    state['state'] = 'bringing the grabber up to date'
    if not os.path.isdir(os.path.join(EPG, '.git')):
        shutil.rmtree(EPG, ignore_errors=True)
        rc, _, err = run(['git', 'clone', '--depth', '1', '-b', 'master', REPO, EPG], timeout=900)
        if rc != 0:
            state['note'] = 'the grabber could not be fetched (%s)' % err.strip()[-200:]
            return False
        lock_before = None
    else:
        lock_before = file_text(os.path.join(EPG, 'package-lock.json'))
        rc, _, err = run(['git', 'fetch', '--depth', '1', 'origin', 'master'], cwd=EPG, timeout=900)
        if rc == 0:
            rc, _, err = run(['git', 'reset', '--hard', 'origin/master'], cwd=EPG, timeout=300)
        if rc != 0:
            state['note'] = 'the grabber could not be brought up to date (%s); the one here is used' % err.strip()[-200:]
    if lock_before is None or lock_before != file_text(os.path.join(EPG, 'package-lock.json')) \
            or not os.path.isdir(os.path.join(EPG, 'node_modules')):
        rc, _, err = run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=EPG, timeout=3600, env=node_env())
        if rc != 0:
            state['note'] = 'its packages could not be set up (%s)' % err.strip()[-200:]
            return False
    rc, _, err = run(['npm', 'run', 'api:load'], cwd=EPG, timeout=900, env=node_env())
    if rc != 0:
        state['note'] = 'the channel data could not be fetched (%s); the last is used' % err.strip()[-200:]
    global index
    index = build_index()
    state['index'] = len(index)
    state['updated_at'] = time.time()
    save_json('state.json', state)
    say('the grabber up to date: %d channels on %d listing sites' % (len(index), len(SITES)))
    return True


def file_text(p):
    try:
        with open(p, encoding='utf-8') as f:
            return f.read()
    except OSError:
        return None


def esc(s):
    return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def grab():
    """The matched channels' next days read by the grabber, and their shows kept."""
    with lock:
        picked = {k: m for k, m in matches.items() if 'site' in m}
    if not picked:
        state['grab_note'] = 'no channel matched yet'
        return False
    state['state'] = 'reading the guide (%d channels)' % len(picked)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<channels>']
    for key, m in sorted(picked.items()):
        lines.append('  <channel site="%s" lang="%s" xmltv_id="%s" site_id="%s">%s</channel>' % (
            esc(m['site']), esc(m.get('lang') or 'en'), esc(key), esc(m['site_id']), esc(m['site_name'])))
    lines.append('</channels>')
    chan_file, out_file = path('wanted.channels.xml'), path('guide.xml')
    with open(chan_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        os.remove(out_file)
    except OSError:
        pass
    t0 = time.time()
    if GRAB_CMD:
        cmd = GRAB_CMD.split() + [chan_file, out_file]
        rc, out, err = run(cmd, timeout=600)
    else:
        cmd = ['npm', 'run', 'grab', '--', '--channels=' + chan_file, '--output=' + out_file, '--days=%d' % DAYS,
               '--maxConnections=2', '--timeout=30000']
        rc, out, err = run(cmd, cwd=EPG, timeout=3 * 3600, env=node_env())
    state['grab_s'] = round(time.time() - t0)
    if not os.path.isfile(out_file):
        state['grab_note'] = 'the grabber wrote no guide (%s)' % (err.strip() or out.strip())[-200:]
        say(state['grab_note'])
        return False
    shows = read_guide(out_file)
    with lock:
        guide.clear()
        guide.update(shows)
    save_json('guide.json', shows)
    state['grabbed_at'] = time.time()
    state['grab_note'] = '%d of %d channels have shows (%d shows)' % (
        sum(1 for v in shows.values() if v), len(picked), sum(len(v) for v in shows.values()))
    save_json('state.json', state)
    say('the guide read: ' + state['grab_note'] + ' in %d s' % state['grab_s'])
    return True


XMLTV_TIME = re.compile(r'(\d{14})\s*([+-]\d{4})?')


def xml_time(s):
    m = XMLTV_TIME.match(s or '')
    if not m:
        return None
    t = time.strptime(m.group(1), '%Y%m%d%H%M%S')
    ms = int(calendar.timegm(t)) * 1000
    z = m.group(2)
    if z:
        sign = -1 if z[0] == '-' else 1
        ms -= sign * (int(z[1:3]) * 3600_000 + int(z[3:5]) * 60_000)
    return ms


def read_guide(p):
    """The grabber's guide (XMLTV): each channel's shows, in order, two hours back to two days ahead."""
    out = {}
    now = int(time.time() * 1000)
    try:
        for _, el in ET.iterparse(p, events=('end',)):
            if el.tag != 'programme':
                continue
            key = el.get('channel') or ''
            a, b = xml_time(el.get('start')), xml_time(el.get('stop'))
            title = (el.findtext('title') or '').strip()
            if key and a and b and b > a and title and b > now - KEEP_BACK and a < now + KEEP_AHEAD:
                desc = (el.findtext('desc') or '').strip()
                sub = (el.findtext('sub-title') or '').strip()
                if sub and sub.lower() != title.lower():
                    desc = (sub + '. ' + desc).strip() if desc else sub
                out.setdefault(key, []).append([a, b, title[:200], desc[:DESC_MOST]])
            el.clear()
    except (OSError, ET.ParseError) as e:
        say('the guide could not be read (%s)' % type(e).__name__)
    for v in out.values():
        v.sort()
        # The same show twice (two days' pages overlapping): once.
        keep = []
        for s in v:
            if keep and s[0] == keep[-1][0]:
                continue
            keep.append(s)
        v[:] = keep
    return out


# ---- The work, in its own thread ----

def worker():
    while True:
        try:
            now = time.time()
            if not os.path.isdir(os.path.join(EPG, 'sites')) or now - state['updated_at'] >= UPDATE_EVERY:
                update()
                if os.path.isdir(os.path.join(EPG, 'sites')):
                    rematch()
                    grab()
            elif state['changed_at'] and now - state['changed_at'] >= SETTLE:
                state['changed_at'] = 0
                rematch()
                grab()
            elif now - state['grabbed_at'] >= GRAB_EVERY and matches:
                grab()
            state['state'] = 'ready' if guide else ('waiting for a TV\'s channels' if not wanted else 'ready (no shows yet)')
        except Exception as e:  # never gives up: said, and looked at again
            state['state'] = 'something failed (%s); again in 10 minutes' % type(e).__name__
            say(state['state'])
            work.wait(600)
            work.clear()
            continue
        work.wait(60)
        work.clear()


# ---- Answers ----

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def answer(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def allowed(self):
        if private(self.client_address[0]):
            return True
        self.send_error(403)
        return False

    def do_GET(self):
        if not self.allowed():
            return
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == '/status':
            with lock:
                n_match = sum(1 for m in matches.values() if 'site' in m)
                return self.answer(200, {'version': VERSION, 'state': state['state'], 'note': state['note'],
                                         'tvs': len(wanted), 'wanted': len(all_wanted_unlocked()), 'matched': n_match,
                                         'with_shows': sum(1 for v in guide.values() if v), 'index': state['index'],
                                         'updated_at': int(state['updated_at']), 'grabbed_at': int(state['grabbed_at']),
                                         'grab_note': state['grab_note'], 'grab_s': state['grab_s'], 'sites': list(SITES)})
        if u.path == '/matches':
            with lock:
                return self.answer(200, {'matches': matches})
        if u.path == '/programs':
            keys = set(x for x in (q.get('keys') or [''])[0].split(',') if x)
            now = int(time.time() * 1000)
            with lock:
                out = {}
                for k, v in guide.items():
                    if keys and k not in keys:
                        continue
                    m = matches.get(k, {})
                    shows = [s for s in v if s[1] > now - KEEP_BACK]
                    if shows and 'site' in m:
                        out[k] = {'from': m['site'], 'name': m.get('site_name', ''), 'programs': shows}
            return self.answer(200, {'at': int(state['grabbed_at'] * 1000), 'channels': out})
        self.answer(404, {'error': 'not here'})

    def do_POST(self):
        if not self.allowed():
            return
        u = urllib.parse.urlsplit(self.path)
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > 4 << 20:
            return self.answer(400, {'error': 'its length'})
        body = self.rfile.read(n)
        if u.path == '/channels':
            try:
                j = json.loads(body)
                tv = str(j.get('tv') or 'tv')[:40]
                chans = []
                for ch in j.get('channels') or []:
                    key, name = str(ch.get('key') or '')[:80], str(ch.get('name') or '')[:120]
                    if not re.match(r'^[a-z0-9]{1,80}$', key) or not name.strip():
                        continue
                    chans.append({'key': key, 'name': name, 'source': str(ch.get('source') or '')[:40]})
            except (ValueError, AttributeError, TypeError):
                return self.answer(400, {'error': 'not a list of channels'})
            with lock:
                before = [(c['key'], c['name']) for c in wanted.get(tv, {}).get('channels', [])]
                if chans:
                    wanted[tv] = {'at': time.time(), 'channels': chans}
                else:
                    wanted.pop(tv, None)  # (1.0.1) a TV with none to ask for: gone
                save_json('wanted.json', wanted)
            changed = before != [(c['key'], c['name']) for c in chans]  # by their names (what's matched)
            if changed:
                state['changed_at'] = time.time() - (SETTLE if TEST else 0)
                work.set()
            return self.answer(200, {'kept': len(chans), 'changed': changed})
        if u.path == '/grab' and TEST:
            state['changed_at'] = time.time() - SETTLE
            work.set()
            return self.answer(202, {'asked': True})
        self.answer(404, {'error': 'not here'})


def all_wanted_unlocked():
    out = {}
    for tv in wanted.values():
        for ch in tv.get('channels', []):
            if ch.get('key') and ch['key'] not in out:
                out[ch['key']] = ch
    return out


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        e = sys.exc_info()[1]
        if not isinstance(e, (ConnectionError, TimeoutError)):
            say('a request failed (%s)' % type(e).__name__)


def main():
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    os.makedirs(DATA, exist_ok=True)
    wanted.update({tv: w for tv, w in load_json('wanted.json', {}).items() if w.get('channels')})
    matches.update(load_json('matches.json', {}))
    guide.update(load_json('guide.json', {}))
    st = load_json('state.json', {})
    for k in ('updated_at', 'grabbed_at', 'grab_note', 'grab_s', 'index'):
        if k in st:
            state[k] = st[k]
    srv = Server(('0.0.0.0', PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    say('Home guide %s, answering on port %d' % (VERSION, PORT))
    worker()


if __name__ == '__main__':
    main()
