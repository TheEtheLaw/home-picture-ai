# Home media

The media server for Home (a TV launcher). When a TV with Home plays a film or an episode, it asks this app for the
file: the app downloads the whole file onto its own USB drive over one steady connection and plays it to the TV as it
comes in. Every jump to a part that's in is instant; a jump to a part that isn't in yet moves the download there
first, then it fills in the rest. While one plays, the next episode can come in ahead of time (after the one playing).

## Setting it up

1. Plug the drive into a powered USB hub, and the hub into one of the Pi's blue USB 3 ports. Only the media drive
   should be plugged in when the app first starts.
2. Install **Home media**, then turn **Protection mode** off on its page: using a drive needs full access to the
   Pi's hardware. It only applies to this app.
3. **Start** it, and keep **Start on boot** and **Watchdog** on.

On its first start the app formats the drive for itself (Linux's ext4, named `HOMEMEDIA`): **everything on the drive
is erased**. It does this once: the **Format a new drive** setting turns itself off afterwards. To format another
drive later (a replacement), turn the setting on again.

The app never touches Home Assistant's own disk or the Pi's SD card, never formats a drive under 100 GB, and never
formats when more than one drive is plugged in. The drive is mounted inside this app only. Unplugged, the app says so
in its log and writes nothing anywhere; the TVs play from the source as before until it's back.

## Settings

- **Format a new drive** (`format_new_drive`): format the drive on the next start if it isn't this app's yet. On
  at first; it turns itself off once it has formatted one.
- **Days to keep** (`keep_days`, 1 to 30, 8 at first): a file that hasn't been watched (the next episode, waiting
  for the weekend, or one stopped partway) goes once nobody has played it for this many days. Eight, so one that came
  in on a Saturday is still there the next Saturday evening.

## What it keeps

The files the TVs asked for, on the drive only, each with a small note of which parts are in. A file goes once it's
been watched, after the days above with nobody playing it if it hasn't been, or (oldest first) when the drive has
less than 50 GB free.
Files being played or coming in are never removed.

Links are kept in memory only: never logged, never written to the drive or anywhere else. After a restart, a file that
wasn't all in waits until a TV asks for it again with its link. The app only answers devices on the home network
(private addresses), and its log says what comes in by name only.

## How it works

Announced on the home network as `_homemedia._tcp`, port 8766.

- `POST /get` with `{"key", "url", "name", "ext", "headers", "prefetch"}`: starts (or goes on with) that file; a
  fresh link replaces an old one. `prefetch: true` (the next episode) waits until what's playing is all in. Answers
  `{"id", "play", "size", "have", "complete", "state", "rate", ...}`.
- `GET /play/<id>`: the file, with ranges (as a player asks for it). A part that isn't in yet is sent as soon as it
  comes in (the download goes there first); a TV waits at most a minute before its connection ends and it asks again.
- `POST /done` with `{"key"}` or `{"id"}`: it's been watched; it goes within ten minutes, once no TV is reading it.
- `GET /status`: the drive (ready or why not, its size and free space) and the files (size, how much is in, how fast
  it's coming in, state). The watchdog uses it.
