# Changelog

## 1.0.3

- A file once watched stops coming in at once (it went on to the end before, alongside the next episode); played again,
  it goes on.
- The TVs' asks are answered at once while a file comes in at full speed: what's in is noted on the drive afterwards,
  not while the TV waits (one ask took 6.8 s).

## 1.0.2

- The next episode, coming in ahead of time, hands its connection over the moment one being played starts coming in, and
  goes on from where it was once that one is all in: one connection to the source at a time.
- A copy Home doesn't want any more (another copy was picked, or it plays straight from the source now) stops coming in
  at once and goes from the drive (`POST /done` with `"why": "dropped"`).

## 1.0.1

- Mounts its drive: Home Assistant now lets an app mount a drive only when it asks for that right, so it asks
  (`SYS_ADMIN`).
- Reads each drive's filesystem and label off the drive itself (an app has no copy of the Pi's device list), so it
  knows its own drive, and Home Assistant's own disk even when that starts from USB.
- Never formats a drive that's in use on the Pi, or one it has already formatted since it started. Turning
  **Format a new drive** on again now formats a replacement drive, as it should.
- The watchdog's checks no longer leave errors in the log.

## 1.0.0

- First version: formats a new drive once (ext4, `HOMEMEDIA`) and mounts it inside the app; downloads what the TVs
  ask for over one connection, playing it to them as it comes in; a jump moves the download there first; the next
  episode comes in after the one playing; files go once watched, after 8 days unused if not watched (the setting), or
  when the drive runs low.
