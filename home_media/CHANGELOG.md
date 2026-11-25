# Changelog

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
