# Home guide

A TV guide for the channels the TVs at home ask for: the ones no better guide covers. Home (on each TV) uses its
providers' guides first where they're good, and asks this app for the rest.

## How it works

The TVs send the names of the channels they need a guide for (`POST /channels` with `{"tv", "channels": [{"key",
"name", "source"}]}`; names only, never an address). Each name is matched to a channel of the TV listing sites the
[iptv-org EPG grabber](https://github.com/iptv-org/epg) reads, of those whose ways are honest (no made-up browser, no way
around a site's checks): tvpassport.com and tvinsider.com (the US and Canada), tv24.co.uk and Virgin TV Go (the UK),
Plex's free channels, and Sky. A name is matched by its words ("ESPN 2" is "ESPN2"; "FS1" is "Fox Sports 1"), its feed
(East or West: hours apart) and its country (a word before it: "US:", "UK|", "USA "). A match must be clear, well ahead
of the next best of another name, or there is none: no guide is better than another channel's.

Twice a day the grabber reads the matched channels' next two days; the TVs take the shows (`GET /programs[?keys=a,b]`:
each channel's shows, two hours back to two days ahead, and the site they're from). The grabber itself is fetched on the
first start (a few minutes) and brought up to date each day, with the channel data it uses, since the sites change.

`GET /status` says how it stands; `GET /matches` lists what each channel was matched to and how sure it was, or why it
wasn't. Only the home network's own devices are answered.
