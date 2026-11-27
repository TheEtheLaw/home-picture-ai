# Changelog

## 1.0.1

- Local stations: matched by their call sign, in brackets or first ("FOX 11 (KTTV) Los Angeles", "KTLA Los Angeles"),
  to the same station's main channel (not its second, "KABC-TV2"), and never to one in another place (a name's city
  against the site's); a network's station named by its place alone ("NBC Los Angeles", "ABC [New York]") when that
  network has one station there (not with a channel number, which the sites don't say: "CBS 9" in Los Angeles is KCAL).
- Other names a little stricter (0.8 alike at least): one word more than a site's name is too far ("Sportsnet LA Lakers"
  isn't SportsNet LA).
- A TV that sends an empty list is no longer kept.

## 1.0.0

- A TV guide for the channels the TVs ask for (`POST /channels`: each channel's key, name and provider; names only).
  Each is matched by its name to a channel of the TV listing sites the iptv-org EPG grabber reads honestly (tvpassport.com,
  tvinsider.com, tv24.co.uk, Virgin TV Go, Plex, Sky): its words, its feed (East or West) and its country; a match must be
  clear, or there's none. The grabber (fetched on the first start, brought up to date each day) reads the matched
  channels' next two days twice a day; the TVs take the shows (`GET /programs`). `GET /matches` lists each match and how
  sure it is; `GET /status` how it stands.
