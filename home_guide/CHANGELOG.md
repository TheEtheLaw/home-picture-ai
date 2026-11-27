# Changelog

## 1.0.0

- A TV guide for the channels the TVs ask for (`POST /channels`: each channel's key, name and provider; names only).
  Each is matched by its name to a channel of the TV listing sites the iptv-org EPG grabber reads honestly (tvpassport.com,
  tvinsider.com, tv24.co.uk, Virgin TV Go, Plex, Sky): its words, its feed (East or West) and its country; a match must be
  clear, or there's none. The grabber (fetched on the first start, brought up to date each day) reads the matched
  channels' next two days twice a day; the TVs take the shows (`GET /programs`). `GET /matches` lists each match and how
  sure it is; `GET /status` how it stands.
