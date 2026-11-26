# Changelog

## 1.5.0

- The bench: times bigger models on this Pi, to pick the ad AI's next model. Started from the home network
  (`POST /bench`), it downloads each one (a fixed version from Hugging Face, checked by its SHA-256), times it on 1
  core and on 3 while no TV is asking (it waits for a quiet moment, so the TVs' answers never slow down), and deletes
  it again. The two models the TVs use now are timed too, to compare. `GET /bench` says how far it is and each
  model's times; `GET /status` says the bench's state (`bench`).
- The models: Apple's MobileCLIP2-S2, -B and -L/14, and Google's SigLIP 2 B/16 and L/16 at 256 (also 8-bit). Each
  needs room for itself plus 1 GB in this app's storage, and enough free memory; one that doesn't fit is skipped and
  said so.
- Nothing changes for the TVs: the same models answer them.

## 1.4.0

- Reads a whole picture too, for the credits: a TV can send a whole frame (`POST /read?side=N`, a PNG or JPEG) and
  the same PP-OCRv4 models look for its lines of letters with the picture's longer side up to N points (256 to 1280,
  in steps of 32). A TV sends a 960x540 picture at 768 near the end of an episode or a film, a few seconds apart, and
  knows the credits by their words ("Directed by", "Executive Producer", the copyright line), even over a scene.
- A logo's spot is read as before (no `side`: 256). Every answer says the side it used (`side`), and `GET /status`
  says the largest a TV can ask for (`read_side_max`), so a TV knows this version can read whole pictures.
- Nothing more to download: the same two models, on one core for a moment when a TV asks.

## 1.3.0

- Remembers the ads the TVs have seen: once a TV's ad screen has covered a break, the TV sends the AI's fingerprints
  of its ads (`POST /ads`, 512 numbers each, never a picture), and each ad is kept once, with no limit. Every answer
  to `POST /look` now says how like the nearest remembered ad the picture is (`known`), so an ad seen before is known
  right away, on any channel and by either TV, whatever the picture's lean.
- Grouped by how they look once there are 8,192, so finding one stays quick however many there are (a million: a few
  thousandths of a second). New ones go on the end of a log, so a break's ads never rewrite the whole memory. A
  ceiling of 2 million (decades of watching) keeps it from filling the storage.
- Kept in the app's own storage (`ads.npz`, `ads.log`: about half a kilobyte an ad) and in Home Assistant's backups;
  nothing to download. `GET /status` says how many it remembers (`ads`, `ads_note`).

## 1.2.0

- Reads a logo's letters: a TV can send the spot where it found a mark that doesn't move (`POST /read`, a PNG or
  JPEG), and PaddleOCR's PP-OCRv4 models find its lines of letters and read them. The TV checks them against the
  channel's name, so it can tell the channel's own logo from a scoreboard or a provider's mark right away, instead of
  waiting for the channel's first break.
- The two models (15 MB) come with the app, in RapidOCR's package (`rapidocr-onnxruntime` 1.4.4), and each is checked
  by its SHA-256 before it's used. Nothing more to download. Reading takes one core for a moment, and only when a TV
  asks (once for each new logo).

## 1.1.0

- A second look, at the channel's logo: a TV can send the spot where it found the channel's own logo along with a
  picture (`POST /look?spot=x0,y0,x1,y1`), and Apple's MobileCLIP2-S0 looks at that spot, zoomed, on its own core at
  the same time as the main AI. The TV keeps what the spot usually looks like and asks only when the logo's shape
  isn't clear on its own.
- The second model (46 MB) downloads once on the first start after the update, checked by its SHA-256. The main AI
  answers as before meanwhile.

## 1.0.1

- Better words for what an ad is and what the program is: documentaries, interviews and a sports channel's own logo
  and ticker on the program's side; app ads, prices and promo codes, staged scenes with a brand's logo, a brand's
  last shot on the ads' side. Fewer program pictures taken for ads.
- Sends `sure`: how far past `ad` a picture's score has to be before a TV knows the channel's look.

## 1.0.0

- First version: CLIP ViT-B/32's picture tower with ONNX Runtime 1.31.1, found by the TVs on the home network.
