# Changelog

## 1.7.1

- Letting go of remembered ads (`POST /ads/forget`) is paused: a channel can keep its logo on through its ads (FS1's
  logo in its bottom bar stays), so a TV's pictures with the logo on can be real ads, not the show. It answers as
  before and lets nothing go, until the TVs know which logos really go away in a break.

## 1.7.0

- The ads it remembers clean themselves: a TV sends the show's own pictures (ones with the channel's logo on that came
  back as ads seen before) to `POST /ads/forget`, and every remembered ad as like one of them as an ad seen again is
  let go. A break that wasn't one (the AI taking a show for ads, then its pictures remembered as the break's) can't go
  on teaching it the show. What's let go stays let go when the app starts again.

## 1.6.0

- An ad's words: `POST /read?side=N&signs=1` reads a whole picture and says what its lines say of an ad (`signs`): web
  addresses, phone numbers, prices, and fine print (its usual words: terms, offers, a car's or a drug's small print;
  and small lines), each with the words found, and `ad` when any of the strong ones is there. Each line read now says
  where it is too (`box`: its left, top, right and bottom, in parts of the picture's width and height).
- The bench on the recordings (`POST /bench?set=read`): pictures from the breaks (where a TV's break screen was up) and
  from the programs (well away from any) in his recordings on Home media are read as a TV reads a whole picture, and it
  says how often each sign was there in each (with the words found), how high their long lines are, and which program
  pictures had an ad's words (maybe a break the screen missed). It also says where Home 2.05 put each break screen's
  change (to a cut, by the channel's logo, the sound or the only cut; by a silence or black picture; or at the AI's
  picture), and times the TVs' own reader at 960 and 1280. Numbers and the signs' words only: no picture is kept.
- Reads every recording on Home media (its list since 1.0.8), not only the newest five.

## 1.5.1

- The bench for one TV at a time (the default; `POST /bench?set=all` is 1.5.0's): what counts is how many pictures a
  second the Pi gets through, since the look ahead gives the AI seconds before a picture is on screen. Each model is
  timed on 1 core, on all 4 one picture at a time, with 4 pictures at once (a core each) and with 4 in one batch;
  `per_second` is its best way (`best`), and `every_picture` says whether that keeps up with one TV (2 a second).
- 8-bit copies made here: of MobileCLIP2-S0, -S2 and -B, both of the matrix math only and of all of it (set on 48 of
  the TVs' recorded pictures from Home media), each checked against the original on 16 others (`fidelity`: how alike
  they see them, 1 is the same).
- `POST /bench?busy=1` times even while a TV is asking (its times then a little high, and said so).
- Adds the ONNX package (1.23.2) for that, a fixed version. Nothing changes for the TVs.

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
