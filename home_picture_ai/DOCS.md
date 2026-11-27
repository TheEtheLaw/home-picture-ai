# Home picture AI

The picture AI for Home's ad screen (Home is a TV launcher). The TVs on this home network send it the small picture
they're watching, about once a second while live TV plays, and it says how much the picture looks like an ad, plus
what it saw, so each TV can tell its program from the ads. The TV decides when to show its ad screen; this only looks.

## Using it

There's nothing to set. Start it once and leave **Start on boot** and **Watchdog** on. The TVs find it by
themselves on the home network (it announces itself there as `_homeai._tcp`, port 8765).

The first start downloads the AI models once (352 MB and 46 MB, from Hugging Face, at fixed versions, each checked
by its SHA-256) into this app's own storage. Until the main one is there, the log says how far it got and the TVs
wait; the second one comes after it. The models are left out of Home Assistant's backups, since they download again
by themselves.

## What it keeps

No pictures. Each picture is looked at and forgotten: never saved, never logged, never sent anywhere. It answers only
devices on the home network (private addresses).

Since 1.3.0 it remembers the ads the TVs have seen, as the AI's fingerprints (512 numbers each, not pictures): once a
TV's ad screen has covered a break, the TV sends the fingerprints of the ads in it, one a second. Each ad is kept
once (a fingerprint 0.95 or more like a kept one is that one, seen again), with no limit: about half a kilobyte each,
roughly 80 MB a year at a few hours of TV a day, in this app's own storage (`ads.npz`, and `ads.log` for the newest,
folded in as it grows), in Home Assistant's backups. A ceiling of 2 million (about 1 GB, decades of watching) keeps
it from ever filling the storage; only then do the ones seen longest ago go. From 8,192 ads on they're grouped by
how they look, so each picture is compared only with the few groups it could be in: with a million ads an answer
takes a few thousandths of a second longer. Every answer says how like the nearest remembered ad the picture is, so
an ad seen before is known right away, on any channel and by either TV. A different main model starts the memory
over.

## The bench (since 1.5.0)

To pick the ad AI's next model, `POST /bench` (from the home network) times bigger models on this Pi, one at a time:
Apple's MobileCLIP2-S2, -B and -L/14 and Google's SigLIP 2 B/16 and L/16 at 256 (also 8-bit), and the two models the
TVs use now, to compare. Each is downloaded (a fixed version from Hugging Face, checked by its SHA-256), timed on 1
core and on 3, and deleted again; nothing is kept, and the TVs keep their models. A timing starts only once no TV has
asked for 20 seconds, so the TVs' answers never slow down. It takes about half an hour (the downloads are about 3.7
GB in all). `GET /bench` says how far it is: for each model its size, its time for a picture on 1 core (`ms_1`) and on
3 (`ms_3`), how many pictures a second that is (`per_second`), its memory, and why one wasn't timed (no room, too
little memory, a download that didn't come or didn't check out).

Since 1.5.1 the bench is for one TV at a time (`POST /bench`; the first one is `POST /bench?set=all`): what counts is
how many pictures a second the Pi gets through, since the look ahead gives the AI seconds before a picture is on
screen. Each model is timed on 1 core, on all 4 one picture at a time, with 4 pictures at once (a core each) and with 4
in one batch: `per_second` is its best way (`best`), `every_picture` whether that keeps up with one TV (2 a second).
It also makes 8-bit copies of Apple's models here, of their matrix math only and of all of them (set on 48 of the
TVs' recorded pictures, read from Home media on this Pi), and checks each against the original on 16 others
(`fidelity`, 1 is the same). Everything it downloads or makes is deleted when it's done. `POST /bench?busy=1` times
even while a TV is asking (its times then a little high, and said so).

Since 1.6.0 `POST /bench?set=read` reads the screen's words on his recordings (on Home media, on this Pi): up to 120
pictures from the breaks (where a TV's break screen was up) and 120 from the programs (at least 30 seconds from any),
read on 2 cores as a TV reads a whole picture. `GET /bench` then says, for each, how often an ad's signs were there
(`share`: web addresses, phone numbers, prices, fine print's words, small lines, any of them), the words found
(`examples`), how high the long lines are (`long_lines_by_height`), and which program pictures had an ad's words
(`program_with_ad_signs`: maybe a break the screen missed). It also says where Home 2.05 put each break screen's change
(`edges`) and how long the TVs' own reader takes at 960 and 1280 (`tv_reader_ms`). Nothing is kept but those numbers
and words.

Since 1.8.0 `POST /bench?set=score` scores the models on his recordings, with the answers as JSON: `{"tz_minutes":
-300, "sets": [{"name", "channel", "from", "to", "every", "ads": [[from, to], ...], "programs": [[from, to], ...],
"look", "probe", "line_from"}]}`. Times are ms since 1970; `channel`, `from` and `to` pick the recordings (the
channel's name without spaces or signs, the first and last minutes as the recordings' names start); `every` is the
least ms between two pictures of a span; `look` also scores the likeness to the program's look (made from the first
30% of the set's program, tried on the rest), `probe` a lean learned from the first half of the set's answers (tried
on the second), and `line_from` scores this set by an earlier set's line and learned lean. Each model's lean comes from
its own text half (CLIP's word pieces, the same 14 kinds of ad and 15 of program for every model). `GET /bench` then
says under `score`: each set's pictures (`files`, `ads`, `programs`, its `first` and `last` at the set's time), and for
each model, set by set: `auc` (how often an ad's picture leans further than a program's: 0.5 a coin's toss, 1 always),
`caught` (the share of ads over the `line` that lets 2% of the program's pictures over), `look_auc`, `look_caught`,
`both_auc`, `both_caught`, `probe_auc`, `probe_caught`, `spans` (each answered span's share over the earlier set's
line, and how many pictures) and `probe_from_auc`; for the TVs' model now also `now_auc` and `now_over_ad` (by the lean
the TVs use now) and `words_like_now` (how like that lean the words' is); and `ms`, its time for a picture on all
cores. Text halves and new models are downloaded (fixed versions, checked by SHA-256) and deleted after.

## Each channel's own lesson (since 1.9.0)

The AI's lean is the same for every channel, but each channel's ads and programs look their own way, and a lean learned
from a channel's own breaks parts them much better. So a TV says which channel each picture is of and the picture's own
time (`POST /look?ch=NAME&t=MS`), and tells the app how each run of the AI's ads on it ended (real or not, and why), when
its screen was up and which show was on (`POST /lessons/event` with `{"channel", "kind": "run", "from", "to", "real",
"why"}`, `{"channel", "kind": "screen", "from", "to"}` or `{"channel", "kind": "show", "at", "name"}`). What the AI saw
of each picture (its 512 numbers, a byte each, never the picture) is kept by channel for 21 days, a file a day; the
words too. A replay's (`replay=1` with either) are not kept.

Each night, once no TV has asked for half an hour and 20 hours after the last time, each channel's lesson is learned
again: its ad pictures from inside its real breaks (their first and last 10 seconds left out), its program pictures at
least 40 seconds from any run of the AI's ads and any screen, one each 4 seconds. Before a lesson is used it's tested on
the newest day it didn't learn from (with no other day yet, a long day's later part): its AUC at least 0.97, 85% of
that day's ad pictures over its line (the line lets 2% of the program over), the screen it would put up (3 ad words in
a row start it, 2 of the program end it) covering at least 80% of the real breaks and up outside them at most 30
seconds an hour. A show watched at least 15 minutes with more than 60 seconds an hour of it outside real breaks is held
(the TV goes by its own rules there). A channel whose own lesson hasn't passed can use another channel's that passes
this channel's own test (the same channel from another provider, a sister channel), at most a quarter of its watching
held. A lesson in use goes with every answer for its channel: `"lesson": {"score", "line", "from", "held"}` (`from`:
`own` or the channel it's from). `GET /lessons` says how each channel's went (learned, tested, used or not, why) and
what's kept; `POST /lessons/learn[?before=MS][&channels=A,B]` learns now (a test can learn from what came before a
time). Another main model starts the lessons over.

## How it works

- `POST /look` with a JPEG: `{"model", "score", "embedding", "ad", "like", "sure", "known", "ms"}`. `score` is how
  far the picture leans toward an ad (over `ad`: an ad; until the TV knows the channel, it has to be past `ad` by
  `sure`), `embedding` is what the AI saw (512 numbers, float32, base64) for the TV to compare with its program's
  usual look (under `like`: not the program), and `known` (since 1.3.0) how like the nearest remembered ad it is (0
  to 1; 0.95 or more: an ad seen before). The lean is measured against `direction.bin`: the average of 14 descriptions of
  ads minus the average of 15 descriptions of programs, made once with CLIP's text half.
- `POST /look?spot=x0,y0,x1,y1`: the same, and a second look at that spot of the picture (where the TV found the
  channel's own logo): `{"spot_model", "spot", "spot_like", "spot_ms"}` besides. `spot` is what the second model saw
  there (512 numbers, float32, base64), for the TV to compare with the spot's usual look (at or over `spot_like`: the
  logo's there). Left out while the second model isn't ready.
- `POST /read` with a PNG or JPEG of a logo's spot (since 1.2.0): `{"read_model", "lines", "read_ms"}`, where
  `lines` are the logo's lines of letters, top to bottom, each `{"text", "score"}`. The TV sends a few pictures of the
  spot put together (so what moves behind the logo fades) and checks the letters against the channel's name.
- `POST /read?side=N` with a PNG or JPEG of a whole picture (since 1.4.0): the same answer, with the letters looked for
  at the picture's longer side N (256 to 1280, in steps of 32; `side` in the answer says which). A TV sends a 960x540
  picture at 768 near the end of an episode or a film, and knows the credits by their words. Without `side`, 256.
  Since 1.6.0 each line also says where it is (`box`: left, top, right, bottom, in parts of the picture's width and
  height), and with `signs=1` the answer says what the lines say of an ad (`signs`): `url`, `phone`, `price` and
  `words` (fine print's usual words), each the words found; `small` (small lines of 12 letters or more); `ad` (a web
  address, phone number or price, or fine print's words); and `why`.
- `POST /ads` (since 1.3.0) with `{"model", "ads": [[scale, fingerprint], ...]}`, each fingerprint 512 numbers of a
  byte each (base64), times its scale: a covered break's ads, at most 600 at a time. Answers `{"new", "again",
  "ads"}`: how many were new, how many were seen before, and how many it remembers now. Fingerprints from another
  model are turned away (409).
- `POST /ads/forget` (since 1.7.0) with `{"model", "shows": [[scale, fingerprint], ...], "logo": "goes"}`: the show's
  own pictures (a TV's, the channel's logo on), each fingerprint as for `POST /ads`. Every remembered ad as like one of
  them as an ad seen again is let go (a break that wasn't one taught the memory the show). Answers `{"forgot", "ads"}`:
  how many were let go, and how many it remembers now. Since 1.7.2 only with `"logo": "goes"` (the TV has seen that
  logo go away in the channel's breaks: Home 2.08); without it nothing is let go, as a channel can keep its logo on
  through its ads (paused altogether in 1.7.1).
- `GET /status`: `{"model", "version", "ready", "note", "looks", "ms", "up", "spot_model", "spot_ready",
  "spot_note", "spot_looks", "spot_ms", "read_model", "read_ready", "read_note", "reads", "read_ms", "read_side_max",
  "ads", "ads_note"}`. `read_side_max` (since 1.4.0): the largest `side` a TV can ask for.

The main model is the picture half of OpenAI's CLIP ViT-B/32 (MIT license), from Xenova's ONNX copy
(`Xenova/clip-vit-base-patch32`), run with Microsoft's ONNX Runtime (MIT license) on 3 of the processor's cores. The
second is the picture half of Apple's MobileCLIP2-S0, under Apple's research license (`apple-amlr`), from plhery's
ONNX copy (`plhery/mobileclip2-onnx`), on the fourth core. Logos are read by PaddleOCR's PP-OCRv4 detection and
recognition models (Apache-2.0), from RapidOCR's ONNX copies (Apache-2.0, in its package, which comes with the app),
on one core for a moment when a TV asks.
