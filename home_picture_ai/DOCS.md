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
- `POST /ads` (since 1.3.0) with `{"model", "ads": [[scale, fingerprint], ...]}`, each fingerprint 512 numbers of a
  byte each (base64), times its scale: a covered break's ads, at most 600 at a time. Answers `{"new", "again",
  "ads"}`: how many were new, how many were seen before, and how many it remembers now. Fingerprints from another
  model are turned away (409).
- `GET /status`: `{"model", "version", "ready", "note", "looks", "ms", "up", "spot_model", "spot_ready",
  "spot_note", "spot_looks", "spot_ms", "read_model", "read_ready", "read_note", "reads", "read_ms", "read_side_max",
  "ads", "ads_note"}`. `read_side_max` (since 1.4.0): the largest `side` a TV can ask for.

The main model is the picture half of OpenAI's CLIP ViT-B/32 (MIT license), from Xenova's ONNX copy
(`Xenova/clip-vit-base-patch32`), run with Microsoft's ONNX Runtime (MIT license) on 3 of the processor's cores. The
second is the picture half of Apple's MobileCLIP2-S0, under Apple's research license (`apple-amlr`), from plhery's
ONNX copy (`plhery/mobileclip2-onnx`), on the fourth core. Logos are read by PaddleOCR's PP-OCRv4 detection and
recognition models (Apache-2.0), from RapidOCR's ONNX copies (Apache-2.0, in its package, which comes with the app),
on one core for a moment when a TV asks.
