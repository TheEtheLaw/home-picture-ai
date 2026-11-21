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

Nothing. Each picture is looked at and forgotten: never saved, never logged, never sent anywhere. It answers only
devices on the home network (private addresses).

## How it works

- `POST /look` with a JPEG: `{"model", "score", "embedding", "ad", "like", "sure", "ms"}`. `score` is how far the
  picture leans toward an ad (over `ad`: an ad; until the TV knows the channel, it has to be past `ad` by `sure`),
  `embedding` is what the AI saw (512 numbers, float32, base64) for the TV to compare with its program's usual look
  (under `like`: not the program). The lean is measured against `direction.bin`: the average of 14 descriptions of
  ads minus the average of 15 descriptions of programs, made once with CLIP's text half.
- `POST /look?spot=x0,y0,x1,y1`: the same, and a second look at that spot of the picture (where the TV found the
  channel's own logo): `{"spot_model", "spot", "spot_like", "spot_ms"}` besides. `spot` is what the second model saw
  there (512 numbers, float32, base64), for the TV to compare with the spot's usual look (at or over `spot_like`: the
  logo's there). Left out while the second model isn't ready.
- `POST /read` with a PNG or JPEG of a logo's spot (since 1.2.0): `{"read_model", "lines", "read_ms"}`, where
  `lines` are the logo's lines of letters, top to bottom, each `{"text", "score"}`. The TV sends a few pictures of the
  spot put together (so what moves behind the logo fades) and checks the letters against the channel's name.
- `GET /status`: `{"model", "version", "ready", "note", "looks", "ms", "up", "spot_model", "spot_ready",
  "spot_note", "spot_looks", "spot_ms", "read_model", "read_ready", "read_note", "reads", "read_ms"}`.

The main model is the picture half of OpenAI's CLIP ViT-B/32 (MIT license), from Xenova's ONNX copy
(`Xenova/clip-vit-base-patch32`), run with Microsoft's ONNX Runtime (MIT license) on 3 of the processor's cores. The
second is the picture half of Apple's MobileCLIP2-S0, under Apple's research license (`apple-amlr`), from plhery's
ONNX copy (`plhery/mobileclip2-onnx`), on the fourth core. Logos are read by PaddleOCR's PP-OCRv4 detection and
recognition models (Apache-2.0), from RapidOCR's ONNX copies (Apache-2.0, in its package, which comes with the app),
on one core for a moment when a TV asks.
