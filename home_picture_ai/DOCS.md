# Home picture AI

The picture AI for Home's ad screen (Home is a TV launcher). The TVs on this home network send it the small picture
they're watching, about once a second while live TV plays, and it says how much the picture looks like an ad, plus
what it saw, so each TV can tell its program from the ads. The TV decides when to show its ad screen; this only looks.

## Using it

There's nothing to set. Start it once and leave **Start on boot** and **Watchdog** on. The TVs find it by
themselves on the home network (it announces itself there as `_homeai._tcp`, port 8765).

The first start downloads the AI model once (352 MB, from Hugging Face, at a fixed version, checked by its SHA-256)
into this app's own storage. Until it's there, the log says how far it got and the TVs wait. The model is left out
of Home Assistant's backups, since it downloads again by itself.

## What it keeps

Nothing. Each picture is looked at and forgotten: never saved, never logged, never sent anywhere. It answers only
devices on the home network (private addresses).

## How it works

- `POST /look` with a JPEG: `{"model", "score", "embedding", "ad", "like", "sure", "ms"}`. `score` is how far the
  picture leans toward an ad (over `ad`: an ad; until the TV knows the channel, it has to be past `ad` by `sure`),
  `embedding` is what the AI saw (512 numbers, float32, base64) for the TV to compare with its program's usual look
  (under `like`: not the program). The lean is measured against `direction.bin`: the average of 14 descriptions of
  ads minus the average of 15 descriptions of programs, made once with CLIP's text half.
- `GET /status`: `{"model", "version", "ready", "note", "looks", "ms", "up"}`.

The model is the picture half of OpenAI's CLIP ViT-B/32 (MIT license), from Xenova's ONNX copy
(`Xenova/clip-vit-base-patch32`), run with Microsoft's ONNX Runtime (MIT license) on 3 of the processor's cores.
