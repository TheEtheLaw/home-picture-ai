# Home picture AI

A Home Assistant app (add-on) that runs the picture AI for Home's ad screen: the TVs on the home network send it
the small picture they're watching and it says whether it's an ad, and it remembers the ads they've seen (as the
AI's fingerprints, not pictures). It runs on the Home Assistant machine (a Raspberry Pi 4 here) so the TVs don't
have to.

## Installing

1. In Home Assistant: **Settings**, then **Apps** (or **Add-ons**), then the store, then the menu (⋮),
   **Repositories**, and add `https://github.com/TheEtheLaw/home-picture-ai`.
2. Open **Home picture AI**, **Install**, then **Start**. Keep **Start on boot** and **Watchdog** on.

More in [home_picture_ai/DOCS.md](home_picture_ai/DOCS.md).

## Licenses

The AI models aren't in this repository: the app downloads them on first start. The main one is the picture half of
OpenAI's CLIP ViT-B/32 (MIT license), as converted to ONNX by Xenova. The second, which looks at the channel's logo,
is the picture half of Apple's MobileCLIP2-S0, under Apple's research license (`apple-amlr`), as converted to ONNX by
plhery. Logos are read by PaddleOCR's PP-OCRv4 models (Apache-2.0), as converted to ONNX by RapidOCR (Apache-2.0),
which come with the app in RapidOCR's package. ONNX Runtime is Microsoft's (MIT license).
`direction.bin` is 512 numbers made once with CLIP's text half from two sets of words: TV commercials of every
kind, and TV programs (sports, news, shows, films).
