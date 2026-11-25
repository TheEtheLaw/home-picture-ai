# Home apps

Home Assistant apps (add-ons) for Home, a TV launcher. They run on the Home Assistant machine (a Raspberry Pi 4
here) so the TVs don't have to.

- **Home picture AI**: the picture AI for Home's ad screen and its credits. The TVs on the home network send it the
  small picture they're watching and it says whether it's an ad, it remembers the ads they've seen (as the AI's
  fingerprints, not pictures), and near the end of an episode or a film it reads the words on the screen, so the TVs
  know when the credits start. More in [home_picture_ai/DOCS.md](home_picture_ai/DOCS.md).
- **Home media**: the media server. It downloads what the TVs play onto its own USB drive and plays it to them from
  there, so every jump is instant, and gets the next episode ready ahead of time. More in
  [home_media/DOCS.md](home_media/DOCS.md).

## Installing

1. In Home Assistant: **Settings**, then **Apps** (or **Add-ons**), then the store, then the menu (⋮),
   **Repositories**, and add `https://github.com/TheEtheLaw/home-picture-ai`.
2. Open the app, **Install**, then **Start**. Keep **Start on boot** and **Watchdog** on. For **Home media**, turn
   **Protection mode** off before starting it (it needs the drive), and read its DOCS first: its first start
   formats the drive.

## Licenses

The AI models aren't in this repository: Home picture AI downloads them on first start. The main one is the picture
half of OpenAI's CLIP ViT-B/32 (MIT license), as converted to ONNX by Xenova. The second, which looks at the
channel's logo, is the picture half of Apple's MobileCLIP2-S0, under Apple's research license (`apple-amlr`), as
converted to ONNX by plhery. Logos are read by PaddleOCR's PP-OCRv4 models (Apache-2.0), as converted to ONNX by
RapidOCR (Apache-2.0), which come with the app in RapidOCR's package. ONNX Runtime is Microsoft's (MIT license).
`direction.bin` is 512 numbers made once with CLIP's text half from two sets of words: TV commercials of every
kind, and TV programs (sports, news, shows, films).

Home media uses Python's own library, Debian's drive tools (util-linux, e2fsprogs) and python-zeroconf (LGPL-2.1),
each as published.
