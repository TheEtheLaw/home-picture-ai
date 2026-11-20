# Changelog

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
