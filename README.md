# Knuckle liveness check

A small Streamlit app that checks whether a photo shows a **real hand** or a **replay attack** (the hand shown on a phone or monitor screen). Three models score each photo, and each one gives a verdict: ✅ LIVE or ⛔ ATTACK.

| Model | Weights | Backbone |
|---|---|---|
| MobileNetV3 | `weights/mobilenet_v3_large_replay.pt` (full model) | torchvision MobileNetV3-Large |
| DINOv2 + LoRA | `weights/dinov2_lora_replay.pt` (LoRA + head) | DINOv2 ViT-S/14, downloaded on first run |
| FoundPAD (CLIP) | `weights/foundpad_replay.pt` (LoRA + head) | CLIP ViT-B/16, downloaded on first run |

Thresholds and preprocessing are in `weights/config.json`.

## How to test it

1. Open the app and allow camera access.
2. On the **📷 Camera** tab, show the **back of your hand** with all of your fingers in view, then take a photo. You can also use the **📁 Upload** tab to upload 1–5 photos of one attempt.
3. Try both kinds of attempt:
   - **Real:** your actual hand.
   - **Attack:** a photo or video of a hand shown on another phone or monitor.
4. Read the verdict table and tell me which models got it right or wrong.

The first photo takes a little longer while the models load.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10–3.12 is recommended. The first run downloads DINOv2 (~90 MB) and CLIP ViT-B/16 (~350 MB).

## Credits

`TPO_repo/` contains the FoundPAD / TPO code and base checkpoint by its original authors (https://github.com/gurayozgur/TPO). It is licensed under **CC BY-NC-SA 4.0**; see `TPO_repo/LICENSE`. Because the FoundPAD weights here are fine-tuned from that checkpoint, they fall under the same non-commercial share-alike terms.
