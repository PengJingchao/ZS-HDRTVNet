# ZS-HDRTVNet

## Environment

Tested with Python 3.10.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Place the image dataset and HDR video dataset like this:

```text
data/
  RGBTHDRdataset/
    hdr_reduced_size/
    registered_RGB_reduced_size/
    infrared/
    trainingset.txt
    testset.txt
  HDRVideo/
    <scene_name>/
      *.hdr or *.exr
```

## Train

Stage 1:

```bash
python scripts/train_stage1_image.py
```

Stage 2:

```bash
python scripts/train_stage2_video.py
```

If needed, edit the checkpoint path in `configs/train_stage2_video.yaml` so it points to the stage-1 best checkpoint.

## Inference

Single-image set inference:

```bash
python scripts/infer_image_config.py --config configs/infer_image.yaml
```

Video inference:

```bash
python scripts/infer_video_config.py --config configs/infer_video.yaml
```

Both inference scripts only save `.hdr` and tone-mapped `.png` outputs. They do not compute metrics.
