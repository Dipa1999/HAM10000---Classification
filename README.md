# HAM10000 skin lesion classification

PyTorch pipeline for seven-class skin lesion classification on the HAM10000 dataset. Five models are trained under one recipe: three CNNs trained from scratch (HAMCNN, Enhanced CNN, Double CNN) and two pretrained transfer models (ViT-DINOv2 and EfficientNet-B3). The splits are grouped by `lesion_id`, the outlier filter is fitted on the training split only, and the calibration and ensemble weights are fitted on the validation split. The test split is read once, in `evaluate.py --final`.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Results (test split, calibrated)

| Model            | Accuracy | Macro-F1 | Balanced acc. | ECE raw | ECE cal |
|------------------|---------:|---------:|--------------:|--------:|--------:|
| HAMCNN           | 0.635    | 0.469    | 0.547         | 0.174   | 0.091   |
| Enhanced CNN     | 0.647    | 0.479    | 0.566         | 0.210   | 0.077   |
| Double CNN       | 0.655    | 0.478    | 0.587         | 0.212   | 0.060   |
| ViT-DINOv2       | 0.828    | 0.720    | 0.705         | 0.164   | 0.020   |
| EfficientNet-B3  | 0.784    | 0.646    | 0.663         | 0.227   | 0.035   |
| Ensemble (5)     | 0.816    | 0.727    | 0.730         | 0.272   | 0.019   |

Single seed, single split. The difference between the ensemble and ViT-DINOv2 is small and was not tested for significance. The `df` and `vasc` classes have 17 and 21 test images, so their per-class scores are unstable.

## Dataset

The dataset is not included. Download HAM10000 (Tschandl et al., 2018; available on Harvard Dataverse, doi:10.7910/DVN/DBW86T) and check its license on the dataset page before any redistribution.

Place the files where `prep_config.json` expects them:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

## Setup

Tested with Python 3.12 and the following versions:

- PyTorch 2.6.0 (CUDA 12.4 build)
- timm 1.0.30
- albumentations 2.0.8
- scikit-learn 1.8.0
- numpy, pandas, scipy, opencv-python, pillow, matplotlib, tqdm, jupyter

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.30 albumentations==2.0.8 scikit-learn==1.8.0 \
    numpy pandas scipy opencv-python pillow matplotlib tqdm jupyter
```

The code is designed for 5 GB VRAM and 10 GB RAM. Run every command from the project root, because all paths in `config.json` and `prep_config.json` are relative to it.

## Pipeline

1. **Splits and EDA.** Run `01_EDA_Preparation.ipynb`, or `python prepare.py`. This writes `processed/{train,val,test}.csv`, `processed/splits_manifest.json` and the figures in `processed/eda/`.
2. **Training, evaluation, plots, Grad-CAM.** Run `02_Run_Pipeline.ipynb`. It builds the 224x224 uint8 cache, trains the models, fits the calibration on validation and produces the figures. The final test cell should run only once, when the validation calibration is final.

Equivalent command-line steps:

```bash
python train.py --model vit_dinov2 [--resume]   # one model; repeat for each key in config.json
python evaluate.py                              # validation: fits temperatures, ensemble, tau
python evaluate.py --final                      # test: once, with the frozen calibration
python plots.py --split val                     # add --split test after --final
python gradcam.py                               # transfer models, validation split
```

Outputs are written to `runs/<config_hash>/`, where `<config_hash>` is the first 8 characters of the MD5 of `config.json` without path keys. Changing a training or evaluation setting creates a new run directory and leaves the old ones untouched.

## Code structure

| File | Role |
|------|------|
| `config.json` | All hyperparameters, model specs, paths |
| `prep_config.json` | Split ratios, seed, outlier filter, metadata paths |
| `prepare.py` | Cleaning, lesion-grouped split, outlier flags, manifest |
| `data.py` | Image cache, transforms, weighted sampler, mixup/CutMix |
| `models.py` | Architectures and the `MODELS` registry |
| `losses.py` | Focal loss, mixed-batch loss, reconstruction loss |
| `optim.py` | SAM, optimizer and scheduler builders |
| `train.py` | Training loop, checkpoints, resume, early stopping |
| `evaluate.py` | TTA probabilities, calibration, ensemble, metrics |
| `gradcam.py` | Grad-CAM for the transfer models |
| `plots.py` | Training curves, confusion matrices, per-class F1 |

## Design constraints

- Test data is never used for training, calibration, ensemble fitting or model selection.
- Ensemble weights are bounded to [0, 1], sum to 1, and are shrunk toward uniform weights with factor `ensemble_shrink`. The objective is severity-weighted per-class recall, with severities akiec 2, bcc 2.5, mel 3, all others 1.
- The logit adjustment `tau * log(pi)` is fitted on validation. In the reported run the selected `tau` is 0, so it has no effect.
- Each checkpoint and cached result stores the split hash. Changing the split CSVs without rerunning `prepare.py` makes the code fail loudly.

## Limitations

- One seed and one split. No confidence intervals.
- Near-duplicate images are not detected. Grouping uses `lesion_id` only, so residual leakage across splits cannot be excluded.
- Validation is used for early stopping, checkpoint selection and all post-hoc fitting, so validation metrics are optimistic.
- The ensemble is fitted on severity-weighted recall, but the reported metric is macro-F1. The effect of this mismatch was not tested.
- The dataset comes from a single clinical source with limited coverage of darker skin tones.
- The models are unreliable on inputs outside the dermoscopic domain, and no out-of-distribution detection is implemented.

This project is for research and education only. It is not a diagnostic tool and must not be used for clinical decisions.

## License

Code: MIT (see `LICENSE`). The dataset is distributed under its own license, which is separate from this one.
