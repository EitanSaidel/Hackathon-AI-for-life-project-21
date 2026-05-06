# Generative AI for Medical Imaging

[Presentation](https://canva.link/ujw5v02wmnhomyc)

## Enviroment

- `scikit-learn`
- `monai`
- `nibabel`
- `monai-generative`
- `einops`

```bash
pip install scikit-learn monai nibabel monai-generative einops
```

## Dataset

Download `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`, a collection of 1251 patient scans

[BRaTS 2023](https://www.synapse.org/Synapse:syn51514105) (**training** data, 12.3 GB)

> Requires an account

Each subfolder (named `BRaTS-GLI-{id}`), contains five `.nii.gz` files. Namely:
- `BRaTS-GLI-{id}-seg.nii.gz` (Segmentation mask)
- `BRaTS-GLI-{id}-t1c.nii.gz` (T1 contrast scan)
- `BRaTS-GLI-{id}-t1n.nii.gz` or `BRaTS-GLI-{id}-t1.nii.gz` (T1 scan)
- `BRaTS-GLI-{id}-t2f.nii.gz` (FLAIR scan)
- `BRaTS-GLI-{id}-t2w.nii.gz` (T2 scan)

The purpose of each file is:
- **T1** - baseline MRI scan, shows just the anatomy.
- **T1c** (T1 contrast) - MRI scan after a contrast agent has been injected, highlights tissue growth.
- **T2** - MRI scan sensitive to fluids, highlights inflamation.
- **FLAIR** (Fluid Attenuated Inversion Recovery) - MRI scan timed to ignore fluids, highlights absormal tissue.
- **Segmentation mask** - manually drawn mask where each pixel has been labeled (e.g. "healthy tissue").

> By combining these images, we can identify different tissue types.

These will then be split into a training set (80%) and a validation set (20%).

## Progress

- [x] Download and parse dataset

- [ ] Build variational autoencoder (VAE)

## Process

1. Load data and fix a 128^3 image for each scan
2. Compress to 'latent space' via a VAE (raw pixel data is too large for computations)
3. ...
4. Revert to a 3D via (again via VAE)
5. Manually confirm outputs
