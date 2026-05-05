# Generative AI for Medical Imaging

## Enviroment

- `scikit-learn`
- `monai`

```bash
pip install scikit-learn monai
```

## Dataset

[BRaTS 2023](https://www.synapse.org/Synapse:syn51514105) (**training** data, 12.3 GB)

> Requires an account

A collection of patient scans, each includes four MRI scans:

- **T1** - baseline scan
- **T1c** (T1 contrast) - scan after a contrast agent has been injected
- **T2** - scan sensitive to fluids
- **FLAIR** (Fluid Attenuated Inversion Recovery) - scan timed to ignore fluids

**T1** just shows the anatomy. **T1c** shows where there is tissue growth. **T2** shows were there is swelling. **FLAIR** highlights abnomal tissues.

> By combining these images, we can identify different tissue types.

Additionaly, there is a _segmentation mask_. This is a manually drawn mask where each pixel has been labeled (e.g. "healthy tissue").

## Progress
