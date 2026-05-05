import os
import glob
from sklearn.model_selection import train_test_split
data_root = "/Users/eitansaidel/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
patient_folders = sorted(glob.glob(os.path.join(data_root, "BraTS-GLI-*")))
datalist = []

for folder in patient_folders:
    if not os.path.isdir(folder):
        continue

    t1_files = glob.glob(os.path.join(folder, "**", "*t1n.nii.gz"), recursive=True)

    if not t1_files:
        t1_files = glob.glob(os.path.join(folder, "**", "*t1.nii.gz"), recursive=True)

    t1c_files = glob.glob(os.path.join(folder, "**", "*t1c.nii.gz"), recursive=True)
    t2_files = glob.glob(os.path.join(folder, "**", "*t2w.nii.gz"), recursive=True)
    flair_files = glob.glob(os.path.join(folder, "**", "*t2f.nii.gz"), recursive=True)
    seg_files = glob.glob(os.path.join(folder, "**", "*seg.nii.gz"), recursive=True)

    # Check if any sequence is missing
    if not (t1_files and t1c_files and t2_files and flair_files and seg_files):
        print(f" Skipping folder (Missing files): {os.path.basename(folder)}")
        continue
    patient_entry = {
        "image": [t1_files[0], t1c_files[0], t2_files[0], flair_files[0]],
        "label": seg_files[0]
    }
    datalist.append(patient_entry)

print(f"\n Successfully organized {len(datalist)} patients.")
if len(datalist) > 0:
    train_files, val_files = train_test_split(datalist, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print("\nExample Patient Dictionary:")
    print(train_files[0])
else:
    print(" No patients were added.")

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    NormalizeIntensityd,
    EnsureTyped
)
from monai.data import Dataset, DataLoader

print("\nBuilding the MONAI Transform Pipeline")

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, num_workers=0)

print("\nPipeline built successfully")
print("\nFetching the first patient through the pipeline")
for batch_data in train_loader:
    inputs, labels = batch_data["image"], batch_data["label"]
    print(f"Loaded Image Shape: {inputs.shape}")
    print(f"Loaded Label Shape: {labels.shape}")
    break
print("\nData check complete.")
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropd,
    EnsureTyped
)
from monai.data import Dataset, DataLoader
print("\nBuilding the updated MONAI Transform Pipeline...")
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[128, 128, 128],
            random_size=False
        ),

        EnsureTyped(keys=["image", "label"]),
    ]
)
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, num_workers=0)
print("Pipeline updated successfully!")
print("\nFetching the first patient through the updated pipeline")
for batch_data in train_loader:
    inputs, labels = batch_data["image"], batch_data["label"]
    print(f"Loaded Image Shape: {inputs.shape}")
    print(f"Loaded Label Shape: {labels.shape}")
    break

print("\nData check complete.")

import torch
from monai.networks.nets import AutoencoderKL

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

vae = AutoencoderKL(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    latent_channels=4,
    channels=(32, 64, 128),
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)

vae.eval()

inputs = inputs.to(device)
with torch.no_grad():
    latent_space = vae.encode_stage_2_inputs(inputs)

print(f"Original Volume: {inputs.shape}")
print(f"Compressed Latent Space: {latent_space.shape}")

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

print("\nInitializing the Latent Diffusion Model (LDM)")

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    beta_start=0.0015,
    beta_end=0.0195,
)

unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    num_res_blocks=1,
    channels=(32, 64, 64),
    attention_levels=(False, True, True),
    num_head_channels=(0, 32, 32),
).to(device)

unet.eval()

print(" Diffusion Engine successfully initialized!")

pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"Total trainable parameters in UNet: {pytorch_total_params:,}")







