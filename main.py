import os
import glob
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

# MONAI & Generative Extensions
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    CropForegroundd, NormalizeIntensityd,
    RandSpatialCropd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AutoencoderKL
from monai.networks.schedulers import DDPMScheduler
from generative.networks.nets import DiffusionModelUNet, ControlNet

# --- 1. DATA PREPARATION ---
# Organizing 1,251 patients from the BraTS dataset
data_root = "/Users/eitansaidel/Downloads/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
patient_folders = sorted(glob.glob(os.path.join(data_root, "BraTS-GLI-*")))
datalist = []

for folder in patient_folders:
    if not os.path.isdir(folder): continue
    t1 = glob.glob(os.path.join(folder, "**", "*t1n.nii.gz"), recursive=True)
    t1c = glob.glob(os.path.join(folder, "**", "*t1c.nii.gz"), recursive=True)
    t2 = glob.glob(os.path.join(folder, "**", "*t2w.nii.gz"), recursive=True)
    flair = glob.glob(os.path.join(folder, "**", "*t2f.nii.gz"), recursive=True)
    seg = glob.glob(os.path.join(folder, "**", "*seg.nii.gz"), recursive=True)

    if all([t1, t1c, t2, flair, seg]):
        datalist.append({"image": [t1[0], t1c[0], t2[0], flair[0]], "label": seg[0]})

train_files, _ = train_test_split(datalist, test_size=0.2, random_state=42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. THE TRANSFORMATION PIPELINE ---
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
    EnsureTyped(keys=["image", "label"]),
])

loader = DataLoader(Dataset(data=train_files, transform=transforms), batch_size=1)
inputs = next(iter(loader))["image"].to(device)

# --- 3. VAE (COMPRESSION) ---
vae = AutoencoderKL(
    spatial_dims=3, in_channels=4, out_channels=4, latent_channels=4,
    channels=(32, 64, 128), num_res_blocks=1, norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)
vae.eval()

with torch.no_grad():
    latent_space = vae.encode_stage_2_inputs(inputs)
    # Sanity Check: Visualizing the reconstruction to check MSE
    recon = vae.decode_stage_2_outputs(latent_space)
    mse = torch.mean((inputs - recon) ** 2).item()
    print(f"✅ Latent Shape: {latent_space.shape}")
    print(f"⚠️ VAE MSE: {mse:.4f}")

# --- 4. GENERATIVE ENGINE & CONTROLNET PATCH ---
unet = DiffusionModelUNet(
    spatial_dims=3, in_channels=4, out_channels=4, num_res_blocks=1,
    num_channels=(32, 64, 64), attention_levels=(False, True, True),
    num_head_channels=(0, 32, 32),
).to(device)

controlnet = ControlNet(
    spatial_dims=3, in_channels=4, num_channels=(32, 64, 64),
    num_res_blocks=1, attention_levels=(False, True, True),
    num_head_channels=(0, 32, 32),
).to(device)

# MANUAL OVERWRITE: Fixes 'conditioning_embedding' TypeError and channel mismatches
controlnet.controlnet_cond_embedding = nn.Sequential(
    nn.Conv3d(1, 16, kernel_size=3, padding=1),
    nn.SiLU(),
    nn.Conv3d(16, 32, kernel_size=3, padding=1)
).to(device)

controlnet.load_state_dict(unet.state_dict(), strict=False)
controlnet.eval()
unet.eval()

# --- 5. THE FIXED GROWTH LOOP ---
# 1-channel mask for spatial guidance
mask = torch.zeros((1, 1, 32, 32, 32)).to(device).float()
mask[:, :, 14:18, 14:18, 14:18] = 1.0

scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(50)
current_latents = torch.randn_like(latent_space).to(device)

print("🚀 Starting Synthesis...")
for t in scheduler.timesteps:
    with torch.no_grad():
        # Step A: Get ControlNet residuals
        down_samples, mid_sample = controlnet(
            current_latents,
            timesteps=torch.tensor([t]).to(device),
            controlnet_cond=mask
        )

        # Step B: UNet Denoising
        noise_pred = unet(
            current_latents,
            timesteps=torch.tensor([t]).to(device),
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample
        )

        # Step C: FIX - Tuple indexing for the scheduler result
        step_result = scheduler.step(noise_pred, t, current_latents)
        current_latents = step_result[0]  # Accessing the first element of the tuple

# --- 6. DECODING & SAVING ---
print("✅ Decoding final synthetic volume...")
with torch.no_grad():
    synthetic_image = vae.decode_stage_2_outputs(current_latents)

synthetic_array = synthetic_image.detach().cpu().numpy().squeeze()
flair_final = synthetic_array[3, :, :, :]
nib.save(nib.Nifti1Image(flair_final, np.eye(4)), "final_synthetic_tumor.nii.gz")
print("✅ SAVED: final_synthetic_tumor.nii.gz")