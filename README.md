Contains 2 pretrained models.

A from-scratch implementation of a denoising diffusion model trained on the MNIST handwritten digit dataset, built for DAT494 (Deep Learning Structures).

## Overview

This model implements the core idea behind diffusion models: **add noise to images, then train a neural network to predict and remove that noise.** Given a noisy image, the model learns to recover the clean original.

## Architecture

The model uses a **U-Net** with skip connections:

| Layer | Type | Channels |
|-------|------|----------|
| Down 1 | Conv2d + GELU + MaxPool | 1 → 32 |
| Down 2 | Conv2d + GELU + MaxPool | 32 → 64 |
| Bottleneck | Conv2d + GELU | 64 → 64 |
| Up 1 | Upsample + Conv2d + GELU | 128 → 64 |
| Up 2 | Upsample + Conv2d + GELU | 96 → 32 |
| Up 3 | Upsample + Conv2d | 32 → 1 |

## Noising Strategy

Images are corrupted using linear interpolation between the original and uniform random noise, with a noise amount sampled from `[0, 1]`:

```python
torch.lerp(input, noise, noise_amount)
```

## Training

- **Dataset:** MNIST (60,000 training images)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=0.0001)
- **Batch size:** 128
- **Epochs:** 15

## Inference / Sampling

Samples are generated iteratively from pure noise using a progressive denoising loop, stepping from fully noisy to increasingly clean images over multiple passes.

## Requirements
```bash
pip install torch torchvision matplotlib
```


## Results

After 15 epochs, the model successfully denoises MNIST digits across the full noise spectrum and can generate recognizable digits from random noise through iterative refinement.
