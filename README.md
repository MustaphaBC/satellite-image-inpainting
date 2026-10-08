# Context-Aware Satellite Image Inpainting

A deep-learning project that reconstructs masked regions in satellite images. The notebook implements a context-aware GAN workflow with a generator, PatchGAN-style discriminator, perceptual loss, and image-quality evaluation.

## Method

- Uses the UC Merced Land Use image collection (21 classes, 2,100 images) as the source dataset.
- Creates masked training examples and trains a generator to fill missing image regions.
- Evaluates reconstructions with PSNR, SSIM, and LPIPS metrics.
- Includes a detailed project report in `rapport detailler.pdf`.

## Requirements

The notebook uses Python, PyTorch/torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-image, and LPIPS. A CUDA-capable GPU is recommended for training. Install compatible versions for your platform, for example:

```bash
python -m pip install torch torchvision opencv-python pillow numpy matplotlib scikit-image lpips
```

## Run

1. Place the UC Merced images under `UCMerced_LandUse/Images` or update the dataset path in the notebook.
2. Open `context-aware-satellite-image-inpainting-v-2-0 (1).ipynb` in Jupyter.
3. Run the cells in order to prepare data, train, and evaluate the model.

## Data and artifacts

The dataset, trained checkpoints, generated images, metrics, and saved models are excluded from the public repository.
