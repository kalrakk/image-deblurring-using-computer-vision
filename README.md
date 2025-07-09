
# Image Deblurring Using Computer Vision

This repository presents our final project proposal for developing a deep learning-based image deblurring framework. The goal is to address key challenges in dataset accessibility, preprocessing, and model generalization by building a modular system that simplifies training and evaluation of state-of-the-art models.

## Project Overview

Image deblurring remains a challenging task due to variations in blur patterns and lack of accessible, domain-specific datasets. This project proposes a robust solution through a custom Python package (`DBlur`) and evaluation on curated datasets with deep learning models.

## Objectives

1. Develop a deep learning-based image deblurring framework capable of generalizing to real-world scenarios.
2. Benchmark the performance of existing deblurring models on multiple datasets.
3. Build and release a Python package (`DBlur`) for streamlined training and testing.
4. Evaluate performance using PSNR and SSIM metrics.

## Datasets

The project leverages diverse and domain-relevant datasets:

- **CelebA Dataset**: Face deblurring dataset (2 million training images, 1,299 validation, 1,300 test) generated using blurred kernels.
- **Helen Dataset**: Smaller face dataset (2,000 training, 155 validation, 155 test) also using synthetic blur kernels.

Dataset references:

- https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets
- https://www.datacamp.com/blog/computer-vision-projects

## Methodology

### Data Preprocessing

- Normalization and resizing  
- Data augmentation (random crop, flip)  
- Noise removal and pre-filtering  

### Model Selection and Training

- **U-Net Based Deblurring**: Optimized CNN for image-to-image translation  
- **Transformer-Based Deblurring**: Leveraging attention mechanisms for improved reconstruction  
- Hyperparameter tuning and GPU-accelerated training

### Model Evaluation

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures fidelity of restored images  
- **SSIM (Structural Similarity Index Measure)**: Evaluates perceptual similarity  
- Cross-validation across datasets  
- Benchmark against traditional deblurring techniques

## Expected Outcomes

- A high-performance deep learning framework for deblurring various types of images
- A command-line ready Python package `DBlur` for model training and evaluation
- Curated datasets packaged and documented for researchers
- Public benchmark results for comparison and future research

## Conclusion

This project addresses critical needs in image restoration research by offering a unified framework for image deblurring. By utilizing modern architectures and curated datasets, it aims to become a valuable tool for both academic and applied computer vision communities.

## References

1. Zhang, K. et al. (2022). [Comprehensive Review on Deep Learning for Deblurring](https://arxiv.org/pdf/2201.10700)
2. Denis, L. (2013). [Blind Deconvolution: Inverse Problems and Applications](https://www.ens-lyon.fr/PHYSIQUE/Equipe3/SIERRA/old/pdf/DENIS_L_slides.pdf)
3. Zhang, B. et al. (2024). [Image Deblurring via Residual Wavelet Transformer](https://doi.org/10.1016/j.eswa.2023.123005)
4. Shi, X.-P. et al. (2024). [Multi-Scale U-Net with Dilated Convolution](https://doi.org/10.1177/00368504241231161)
