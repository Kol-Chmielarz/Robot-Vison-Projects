MNIST and CIFAR-10 Neural Network Project

This project explores deep learning techniques using PyTorch on the MNIST and CIFAR-10 datasets. The neural networks are trained to classify handwritten digits and colored images across 10 classes. The implementation includes model training, evaluation, visualization of losses and accuracies, and predictions on sample test images.


Features
- Custom convolutional neural networks (ConvNet) for image classification.
- Train/test pipelines for both MNIST and CIFAR-10 using PyTorch.
- Accuracy and loss plots over epochs.
- Tensorboard-compatible logs.
- Visualization of predicted classes alongside actual labels.
- Parameterized CLI (batch size, learning rate, epochs, mode).


Datasets
- [MNIST](http://yann.lecun.com/exdb/mnist/) - Handwritten digit classification
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - 10 object categories


Tech Stack
- Python
- PyTorch
- Torchvision
- Matplotlib
- argparse

Image Filtering and Warping

This project demonstrates core computer vision techniques such as image filtering, edge detection, and geometric warping using Python and OpenCV. It simulates key operations found in real-time visual pipelines used in robot vision systems.

Features
- Convolution with custom kernels (Gaussian, Sobel, etc.).
- Edge detection with filters and thresholding.
- Perspective warping and homography transformations.
- Image transformation visualization (before/after).
- Implemented in a clean, modular Jupyter notebook format.

- Tools & Libraries
- Python
- NumPy
- OpenCV
- Matplotlib


Inputs
- Local images (e.g., road scenes, printed patterns)
- Sample images processed with custom filters and matrix transformations


Output
- Side-by-side filtered image comparisons
- Warped outputs using selected source/destination points
