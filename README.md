# MNIST Handwritten Digit Classifier (CNN)

This repository contains a Convolutional Neural Network (CNN) that learns to recognize handwritten digits (0 to 9) using the MNIST dataset. The workflow includes data preprocessing, model training, and evaluation using test accuracy and a confusion matrix.

---
## Objectives
1. Load the MNIST dataset of handwritten digit images.
2. Preprocess images to a format suitable for CNN training.
3. Train a neural network to classify digits from 0 to 9.
4. Evaluate the trained model on the official MNIST test set.
---
## Dataset
MNIST is a standard benchmark dataset for handwritten digit recognition.

1. Training set: 60,000 images
2. Test set: 10,000 images
3. Image size: 28 × 28 pixels
4. Image type: grayscale (single channel)
5. Labels: integers from 0 to 9

The dataset is loaded using:
`tf.keras.datasets.mnist.load_data()`

---
## Method
### Preprocessing

The preprocessing steps applied are:

1. Type conversion: convert pixels to `float32`
2. Normalization: scale pixel values from `[0, 255]` to `[0, 1]`
3. Reshaping: add a channel dimension for CNN input  
   `(28, 28) → (28, 28, 1)`
---
### Model Architecture
---
The classifier is a CNN designed for MNIST-scale images.

1. Convolution layer: 32 filters, kernel 3 × 3, ReLU, same padding
2. Max pooling: 2 × 2
3. Convolution layer: 64 filters, kernel 3 × 3, ReLU, same padding
4. Max pooling: 2 × 2
5. Flatten: convert feature maps into a single vector
6. Dropout: 0.3 to reduce overfitting
7. Dense layer: 128 units, ReLU
8. Output layer: 10 units, Softmax (probabilities for digits 0 to 9)
---
### Training Configuration
---
1. Loss function: SparseCategoricalCrossentropy  
   (appropriate because labels are integers 0 to 9)
2. Optimizer: Adam with learning rate 1e-3
3. Metric: accuracy
4. Validation split: 0.1 (10% of training data used for validation)
5. Batch size: 128
6. Max epochs: 15
7. Early stopping: monitors validation accuracy, patience = 3, restores best weights
---
### Evaluation
---
After training, the script reports:

1. Test accuracy on the official test set
2. Test loss
3. Confusion matrix  
   Rows represent true labels, columns represent predicted labels

Interpretation notes:
1. Large values on the diagonal mean correct classification.
2. Off-diagonal values represent mistakes, usually between visually similar digits.
---
## Results
Typical performance for this architecture on MNIST is approximately:
1. Test accuracy around 98.5% to 99.3% (varies slightly by system and run)

Your run prints the final metrics and the confusion matrix.

---
## Requirements
1. Python 3.11 (recommended for TensorFlow compatibility on Windows)
2. Packages:
   1. numpy
   2. tensorflow

---

## Installation

### Option A: Install packages globally (simplest)
```bash
pip install numpy tensorflow

```
---

## Author

Minahil Ahsan Awan   
Email: minahilahsaanawan@gmail.com  

LinkedIn: https://linkedin.com/in/minahilahsaanawan
