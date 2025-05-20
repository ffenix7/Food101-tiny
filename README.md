# FOOD101-tiny Classifier

This project implements a deep learning model to classify images from a subset of the FOOD-101 dataset, referred to as FOOD101-tiny. The model utilizes a modified VGG16 architecture with DropBlock for regularization and is built using PyTorch.

## Notebook Overview

The `food101.ipynb` Jupyter Notebook contains the complete workflow:
1.  **Data Loading and Preprocessing**: Loads the FOOD-101-MINI dataset, applies transformations for data augmentation (random resized crop, horizontal flip, rotation, color jitter, random erasing) and normalization.
2.  **Data Visualization**: Displays sample images from each class in the training set.
3.  **Model Architecture**: Defines a modified VGG16 model with Batch Normalization and DropBlock2D layers. The classifier head is adjusted for the number of classes in the FOOD100-tiny dataset.
4.  **Model Training**: Trains the model using Adam optimizer and CrossEntropyLoss. It includes a simple early stopping mechanism based on validation loss. Training and validation loss/accuracy are printed for each epoch.
5.  **Evaluation**: After training, the model is evaluated on the test set. It prints the overall accuracy, a classification report (precision, recall, F1-score per class), and displays a confusion matrix.
6.  **Results Visualization**: Plots training vs. validation loss and accuracy over epochs.

## Dataset

The model is trained on the FOOD101-tiny dataset, which is a smaller version of the original FOOD-101 dataset. The data is organized in the `Data/food-101-tiny/` directory with `train` and `valid` subdirectories, each containing subfolders for the respective food classes.

The 10 classes included are:
- apple_pie
- bibimbap
- cannoli
- edamame
- falafel
- french_toast
- ice_cream
- ramen
- sushi
- tiramisu

## Model Architecture

The model is a modified version of `VGG16_bn` (VGG16 with Batch Normalization) pre-trained on ImageNet.
- The original features part of VGG16 is used.
- A `DropBlock2D` layer (block_size=10, drop_prob=0.3) is inserted after the feature extraction layers for regularization.
- An `AdaptiveAvgPool2d` layer is used to reduce the feature map dimensions to (1, 1).
- The classifier consists of a `Flatten` layer, a `Dropout` layer (p=0.5), and a `Linear` layer to output scores for the 10 food classes.

## Setup and Installation

1.  **Clone the repository (if applicable):**
   
    ```bash
    git clone https://github.com/ffenix7/Food101-tiny
    cd Food101-tiny
    ```

2.  **Create a virtual environment (recommended):**
   
    ```bash
    python -m venv venv
    # On Windows
    venv\\Scripts\\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
   
    The project relies on several Python libraries. These can be installed using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include:
    *   `torch`
    *   `torchvision`
    *   `scikit-learn`
    *   `matplotlib`
    *   `dropblock` (as per the notebook imports)

## Usage

1.  Ensure your dataset is correctly placed in the `Data/food-101-tiny/` directory.
2.  Open and run the `food101.ipynb` notebook using Jupyter Lab or Jupyter Notebook.
    ```bash
    jupyter lab food101.ipynb
    # or
    jupyter notebook food101.ipynb
    ```
3.  Execute the cells in the notebook sequentially to load data, train the model, and view the results.

## Important Variables

The notebook defines several important variables that can be configured:
- `img_size`: The size to which images are resized (default: 128).
- `batch_size`: The number of samples per batch (default: 64).
- `num_epochs`: The maximum number of training epochs (default: 25).
- `patience`: The number of epochs to wait for improvement in validation loss before potential early stopping (default: 3, though the early stopping logic in the provided training loop might need refinement for full functionality).

## License

This project is licensed under the terms of the LICENSE file.

The DropBlock regularization technique is based on the paper:
"DropBlock: A regularization method for convolutional networks"
Ghiasi, Golnaz, Tsung-Yi Lin, and Quoc V. Le.
arXiv preprint arXiv:1810.12890 (2018).
Link: https://arxiv.org/abs/1810.12890
