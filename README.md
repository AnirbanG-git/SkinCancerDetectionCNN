# Skin Cancer Detection with Convolutional Neural Networks

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Developed](#models-developed)
  - [Model 1: Vanilla CNN](#model-1-vanilla-cnn)
  - [Model 2: Augmented CNN using Keras](#model-2-augmented-cnn-using-keras)
  - [Model 3: Augmented CNN using Augmentor](#model-3-augmented-cnn-using-augmentor)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview
This project aims to build a reliable multiclass classification model to detect melanoma, a deadly skin cancer, using TensorFlow. Utilizing images from the International Skin Imaging Collaboration (ISIC), the model assists in reducing the manual effort required for diagnosis by dermatologists.

## Dataset
The dataset comprises 2357 images of various skin diseases, with a dominance of melanoma and moles images. It includes:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

Note: The dataset is highly imbalanced, with Seborrheic Keratosis having the fewest samples.

## Models Developed
### Model 1: Vanilla CNN
- Initial attempts showed the model overfitting with a 45% discrepancy between training and validation accuracies.

### Model 2: Augmented CNN using Keras
- Incorporation of a data augmentation layer in Keras reduced overfitting significantly, with a narrower 5% gap between training and validation accuracies. However, there was a slight decrease in overall accuracy.

### Model 3: Augmented CNN using Augmentor
- Further refinement using the Augmentor library addressed overfitting effectively, achieving approximately 80% accuracy. There remains scope for further improvements.

## Technologies Used
As the libraries versions keep on changing, it is recommended to mention the version of library used in this project:
- **Python** - version 3.8.18
- **pandas** - version 2.0.3
- **numpy** - version 1.22.3
- **matplotlib** - version 3.7.2
- **seaborn** - version 0.12.2
- **anaconda** - version 23.5.2

## How to Use
Details on how to setup, train, and evaluate the models can be found in the subsequent sections of this README.

## Contributing
Contributions to improve the models or extend the dataset are welcome. Please submit a pull request or open an issue to discuss potential changes.

## Acknowledgments
- International Skin Imaging Collaboration (ISIC) for providing the dataset used in this project.

## Contact
Created by [@AnirbanG-git] - feel free to contact me!
