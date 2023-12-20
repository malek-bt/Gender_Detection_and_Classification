# Gender_Detection_and_Classification


This project focuses on detection and classification of gender using machine learning models, including MobileNetV2, EfficientNetB0, and Xception.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Techniques Used](#techniquesUsed)
- [Models](#models)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Testing](#testing)
- [Future Work](#Contributors)

## Introduction

Gender Detection and Classification is a machine learning project aimed at recognizing and classifying gender from images. The project utilizes three different deep learning models: MobileNetV2, EfficientNetB0, and Xception, to achieve accurate classification.

## Dataset

The dataset used for training and testing the models can be found on Kaggle. The gender detection and classification image dataset is available at:

[Gender detection and classification image dataset on Kaggle](https://www.kaggle.com/datasets/trainingdatapro/gender-detection-and-classification-image-dataset/)

This dataset consists of labeled images of persons. Each image is labeled with the corresponding gender (e.g., man , women).


## Techniques Used


### Data Augmentation

Data augmentation is a key technique used to artificially increase the size of the dataset by applying various transformations to the existing images. Common augmentations include rotation, flipping, zooming, and changes in brightness. This helps improve the model's ability to generalize and handle variations in input data.

### Early Stopping

To prevent overfitting and find the optimal number of training epochs, we employed early stopping during model training. Early stopping monitors the model's performance on a validation set and halts training when the performance stops improving, preventing the model from learning noise in the training data.


## Models

1. **MobileNetV2**: A lightweight convolutional neural network architecture designed for mobile and edge devices.

2. **EfficientNetB0**: Part of the EfficientNet family, known for balancing model efficiency and accuracy.

3. **Xception**: A highly efficient convolutional neural network architecture known for its exceptional performance in image classification tasks.


## Usage

In this example, users are instructed to open the Colab notebook and run a specific cell that installs the required libraries using the `!pip install` command. Adjust the dependencies in the command according to the libraries used in your project.

## Training

### Model Training Example (DenseNet201)

Below is an example of training the Chess Men Classification model using MobileNetV2 as the base model:

```python

from tensorflow.keras.applications import MobileNetV2


# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model.fit(train_generator, epochs=num_epochs, validation_data=test_generator, callbacks=[early_stopping])

# Save the model
model.save('/content/mobileNet_gender_classification.keras')


```
## Evaluation

Evaluate the performance of the trained models using the testing dataset. The evaluation script computes metrics such as accuracy , precision, recall, and F1 score. Run the evaluation script to obtain performance metrics and analyze the model's effectiveness.

Example command:

```bash
# Evaluate the model on the test set
eval_result = model.evaluate(test_generator)
print("Test Accuracy:", eval_result[1])


```

## Results

### Model Performance

After evaluating the trained models, MobileNetV2 demonstrated the highest accuracy among the models, achieving an accuracy of 85%. This indicates that MobileNetV2 performed exceptionally well in classifying chess pieces on the validation set.


## Acknowledgments

This project has been made possible through the contributions and support of various individuals and organizations. We extend our sincere thanks to:

- **Kaggle Community:** For providing the Chessman Image Dataset used in this project. The Kaggle community has been instrumental in fostering a collaborative environment for data science and machine learning.

- **TensorFlow and Keras Developers:** We express our appreciation to the developers of TensorFlow and Keras for creating powerful deep learning frameworks that facilitated the implementation of complex models in this project.

- **Colab Notebooks:** The project extensively utilized Google Colab Notebooks for its ease of use and access to GPU resources. Colab greatly accelerated model training and experimentation.

- **Open Source Contributors:** Many open-source libraries and tools have played a crucial role in the development of this project. We are grateful for the efforts of the open-source community that continually contributes to the field of machine learning.


## Testing

To test the trained models on new images, follow the steps below:

1. **Add a New Model:**
   - If you have a new model you'd like to test, make sure the model is saved in a compatible format (e.g., Keras model in HDF5 format).

2. **Download or Prepare a New Image:**
   - Select or obtain an image that you want to use for testing. Ensure that the image is in a supported format (e.g., JPEG or PNG).

3. **Update the Test Code:**
   - Open the provided test script or code snippet 
   - Locate the section where the model is loaded, and update the model path to point to your new model.

   ```python
   # Load the saved model (update the model path)
   loaded_model = load_model('/path/to/your/new_model.h5')
   ```

## Contributors

Thanks to the following people who have contributed to this project:

- [Malek Bentaher](https://github.com/malek-bt)
- [Hatem Henchir](https://github.com/hatemhenchir)
- [Salma Missaoui]()

We welcome contributions and ideas from the community to further advance this project.


