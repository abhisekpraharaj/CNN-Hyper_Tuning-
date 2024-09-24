# CNN-Hyper_Tuning(War_in_Progress)[Auto_Hypertuning_CNN]-
This repository focuses on a model where hyperparameters are automatically tuned to optimize performance. The best model, after tuning, is then evaluated on the test data to determine its accuracy and overall performance.


# Citation:
Citation: The "Dogs vs. Cats" dataset is sourced from Kaggle: https://www.kaggle.com/competitions/dogs-vs-cats


# Importing Libraries
Importing TensorFlow and Keras libraries:
TensorFlow is imported for deep learning functionalities.
Various Keras modules are imported for building and configuring neural networks.
## Importing Keras Tuner:
This library is used for hyperparameter tuning of neural networks.
## Importing image processing libraries:
OpenCV (cv2) and Pillow (PIL) are imported for image manipulation.
## Importing data manipulation libraries:
Pandas is imported for data handling and analysis.
## Importing system libraries:
The 'os' module is imported for file and directory operations.
## Suppressing warnings:
All warnings are being ignored using the warnings module.
## Creating an output directory:
A directory named 'tuning_output' is created in the '/kaggle/working/' path for storing output files.

# Feature Extraction:
The code iterates through a list of training images (presumably stored in train_images).
For each image, it extracts and stores several features:
a. Image name: The full filename of the image.
b. Category: The name of the image without the file extension (assumed to be either 'dog' or something else).
c. Code: A binary classification where 1 represents 'dog' and 0 represents 'not dog'.
d. Size: The dimensions of the image (height, width).
e. Aspect ratio: The ratio of height to width.
It uses OpenCV (cv2) to read each image and extract its dimensions.
The extracted features are stored in separate lists: image_name, category, code, size, and aspect_ratio.

Outputs:

Five lists are populated:
# image_name: Contains the full filenames of all images.
# category: Contains the image names without file extensions.
# code: Contains 1 for dog images, 0 for others.
# size: Contains tuples of (height, width) for each image.
# aspect_ratio: Contains the height/width ratio for each image.

These outputs are not directly displayed but are stored in memory for further processing. This feature extraction step is likely preparing the data for analysis or for input into a machine learning model. The binary classification (dog/not dog) suggests this might be part of a dog breed classification task or a dog detection task.

### Model Building
This is a Convolutional Neural Network (CNN) model with several layers designed for binary classification, likely based on your earlier project with the 'Dogs vs. Cats' dataset. The model structure combines convolutional layers, batch normalization, pooling, and dropout layers, followed by dense layers for classification. It incorporates hyperparameter tuning using `keras_tuner` to optimize the model during training. Let's break down its architecture:

### 1. **Input Layer**
   - Takes an input image of shape `(img_width, img_height, 3)` (3 channels for RGB).
   
### 2. **Convolutional Layer 1**
   - **Filters**: Variable, determined by hyperparameter tuning between 32 and 128, incremented by 32.
   - **Kernel Size**: Either 3x3 or 5x5, depending on the tuning.
   - **Activation Function**: ReLU.
   - **Padding**: `same` padding to maintain spatial dimensions.
   - **Regularization**: L2 regularization applied with a tunable factor between 1e-5 and 1e-1.
   
### 3. **Batch Normalization 1**
   - Normalizes the output of the first convolutional layer, helping stabilize and speed up the training process.
   
### 4. **Max Pooling 1**
   - Pooling size is (2, 2), reducing the spatial dimensions by half.
   
### 5. **Convolutional Layer 2**
   - **Filters**: Tunable between 64 and 128, with a step size of 64.
   - **Kernel Size**: Tunable to either 3x3 or 5x5.
   - **Activation Function**: ReLU.
   - **Padding**: `same`.
   - **Regularization**: L2 regularization between 1e-3 and 1e-2.
   
### 6. **Batch Normalization 2**
   - Normalizes the output of the second convolutional layer.

### 7. **Max Pooling 2**
   - Pooling size is (2, 2), reducing the spatial dimensions.

### 8. **Dropout 1**
   - Dropout rate is tunable between 0.1 and 0.4, used to prevent overfitting by randomly turning off a percentage of neurons during training.

### 9. **Convolutional Layer 3**
   - **Filters**: Tunable between 128 and 512, with a step size of 128.
   - **Kernel Size**: Tunable to either 3x3 or 5x5.
   - **Activation Function**: ReLU.
   - **Padding**: `same`.
   - **Regularization**: L2 regularization between 1e-4 and 1e-2.
   
### 10. **Batch Normalization 3**
   - Normalizes the output of the third convolutional layer.

### 11. **Max Pooling 3**
   - Pooling size is (2, 2), reducing the spatial dimensions.

### 12. **Dropout 2**
   - Dropout rate is tunable between 0.2 and 0.3.

### 13. **Flatten Layer**
   - Converts the multi-dimensional tensor from the convolutional layers into a one-dimensional tensor.

### 14. **Dense Layer 1**
   - **Units**: Tunable between 64 and 512, in steps of 64.
   - **Activation Function**: ReLU.
   - **Regularization**: L2 regularization between 1e-3 and 1e-2.

### 15. **Dropout 3**
   - Dropout rate is tunable between 0.1 and 0.2.

### 16. **Output Layer**
   - **Units**: 1 (single neuron).
   - **Activation Function**: Sigmoid, used for binary classification output.
   
### 17. **Optimizer and Learning Rate Schedule**
   - **Optimizer**: Adam optimizer with an exponentially decaying learning rate, starting at `1e-4`, decaying every 10,000 steps by a factor of 0.9.
   
### 18. **Loss Function**
   - **Binary Crossentropy**: Used for binary classification problems.

### 19. **Metrics**
   - **Accuracy**: Used to evaluate model performance during training.

### Hyperparameters Tuned
- Number of filters for Conv layers.
- Kernel size for Conv layers.
- L2 regularization factor for different Conv and Dense layers.
- Dropout rates.
- Number of units in the first Dense layer.

This model is designed to optimize both accuracy and generalization with adjustable parameters for different layers and regularization techniques (L2 and Dropout). Hyperparameter tuning will help identify the best-performing configuration.


## Tuner Search Summary
Search space summary
Default search space size: 11
conv_1_filter (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 128, 'step': 32, 'sampling': 'linear'}
conv_1_kernel (Choice)
{'default': 3, 'conditions': [], 'values': [3, 5], 'ordered': True}
l2 (Float)
{'default': 1e-05, 'conditions': [], 'min_value': 1e-05, 'max_value': 0.1, 'step': None, 'sampling': 'log'}
conv_2_filter (Int)
{'default': None, 'conditions': [], 'min_value': 64, 'max_value': 128, 'step': 64, 'sampling': 'linear'}
conv_2_kernel (Choice)
{'default': 3, 'conditions': [], 'values': [3, 5], 'ordered': True}
dropout_1 (Float)
{'default': 0.1, 'conditions': [], 'min_value': 0.1, 'max_value': 0.4, 'step': 0.1, 'sampling': 'linear'}
conv_3_filter (Int)
{'default': None, 'conditions': [], 'min_value': 128, 'max_value': 512, 'step': 128, 'sampling': 'linear'}
conv_3_kernel (Choice)
{'default': 3, 'conditions': [], 'values': [3, 5], 'ordered': True}
dropout_2 (Float)
{'default': 0.2, 'conditions': [], 'min_value': 0.2, 'max_value': 0.3, 'step': 0.1, 'sampling': 'linear'}
dense_1_units (Int)
{'default': None, 'conditions': [], 'min_value': 64, 'max_value': 512, 'step': 64, 'sampling': 'linear'}
dropout_3 (Float)
{'default': 0.1, 'conditions': [], 'min_value': 0.1, 'max_value': 0.2, 'step': 0.1, 'sampling': 'linear'}

Best Model:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 180, 180, 96)   │         7,296 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 180, 180, 96)   │           384 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 90, 90, 96)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 90, 90, 128)    │       110,720 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 90, 90, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 45, 45, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 45, 45, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 45, 45, 128)    │       409,728 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 45, 45, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 22, 22, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 22, 22, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 61952)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 384)            │    23,789,952 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 384)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           385 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 24,319,489 (92.77 MB)
 Trainable params: 24,318,785 (92.77 MB)
 Non-trainable params: 704 (2.75 KB)



# Generators 
Batch size is set to 16.
Data augmentation for training images:
An ImageDataGenerator is created for training data with various augmentation techniques:
Preprocessing using ResNet50's preprocess_input function
Width and height shifts
Shear and zoom transformations
Horizontal flipping
Training data generator:
Uses flow_from_dataframe to create a generator from a pandas DataFrame
Specifies the directory containing images, target image size, and batch size
Sets 'binary' as the class mode, confirming this is a binary classification task
Validation data generator:
Another ImageDataGenerator is created for validation data
Only applies preprocessing, no augmentation
Early stopping callback:
Set up to monitor validation loss with a patience of 5 epochs
Will restore the best weights when stopping
Calculation of steps:
validation_steps and steps_per_epoch are calculated based on the number of samples and batch size
Outputs:

train_generator and validation_generator: These are not direct outputs but are set up to provide batches of preprocessed and potentially augmented images during training.
early_stopping: A callback object to be used during model training.
validation_steps and steps_per_epoch: Integer values representing the number of batches to use in each epoch for validation and training respectively.
This setup suggests a robust training process with:

Data augmentation to increase the effective size and variety of the training set
Separate handling of training and validation data
Use of early stopping to prevent overfitting
Efficient batch processing of data
The use of ResNet50's preprocessing function also hints that transfer learning or a ResNet50-based model might be used in the full pipeline.




# Output


## Test Metrics:
# Test Loss: 1.0851091146469116
This value represents the average loss on the test dataset. The loss function used here is binary crossentropy (since this is a binary classification task). The loss is higher than expected for a well-performing model, which could indicate overfitting, issues with hyperparameter tuning, or further optimization needed for the model.


# Test Accuracy: 0.7483999729156494 (about 74.8%) While the RESNet is 98% accurate
This indicates that the model correctly classified around 74.84% of the images in the test dataset. While this is a decent accuracy, there is likely room for improvement, especially given the simplicity of the binary classification task (dogs vs. cats). A higher accuracy, typically above 85-90%, is often expected for well-optimized models in such tasks.



Further hyperparameter tuning
Trying different model architectures
Collecting more training data
Applying more sophisticated data augmentation techniques
Using transfer learning from a pre-trained model

Overall, while the model shows definite learning and some predictive power, there's potential for improvement depending on the specific requirements of your project.

# Confusion Matrix
Matrix Structure:
The y-axis represents the true labels (actual class)
The x-axis represents the predicted labels (model's predictions)




## Interpretation:
The model's performance is only marginally better than random guessing (50% accuracy for a binary classification).
There's a significant room for improvement in the model's ability to distinguish between cats and dogs.
The nearly balanced nature of correct and incorrect predictions suggests that the model might not have learned very meaningful features to differentiate between cats and dogs.
