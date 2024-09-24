# CNN-Hyper_Tuning(War_in_Progrss)-
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

# Model Building
This block defines a function `build_model` that creates a convolutional neural network (CNN) architecture using Keras. The function is designed to work with Keras Tuner for hyperparameter optimization. Here's a breakdown of what's happening:

1. The function takes a `hp` (hyperparameters) argument from Keras Tuner.

2. It creates a Sequential model with the following structure:
   - Input layer: Expects images of size (img_width, img_height, 3)
   - Three Convolutional layers, each followed by BatchNormalization and MaxPooling2D
   - Dropout layers after the second and third convolutional blocks
   - A Flatten layer to transition from convolutional to dense layers
   - One Dense layer
   - A final Dense layer with sigmoid activation for binary classification

3. Each layer has hyperparameters that can be tuned:
   - Number of filters in Conv2D layers
   - Kernel sizes for Conv2D layers
   - L2 regularization strength
   - Dropout rates
   - Number of units in the Dense layer

4. The model is compiled with:
   - Adam optimizer (with tunable learning rate)
   - Binary crossentropy loss (suitable for binary classification)
   - Accuracy metric

5. After defining the function, it's called once with a new HyperParameters object.

Outputs:
- The function itself doesn't produce any direct output when defined.
- The last line `build_model(keras_tuner.HyperParameters())` creates and returns a model instance, but this instance isn't stored or used directly in this block.

This architecture suggests a binary image classification task (likely the dog/not dog classification hinted at in the previous block). The use of Keras Tuner indicates that this model will be optimized by searching over the defined hyperparameter space to find the best configuration.

The flexibility in the architecture (tunable hyperparameters) allows for finding an optimal model structure for the specific problem at hand.

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

# Model Accuracy and Model Loss
graph:

## Model Accuracy (Left Graph):

The blue line represents the training accuracy.
The orange line represents the validation accuracy.
Both accuracies start around 0.60-0.65 and generally increase over time.
There's significant fluctuation in both lines, especially in the validation accuracy.
The highest validation accuracy appears to be around 0.75-0.76.
Towards the end (after epoch 25), there's a noticeable drop in validation accuracy while training accuracy continues to rise, which might indicate overfitting.


## Model Loss (Right Graph):

Again, the blue line is for training loss and the orange for validation loss.
Both losses start high (around 1.5-1.6) and generally decrease over time.
The validation loss is more volatile than the training loss, especially in the early epochs.
After about epoch 10, the training and validation losses converge and follow similar patterns.
The final losses are around 1.1-1.2.



Key Observations:

The model is learning, as indicated by the general increase in accuracy and decrease in loss over time.
There's significant volatility in the validation metrics, suggesting the model might be sensitive to the specific examples in each validation batch.
The divergence between training and validation accuracy towards the end might indicate the onset of overfitting.
The model doesn't seem to plateau completely, suggesting that further training or adjustments might yield improvements.
The final validation accuracy is around 65-70%, which may or may not be satisfactory depending on the specific problem and requirements.


# Output
## Test Process:
The model was evaluated on 313 steps (batches) of test data.
It took approximately 19 seconds total, with an average of 54ms per step.

## Test Metrics:
# Test Loss: 1.041985273361206
This is the average loss (error) of the model on the test set.
Lower values indicate better performance.
The loss function used was likely binary cross-entropy, given the binary classification task.


# Test Accuracy: 0.7139999866485596 (about 71.4%)
This represents the proportion of correct predictions on the test set.
It means the model correctly classified about 71.4% of the test samples.

## Interpretation:
The test accuracy (71.4%) is slightly higher than the final validation accuracy seen in the training graphs (which appeared to be around 65-70%).
This suggests that the model generalizes reasonably well to unseen data.
A 71.4% accuracy for a binary classification task is above random guessing (50%), indicating that the model has learned useful patterns.
However, depending on the specific problem and requirements, this accuracy might or might not be considered good enough. For many real-world binary classification tasks, higher accuracy would typically be desired.


## Considerations:
The test loss (1.04) is slightly lower than the final validation loss seen in the training graphs (which was around 1.1-1.2). This is a good sign, indicating that the model isn't overfitting to the validation set.
The accuracy achieved suggests there's room for improvement. Depending on the application, you might consider:

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

## Classification Results:

True Positives (Cats correctly identified as cats): 1548
True Negatives (Dogs correctly identified as dogs): 1018
False Positives (Dogs incorrectly classified as cats): 952
False Negatives (Cats incorrectly classified as dogs): 1482


## Model Performance:

The model seems to have a bias towards classifying images as cats, as evidenced by the higher number of false positives for cats compared to false negatives.
The accuracy for identifying cats (1548 / (1548 + 1482) ≈ 51.1%) is slightly better than random guessing.
The accuracy for identifying dogs (1018 / (1018 + 952) ≈ 51.7%) is also slightly better than random guessing.


## Overall Performance:

Total correct predictions: 1548 + 1018 = 2566
Total predictions: 1548 + 1482 + 952 + 1018 = 5000
Overall accuracy: 2566 / 5000 = 51.32%


## Interpretation:
The model's performance is only marginally better than random guessing (50% accuracy for a binary classification).
There's a significant room for improvement in the model's ability to distinguish between cats and dogs.
The nearly balanced nature of correct and incorrect predictions suggests that the model might not have learned very meaningful features to differentiate between cats and dogs.
