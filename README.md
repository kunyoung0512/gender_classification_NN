# Introduction
Gender classification plays an important role in modern society, we see it in the way of how they are utilized for security and forensic purposes in the way of facial recognition.
Additionally, given the rise of social media and social media platforms, social media marketing which is an extension of personalized advertising utilizing demographic and gender info is also on the rise.
As the name implies, gender classification is a process whereby a system receives input(s) of a face of a person from a given image and tries to determine the gender of the given person.
Current methods do exist already for this classification task however, given its widespread usages and accuracy of the classification being of utmost importance, more could be done to increase the prediction accuracy of a model.

# Project Scope
The main goal of this project thus aims to classify the gender of faces in an image with the highest possible accuracy by means of optimizing model selection and the tuning different hyperparameters.

# Dataset

- Adience Dataset
For testing the accuracy of our model, there was an option between 2 datasets from the adience data, one being the faces dataset where it contained face images that were cropped and the second being the aligned dataset where it contained face images that were cropped and aligned.
For the purpose of accuracy, we chose the aligned dataset as having faces which were already preprocessed to be aligned will substantially boost the performance of our training and prediction.
The 5 different fold_frontal_data.txt (from 0-4) provides us the information we need to do our data preprocessing.
The aligned dataset initially containing 26,580 photos and 2284 subjects was reduced to 12,194 after data preprocessing removing incorrectly labelled records and removing records which did not have a class label f (female) or m (male).

- CelebA Dataset
CelebA Dataset is a large-scale face attributes dataset with the following features:
- 202599 face images
- 40 binary attributes annotations for images

# Data Pre-processing
- Adience Dataset
To load the aligned dataset into the model for training, we first perform the data preprocessing as
follows:
1) Load the 5 different fold_frontal_data.txt files
2) Loop through each fold and concatenate each ‘user_id’, ‘face_id’, ‘original_image’ into a link which identifies the absolute path of where each image is located and change gender label m and f into binary form (0 for f, 1 for m), E.g.

![image](https://github.com/user-attachments/assets/a85b4765-8164-4942-9f49-20de79bc654c)

- CelebA Dataset

![image](https://github.com/user-attachments/assets/7e676779-f398-41c5-a7be-055fa31bda01)

# Libraries
- Wandb
- Tensorflow
- Pandas
- Numpy

# Review of Existing Techniques
To select the best model to use for the gender classification task, we perform some comparison with several existing model architectures. All training done was done with a 80% train and 20% test data split. 

- Hassner CNN
Similar to the existing research paper, we first evaluate the Hassner CNN model architecture.

Network Architecture generalized:
3 convolutional layers with differing filter, pooling and stride size with rectified linear (ReLu) hidden activation function.
Followed by 2 custom fully connected layers with dropout 0.5 and ReLu hidden activation function.
A last fully connected layer with sigmoid activation function mapping to the final classes for gender.
After training the model, we evaluated its accuracy with model.evaluate() and obtain a validation accuracy of 0.8798688054084778 ≈ 0.88 / 88%.

- InceptionV3 Model
The InceptionV3 model will first perform pre-processing of the image input before generating the train and test split. The top inception_v3 layer then accepts the processed inputs of shape (299, 299, 3).

Network Architecture generalized:
InceptionV3 as the first layer using pretrained weights from ‘imagenet’ and average pooling.
Followed by 3 fully connected layers, each with a ReLu hidden activation and a batch normalization layer.
A last fully connected layer with sigmoid activation function mapping to the final classes for gender.
After training the model, we evaluated its accuracy with model.evaluate() and obtain a validation accuracy of 0.876588761806488 ≈ 0.88 / 88%.

- Resnet50V2 Model
The Resnet50v2 model will first perform pre-processing of the image input before generating the train and test split. The top resnet50V2 layer then accepts the processed inputs of shape (224. 224. 3).

Network Architecture generalized:
Resnet50V2 as the first layer using pretrained weights from ‘imagenet’ and average pooling.
Followed by 3 fully connected layers, each with a ReLu hidden activation and a batch normalization layer.
A last fully connected layer with sigmoid activation function mapping to the final classes for gender.
After training the model, we evaluated its accuracy with model.evaluate() and obtain a validation accuracy of 0.891758918762207 ≈ 0.89 / 89%.

# Comparison of Model Architectures
The Resnet50V2 model seemed to have the highest accuracy/val_accuracy and lowest loss/val_loss values out of the 3 models tested.

![image](https://github.com/user-attachments/assets/ddf3fa2f-dad3-40e3-86db-fb011795ba5e)

We can see from the table above that in all categories, Resnet50V2 is the clear choice of model to use with the highest accuracy/val_accuracy values and lowest loss/val_loss values, this agrees with our previous observation as well.
The Resnet50V2 model will then be used in our later experiments.

![image](https://github.com/user-attachments/assets/d8475d25-95d1-41ab-a047-370316ed2464)

# Hyperparameter Tuning
- Batch Size Tuning
For batch size tuning, the 6 batch size values of [1, 4, 8, 16, 32, 64] were tested and evaluated.
We observed from testing that batch sizes of 16, 32 and 64 have roughly equal validation accuracy however by the total average of val_accuracy obtained, batch size of 32 was determined as the best batch size to use.

![image](https://github.com/user-attachments/assets/4ee41d58-5816-4da0-9ffc-576f8d8faa0a)

- Learning Rate Tuning
Utilizing the best batch size of 32 for learning rate tuning, the 4 learning rate values of [0.1, 0.001, 0.0003, 0.00001] were tested and evaluated.
We observed from testing that the learning rate of 0.001 / 1e3 produces the best validation accuracy by a decent margin, we thus determined 0.001 as the best learning rate to use.

![image](https://github.com/user-attachments/assets/f4890f6d-79ec-44ee-8212-ec45b90e6989)

- Number of Neurons Tuning
Utilizing the previously tuned batch sizes and learning rates for the tuning of the number of neurons at each fully connected layer, the 4 sets of values described by table 1 were tested and evaluated. 
We observed from testing that the 3rd set of number of neurons produces the best validation accuracy by a decent margin, we thus determined 1024, 512 and 256 as the best number of neurons to be used at each fully connected layer respectively.

![image](https://github.com/user-attachments/assets/0bc2f464-2b75-47bb-a485-dc5f72c094ba)

- Drop Out Rate Tuning
For dropout rate tuning, the 6 dropout rate values of [0, 0.1, 0.2, 0.3, 0.4, 0.5] were tested and evaluated for each fully connected layer.
We observed from testing that the dropout rate of 0.5 for each fully connected layer produces the best validation accuracy by a decent margin, we determined 0.5 to be the best learning rate to use at each fc layer.

![image](https://github.com/user-attachments/assets/13fd8b09-3a47-46fe-b186-d7f1d03082b0)

# Experiments and Results
With our tuned parameters and selected model, we performed the gender classification task once more.
The comparison will be between 2 models with the same tuned hyperparameters, however, one will utilize the pretrained weights learned from the CelebA dataset and one without.
The experiments will run for the same 20 epochs as per the previous models. The tuned parameters and comparison results are shown below:

![image](https://github.com/user-attachments/assets/d3f17948-fb25-4e23-ab44-32f835b9c43d)

![image](https://github.com/user-attachments/assets/86737e0d-8a5f-4d5e-950e-aef9533affea)

During the comparisons, the pre-training with CelebA produces higher accuracy/val_accuracy compared to that of the initial untuned model and the tuned model without pretraining.
However, we observe that the training accuracy of the tuned model is significantly lower than that of the initial model, this could be explained by the use of dropouts for the tuned models which indicates a higher variance at some of the fully connected layers as compared to the initial evaluation model without dropout. 
Overall, with tuning and pretraining, we are able to obtain an approximate 2% increase in performance.

![image](https://github.com/user-attachments/assets/5040142b-af99-4f89-bacc-5aabf6bbd332)

# Demo Experiment
In this demo experiment, we put our best model to the test against 10 random faces from the internet and see whether it can predict their gender correctly. The result from this experiment is as follows:

![image](https://github.com/user-attachments/assets/0d22e11b-6c28-4fb2-b973-8f6f5722eb23)

# Future Considerations
To increase the accuracy further and to solve overfitting issues, some considerations for future work could be to utilize and tune other parameters such as:
1) L2 Regularization - Penalty term discouraging weights from attaining large values
2) Random Sampling (K-Fold Cross Validation) – Testing Technique which enables targeted tuning of parameters based on certain evaluations on unseen data
3) Learning Rate Scheduler – Adjust weights during training to improve adaptability of model

# Conclusion
In conclusion, we performed automatic gender classification of face images utilizing convolutional neural networks (CNN) on a variety of models on the Adience dataset.
We experimented with the different models (Hassner, Inception, Resnet) and different hyperparameters to eventually obtain the best model and tuned parameters for the purpose of achieving the highest accuracy of gender classification.
We also experimented using pre-trained weights from the CelebA dataset and achieved the highest val_accuracy observed from all experiments conducted of 0.918 ≈ 0.92 / 92%. 
