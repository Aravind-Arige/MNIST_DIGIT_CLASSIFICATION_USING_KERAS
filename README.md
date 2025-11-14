# MNIST_DIGIT_CLASSIFICATION_USING_KERAS
1. problem definition 
We are going to build a simple digit recognition and classification algorithm, using keras and simple neural network.
2. Data
The data used for this project is the MNIST (Modified National Institute of Standards and Technology) dataset first released in 1999. It contains 60,000 28x28 grayscale training images and 10,000 testing images of handwritten digits. The dataset became a classical learning tool for computer vision and classification algorithms.

3. Tools
For this project, we are going to use Keras, a TensorFlow library that provides a Python interface for artificial neutal networks. The basic data structures in Keras are layers and models. After building our model, we will use scikit-learn library to evaluate its peroformance.
4. Visualization
First, let‘s visualize the digit samples from our dataset.
To disaply the actual image in a human-friendly way, we have several methods. One of them is to use the matplotlib function imshow().
5. Classes to categories
Next step is to convert our 10 classes to binary categories. For example, if a model classifies a digit on an image as a number 3, it will correspond with 1, while other categories will be 0
6. Data pre-processing
The next step is data preprocessing. For the machine learning algorithm to understand our data better, we need to do two things: normalize the data, and flatten it into a vector.
7. Reshape data to a vector (flatten)
The primary reason behind why we flatten images (multi-dimensional arrays) into 1-D array is because they take less memory, and as a result reduce the amount of time needed to train the model. 
![alt text](<Screenshot 2025-11-14 063844.png>)
8. Fully Connected Neural Network
Now it‘s time to build our first neural network! For this project, we will build a simple fully connected neural network which consists of a serious of layers in which every neuron in one layer is connected to every neuron in the other layer. The main advantage of fully connected neural networks is that they can be used for a broad range of pattern recognition tasks. Compared to more "specialized" artificial networks, such as CNN or RNN, however, they tend to perform worse for specific problems.
9. Model
When we are building a model in Keras, we have two optins: to build a Sequential model, of a Functional model. Sequential models are the most common type, and we can use it to create the models by stacking layers in a particular sequence that we choose. Functional models, on the other hand, are used to create models with extra flexibility, such as connecting layers to more then one previous and one next layer. In other words, it allows us to create more complex and sophisticated networks.

For this project, we will create a Sequential model.
10. Layers¶
There are 3 types of layers in a neural network: an input layer, a hidden layer(s), and an output layer. In our case, we will create a simple neural network with 1 input layer, 1 output layer, and 2 hidden layers: a dense and a dropout layer.

Dense layer
Dense layer is the type of a layer in which all neurons are connected with all neurons of the previous layer. As the first parameter, Dense() layer accepts the number of neurons we want the layer to have.
Dropout Layer¶
Dropout layer is used as a method of regularization. Dropout layer randomly drops nodes (neurons), and forces the neurons in previous layers to take more or less responsibilites, and thus eliminates overfitting.
11. Activation functions¶
Each layer in a neural network has an activation function. It tells each neuron how to transform the data it has, before sending it to the next layer. The most common activation function for hidden layers is ReLU, sigmoid, and tanh.

The most common activation functions for the output layer are linear, logistic (sigmoid), and softmax. The choice of the output function depends on the type of the problem we are trying to solve. For regression problems, linear activation function is common. For classification problems, we can use a sigmoid for binary or multilabel, and softmax for multiclass classification.

For our problem today, we will use ReLU as our hidden layer activation function, because we are dealing with images, and a softmax activation for an output layer , because it is a multiclass classification problem.
Loss and Optimizer functions¶
Apart from the activation functions, we also need to decide on loss and optimizer functions, when building a neural network.

Loss
Loss function calculates how wrong the prediction is. For classification problems, there are several options of loss functions:

Binary Classification:

Binary Cross-Entropy
default loss function for binary classification

Hinge Loss
primary developed for Support Vector Machines (SVM)

Multi-Class Classification:
Multi-Class Cross-Entropy Loss
default loss function for multiclass classification (can belong to more than one class)

Categorical cross-entropy
each output can belong to only one class

Sparse Multiclass Cross-Entropy Loss
no need for one-hot encoding, good for problems with large number of labels

For our project, we will use the categorical cross-entropy function, because each image can only signify one number.
12. Optimizer
Optimizer helps us correctly modify the weights of the neurons and the learning rate to reduce the loss and improve accuracy. Optimizers can be broadly divided into two categories: Gradient descent and Adaptive optimizers. The principal difference between the two groups is that for gradient descent optmizers, we need to tune the learning rate manually, while adaptive optimizers do it automatically.

For this project, we will use an adaptive Adam optimzier, which is currently considered to be the best optimizer overall.
13. Batch size
Batch size refers to the number of samples used in one iteration. Typically, we have 3 options to choose from.

Batch mode Using all the data in the dataset. Has long iteration times.
Mini-batch mode Using parts of the dataset. Faster learning than in the batch mode. Typical mini-batch sizes are 32, 64, 128, 256, and 512. The advised mini-batch size is 32.
Stochastic mode Using one sample per iteration.
We are going to use a mini-batch mode, and because our problem is very simple, and we will use 512 batch size.

14. Epochs
When all data we have has updated the internal parameters, that equals to one epoch. There are many strategies to chose the best number of epochs. Some people swear by 11 epochs. Other people give other numbers. But the best way to decide if you have too many or too little epochs is to look at the model perofrmance. If the model keeps improving, it is advisable to try a higher numebr of epochs. If the model stopped improving way before the final epoch, it is advisable to try a lower number of epochs.
15. Model evaluation
Now that we have compiled and fitted our model, it is time to evaluate its performance. Since it is a classification problem, we can use the typical classification evaluation tools, such as accuracy, and loss. Similarly to machine learning problems, we evaluate the model on test data - the data that our algorithm hasn‘t seen yet. If we want to compare test data evaluation with train data performance, we can also see both.
16. Make predictions
Now that we trained and evaluated out model, let‘s try making predictions! Softmax activation function that we used for our output layer returns the predictions as a vector of probabilities. For example, "this image is 86% likely to be number 7, and this image has a 14% probability of being number 2."
17. Confusion matrix
Another effective way to see the accuracy of a classification model is by making a confusion matrix.
![alt text](<Screenshot 2025-11-14 063942.png>)
18. Conclusion
As we can see, our model performed pretty well, and the absolute majority of the labels were correct. In this notebook, we learned how to create a simple neural network, about the types of the layers, activation and loss functions, and optimizers. We also learned how to pre-process images through normalization, and how to evaluate our model after fitting it with accuracy and confusion matrix.