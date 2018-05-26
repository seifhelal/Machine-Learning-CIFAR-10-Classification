
I implemented two Neural Nets; one is my own implementation following Stanford’s cs231n Skelton, the other was implemented using Keras(built on top of tensorflow) API. 

**My Own Neural Network**

It consists of 4 main classes: 

1-	Neurons: this class has the implementation of the layers and their contents of forward and backward affine functions, leaky ReLU activation function and the batch-normalization. 

2-	FullyConnectedNet: This class is used to dynamically initialize the Neural Network and take the parameters regarding the activation of batch-normalization and regularization. It has a loss function that is used to implement the forward path of the neural network and the backward path and return the gradients and the loss. It has the random seed that is used to initialize the weights.

3-	GD_optimizers: this class has the implementations of the stochastic gradient descent, rmsprop optimizer and adam optimizer. 

4-	Trainer: this class is used to accept both training and validation data and labels so it can periodically check classification accuracy on both training and validation data to watch out for overfitting. It has a train function that is used to train the model and run the optimization procedure. 
CIFAR-10_NN file is the main implementation that has the values of the hyper parameters and the real implementation of the neural network. 

**Keras Neural Network**

	I used Keras library that is built on top of tensorflow to implement my design. I preprocessed the data and constructed my layers and calculated the total validation and training losses. The file’s name is Keras_NN.ipynb


**Data Pre-Processing:**

**My Own Neural Network**

I have made some preprocessing for the data to increase the accuracy of my neural network. 

-	First, I normalized the data using an exponential running mean and variance for the data. And I used the model suggested by the link below and it yielded a very good result: 
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html 

-	I wanted to increase the size of the available data. I flipped the training images horizontally and added them to the original training data and shuffled them. This process produced 100,000 training images. This method increased the testing accuracy at the end by ~4%. 

**Neural Network Using Keras Library**

I did the same things that I did with my own implementation. Moreover, I randomly rotated the images, randomly shifted them horizontally and vertically. 


**CCRNs**

**My own Neural Network**

-	CCRn of plane is:  0.629000
-	CCRn of car is:  0.692000
-	CCRn of bird is:  0.439000
-	CCRn of cat is:  0.416000
-	CCRn of deer is:  0.555000
-	CCRn of dog is:  0.475000
-	CCRn of frog is:  0.652000
-	CCRn of horse is:  0.636000
-	CCRn of ship is:  0.726000
-	CCRn of truck is:  0.659000

**Keras Neural Network**

-	CCRn of plane is:  0.737000
-	CCRn of car is:  0.808000
-	CCRn of bird is:  0.545000
-	CCRn of cat is:  0.489000
-	CCRn of deer is:  0.569000
-	CCRn of dog is:  0.595000
-	CCRn of frog is:  0.790000
-	CCRn of horse is:  0.770000
-	CCRn of ship is:  0.783000
-	CCRn of truck is:  0.755000

**Comparing CCRN of my own Neural Net with LLS classifier:**

**My Neural Network**

-	CCRn of plane is:  0.629000
-	CCRn of car is:  0.692000
-	CCRn of bird is:  0.439000
-	CCRn of cat is:  0.416000
-	CCRn of deer is:  0.555000
-	CCRn of dog is:  0.475000
-	CCRn of frog is:  0.652000
-	CCRn of horse is:  0.636000
-	CCRn of ship is:  0.726000
-	CCRn of truck is:  0.659000

**LLS classifier**

-	CCRn of plane is:  0.469000
-	CCRn of car is:  0.445000
-	CCRn of bird is:  0.207000
-	CCRn of cat is:  0.177000
-	CCRn of deer is:  0.243000
-	CCRn of dog is:  0.285000
-	CCRn of frog is:  0.449000
-	CCRn of horse is:  0.426000
-	CCRn of ship is:  0.508000
-	CCRn of truck is:  0.428000

As we see there is a huge improvement in using Neural Networks compared to the LLS normal classifier. 

**ACCR**

My own Neural Network

-	Validation set accuracy:  0.6125
-	Test set accuracy ACCR:  0.5879

Keras Neural Network

-	Test set accuracy ACCR:  0.6850

Average Accuracy of both NNs combined

-	Avg= 63.65 accuracy. 


