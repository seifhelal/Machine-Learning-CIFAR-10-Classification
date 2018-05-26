
I have implemented 2 designs for CNN architecture one is my own implementation and the other is a keras model that is built to follow the architecture of Residual Neural Network which is known to achieve the highest accuracy.

1- Data preprocessing:

- in my own implementation, I used some data augmentation to increase the variety of the dataset. First I rotated the images with a margin in addition to shifting the images horizontally and vertically and flipping it on its vertical axis.

- Keras implementation, I used that data augmentation library to perform different image generating techniques to increase the data and hence increase the accuracy.

2- Architecture:

- My own implementation neural net architecture was built gradually to know the best number of layers as I started with one neuron and increased it until I reached my final architecture which is 4 convolution neural nets followed by 4 hidden connected nets. The hyber parameters where tested randomly as I did in my last assignment on a small data set and the highest
accuracy parameters where used.

- Keras implementation: It was built on the winning architecture of image recognition in 2015 and the parameters where obtained from some previous models that got high accuracy in this field. 

3- ACCR for:
- Keras: ACCR: 0.9273
- Own implementation: 0.8286

Average ACCR between two mdels is: 87.8% accuracy.

5- CCRn for:

- Keras:

CCRn of airplane is: 0.940000 

CCRn of automobile is: 0.978000 

CCRn of bird is: 0.906000

CCRn of cat is: 0.839000

CCRn of deer is: 0.940000 

CCRn of dog is: 0.868000

CCRn of frog is: 0.961000 

CCRn of horse is: 0.934000 

CCRn of ship is: 0.958000 

CCRn of truck is: 0.949000


- Own Implmentation:

- CCRn of airplane is: 0.873000

- CCRn of automobile is: 0.883000

- CCRn of bird is: 0.752000

- CCRn of cat is: 0.634000

- CCRn of deer is: 0.776000

- CCRn of dog is: 0.772000

- CCRn of frog is: 0.940000

- CCRn of horse is: 0.852000

- CCRn of ship is: 0.848000

- CCRn of truck is: 0.956000

