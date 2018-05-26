#K nearest neighbor (k-NN) classifier that can recognize the 10 different classes in the CIFAR-10 dataset.

•	The accuracies for validation in 3 cross fold validation and its K:

-	k = 1, accuracy = 0.307808
-	k = 1, accuracy = 0.303904
-	k = 1, accuracy = 0.299700
-	k = 3, accuracy = 0.292793
-	k = 3, accuracy = 0.287387
-	k = 3, accuracy = 0.300300
-	k = 5, accuracy = 0.306306
-	k = 5, accuracy = 0.292492
-	k = 5, accuracy = 0.309910
-	k = 8, accuracy = 0.311111
-	k = 8, accuracy = 0.295495
-	k = 8, accuracy = 0.317718
-	k = 10, accuracy = 0.307207
-	k = 10, accuracy = 0.292492
-	k = 10, accuracy = 0.321622
-	k = 20, accuracy = 0.312312
-	k = 20, accuracy = 0.296697
-	k = 20, accuracy = 0.314414
-	k = 40, accuracy = 0.312613
-	k = 40, accuracy = 0.286186
-	k = 40, accuracy = 0.304505
-	k = 50, accuracy = 0.304505
-	k = 50, accuracy = 0.285586
-	k = 50, accuracy = 0.299099
-	k = 100, accuracy = 0.292192
-	k = 100, accuracy = 0.277477
-	k = 100, accuracy = 0.286186

It was found that K=8 is the best K to be used. It was chosen because it has the minimum error bar between all Ks and lowest standard deviation with high mean. 

•	Correct Classification Rate for each of the 10 Classes (CCRn) and Average Correct Classification Rate (ACCR) is reported below: 
-	ACCR of the testing is: 0.324300
-	CCRn of plane is:  0.577000
-	CCRn of car is:  0.188000
-	CCRn of bird is:  0.464000
-	CCRn of cat is:  0.145000
-	CCRn of deer is:  0.423000
-	CCRn of dog is:  0.183000
-	CCRn of frog is:  0.268000
-	CCRn of horse is:  0.209000
-	CCRn of ship is:  0.601000
-	CCRn of truck is:  0.185000

Note: The 10,000 pictures used in the training has an equal amount of each class, moreover, the testing data used to provide CCRn and ACCR were the 10,000 pictures provided by CIFAR-10 for testing.  

#Linear least square classifier (LLS) that can best recognize the 10 different classes in the CIFAR-10 dataset.

•	Correct Classification Rate for each of the 10 Classes (CCRn) and Average Correct Classification Rate (ACCR) is reported below: 

-	ACCR of the testing is: 0.363700
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

•	We can check for over-fitting in LLS by comparing the predictions for training and testing data. Over fitting means that the model works fine for the training data while it works with a significant low accuracy for the testing data. ACCR for testing was 0.363700 and ACCR for training data was 0.509440. In my view, I see the difference between the ACCR for both the training and testing sets is not a huge difference and reflects that there is no over fitting. 

Another point to look at to see if there is overfitting or not, is looking at the weights and see if they have a huge positive or negative value or no. In my case, there was no anomaly or huge values in the weights. 


 
 
