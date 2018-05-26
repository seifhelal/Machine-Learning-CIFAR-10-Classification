import numpy as np
class NearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        pass

    def train(self, Xes, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = Xes
        self.ytr = y
    def predict(self, X, k ,  l='L1'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        Ys_k=[]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            #sort the distances of the required points
            sorted_distances=np.argsort(distances)
            #take the nearest K neighbors to the required point  
            sorted_distances=sorted_distances[:k].tolist()
            #reterive the labels of the sorted distances
            Ys_k=self.ytr[sorted_distances]
            #calculate the number of occurrences of each label 
            occurrences= np.bincount(Ys_k)
            #assign te highest repeated label to the testing point
            Ypred[i]=np.argmax(occurrences)
            
        return Ypred
