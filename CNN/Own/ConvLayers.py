import pyximport
pyximport.install()
from ConvNeurons import *
from FCHiddenLayer import *
from Softmax import *
import sys
import numpy as np

from augment import *



def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

class NETWORK(object):
    def __init__ (self):
        self.layers= []

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_augmentation(self,rotation_range=0,height_shift_range=0, width_shift_range=0, img_row_axis=1, img_col_axis=2,
 img_channel_axis=0,horizontal_flip=False,vertical_flip=False):
        
        self.rotation_range = float(rotation_range)
        self.height_shift_range = float(height_shift_range)
        self.width_shift_range= float(width_shift_range)
        self.img_row_axis= int(img_row_axis)
        self.img_col_axis= int(img_col_axis)
        self.img_channel_axis= int(img_channel_axis)
        self.horizontal_flip= bool(horizontal_flip)
        self.vertical_flip= bool(vertical_flip)
        self.augment = True
    
    def save_best(self):
        for i in range(len(self.layers)):
            if self.layers[i].name == "hidden_layer" or self.layers[i].name == "convolution":
                self.layers[i].save_params()
    
    def load_best(self):
        for i in range(len(self.layers)):
            if self.layers[i].name == "hidden_layer" or self.layers[i].name == "convolution":
                self.layers[i].load_best_params()
    
    
    def loss_softmax(self,X_train, batch_size= 250,y_train=None):
        for i in range(len(self.layers)-1):
            #self.layers[i].who()
            if i == 0:
                out = self.layers[i].forward(X_train)
            else:
                out = self.layers[i].forward(out)
                
        scores = out
        #print("scores shape", scores.shape)
        if y_train is None:
            return scores
        #.layers[len(self.layers)-1].who()
        loss, dx = self.layers[len(self.layers)-1].forward(scores, y_train)
        
        for i in reversed(range(len(self.layers)-1)):
            #self.layers[i].who()
            dx = self.layers[i].backward(dx)    
        
        return loss
    
    def test(self, X, batch_size=100):
        N = X.shape[0]
        num_layers = len(self.layers)
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            output = None
            for j in range(num_layers - 1):
                if j == 0:
                    output = self.layers[j].forward(X[start:end])
                else:
                    output = self.layers[j].forward(output)
        
            y_pred.append(np.argmax(output, axis=1))
        y_pred = np.hstack(y_pred)
        return y_pred
    
    def train(self, X_train, y_train, X_val, y_val, num_epochs= 10, learning_rate = 1e-4, 
                  learning_rate_decay = 0.95, lambda_reg = 1e-5, iterations = 100, 
                  batch_size = 200, verbose = False, print_every=10):
            self.all_loss = []
            self.all_train_acc = []
            self.all_val_acc = []
            self.best_accuracy = 0.0
            self.lr = learning_rate
            epoch_it = int(X_train.shape[0] / batch_size)
            print("epoch it is", epoch_it)
            iterations = num_epochs * epoch_it
            epochs = 0
           
            for t in range(0,iterations):
                learning_rate= self.lr
                idxs = np.random.choice(X_train.shape[0], batch_size)
                X_batch = X_train[idxs]
                y_batch = y_train[idxs]
                
                
                if self.augment:
                    X_batch = augment_batch(X_batch,
                                      rotation_range=self.rotation_range,
                                      height_shift_range=self.rotation_range,
                                      width_shift_range=self.width_shift_range,
                                      img_row_axis=self.img_row_axis,
                                      img_col_axis=self.img_col_axis,
                                      img_channel_axis=self.img_channel_axis,
                                      horizontal_flip=self.horizontal_flip,
                                      vertical_flip=self.vertical_flip)

                loss = self.loss_softmax(X_batch, y_train=y_batch, batch_size = batch_size)
                self.all_loss.append(loss)

                if (t + 1) % epoch_it == 0:
                    epochs += 1
                    #print("iam in epocn no.", epochs)
                    learning_rate *= learning_rate_decay


                if (t == 0) or ((t + 1) % epoch_it == 0) or (t == iterations + 1):

                    print ('inside loop: iteration', t+1, '/ ', iterations, ': loss', loss)
                    #for training
                    y_pred = []

                    num_samples=1000
                    N = X_train.shape[0]
                    if num_samples is not None and N > num_samples:
                        mask = np.random.choice(N, num_samples)
                        N = num_samples
                        X = X_train[mask]
                        y = y_train[mask]

                    num_batches_train = int(N / batch_size)
                    if N % batch_size != 0:
                        num_batches_train += 1
                    start =0 
                    end = 0
                    for i in range(0, num_batches_train):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        y_pred.append(self.predict(X[start:end]))
                    y_pred = np.hstack(y_pred)
                    train_acc = np.mean(y_pred == y)
                    self.all_train_acc.append(train_acc)


                    #for validation 
                    y_pred = []
                    num_batches_val = int(X_val.shape[0] / batch_size)
                    if X_val.shape[0] % batch_size != 0:
                        num_batches_val += 1
                    for i in range(0, num_batches_val):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        y_pred.append(self.predict(X_val[start:end]))
                    y_pred = np.hstack(y_pred)
                    val_acc = np.mean(y_pred == y_val)
                    self.all_val_acc.append(val_acc)

                    if verbose:
                        print ('Finished epoch ', epochs,'/', num_epochs, ': train: ', train_acc, ', val: ',val_acc)

                    if val_acc > self.best_accuracy:
                        self.best_accuracy = val_acc
                        self.save_best()

            self.load_best()
            print("Finished")   

    def predict(self, X_test):
        scores = self.loss_softmax(X_test)
        y_pred = np.argmax(scores, axis=1)
        return y_pred