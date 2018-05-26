import numpy as np
class softmax(object):
    def __init__(self):
        pass
    
    def name (self):
        print("softmax")
    
    def forward(self, X, Y):
        scores = X
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs /= np.sum(probs, axis = 1, keepdims = True)
        data_loss = -np.sum(np.log(probs[np.arange(scores.shape[0]), Y]))/scores.shape[0]

        #get derivative of scores
        dx_scores = probs.copy()
        dx_scores[np.arange(scores.shape[0]), Y] -= 1
        dx_scores /= scores.shape[0] 
        
        return data_loss, dx_scores