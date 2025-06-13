import numpy as np


class LogisticRegression():

    def __init__(self, iterations=1000, l2_penalty=0, random_state=1):
        
        self.iterations = 1000
        self.l2_penalty = 0
        self.parameters = []

    def _init_params(self, n_features):

        self.parameters = np.random.normal(size=(n_features + 1)) #initalize n weights + 1 bias randomly
        self.parameters = self.parameters.reshape(-1, 1) # make it matrix

    @staticmethod
    def _sigmoid(logits):
        return 1 / (1 + np.exp(-logits))

    def _compute_logits(self, X):
        #X is design matrix (m, n_features + 1)
            
        return (X @ self.parameters)  #(m, n) x (n, 1) =  (m, 1) #logits
    
    
        
    

         
        


        


    

