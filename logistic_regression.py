import numpy as np

class LogisticRegression():

    def __init__(self, lr=1e-4, iterations=1000, l2_penalty=0):
        
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.lr = lr
        self.parameters = []

    def _init_params(self, n_features):

        self.parameters = np.random.normal(size=(n_features + 1), scale=0.1) #initalize n weights + 1 bias randomly
        self.parameters = self.parameters.reshape(-1, 1) # make it matrix

    @staticmethod
    def _get_design_matrix(X):
        n, d = X.shape
        X_d = np.empty((n, d+1), dtype=X.dtype)
        X_d[:, 0] = 1 #column of ones
        X_d[:, 1:] = X
        return X_d

    @staticmethod
    def _sigmoid(logits):
        return 1 / (1 + np.exp(-logits))

    def _compute_logits(self, X):        
        return (X @ self.parameters)  #(m, n) x (n, 1) =  (m, 1) #logits
    
    @staticmethod
    def _binary_cross_entropy_loss(Y_true, Y_pred):
        
        loss = - (Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        return loss.mean()
    
    def _compute_gradient(self, X_d, Y_true, Y_pred):
        """
        X_d:    (m × (d+1)) design matrix, X_d[:,0] == 1
        Y_true: (m × 1) true labels
        Y_pred: (m × 1) predicted probabilities
        """
        
        m = Y_true.shape[0]
        gradient = X_d.T @ (Y_pred - Y_true)  / m  #  (n, m)  @ (m, 1) = (n, 1) #gradient

        #l2 regularization
        gradient[1:] += (self.parameters[1:] * self.l2_penalty) / m #add sum of weights (except bias term)

        return gradient
    
    def fit(self, X, Y):

        if not (isinstance(X, np.ndarray) and isinstance(Y, (np.ndarray, np.array, list))):
            raise ValueError("Invalid type for inputs. Expected numpy.ndarray")
        if len(Y.shape) != 2:
            raise ValueError("Y must be matrix. Hint: do Y.reshape(-1, 1)")
        
        self.n_features = X.shape[1]
        self._init_params(self.n_features)

        #create design matrix
        X_d  = self._get_design_matrix(X)

        #Training loop
        for i in range(self.iterations):
            
            logits = self._compute_logits(X_d)
            Y_pred = self._sigmoid(logits)

            #compute loss
            loss = self._binary_cross_entropy_loss(Y, Y_pred)
            print(f"ITERATION {i + 1} | BCE LOSS = {float(loss.round(6))}")

            #compute gradient
            gradient = self._compute_gradient(X_d, Y, Y_pred)

            #update parameters
            self.parameters -= self.lr * gradient

    def predict_proba(self, X):

        X_d = self._get_design_matrix(X)
        logits = self._compute_logits(X_d)
        probs = self._sigmoid(logits)

        return probs



    

         
        


        


    

