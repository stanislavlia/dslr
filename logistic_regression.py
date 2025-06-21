import numpy as np
import json

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
    def _binary_cross_entropy_loss(Y_true, Y_pred, eps=1e-15):
        # clip to avoid log(0)
        p = np.clip(Y_pred, eps, 1 - eps)

        loss = - np.mean( Y_true * np.log(p) + (1 - Y_true) * np.log(1 - p) )
        return loss

    
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

            #compute gradient
            gradient = self._compute_gradient(X_d, Y, Y_pred)

            #update parameters
            self.parameters -= self.lr * gradient

    def predict_proba(self, X):

        X_d = self._get_design_matrix(X)
        logits = self._compute_logits(X_d)
        probs = self._sigmoid(logits)

        return probs



class OVALogisticRegression():
    """One vs All Logistic Regression for Muticlass Classification"""

    def __init__(self, lr=1e-4, iterations=1000, l2_penalty=0):
        
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.lr = lr

        self.binary_logregs = []
    
    def _init_params(self, n_features, num_classes):
        
        self.binary_logregs = [LogisticRegression(lr=self.lr,
                                                  iterations=self.iterations,
                                                  l2_penalty=self.l2_penalty) for _ in range(num_classes)]
        #init params
        for logreg in self.binary_logregs:
            logreg._init_params(n_features)
    
    def dump_parameters(self, filepath):
        """
        Serialize the OVA model parameters to a JSON file.
        """
        data = {
            'lr':          self.lr,
            'iterations':  self.iterations,
            'l2_penalty':  self.l2_penalty,
            'n_features':  self.n_features,
            'n_classes':   self.n_classes,
            'parameters': [model.parameters.flatten().tolist()
                           for model in self.binary_logregs]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_parameters(self, filepath):
        """
        Load OVA model parameters from a JSON file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        #restore params
        self.lr         = data.get('lr',    self.lr)
        self.iterations = data.get('iterations', self.iterations)
        self.l2_penalty = data.get('l2_penalty', self.l2_penalty)

        self.n_features = data['n_features']
        self.n_classes  = data['n_classes']
        # re-init sub-models
        self._init_params(self.n_features, self.n_classes)

        # assign saved parameters back to each sub-model
        for lr_model, params_list in zip(self.binary_logregs, data['parameters']):
            arr = np.array(params_list).reshape(self.n_features + 1, 1)
            lr_model.parameters = arr
    
    @staticmethod
    def _softmax(logits):
        """
        Numerically stable softmax:
          softmax_i = exp(x_i − max_row) / sum_j exp(x_j − max_row)
        """
        # subtract row-wise max to avoid huge exponents
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def _onehot_labels(Y):
        "OneHot Encoder for Y"

        n = Y.shape[0]
        K = Y.max() + 1
        one_hot = np.zeros((n, K), dtype=int)
        idx = Y.ravel()
        one_hot[np.arange(n), idx] = 1 #set 1 in correct position correspodning to label
        return one_hot
    
    def predict_proba(self, X):

        X_d = LogisticRegression._get_design_matrix(X)

        logits = []
        for j in range(self.n_classes):
            #compute logits for jth class
            jth_class_logits = (self.binary_logregs[j]._compute_logits(X_d))
            logits.append(jth_class_logits)
        #Transpose
        logits = np.hstack(logits)  #(m, n_classes)

        #get probs
        probs = self._softmax(logits)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        predicts = np.argmax(probs, axis=1).reshape(-1, 1)
        return predicts

    def fit(self, X, Y):

        if not (isinstance(X, np.ndarray) and isinstance(Y, (np.ndarray, np.array, list))):
            raise ValueError("Invalid type for inputs. Expected numpy.ndarray")
        if len(Y.shape) != 2:
            raise ValueError("Y must be matrix. Hint: do Y.reshape(-1, 1)")

        self.n_classes = len(np.unique(Y))
        self.n_features = X.shape[1]
        self._init_params(self.n_features, self.n_classes)

        Y_one_hot = self._onehot_labels(Y)
        
        #train binary logreg for each class separetely
        for j, logreg in enumerate(self.binary_logregs):

            y_class_labels = Y_one_hot[:, j].reshape(-1, 1) #take jth column from One-hot matrix
            logreg.fit(X, y_class_labels)
