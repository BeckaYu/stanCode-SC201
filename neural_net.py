from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object): 
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    D, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function on the weight matrices. The
    network uses a ReLU nonlinearity after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def cost(self, X, y=None):
        """
        Compute the cost and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample of dimension
          D, and there are N samples.
        - y: Vector of training labels, of length N. y[i] is the label for X[i], and
          each y[i] is an integer in the range 0 <= y[i] <= C - 1. This parameter is 
          optional; if it is not passed then we only return scores, and if it is 
          passed then we instead return the cost and gradients.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - cost: cost for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the cost function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1'] # W1.shape=(D*H), b1.shape(H,)
        W2, b2 = self.params['W2'], self.params['b2'] # W2.shape=(H*C), b1.shape(C,)
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the first step of the forward pass by computing the class   #
        # scores for the input. Store the result in the scores variable, which      #
        # should be an array of shape (N, C). Our solution uses 3 lines of code.    #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        K = np.dot(X,W1) + b1.reshape((1, b1.shape[0])) # K.shape=(N,H), X.shape=(N,D), b1.shape=(1,H)
        A = np.maximum(0, K) # A.shape=(N*H)
        scores = np.dot(A, W2) + b2.reshape((1, b2.shape[0])) # scores.shape=(N,C), b2.shape=(1,c)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the cost
        cost = None
        #############################################################################
        # TODO: Finish the forward pass by computing the cost. Store the result in  #
        # the variable cost, which should be a scalar. Use the softmax classifer    #
        # cost. Our solution uses 5 lines of code.                                  #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores -= np.amax(scores, axis=1, keepdims=True)
        sigma = np.exp(scores) / np.sum((np.exp(scores)), axis=1, keepdims=True) # sigma.shape=(N,D)
        sigma_true = sigma[range(N), y] # sigma_true.shape=(N,)
        cost = (1/N) * np.sum(-(np.log(sigma_true)))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        self.grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same      #
        # size. Our solution uses 9 lines of code.                                  #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # dL/dscores
        sigma[range(N), y] -= 1 # In-place operation; sigma.shape=(N,C)-> unchanged
        grads_classScores = sigma / N # cost function has factor 1/N
        
        # dL/db2
        self.grads['b2'] = np.sum(grads_classScores, axis=0) # grads['b2'].shape=(C,)

        # dL/dW2
        self.grads['W2'] = np.dot(A.T, grads_classScores) # grads['w2'].shape=(H,C), A.shape=(N,H)

        # dL/db1
        dK1 = np.dot(grads_classScores, W2.T) * (np.where(K>0, 1, 0))
        self.grads['b1'] = np.sum(dK1, axis=0) # grads['b1'].shape=(H,)
        
        # dL/dw1
        self.grads['W1'] = np.dot(X.T, dK1) # grads['w1'].shape=(4,10)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return cost, self.grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95, 
              num_iters=100, batch_size=200,
              verbose=False):
        """
        Train this neural network using batch gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        # Initialize the random number generator
        rng = np.random.default_rng(seed=42)

        # Use batch GD to optimize the parameters in self.model
        cost_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively. Our solution uses 3 lines   #
            # of code.                                                              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            indices = rng.choice(num_train, batch_size, replace=True)
            X_batch = X[indices] # advanced indexing; mini=batch data
            y_batch = y[indices] # advanced indexing; mini=batch sample
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute cost and gradients using the current minibatch
            cost, grads = self.cost(X_batch, y=y_batch)
            cost_history.append(cost)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using batch gradient descent. You'll need to use the gradients        #
            # stored in the grads dictionary defined above. Our solution uses 4     #
            # lines of code.                                                        #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.params['W1'] = self.params['W1'] - learning_rate * self.grads['W1']
            self.params['b1'] = self.params['b1'] - learning_rate * self.grads['b1']
            self.params['W2'] = self.params['W2'] - learning_rate * self.grads['W2']
            self.params['b2'] = self.params['b2'] - learning_rate * self.grads['b2']
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: cost %f' % (it, num_iters, cost))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'cost_history': cost_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function. Our solution uses 4 lines of code.       #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1.reshape((1, b1.shape[0])))
        scores = np.dot(hidden_layer, W2) + b2.reshape((1, b2.shape[0])) # scores.shape=(N,C), b2.shape=(1,c)
        y_pred = np.argmax(scores, axis=1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
