"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    N,D = np.shape(X)
    D,C = np.shape(W)
    scores = np.dot(X, W)
    dscores = np.zeros_like(scores)
    temp = np.sum(np.exp(scores),1)
    for i in range(0,N):
        for j in range(0,C):
            scores[i,j] = np.exp(scores[i,j])/ temp[i]
            dscores[i,j] = scores[i,j]
            if j == y[i]:
                loss += -np.log(scores[i,j])
                dscores[i,j] = scores[i,j] - 1
    for i in range(0, N):
        dW +=  np.dot(X[i,:].reshape(D,1), dscores[i,:].reshape(1,C))
        #for j in range(0, C):
    for i in range(0, D):
        for j in range(0, C):
            loss += 1/2 * reg * W[i,j] * W[i,j]
            dW += reg * W[i,j] 
    loss /= N
    dW /= N
    
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    scores = np.dot(X,W)
    scores -=np.max(scores)
    N = X.shape[0]
    exp_scores = np.exp(scores)
    
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    correct_logprobs = -np.log(probs[range(N),y])
    
    data_loss = np.sum(correct_logprobs)/N
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N
    
    dW = np.dot(X.T, dscores)
    dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7,2e-7, 5e-7]
    regularization_strengths = [1e4, 2e4, 3e4, 4e4, 5e4,]
    num_iters = [2000]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    
    for lr in learning_rates:
        for regs in regularization_strengths:
            for num_iter in num_iters:
                softmax = SoftmaxClassifier()
                loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=regs,
                              num_iters=num_iter, verbose=True)
                y_train_pred = softmax.predict(X_train)
                train_accuracy = np.mean(y_train == y_train_pred)
                y_val_pred = softmax.predict(X_val)
                val_accuracy = np.mean(y_val == y_val_pred)

                all_classifiers.append((softmax,val_accuracy))
                results[(lr,regs,num_iter)] = (train_accuracy, val_accuracy)

                if(val_accuracy > best_val):
                    best_val = val_accuracy
                    best_softmax = softmax
            
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg,num_iter) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg,num_iter)]
        print('lr %e reg %e num_iters %e train accuracy: %f val accuracy: %f' % (
              lr, reg, num_iter, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
