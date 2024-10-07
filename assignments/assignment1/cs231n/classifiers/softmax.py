from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n = X.shape[0]
    c = W.shape[1]

    for i in range(n):
      score = X[i].dot(W)
      max_score = np.max(score)
      score -= max_score#保证数据的稳定性
      exp_score = np.exp(score)
      pro = exp_score/ np.sum(exp_score)
      correct_score_pro = pro[y[i]]
      
      loss -= np.log(correct_score_pro)

      #计算梯度
      for j in range(c):
        if j == y[i]:
          dW[:,j] += (pro[j]-1)*X[i]
        else:
          dW[:,j] += pro[j]*X[i]

    loss /= n
    dW /= n

    loss += reg*np.sum(W*W)
    dW += 2*reg*W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]

    scores = X.dot(W)
    #保持数值的稳定性
    scores -= np.max(scores,axis =1 ,keepdims = True)

    exp_score = np.exp(scores)
    pro = exp_score / np.sum(exp_score,axis =1 ,keepdims = True)

    correct_class_pro = pro[np.arange(N),y]

    loss = -np.sum(np.log(correct_class_pro)) / N
    loss += reg*np.sum(W*W)

    #对于梯度的计算
    tran = pro
    tran[range(N),y] -= 1
    dW = X.T.dot(tran) / N
    dW += 2*reg*W

    

    

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
