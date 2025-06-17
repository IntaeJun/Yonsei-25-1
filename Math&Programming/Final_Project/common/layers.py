import numpy as np
from common.functions import *
from common.util import im2col, col2im

class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None # stores output for use in backward pass

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W # weight matrix
        self.b = b # bias vector
        
        self.x = None
        self.original_x_shape = None # used to reshape gradient
        self.dW = None # gradient of weight
        self.db = None # gradient of bias

    def forward(self, x):
        self.original_x_shape = x.shape # store input shape
        x = x.reshape(x.shape[0], -1) # flatten input (N, d1, d2, ...)
        self.x = x

        out = np.dot(self.x, self.W) + self.b 

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # gradient w.r.t. input
        self.dW = np.dot(self.x.T, dout) # gradient w.r.t. weights
        self.db = np.sum(dout, axis=0) # gradient w.r.t. biases
        
        dx = dx.reshape(*self.original_x_shape)  # reshape back to input shape
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # loss value
        self.y = None    # output of softmax
        self.t = None    # true labels (one-hot or label index)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # convert scores to pr. dist.
        self.loss = cross_entropy_error(self.y, self.t) # loss by CEE
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # Case 1: one-hot encoded labels
        if self.t.size == self.y.size: 
            dx = (self.y - self.t) / batch_size
        # Case 2: integer labels
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    Implements dropout regularization.
    Reference: http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio # pr. of dropping a neuron
        self.mask = None # binary mask applied during training

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio # keep neurons with (1-dropout_ratio) pr.
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio) # scale outputs during inference

    def backward(self, dout):
        return dout * self.mask # propagate gradients only thru active neurons


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # filter weights (FN, C, FH, FW)
        self.b = b # filter biases (FN, )
        self.stride = stride
        self.pad = pad
        
        # intermediate varialbes (for backward pass)
        self.x = None   
        self.col = None
        self.col_W = None
        
        # gradients for weights and biases
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape # number of filters, channels, filter height/width
        N, C, H, W = x.shape  # batch size, input channels, height, width
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # reshape input and weights for matrix multiplication
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # perform convolution via matrix multiplication
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # reshape dout to (N*out_h*out_w, FN)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # compute gradients for bias and weights
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # compute graident for input
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        
        dcol = np.zeros((self.arg_max.size, pool_size))
        
        dcol[np.arange(self.arg_max.size), self.arg_max] = dout.flatten()

        N, C, H, W = self.x.shape
        out_h = (H + 2*self.pad - self.pool_h)//self.stride + 1
        out_w = (W + 2*self.pad - self.pool_w)//self.stride + 1
        dcol = dcol.reshape(N*out_h*out_w, -1)
        
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx