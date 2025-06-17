import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import Convolution, Pooling, Affine, ReLU, SoftmaxWithLoss

class CNN:
    """
    conv - relu - pool -
    conv - relu - pool -
    affine - relu - affine - softmax
    """

    def __init__(self, input_dim=(1,28,28),
                 conv_param_1={'filter_num':16,'filter_size':3,'pad':1,'stride':1},
                 conv_param_2={'filter_num':32,'filter_size':3,'pad':1,'stride':1},
                 hidden_size=100, output_size=10):
        
        # 1) Initialize Parameters
        pre_node_nums = np.array([
            input_dim[0]*3*3,
            conv_param_1['filter_num']*3*3,
            hidden_size
        ])
        scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        C1, C2 = conv_param_1['filter_num'], conv_param_2['filter_num']
        # conv1
        self.params['W1'] = scales[0] * np.random.randn(
            C1, input_dim[0], 3, 3)
        self.params['b1'] = np.zeros(C1)
        # conv2
        self.params['W2'] = scales[1] * np.random.randn(
            C2, C1, 3, 3)
        self.params['b2'] = np.zeros(C2)
        # affine1
        flat_size = C2 * 7 * 7
        self.params['W3'] = scales[2] * np.random.randn(flat_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        # affine2
        self.params['W4'] = np.sqrt(2.0/hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 2) Construct layer structure
        self.layers = [
            Convolution(self.params['W1'], self.params['b1'],
                        conv_param_1['stride'], conv_param_1['pad']),
            ReLU(),
            Pooling(pool_h=2, pool_w=2, stride=2),

            Convolution(self.params['W2'], self.params['b2'],
                        conv_param_2['stride'], conv_param_2['pad']),
            ReLU(),
            Pooling(pool_h=2, pool_w=2, stride=2),

            Affine(self.params['W3'], self.params['b3']),
            ReLU(),
            Affine(self.params['W4'], self.params['b4'])
        ]
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            x = layer.forward(x) 
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0
        for i in range(x.shape[0]//batch_size):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # 3) grads: W1, b1, W2, b2, W3, b3, W4, b4
        grads = {
            'W1': self.layers[0].dW, 'b1': self.layers[0].db,
            'W2': self.layers[3].dW, 'b2': self.layers[3].db,
            'W3': self.layers[6].dW, 'b3': self.layers[6].db,
            'W4': self.layers[8].dW, 'b4': self.layers[8].db,
        }
        return grads
    
    def save_params(self, file_name="simple_cnn_params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Successfully Saved: {file_name}")
    
    def load_params(self, file_name="simple_cnn_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val
        
        self.layers[0].W, self.layers[0].b = self.params['W1'], self.params['b1']
        self.layers[3].W, self.layers[3].b = self.params['W2'], self.params['b2']
        self.layers[6].W, self.layers[6].b = self.params['W3'], self.params['b3']
        self.layers[8].W, self.layers[8].b = self.params['W4'], self.params['b4']

        print(f"Successfully Loaded: {file_name}")