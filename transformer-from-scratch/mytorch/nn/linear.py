import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        self.input_shape = A.shape
        batch_size = np.prod(self.input_shape[:-1])
        S2flatZ = A.reshape(batch_size, self.W.shape[1]) @ self.W.T + self.b
        Z = S2flatZ.reshape(self.input_shape[:-1] + (self.W.shape[0],))
        
        # Store input for backward pass
        self.A = A
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZf = dLdZ.reshape(batch_size, self.W.shape[0])
        self.dLdA = dLdZf @ self.W
        self.dLdW = dLdZf.T @ self.A.reshape(batch_size, self.W.shape[1])
        self.dLdb = dLdZf.sum(axis=0)
        self.dLdA = self.dLdA.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
