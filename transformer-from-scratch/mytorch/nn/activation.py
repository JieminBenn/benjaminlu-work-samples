import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        First = np.max(Z, axis=self.dim, keepdims=True)
        expZ = np.exp(Z - First)
        Second = np.sum(expZ, axis=self.dim, keepdims=True)
        self.A = expZ / Second
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        if self.dim >= 0:
            dim = self.dim
        else:
            dim = self.dim + len(shape)
        
        mss = np.moveaxis(np.empty(shape), dim, -1).shape
           
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = np.moveaxis(self.A, dim, -1).reshape(-1, C)
            dLdA = np.moveaxis(dLdA, dim, -1).reshape(-1, C)
        else:
            self.A = self.A.reshape(-1, C)
            dLdA = dLdA.reshape(-1, C)


        dLdZ = self.A * (dLdA - np.sum(dLdA * self.A, axis=1, keepdims=True))

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = np.moveaxis(self.A.reshape(mss), -1, dim)
            dLdZ = np.moveaxis(dLdZ.reshape(mss), -1, dim)
        else:
            dLdZ = dLdZ.reshape(shape)
            self.A = self.A.reshape(shape)

        return dLdZ
 