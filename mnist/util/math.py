import numpy as np
def softmax(x):
    x_exp = np.exp(x)
    denom = sum(x_exp)
    return x_exp / denom
