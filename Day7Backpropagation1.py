import numpy as np

# raw outputs of network, shape (N, C)
z = np.array([[2.0, 1.0, 0.1],
              [0.5, 2.5, 0.3]])

# label for z, shape (N,), shows which one is correct
y = np.array([0, 1])

def softmax(value):
    local_e = np.exp(value - value.max(axis=1, keepdims=True)) # Avoid overflow for too large of an e^. This subtracts each row by its max value, doesn't change the math because of how e^ works by distribution.
    result = local_e / local_e.sum(axis=1, keepdims=True) # Makes the shape uniform align
    return result

def cross_entropy(value, label):
    row = value.shape[0]
    # print(row)
    output=(-np.log(softmax(value)[np.arange(row), label])).mean()
    # in here we first used the label to identify the correct index in each row, and then took the -log of the p to get their relative cross entropies
    # then we meaned them per definition of cross entropy
    return output

# Check
print(cross_entropy(z, y))                                        # ≈ 0.3185
print(cross_entropy(np.zeros((5, 10)), np.zeros(5, dtype=int)))   # ≈ 2.3026
print(softmax(np.array([[1000., 1001., 1002.]])))                 # ≈ [[0.090, 0.245, 0.665]]
print(cross_entropy(np.array([[50., 0., 0.]]), np.array([1])))    # ≈ 50.0
print(cross_entropy(np.array([[1000., 0., 0.]]), np.array([1])))  # inf, plus a warning

