import numpy as np
rng = np.random.default_rng(1)

# randomize 100x5 columns
matrix=rng.integers(low=3, high=10, size=(100,5))
print(matrix.shape)
final=((matrix-matrix.mean(axis=0))/matrix.std(axis=0))
print(final)

print()
print("================================")
print()

# Pairwise distance matrix between A(nx3) and B(mx3), no loops -- Crossing + 2nd Degrees? Estimate: shape(2, 4, 3) for total difference, shape(2,4) for final distance
a=rng.integers(1, 5, size=(2,3))
b=rng.integers(1, 5, size=(4,3))
diff = -b[np.newaxis,:,:]+a[:,np.newaxis,:]
# print(diff, diff.shape)
distance=np.sqrt(((diff**2).sum(axis=2)))
print(distance, distance.shape)

'''
Notes:
Task 1 & 2 - axis configurations are particularly useful, in changing the dimensions. The axis value corresponds to the index on the shape. 
Task 2 - The dimensions are not separate values, but a way of packaging that retains the information. 
'''