import numpy as np

T = np.load("T_final.npy")
print(T.shape, T.min(), T.max(), T.mean())