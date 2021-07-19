from UDM import UDM

import pandas as pd
import numpy as np
np.set_printoptions(precision=5, suppress=True, edgeitems=30, linewidth=100000)   
from joblib import dump
import matplotlib.pyplot as plt

# df = pd.read_csv("Tom/table3.csv")
# ordinal = ["O" in c for c in df.columns]
# X = df.values

df = pd.read_csv("Tom/adult.csv", index_col=0)
ordinal = [True, False, True, False, False, False, True]
X = df.values[:,:-1] # Ignore compensation column.

udm = UDM(X, ordinal) 
dump(udm, "Tom/adult_udm.joblib")

# print(udm.R)
# print(udm.phi[0])
# plt.imshow(udm.phi[0])
# plt.show()

# mask = np.ones((X.shape[0], X.shape[0]), dtype=bool)
# mask[2,0] = 0
# mask[0,2] = 0
# mask[1,3] = 0

# print(udm(X, mask=mask, placeholder=np.inf))

# print(udm(X[2], X[3]))