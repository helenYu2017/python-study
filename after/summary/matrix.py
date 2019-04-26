import numpy as np 

#A = np.random.random([10000]) 
#np.savez("mydata.npz", A=A)

files = np.load("mydata.npz")
A = files["A"]
