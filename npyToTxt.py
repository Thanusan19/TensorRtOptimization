import numpy as np


X = np.load('/home/localadmin/Documents/TEST_SET_S2S_X.npy')
f=open("X.txt","w+")
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		for k in range(X.shape[2]):
			f.write("%f," % X[i][j][k])
print(X)
