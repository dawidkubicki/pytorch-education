import numpy as np

def softmax(L: list):
	results = []
	for i in L:
		num = np.exp(i)/np.sum(np.exp(L), axis=0)
		results.append(num)
	return results			


print(softmax([1,2,3,4,5,6]))
