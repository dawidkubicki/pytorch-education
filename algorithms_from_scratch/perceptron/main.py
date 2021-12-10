import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")
df.columns=["x1", "x2", "y"]

def step_Function(t):
    if t > 0:
        return 1
    return 0

def prediction(X, W, b):
    return step_Function((np.matmul(X,W)+b)[0])

data = df.to_numpy()

np.random.shuffle(data)

X_data = data[:,:2]
y_data = data[:,2]

'''

Splitting dataset into traning and testing

'''

X_train = X_data[:-15]
y_train = y_data[:-15]

X_test = X_data[-15:]
y_test = y_data[-15:]

'''
Normalizing later would be great

'''

def plot_perceptron(X_train, y_train, a0, a1):
    plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], marker="o", color='red', label='Class 1')
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], marker="s", color='green', label='Class 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.plot(a0, a1)	
    plt.legend()
    plt.show()


''' 

Initialize random weights as floats

'''

W = np.random.rand(2,1)
b = np.ones(1)

'''

Prediction function

'''

def prediction(X, W, b):
	if (np.dot(X,W) + b) > 0:
		return 1
	else:
		return 0

'''

Train step function

'''

def train_step(X, y, W, b, lr):

	boundary_lines = []

	for i in range(X.shape[0]):
		y_hat = prediction(X[i], W, b)
		if y[i] - y_hat == 1:
			W[0] = W[0] + lr*X[i][0]
			W[1] = W[1] + lr*X[i][1]
			b += lr		

		elif y[i] - y_hat == -1:
			W[0] = W[0] - lr*X[i][0]
			W[1] = W[1] - lr*X[i][1]
			b -= lr			

		boundary_lines.append((-W[0]/W[1], -b/W[1]))
	
	return W,b,boundary_lines

epochs = 5

for i in range(epochs):
	W,b,bl = train_step(X_train, y_train, W, b, 0.1)

x0_min = np.min(X_train)
x0_max = np.max(X_train)

# x0*w0 + x1*w1 + b = 0
# x1 = (-x0*w0 -b) /w1


x1_min = ((-(W[0]*x0_min) -b) / W[1])
x1_max = ((-(W[0]*x0_max) -b) / W[1])

plot_perceptron(X_train, y_train, [x0_min, x0_max], [x1_min, x1_max])


def evaluate():
	for idx, i in enumerate(range(y_test.shape[0])):
		pred_y = np.where(np.dot(X_test[i], W) + b > 0., 1, 0)

		print(f" ({idx}) Predicted: {int(pred_y)}")
		print(f" ({idx}) Real value: {int(y_test[0])}\n")



evaluate()







	

