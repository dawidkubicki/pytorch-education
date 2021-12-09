import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.read_csv("data.csv")
df.columns=["x1", "x2", "y"]

def step_Function(t):
    if t > 0:
        return 1
    return 0

def prediction(X, W, b):
    return step_Function((np.matmul(X,W)+b)[0])

data = df.to_numpy()

X_train = data[:,:2]
y_train = data[:,2]

def plot_perceptron(X_train, y_train):
    plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], marker="o", color='red', label='x1')
    plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], marker="s", color='green', label='x2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()




