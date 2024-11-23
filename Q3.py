from matplotlib import pyplot as plt
import numpy as np
import time

#function to read from txt file
def read_data(file_name):
    with open(file_name) as f:
        data = f.read().splitlines()
    #features
    x = []
    for i in data:
        x.append([float(i.split()[0]), float(i.split()[1])])

    #labels
    y = []
    for i in data:
        y.append(i.split()[2])

    #print(data)
    #print(x)
    #print(y)
    
    return x, y

def split_data(x, y):
    X_train = x[:int(0.8*len(x))]
    Y_train = y[:int(0.8*len(y))]
    X_test = x[int(0.8*len(x)):]
    Y_test = y[int(0.8*len(y)):]
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression_batch_gradient(X, Y, w, learning_rate, iterations):
    m = len(Y)
    cost_history = []
    for i in range(iterations):
        gradient = np.dot(X.T, (sigmoid(np.dot(X, w)) - Y)) / m
        w -= learning_rate * gradient
        cost = -np.mean(Y * np.log(sigmoid(np.dot(X, w))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, w))))
        cost_history.append(cost)
    return w, cost_history

def plot_loss(cost_history, learning_rate, filename):
    plt.plot(cost_history, label='learning_rate = ' + str(learning_rate))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def main():
    x, y = read_data('P3input2024_pre.txt')
    #turn y into binary array
    y = np.array([1 if i == '0' else 0 for i in y])
    #split data into test and train
    X_train, Y_train, X_test, Y_test = split_data(x, y)

    learning_rates = [0.0004, 0.0006, 0.0008, 0.0001]
    iterations = 200000

    

    for learning_rate in learning_rates:
        w = np.zeros(X_train.shape[1])
        w, cost_history = logistic_regression_batch_gradient(X_train, Y_train, w, learning_rate, iterations)
        print(f'Learning rate: {learning_rate}, Final weights: {w}')
        plot_loss(cost_history, learning_rate, f'loss_curve_lr_{learning_rate}.png')

if __name__ == '__main__':
    main()