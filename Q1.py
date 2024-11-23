#finding decision boundary for logisitic regression of 2D data

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

#function to split data into test and train
def split_data(x, y):
    X_train = x[:int(0.8*len(x))]
    Y_train = y[:int(0.8*len(y))]
    X_test = x[int(0.8*len(x)):]
    Y_test = y[int(0.8*len(y)):]
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#find decision boundary of logistic regression with batch gradient descent and equation of the form y = a1+ a2*x + a3*x^2 + a4sin(x)
def logistic_regression_batch_gradient(X, Y, w, learning_rate, iterations):
    for i in range(iterations):
        #calculate predictions
        predictions = sigmoid(np.dot(X, w))
        #calculate error
        error = Y - predictions
        #update weights
        w += learning_rate * np.dot(X.T, error)
    print(w)
    return w

#Stochastic gradient descent
def logistic_regression_stochastic_gradient(X, Y, w, learning_rate, iterations):
    for i in range(iterations):
        for j in range(len(X)):
            #calculate prediction
            prediction = sigmoid(np.dot(X[j], w))
            #calculate error
            error = Y[j] - prediction
            #update weights
            w += learning_rate * X[j] * error
    return w

#Mini-batch gradient descent
def logistic_regression_mini_batch_gradient(X, Y, w, learning_rate, iterations, batch_size):
    for i in range(iterations):
        for j in range(0, len(X), batch_size):
            #calculate predictions
            predictions = sigmoid(np.dot(X[j:j+batch_size], w))
            #calculate error
            error = Y[j:j+batch_size] - predictions
            #update weights
            w += learning_rate * np.dot(X[j:j+batch_size].T, error)
    return w

#main function
def main():
    x, y = read_data('P3input2024_pre.txt')
    #turn y into binary array
    y = np.array([1 if i == '0' else 0 for i in y])
    #split data into test and train
    X_train, Y_train, X_test, Y_test = split_data(x, y)
    #print(X_train)
    #print(Y_train)
    #print(X_test)
    #print(Y_test)
    #print('Data split into test and train')

    #run logistic regression with 200,000 iterations and learning rate of 0.1 and batch gradient descent

    #initialize weights
    w = np.zeros(X_train.shape[1])
    #run logistic regression with batch gradient descent
    #calculate time taken
    start = time.time()
    w = logistic_regression_batch_gradient(X_train, Y_train, w, 0.1, 200000)
    end = time.time()
    print(f'Time taken: {end - start}')
    print('Logistic regression complete batch gradient descent')

    #calculate predictions
    predictions = sigmoid(np.dot(X_test, w))
    #turn predictions into binary array
    predictions = [1 if i > 0.5 else 0 for i in predictions]

    #calculate accuracy
    accuracy = np.mean(predictions == Y_test)
    print(f'Accuracy: {accuracy}')

    #run logistic regression with stochastic gradient descent
    #initialize weights
    w = np.zeros(X_train.shape[1])
    #run logistic regression with stochastic gradient descent
    #calculate time taken
    start = time.time()
    w = logistic_regression_stochastic_gradient(X_train, Y_train, w, 0.1, 200000)
    end = time.time()
    print(f'Time taken: {end - start}')
    print('Logistic regression complete stochastic gradient descent')

    #calculate predictions
    predictions = sigmoid(np.dot(X_test, w))
    #turn predictions into binary array
    predictions = [1 if i > 0.5 else 0 for i in predictions]
    
    #calculate accuracy
    accuracy = np.mean(predictions == Y_test)
    print(f'Accuracy: {accuracy}')

    #run logistic regression with mini-batch gradient descent
    #initialize weights
    w = np.zeros(X_train.shape[1])
    #run logistic regression with mini-batch gradient descent
    #calculate time taken
    start = time.time()
    w = logistic_regression_mini_batch_gradient(X_train, Y_train, w, 0.1, 200000, 10)
    end = time.time()
    print(f'Time taken: {end - start}')
    print('Logistic regression complete mini-batch gradient descent')

    #calculate predictions
    predictions = sigmoid(np.dot(X_test, w))
    #turn predictions into binary array
    predictions = [1 if i > 0.5 else 0 for i in predictions]

    #calculate accuracy
    accuracy = np.mean(predictions == Y_test)
    print(f'Accuracy: {accuracy}')



if __name__ == '__main__':
    main()