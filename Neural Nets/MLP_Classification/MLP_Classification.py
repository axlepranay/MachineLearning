
# Loading datatest
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#Load MNIST dataset 
digits = datasets.load_digits(n_class=10)
# Create our X and y data
X = digits.data
Y = digits.target
print(X.shape, Y.shape)
num_examples = X.shape[0]      ## training set size
nn_input_dim = X.shape[1]      ## input layer dimensionality
nn_output_dim = len(np.unique(Y))       ## output layer dimensionality

params = {
    "lr":1e-5,        ## learning_rate
    "max_iter":1000,
    "h_dimn":40,     ## hidden_layer_size
}

# Building helper functions:

def build_model():
    hdim = params["h_dimn"]
    # Initialize the parameters to random values.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, hdim))
    W2 = np.random.randn(hdim, nn_output_dim) / np.sqrt(hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

def softmax(x):
    exp_scores = np.exp(x)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def feedforward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)
    return a1, probs

def backpropagation(model, x, y, a1, probs):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    delta3 = probs
    delta3[range(y.shape[0]), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)
    return dW2, db2, dW1, db1

def calculate_loss(model, x, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation to calculate predictions
    _, probs = feedforward(model, x)
    
    # Calculating the cross entropy loss
    corect_logprobs = -np.log(probs[range(y.shape[0]), y])
    data_loss = np.sum(corect_logprobs)
    
    return 1./y.shape[0] * data_loss

def test(model, x, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate predictions
    _, probs = feedforward(model, x)
    preds = np.argmax(probs, axis=1)
    return np.count_nonzero(y==preds)/y.shape[0]

def train(model, X_train, X_test, Y_train, Y_test, print_loss=True):
    # Gradient descent. For each batch...
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    for i in range(0, params["max_iter"]):

        # Forward propagation
        a1, probs = feedforward(model, X_train)

        # Backpropagation
        dW2, db2, dW1, db1 = backpropagation(model, X_train, Y_train, a1, probs)

        # Gradient descent parameter update
        W1 += -params["lr"] * dW1
        b1 += -params["lr"] * db1
        W2 += -params["lr"] * dW2
        b2 += -params["lr"] * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i % 50 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model, X_train, Y_train)),
                  ", Test accuracy:", test(model, X_test, Y_test), "\n")
    return model
    
    
    
    # Training the model
model = build_model()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

model = train(model, X_train, X_test, Y_train, Y_test)