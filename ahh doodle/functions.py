import numpy as np
import pickle


#----- Neural Network Functions -----
def random_layer(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

def initwb(input_dim, hidden_dim, output_dim):
    W = [random_layer(input_dim, hidden_dim[0])]
    for i in range(1, len(hidden_dim)):
        W.append(random_layer(hidden_dim[i-1], hidden_dim[i]))
    W.append(random_layer(hidden_dim[-1], output_dim))

    b = [np.zeros((1, dim)) for dim in hidden_dim + [output_dim]] 

    return W, b

def ReLU(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # For numerical stability
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def binary_cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return loss

def multiple_cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    return loss

def forward_pass(X, W, b):
    """Performs a forward pass through all layers"""
    activations = [X]
    for l in range(len(W) - 1):
        Z = np.dot(activations[-1], W[l]) + b[l]
        A = ReLU(Z)
        activations.append(A)
    
    # Output layer
    Z = np.dot(activations[-1], W[-1]) + b[-1]
    A = softmax(Z)
    activations.append(A)
    
    return activations

def backward_propagation(y, W, b, activations):

    # m is the number of training examples
    m = len(y)
    dW = [None] * len(W)
    db = [None] * len(b)
    # derivative of MCR and softmax combined simplifies to (A - y) where A is the output of the softmax
    dz = activations[-1] - y
    print(dz.shape)

    for i in range(len(W) - 1, -1, -1):
        # derivative of dW, which is dz * A_prev (activations[i])
        # derivative of db, which is just the sum of dz across samples and normalized by m
        dW[i] = np.dot(activations[i].T, dz) / m
        db[i] = np.sum(dz, axis=0, keepdims=True) / m
        
        if i > 0:  # No need to backpropagate beyond input layer
            da = np.dot(dz, W[i].T)
            dz = da * (activations[i] > 0)  # ReLU derivative
        
    return dW, db
    #dW and db includes all gradients for each layer, which will be used to update weights and biases in the training loop.

def update_parameters(W, b, dW, db, lr):
    for i in range(len(W)):
        W[i] -= lr * dW[i]
        b[i] -= lr * db[i]

def calculate_accuracy(y_true, y_pred):
    # Convert probabilities to class labels
    predictions = np.argmax(y_pred, axis=1)
    # Compare with true labels and take the average
    accuracy = np.mean(predictions == np.argmax(y_true, axis=1))
    return accuracy * 100  # Return as a percentage

#----- Model Saving and Loading -----
def save_model(W, b, filename):
    with open(filename, 'wb') as f:
        pickle.dump((W, b), f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)