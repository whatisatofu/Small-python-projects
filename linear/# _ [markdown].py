# %% [markdown]
# Import libraries

# %%
import matplotlib.pyplot as plt
import csv
from matplotlib.animation import FuncAnimation

# %% [markdown]
# Create a dataset first

# %% [markdown]
# Create functions needed?

# %%
def load_data(file_path):
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    X, y =zip(*data[1:])
    X = list(map(float, X))
    y = list(map(float, y))
    return X, y

def predict(X, w, b):
    return [w * x + b for x in X]

def compute_cost(y_pred, y):
    total = 0
    for i in range(len(y)):
        total += (y_pred[i] - y[i]) ** 2
    cost = total / len(y)
    return cost

def gradient_descent(y_pred, X, y):
    dW = 0
    db = 0
    const = 2 / len(y)
    for i in range(len(y)):
        x = X[i]
        error = y_pred[i] - y[i]
        dW += const * error * x
        db += const * error

    return dW, db



# %%
X, y = load_data('train.csv')
W, b = 0, 0
lr = 0.05

#set up the grid
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', alpha=0.5, label='Data')
line, = ax.plot([], [], color='red', lw=2, label='Model') # Empty line
info_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
ax.legend()

title = ax.set_title("")

def init():
    line.set_data([], [])
    return [line]

def update(frame):
    global W, b
    
    prediction = predict(X, W, b)
    cost = compute_cost(prediction, y)
    dW, db = gradient_descent(prediction, X, y)
    W -= lr * dW
    b -= lr * db

    info_text.set_text(f"Iteration: {frame}\nCost: {cost:.4f}\nW: {W:.2f}\nb: {b:.2f}")
    x_line = [0, 2]
    y_line = [W * x_line[0] + b, W * x_line[1] + b]
    line.set_data(x_line, y_line)
    return [line, info_text]
    
    


# %%
ani = FuncAnimation(fig, update, frames=150, init_func=init, blit=True, interval=50, repeat=False)
plt.show()


