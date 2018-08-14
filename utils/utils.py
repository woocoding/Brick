import numpy as np
import matplotlib.pyplot as plt


def fit(model, examples, labels, nums_epochs, learning_rate=1e-3, print_info=True, **kw):

    losses = []
    for epoch in np.arange(nums_epochs):
        yhat, loss = model.optimize(examples, labels, alpha=learning_rate, **kw)
        losses.append(loss)
        if print_info and (epoch + 1) % (nums_epochs//10 if nums_epochs>=10 else nums_epochs) == 0:
            print(f"Epoch[{epoch + 1}/{nums_epochs}], Loss: {loss:.{4}}")

    result = {
        "learning_rate":learning_rate,
        "losses":losses
    }
    return result

def accuracy(yhat, y, threshold=0.5):
    yhat = np.where(yhat > threshold, 1, 0)
    acc = np.mean(yhat == y)
    return acc

def loss_curve(result):
    plt.plot(result["losses"])
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title(f"Learning rate = {result['learning_rate']}")
    plt.show()

