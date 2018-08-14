import numpy as np
import matplotlib.pyplot as plt


class Dataloader(object):
    
    def __init__(self, examples, labels, batch_size=None, shuffle=False):
        self.examples = examples
        self.labels = labels
        self.m = self.examples.shape[1]
        self.batch_size = batch_size if batch_size else self.m
        self.shuffle = shuffle
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.shuffle:
            idx = np.random.permutation(self.m)
            self.examples = self.examples[:,idx]
            self.labels = self.labels[:,idx]
        while True:
            start = self.batch_count * self.batch_size
            end = start + self.batch_size
            if end > self.m:
                self.batch_count = 0
                raise StopIteration
            self.batch_count += 1
            return self.examples[:, start:end], self.labels[:, start:end]


def fit(model, examples, labels, nums_epochs, learning_rate=1e-3, batch_size=None, print_info=True, **kw):

    losses = []
    dataloader = Dataloader(examples, labels, batch_size)
    for epoch in np.arange(nums_epochs):
        for (batch_examples, batch_labels) in dataloader:
            yhat, loss = model.update(batch_examples, batch_labels, alpha=learning_rate, **kw)
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

