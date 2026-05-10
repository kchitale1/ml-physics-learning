import random
import math

from numpy import size 

class FakeDataset:
    def __init__(self, size):
        self.size = size
        x = [random.uniform(-1, 1) for _ in range(size)]
        y = [math.sin(5*xi) * math.exp(-xi**2) + random.gauss(0, 0.1) for xi in x]
        #y = [2*xi+1+random.gauss(0, 0.1) for xi in x]
        self.x = x
        self.y = y
 
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class FakeLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(self.dataset), self.batch_size):
            batch_x = []
            batch_y = []
            for j in range(i, min(i+self.batch_size, len(self.dataset))):
                x, y = self.dataset[j]
                batch_x.append(x)
                batch_y.append(y)
            yield batch_x, batch_y
    
def train(loader, n_epochs=10, lr=0.01):
    w, b = 0.0, 0.0           # initialize parameters
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xs, ys in loader:
            # 1. forward pass — compute y_pred for every x in the batch
            y_pred = [w*x + b for x in xs]  # forward pass

            # 2. compute residuals r_i = y_pred_i - y_true_i
            r_s = [y_p-y for y_p, y in zip(y_pred, ys)]  # residuals

            # 3. compute batch loss = mean(r_i^2) — for printing
            batch_loss = sum(r**2 for r in r_s) / len(r_s)  # mean squared error

            # 4. compute gradients dw = mean(2*r_i*x_i) and db = mean(2*r_i)
            dw = sum(2*r*x for r, x in zip(r_s, xs)) / len(r_s)  # gradient w.r.t. w      
            db = sum(2*r for r in r_s) / len(r_s)  # gradient w.r.t. 
            
            # 5. update w and b using the gradients and learning rate
            w -= lr * dw  # update w      
            b -= lr * db  # update b    
            epoch_loss += batch_loss
            n_batches += 1
        print(f"epoch {epoch:3d}  loss={epoch_loss / n_batches:.6f}  w={w:.3f}  b={b:.3f}")
    return w, b

if __name__ == "__main__":
    ds = FakeDataset(500)
    loader = FakeLoader(ds, batch_size=5, shuffle=False)
    for xs, ys in ds:
        print(xs, ys)
    w, b = train(loader, n_epochs=20, lr=0.01)
