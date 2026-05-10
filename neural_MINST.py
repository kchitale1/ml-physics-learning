import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Data loading ---

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
# transforms.Compose returns a callable object (it implements __call__).
# Passing it as transform= means MNIST will call transform(image) on each sample.

full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set   = datasets.MNIST('./data', train=False, download=True, transform=transform)
# MNIST subclasses Dataset, which defines __len__ and __getitem__.
# Instantiation here calls Dataset.__init__ (via inheritance) + MNIST's own __init__.

train_set, val_set = torch.utils.data.random_split(full_train, [55000, 5000])
# Tuple unpacking: random_split returns a 2-element sequence;
# Python unpacks it into two names in one assignment.

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)
# DataLoader implements the iterator protocol (__iter__ / __next__).
# Each call to __next__ collates a batch and returns it.
# len(loader) calls DataLoader.__len__, which returns ceil(N / batch_size).

# --- Model definition ---

class MLP(nn.Module):                        # inheritance: MLP IS-A nn.Module
    def __init__(self, h1=128, h2=256):      # default argument values
        #super().__init__()                   # super() finds nn.Module in the MRO;
                                             # must be called so nn.Module can set up
                                             # its internal parameter registry
        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)
        # Assigning nn.Module instances to self.* triggers nn.Module.__setattr__,
        # which auto-registers them as submodules (so .parameters() finds their weights).

    def forward(self, x):                    # method overriding: replaces the abstract
                                             # forward() stub in nn.Module
        x = x.view(x.size(0), -1)           # .view() is a tensor method (like reshape);
                                             # -1 lets Python/PyTorch infer that dimension
        x = torch.relu(self.fc1(x))
        # self.fc1(x) calls fc1.__call__(x), NOT fc1.forward(x) directly.
        # nn.Module.__call__ runs hooks, then calls forward(). This is the standard
        # dunder __call__ pattern: obj(args) → obj.__call__(args).
        x = torch.relu(self.fc2(x))
        return self.fc3(x)                   # same __call__ indirection for fc2, fc3

model     = MLP()                            # instantiation: calls MLP.__init__
loss_fn   = nn.CrossEntropyLoss()           # also an nn.Module; callable via __call__
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# model.parameters() is a generator method (uses yield internally).
# Adam stores a reference to it — it does NOT consume it all at once.

# --- Training utilities ---

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()                            # sets a flag on all submodules (affects
                                             # Dropout, BatchNorm, etc.); method call
    total_loss = 0.0
    for x, y in loader:                      # iterator protocol: loader.__iter__() once,
                                             # then loader.__next__() each iteration;
                                             # x, y = tuple unpacking of each batch
        logits = model(x)                    # model.__call__(x) → model.forward(x)
        loss   = loss_fn(logits, y)          # same __call__ pattern on CrossEntropyLoss
        optimizer.zero_grad()               # clears .grad on every parameter tensor
        loss.backward()                      # autograd: traverses computational graph
                                             # in reverse (reverse-mode AD), populates .grad
        optimizer.step()                     # uses stored .grad values to update weights
        total_loss += loss.item() * x.size(0)
        # loss.item(): converts a 0-d tensor to a plain Python float (__float__ / .item())
        # x.size(0): batch size — needed to weight the loss correctly before averaging
    return total_loss / len(loader.dataset)
    # len(loader.dataset) calls dataset.__len__() — the dunder you implemented in Day F's
    # FakeDataset is the exact same protocol PyTorch's MNIST relies on here.

def evaluate(model, loader, loss_fn):
    model.eval()                             # symmetric to model.train(); disables dropout etc.
    total_loss = 0.0
    correct    = 0
    with torch.no_grad():                    # context manager: __enter__ disables gradient
                                             # tracking, __exit__ re-enables it — same
                                             # protocol as `with open(...) as f`
        for x, y in loader:                  # iterator protocol again
            logits = model(x)
            loss   = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            # == on tensors calls Tensor.__eq__, returning a bool tensor (operator overloading)
            # .sum() reduces it; .item() extracts the Python int

    n = len(loader.dataset)                  # __len__ again
    return total_loss / n, correct / n       # returns a tuple (implicit, no parentheses needed)

# --- Training loop ---

n_epochs = 10
train_losses, val_losses, val_accs = [], [], []  # tuple unpacking of list literals

for epoch in range(n_epochs):               # range implements __iter__/__next__
    train_loss          = train_one_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_acc   = evaluate(model, val_loader, loss_fn)   # tuple unpacking of return value
    train_losses.append(train_loss)         # list.append — plain method call
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{n_epochs}: train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")
    # f-string: evaluated at runtime; :.4f is a format spec (same as str.format)

test_loss, test_acc = evaluate(model, test_loader, loss_fn)   # tuple unpacking
print(f"Test accuracy: {test_acc:.4f}")

# --- Plotting ---
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses,   label='val')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.legend(); plt.title('MNIST MLP training curves')
plt.show()