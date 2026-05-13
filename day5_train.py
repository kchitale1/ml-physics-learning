import torch
import torch.nn as nn
from loss import make_diffusion_data

# --- Model ---
class DiffusionMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(0)   # so train/val/test are reproducible

X_train, y_train = make_diffusion_data(2000)
X_val,   y_val   = make_diffusion_data(500)
X_test,  y_test  = make_diffusion_data(500)

model = DiffusionMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- Training config ---
max_epochs = 5000
patience = 50
checkpoint_path = 'best_diffusion.pt'

best_val_loss = float('inf')
patience_counter = 0
train_history, val_history = [], []

for epoch in range(max_epochs):
    # TRAIN STEP
    # TODO: set train mode, zero grads, forward, loss, backward, step
    # append the train loss (use .item()) to train_history
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    train_loss = loss_fn(pred, y_train)
    train_loss.backward()
    optimizer.step()    
    train_history.append(train_loss.item())

    # VAL STEP
    # TODO: set eval mode, wrap in torch.no_grad(), forward on X_val,
    # compute val loss, append to val_history
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val)
        val_history.append(val_loss.item())

    # CHECKPOINT IF BEST
    # TODO: if val_loss < best_val_loss, save state_dict, reset patience_counter
    # else, increment patience_counter
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), checkpoint_path)
        patience_counter = 0
    else:
        patience_counter += 1

    # EARLY STOP
    # TODO: if patience_counter >= patience, break (and print why)
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} after {patience} epochs without improvement.")
        break   

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  train {train_history[-1]:.2e}  val {val_history[-1]:.2e}")

# --- Reload best & test ---
# TODO: rebuild model (or reuse the existing one), load_state_dict,
# evaluate on (X_test, y_test) under no_grad, print the test loss
best_model = DiffusionMLP()
best_model.load_state_dict(torch.load(checkpoint_path))
best_model.eval()
with torch.no_grad():    
    test_pred = best_model(X_test)
    test_loss = loss_fn(test_pred, y_test)
    print(f"Test loss: {test_loss.item():.2e}")

print("Min loss: ", min(train_history), min(val_history))
print("Final loss: ", train_history[-1], val_history[-1])
print("Best val loss: ", best_val_loss)  
print("Done!")
    
# --- Plot ---
# TODO: plot train_history and val_history on a log y-axis
import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_history, label='train')  
plt.plot(val_history, label='val')
plt.yscale('log')
plt.xlabel('Epoch')
plt.show()
