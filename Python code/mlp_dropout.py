import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import os

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv("data.csv")  
X = df[['T', 'Ek', 'Ed']].values
y = df['P'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=4)

class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, dropout=0.3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()

ini_path = "ini.pth"
if os.path.exists(ini_path):
    state = torch.load(ini_path, map_location="cpu")
   
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"Loaded initial weights from {ini_path}")
else:
    print(f"{ini_path} not found, using random initialization.")

optimizer = optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

epochs = 500
patience = 20  
best_r2 = -np.inf
best_epoch = -1
bad_count = 0
best_model_path = "mlp_model_best.pth"

progress = tqdm(range(epochs), desc="Training", dynamic_ncols=True)
for epoch in progress:
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_test)
        
        y_val_pred_inv = scaler_y.inverse_transform(y_val_pred.detach().numpy())
        y_val_true_inv = scaler_y.inverse_transform(y_test.detach().numpy())
        r2_val = r2_score(y_val_true_inv, y_val_pred_inv)

    progress.set_postfix({"loss": f"{loss.item():.6f}", "val_R2": f"{r2_val:.4f}"})

    if r2_val > best_r2 + 1e-10:
        best_r2 = r2_val
        best_epoch = epoch + 1
        bad_count = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        bad_count += 1

    if bad_count >= patience:
        tqdm.write(f"Early stopping at epoch {epoch+1} (no val R² improvement for {patience} epochs). Best at epoch {best_epoch} with R²={best_r2:.4f}.")
        break

tqdm.write(f"Best validation R²: {best_r2:.4f} at epoch {best_epoch}")


if 'best_model_path' in locals():
    model.load_state_dict(torch.load(best_model_path))

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.numpy())
    y_pred_inv = np.maximum(y_pred_inv, 0) 
    y_true_inv = scaler_y.inverse_transform(y_test.numpy())

r2 = r2_score(y_true_inv, y_pred_inv)
print(f"\nTest-set R²: {r2:.4f}")

print("\nTrue vs Predicted P:")
for t, p in zip(y_true_inv.flatten(), y_pred_inv.flatten()):
    print(f"True: {t:.2f}, Pred: {p:.2f}")

model_path = "mlp_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
