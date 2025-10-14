import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import r2_score

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
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)
    y_pred_inv = scaler_y.inverse_transform(y_pred.numpy())
    y_pred_inv = np.maximum(y_pred_inv, 0)  
    y_true_inv = scaler_y.inverse_transform(y_tensor.numpy())

r2 = r2_score(y_true_inv, y_pred_inv)
print(f"\nTrain-set R²: {r2:.4f}")

print("\nTrue vs Predicted P:")
for t, p in zip(y_true_inv.flatten(), y_pred_inv.flatten()):
    print(f"True: {t:.2f}, Pred: {p:.2f}")

model_path = "mlp_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
