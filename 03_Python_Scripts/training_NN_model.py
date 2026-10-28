"""
Training NN Model after Optimization

This script executes the final training and validation pipeline for the Artificial 
Neural Network (ANN) embedding the hyperparameters dynamically identified by the 
Optuna framework. It incorporates a robust testing routine where the fully trained 
network, safeguarded by early stopping to prevent overfitting, predicts the 
constitutive parameters on an unseen test dataset. To ensure comprehensive evaluation 
and subsequent visualization, the script calculates the global and parameter-specific 
error metrics (R-squared, MAE, and MAPE) on the physical parameter space. It then 
automatically plots the learning curves and exports the network predictions across 
the entire training, validation, and testing sets into formatted CSV files, which 
are essential for generating the final physical correlation scatter plots.
"""

import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. PASTE OPTUNA HYPERPARAMETERS HERE
# ==========================================
hyperparametros = {
    'batch_size': 256,
    'lr': 0.0001,
    'optimizer': 'Adam',
    'n_layers': 4,
    'n_units_l0': 1024,
    'dropout_l0': 0.2,
    'n_units_l1': 512,
    'dropout_l1': 0.4,
    'n_units_l2': 256,
    'dropout_l2': 0.3,
    'n_units_l3': 64,
    'dropout_l3': 0.3,
}

# ==========================================
# 2. DATA CONFIGURATION AND SPLITTING
# ==========================================
DATA_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\data_cpb"

print("\n--- 1. READING TRAIN, VALIDATION AND TEST DATA ---")
X_train_raw = pd.read_csv(os.path.join(DATA_DIR, "x_train.csv"))
y_train_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))

X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "x_test.csv"))
y_test_raw = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

colunas_alvo = ['C12', 'C13', 'C22', 'C23', 'C33', 'C66', 'sigma0', 'k', 'n']

X_full_train = X_train_raw.values
Y_full_train = y_train_raw[colunas_alvo].values
X_test_pd = X_test_raw.values
Y_test_pd = y_test_raw[colunas_alvo].values

# Extract 999 Validation samples from the Training data
X_train_pd, X_val_pd, Y_train_pd, Y_val_pd = train_test_split(
    X_full_train, Y_full_train, test_size=999, random_state=42
)

# ==========================================
# 3. SCALER AND PCA (150 Components)
# ==========================================
print("--- 2. APPLYING SCALER AND PCA ---")
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_pd)
X_val_scaled = scaler_x.transform(X_val_pd)
X_test_scaled = scaler_x.transform(X_test_pd)

pca = PCA(n_components=150)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train_pd)
Y_val_scaled = scaler_y.transform(Y_val_pd)
Y_test_scaled = scaler_y.transform(Y_test_pd)

dim_in = X_train_pca.shape[1] 
dim_out = Y_train_scaled.shape[1] 

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

dataset_treino = TensorDataset(X_train_tensor, Y_train_tensor)
dataloader_treino = DataLoader(dataset_treino, batch_size=hyperparametros['batch_size'], shuffle=True)

# ==========================================
# 4. BUILDING THE CHAMPION NETWORK
# ==========================================
print(f"--- 3. BUILDING NETWORK ({hyperparametros['n_layers']} layers) ---")
layers = []
in_features = dim_in

for i in range(hyperparametros['n_layers']):
    out_features = hyperparametros[f'n_units_l{i}']
    p_dropout = hyperparametros[f'dropout_l{i}']
    
    layers.append(nn.Linear(in_features, out_features))
    layers.append(nn.LayerNorm(out_features))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout(p_dropout))
    in_features = out_features

layers.append(nn.Linear(in_features, dim_out))
modelo = nn.Sequential(*layers)

# ==========================================
# 5. TRAINING WITH EARLY STOPPING (ON VALIDATION SET)
# ==========================================
print("--- 4. TRAINING THE MODEL ---")
criterio_loss = nn.MSELoss()

if hyperparametros['optimizer'] == 'Adam':
    otimizador = torch.optim.Adam(modelo.parameters(), lr=hyperparametros['lr'])
else:
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=hyperparametros['lr'])

best_val_loss = float('inf')
melhor_modelo_pesos = None
patience = 40
patience_counter = 0

historico_epochs = []
historico_loss_tr = []
historico_loss_val = []
historico_r2_tr = []
historico_r2_val = []

for epoch in range(400):
    modelo.train()
    for X_batch, Y_batch in dataloader_treino:
        otimizador.zero_grad()
        Y_prev = modelo(X_batch)
        loss = criterio_loss(Y_prev, Y_batch)
        loss.backward()
        otimizador.step()
        
    # Evaluation on Train and Validation (No Test data here!)
    modelo.eval()
    with torch.no_grad():
        # Train
        Y_train_prev_scaled = modelo(X_train_tensor)
        train_loss = criterio_loss(Y_train_prev_scaled, Y_train_tensor).item()
        r2_train = r2_score(Y_train_tensor.numpy(), Y_train_prev_scaled.numpy())
        
        # Validation
        Y_val_prev_scaled = modelo(X_val_tensor)
        val_loss = criterio_loss(Y_val_prev_scaled, Y_val_tensor).item()
        r2_val = r2_score(Y_val_tensor.numpy(), Y_val_prev_scaled.numpy())
        
    historico_epochs.append(epoch)
    historico_loss_tr.append(train_loss)
    historico_loss_val.append(val_loss)
    historico_r2_tr.append(r2_train)
    historico_r2_val.append(r2_val)
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        melhor_modelo_pesos = copy.deepcopy(modelo.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early Stopping triggered at epoch {epoch}.")
            break

# Restore best epoch weights based on validation loss
modelo.load_state_dict(melhor_modelo_pesos)

# ==========================================
# 6. TRAINING PLOTS (Loss and R2)
# ==========================================
plt.figure(figsize=(14, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(historico_epochs, historico_loss_tr, label='Train', color='green')
plt.plot(historico_epochs, historico_loss_val, label='Validation', color='red')
plt.title('Loss Evolution (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# R2 Plot
plt.subplot(1, 2, 2)
plt.plot(historico_epochs, historico_r2_tr, label='Train', color='green')
plt.plot(historico_epochs, historico_r2_val, label='Validation', color='red')
plt.title('Global $R^2$ Evolution')
plt.xlabel('Epochs')
plt.ylabel('$R^2$ Score')
plt.ylim([0, 1.0])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
caminho_grafico = os.path.join(DATA_DIR, "historico_treino_nn.png")
plt.savefig(caminho_grafico, dpi=300)
print(f"\n✅ Training evolution plot saved to: {caminho_grafico}")

# ==========================================
# 7. FINAL EVALUATION ON UNSEEN TEST SET
# ==========================================
print("\n==========================================")
print("[FINAL RESULTS - TRUE TEST SET]")
print("==========================================")
modelo.eval()
with torch.no_grad():
    Y_test_prev_scaled = modelo(X_test_tensor)
    Y_test_prev_fisico = scaler_y.inverse_transform(Y_test_prev_scaled.numpy())
    Y_test_real_fisico = Y_test_pd

    r2_global = r2_score(Y_test_real_fisico, Y_test_prev_fisico)
    mae_global = mean_absolute_error(Y_test_real_fisico, Y_test_prev_fisico)
    mape_global = mean_absolute_percentage_error(Y_test_real_fisico, Y_test_prev_fisico)

    print(f"R-squared (Global Test): {r2_global:.4f}")
    print(f"MAE (Global Test): {mae_global:.4f}")
    print(f"MAPE (Global Test): {mape_global:.4f}")

    print("\n[R-Squared per Parameter - Test]")
    for i, param in enumerate(colunas_alvo):
        r2_param = r2_score(Y_test_real_fisico[:, i], Y_test_prev_fisico[:, i])
        print(f" - {param}: {r2_param:.4f}")

    # Save Test predictions for plotting
    df_pred_nn = pd.DataFrame(Y_test_prev_fisico, columns=colunas_alvo)
    caminho_csv_nn = os.path.join(DATA_DIR, "y_pred_nn.csv")
    df_pred_nn.to_csv(caminho_csv_nn, index=False)
    print(f"\nTest predictions saved to: {caminho_csv_nn}")

    # Calculate Train R2 for Scatter plots
    Y_train_prev_scaled_final = modelo(X_train_tensor)
    # Save Train and Validation predictions for comprehensive plotting
    Y_train_prev_fisico_final = scaler_y.inverse_transform(modelo(X_train_tensor).detach().numpy())
    Y_val_prev_fisico_final = scaler_y.inverse_transform(modelo(X_val_tensor).detach().numpy())

    pd.DataFrame(Y_train_prev_fisico_final, columns=colunas_alvo).to_csv(os.path.join(DATA_DIR, "y_pred_train_nn.csv"), index=False)
    pd.DataFrame(Y_val_prev_fisico_final, columns=colunas_alvo).to_csv(os.path.join(DATA_DIR, "y_pred_val_nn.csv"), index=False)
    print("✅ NN Train and Validation predictions saved successfully!")
    
    print("\n--- COPY THIS DICT TO YOUR PLOTTING SCRIPT (r2_train_dict) ---")
    print("r2_train_dict = {")
    for i, param in enumerate(colunas_alvo):
        r2_tr = r2_score(Y_train_pd[:, i], Y_train_prev_fisico_final[:, i])
        print(f"    '{param}': {r2_tr:.4f},")
    print("}")
