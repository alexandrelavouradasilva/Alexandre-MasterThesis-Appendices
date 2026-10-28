"""
Training NN Model Optimization

This script implements a dynamic Artificial Neural Network (ANN) architecture using 
PyTorch, coupled with Bayesian Optimization via the Optuna framework, to establish 
an optimal baseline for the parameter identification problem. Building upon the 
data preprocessing foundation (PCA and Standard Scaling), this code introduces an 
advanced hyperparameter search space. Rather than relying on a static network 
topology, the Optuna study conducts 15,000 trials to dynamically select the optimal 
number of hidden layers, neurons per layer, dropout rates, learning rate, batch 
size, and optimizer (Adam vs. AdamW). 

To ensure computational efficiency, the optimization loop integrates an early stopping 
mechanism and a Median Pruner to automatically abort underperforming trials. Once 
the optimization concludes, the script reconstructs and fully trains the "champion" 
network, extracts the R-squared learning curves, and saves the final weights and 
optimized scalers for comparative evaluation against the other models.
"""

import os
import sys
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
import optuna  # <-- Bayesian Optimization Framework

# ==========================================
# 1. SETUP AND DATA SPLITTING (TRAIN / VALIDATION)
# ==========================================
DATA_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\data_cpb"
PASTA_MODELOS = os.path.join(DATA_DIR, "optuna_best_model")
if not os.path.exists(PASTA_MODELOS): os.makedirs(PASTA_MODELOS)

print(f"\n--- 1. READING TRAINING DATA AND SPLITTING VALIDATION SET ---")
try:
    # Read only training files! The real test set is kept separate.
    X_raw = pd.read_csv(os.path.join(DATA_DIR, "x_train.csv"))
    y_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
except Exception as e: 
    print(f"❌ Error reading files: {e}")
    sys.exit()

colunas_alvo = ['C12', 'C13', 'C22', 'C23', 'C33', 'C66', 'sigma0', 'k', 'n']
X_pandas = X_raw.values
Y_pandas = y_raw[colunas_alvo].values

# Hold out 999 samples for Validation
X_train_pd, X_val_pd, Y_train_pd, Y_val_pd = train_test_split(
    X_pandas, Y_pandas, test_size=999, random_state=42
)

print(f"✅ Splitting Complete! Train: {len(X_train_pd)} | Validation: {len(X_val_pd)}")

# ==========================================
# 2. PRE-PROCESSING (SCALER + PCA 150)
# ==========================================
print("\n--- 2. APPLYING SCALER AND PCA (150 Components) ---")
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_pd)
X_val_scaled = scaler_x.transform(X_val_pd)

pca = PCA(n_components=150)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train_pd)
Y_val_scaled = scaler_y.transform(Y_val_pd)

dim_in = X_train_pca.shape[1] 
dim_out = Y_train_scaled.shape[1] 

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32)

# ==========================================
# 3. DYNAMIC NETWORK DEFINITION FOR OPTUNA
# ==========================================
def criar_modelo(trial):
    layers = []
    in_features = dim_in
    
    # Optuna decides how many hidden layers to use (1 to 4)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    
    for i in range(n_layers):
        # Optuna decides the number of neurons per layer from a specific list
        out_features = trial.suggest_categorical(f"n_units_l{i}", [64, 128, 256, 512, 1024])
        # Optuna decides the dropout rate for this layer
        p_dropout = trial.suggest_float(f"dropout_l{i}", 0.05, 0.4, step=0.05)
        
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LayerNorm(out_features))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(p_dropout))
        
        in_features = out_features
        
    layers.append(nn.Linear(in_features, dim_out))
    
    return nn.Sequential(*layers)

# ==========================================
# 4. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # 1. Hyperparameters to test
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    
    # 2. Prepare DataLoader with the chosen Batch Size
    dataset_treino = TensorDataset(X_train_tensor, Y_train_tensor)
    dataloader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
    
    # 3. Build the Model
    modelo = criar_modelo(trial)
    criterio_loss = nn.MSELoss()
    
    if optimizer_name == "Adam":
        otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    else:
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Early Stopping Configuration
    patience = 40
    patience_counter = 0
    best_val_loss = float('inf')
    
    numero_epochs = 350 # No need for 1500 here because early stopping cuts it short
    
    # 5. Training Loop
    for epoch in range(numero_epochs):
        modelo.train()
        for X_batch, Y_batch in dataloader_treino:
            otimizador.zero_grad()
            Y_prev = modelo(X_batch)
            loss = criterio_loss(Y_prev, Y_batch)
            loss.backward()
            otimizador.step()
            
        # Validation evaluation (Early Stopping)
        modelo.eval()
        with torch.no_grad():
            Y_val_prev = modelo(X_val_tensor)
            val_loss = criterio_loss(Y_val_prev, Y_val_tensor).item()
            
        # Pruning: Abort the trial mid-way if it is performing poorly compared to others
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break # Stop training this specific network early

    return best_val_loss # Optuna will attempt to MINIMIZE this

# ==========================================
# 5. START OPTIMIZATION
# ==========================================
print("\n--- 3. STARTING OPTUNA (Searching for the best Network) ---")
# Create the study. Use MedianPruner to avoid wasting time on bad networks.
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

# Execute 15000 trials
study.optimize(objective, n_trials=15000)

print("\n--- OPTIMIZATION COMPLETE! ---")
best_trial = study.best_trial
print(f"Best Validation Loss: {best_trial.value:.4f}")
print("Best Hyperparameters found:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# ==========================================
# 6. TRAIN THE "CHAMPION" NETWORK TO EXTRACT PLOTS
# ==========================================
print("\n--- 4. TRAINING THE CHAMPION MODEL TO EXTRACT PLOTS ---")

# Rebuild the best model found
melhor_batch_size = best_trial.params["batch_size"]
melhor_lr = best_trial.params["lr"]

dataset_treino = TensorDataset(X_train_tensor, Y_train_tensor)
dataloader_treino = DataLoader(dataset_treino, batch_size=melhor_batch_size, shuffle=True)

modelo_final = criar_modelo(best_trial)
criterio_loss = nn.MSELoss()

if best_trial.params["optimizer"] == "Adam":
    otimizador = torch.optim.Adam(modelo_final.parameters(), lr=melhor_lr)
else:
    otimizador = torch.optim.AdamW(modelo_final.parameters(), lr=melhor_lr, weight_decay=best_trial.params["weight_decay"])

historico_epochs, historico_r2_treino, historico_r2_val = [], [], []

best_val_loss = float('inf')
melhor_modelo_pesos = None
patience = 30
patience_counter = 0

for epoch in range(500):
    modelo_final.train()
    for X_batch, Y_batch in dataloader_treino:
        otimizador.zero_grad()
        Y_prev = modelo_final(X_batch)
        loss = criterio_loss(Y_prev, Y_batch)
        loss.backward()
        otimizador.step()
        
    # Calculate R2 at the end of the epoch
    modelo_final.eval()
    with torch.no_grad():
        Y_val_prev = modelo_final(X_val_tensor)
        val_loss = criterio_loss(Y_val_prev, Y_val_tensor).item()
        
        # Validation R2
        r2_val = r2_score(Y_val_tensor.numpy(), Y_val_prev.numpy())
        
        # Training R2 (calculated on the entire block for the plot)
        Y_treino_prev = modelo_final(X_train_tensor)
        r2_treino = r2_score(Y_train_tensor.numpy(), Y_treino_prev.numpy())
        
    historico_epochs.append(epoch)
    historico_r2_treino.append(r2_treino)
    historico_r2_val.append(r2_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        melhor_modelo_pesos = copy.deepcopy(modelo_final.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early Stopping triggered on final training at epoch {epoch}.")
            break

modelo_final.load_state_dict(melhor_modelo_pesos)

# ==========================================
# 7. SAVE EVERYTHING
# ==========================================
# R^2 Plot
plt.figure(figsize=(10, 5))
plt.plot(historico_epochs, historico_r2_treino, label='Train', color='green')
plt.plot(historico_epochs, historico_r2_val, label='Validation', color='red')
plt.title(f'Global $R^2$ Evolution - Best Optuna Model')
plt.xlabel('Epochs')
plt.ylabel('$R^2$ Score')
plt.ylim([0, 1.0])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(PASTA_MODELOS, "evolucao_r2_optuna.png"), dpi=300, bbox_inches='tight')
plt.close()

torch.save(modelo_final.state_dict(), os.path.join(PASTA_MODELOS, "Melhor_Modelo_Optuna.pth"))
with open(os.path.join(PASTA_MODELOS, "scaler_x.pkl"), "wb") as f: pickle.dump(scaler_x, f)
with open(os.path.join(PASTA_MODELOS, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)
with open(os.path.join(PASTA_MODELOS, "pca.pkl"), "wb") as f: pickle.dump(pca, f)

# Save a txt with the best parameters
with open(os.path.join(PASTA_MODELOS, "melhores_hiperparametros.txt"), "w") as f:
    f.write(f"Best final Validation R2: {max(historico_r2_val):.4f}\n")
    for k, v in best_trial.params.items():
        f.write(f"{k}: {v}\n")

print(f"✅ Success! Best model and plots saved in the folder: {PASTA_MODELOS}")
