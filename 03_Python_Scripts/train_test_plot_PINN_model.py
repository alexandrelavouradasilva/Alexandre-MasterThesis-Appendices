"""
Training, Testing and Plot of PINN Model

This script follows the same core training, evaluation, and plotting pipeline as 
the baseline ANN model, but introduces two critical modifications to establish 
the final Physics-Informed Neural Network (PINN). First, it implements feature 
space enrichment by adding the shear test data alongside the cruciform data. 
Second, it replaces the purely data-driven optimization with a physics-guided 
composite loss function. This custom function computes real-time CPB'06 yield 
conditions and Swift hardening evolution directly on the unscaled network 
predictions. 

All other procedures, including the Optuna hyperparameter topology, early-stopping 
safeguards, and parity plot generation, remain identical to the baseline approach, 
with the distinction that the final plotting is executed within this same script.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS AND PATHS
# ==========================================
BASE_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main"
DATA_DIR = os.path.join(BASE_DIR, "data", "data_final_pinn")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "pinn_final_plots")

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

colunas_alvo = ['C12', 'C13', 'C22', 'C23', 'C33', 'C66', 'sigma0', 'k', 'n']

# ==========================================
# 2. PHYSICAL FUNCTIONS (CPB06 & SWIFT)
# ==========================================
def calcular_tensoes_cedencia_direcionais(C_matrix):
    C11 = torch.ones_like(C_matrix[:, 0])
    C12 = C_matrix[:, 0]; C13 = C_matrix[:, 1]
    C22 = C_matrix[:, 2]; C23 = C_matrix[:, 3]
    C33 = C_matrix[:, 4]; C66 = C_matrix[:, 5]

    U1 = (2/3)*C11 - (1/3)*C12 - (1/3)*C13
    U2 = (2/3)*C12 - (1/3)*C22 - (1/3)*C23
    U3 = (2/3)*C13 - (1/3)*C23 - (1/3)*C33
    XX_yield = 1.0 / torch.sqrt(U1**2 + U2**2 + U3**2 + 1e-6)
    
    W1 = -(1/3)*C11 + (2/3)*C12 - (1/3)*C13
    W2 = -(1/3)*C12 + (2/3)*C22 - (1/3)*C23
    W3 = -(1/3)*C13 + (2/3)*C23 - (1/3)*C33
    YY_yield = 1.0 / torch.sqrt(W1**2 + W2**2 + W3**2 + 1e-6)
    
    Biax_yield = 1.0 / torch.sqrt((U1+W1)**2 + (U2+W2)**2 + (U3+W3)**2 + 1e-6)
    Shear_yield = 1.0 / (torch.sqrt(torch.tensor(2.0)) * C66 + 1e-6)
    
    return XX_yield, YY_yield, Biax_yield, Shear_yield

def calcular_losses_pinn_completa(y_prev_scaled, y_real_scaled, medias, desvios, eps_tensor):
    y_prev_fisico = (y_prev_scaled * desvios) + medias
    y_real_fisico = (y_real_scaled * desvios) + medias

    # --- SWIFT LAW CONSTRAINTS ---
    s0_p = torch.clamp(y_prev_fisico[:, 6], min=0.001)
    K_p  = torch.clamp(y_prev_fisico[:, 7], min=1.0)
    n_p  = torch.clamp(y_prev_fisico[:, 8], min=0.05, max=1.0)
    s0_p = torch.min(s0_p, K_p * 0.99)
    
    s0_r = torch.clamp(y_real_fisico[:, 6], min=0.001)
    K_r  = torch.clamp(y_real_fisico[:, 7], min=1.0)
    n_r  = torch.clamp(y_real_fisico[:, 8], min=0.05, max=1.0)
    s0_r = torch.min(s0_r, K_r * 0.99)
    
    e0_p = torch.pow((s0_p / K_p), (1.0 / n_p))
    e0_r = torch.pow((s0_r / K_r), (1.0 / n_r))
    
    curva_p = K_p.unsqueeze(1) * torch.pow((e0_p.unsqueeze(1) + eps_tensor + 1e-6), n_p.unsqueeze(1))
    curva_r = K_r.unsqueeze(1) * torch.pow((e0_r.unsqueeze(1) + eps_tensor + 1e-6), n_r.unsqueeze(1))
    
    loss_swift = nn.functional.mse_loss(curva_p / 1000.0, curva_r / 1000.0)
    loss_limites_n = torch.mean(torch.relu(n_p - 1.0) + torch.relu(-n_p))

    # --- CPB06 YIELD CRITERION CONSTRAINTS ---
    C_matrix_p = y_prev_fisico[:, 0:6]
    C_matrix_r = y_real_fisico[:, 0:6]
    
    XX_p, YY_p, Biax_p, Shear_p = calcular_tensoes_cedencia_direcionais(C_matrix_p)
    XX_r, YY_r, Biax_r, Shear_r = calcular_tensoes_cedencia_direcionais(C_matrix_r)
    
    loss_cpb06 = (nn.functional.smooth_l1_loss(XX_p, XX_r) + 
                  nn.functional.smooth_l1_loss(YY_p, YY_r) + 
                  nn.functional.smooth_l1_loss(Biax_p, Biax_r) + 
                  nn.functional.smooth_l1_loss(Shear_p, Shear_r))
                  
    return loss_swift, loss_cpb06, loss_limites_n

# ==========================================
# 3. ROBUST DATA PREPARATION (CRUCI + SHEAR)
# ==========================================
def preparar_dados_cruci_shear():
    print("Loading and processing data (Cruciform + Shear)...")
    y_train_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))[colunas_alvo].values
    y_test_raw = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))[colunas_alvo].values

    X_cruci_train_raw = pd.read_csv(os.path.join(DATA_DIR, "x_cruciforme_train.csv"), header=None).values
    X_cruci_test_raw = pd.read_csv(os.path.join(DATA_DIR, "x_cruciforme_test.csv"), header=None).values
    
    X_shear_train_raw = pd.read_csv(os.path.join(DATA_DIR, "x_shear_train.csv"), header=None).values
    X_shear_test_raw = pd.read_csv(os.path.join(DATA_DIR, "x_shear_test.csv"), header=None).values

    # Ensure all datasets have matching dimensions
    tamanho_minimo = min(len(y_train_raw), len(X_cruci_train_raw), len(X_shear_train_raw))
    y_train_raw = y_train_raw[:tamanho_minimo]
    X_cruci_train_raw = X_cruci_train_raw[:tamanho_minimo]
    X_shear_train_raw = X_shear_train_raw[:tamanho_minimo]

    # Split for Validation
    indices = np.arange(tamanho_minimo)
    idx_train, idx_val = train_test_split(indices, test_size=897, random_state=42)

    Y_train, Y_val, Y_test = y_train_raw[idx_train], y_train_raw[idx_val], y_test_raw

    # Scale Target Values
    scaler_y = StandardScaler()
    Y_train_s = scaler_y.fit_transform(Y_train)
    Y_val_s = scaler_y.transform(Y_val)
    Y_test_s = scaler_y.transform(Y_test)

    # Process Cruciform Data (Scale + PCA)
    X_cruci_train, X_cruci_val = X_cruci_train_raw[idx_train], X_cruci_train_raw[idx_val]
    scaler_cruci = StandardScaler()
    X_cruci_train_s = scaler_cruci.fit_transform(X_cruci_train)
    X_cruci_val_s = scaler_cruci.transform(X_cruci_val)
    X_cruci_test_s = scaler_cruci.transform(X_cruci_test_raw)

    pca = PCA(n_components=150)
    X_final_train = pca.fit_transform(X_cruci_train_s)
    X_final_val = pca.transform(X_cruci_val_s)
    X_final_test = pca.transform(X_cruci_test_s)

    # Process Shear Data (Scale) and concatenate horizontally (Feature Enrichement)
    X_shear_train, X_shear_val = X_shear_train_raw[idx_train], X_shear_train_raw[idx_val]
    scaler_shear = StandardScaler()
    X_final_train = np.hstack((X_final_train, scaler_shear.fit_transform(X_shear_train)))
    X_final_val = np.hstack((X_final_val, scaler_shear.transform(X_shear_val)))
    X_final_test = np.hstack((X_final_test, scaler_shear.transform(X_shear_test_raw)))

    return {
        'X_train': torch.tensor(X_final_train, dtype=torch.float32),
        'Y_train': torch.tensor(Y_train_s, dtype=torch.float32),
        'X_val': torch.tensor(X_final_val, dtype=torch.float32),
        'Y_val': torch.tensor(Y_val_s, dtype=torch.float32),
        'X_test': torch.tensor(X_final_test, dtype=torch.float32),
        'Y_test_real': Y_test
    }, scaler_y

tensors, scaler_y = preparar_dados_cruci_shear()
dataset_treino = TensorDataset(tensors['X_train'], tensors['Y_train'])
medias_y = torch.tensor(scaler_y.mean_, dtype=torch.float32)
desvios_y = torch.tensor(scaler_y.scale_, dtype=torch.float32)
eps_tensor = torch.linspace(0.0, 1.0, steps=100, dtype=torch.float32).unsqueeze(0)
dim_in = tensors['X_train'].shape[1]

# ==========================================
# 4. FULL PINN TRAINING
# ==========================================
# Loss Function Weights
W_dados = 1.0
W_cpb = 0.0
W_swift = 6.0

print(f"\n=======================================================")
print(f"TRAINING PINN: W_dados={W_dados} | W_cpb={W_cpb} | W_swift={W_swift}")
print(f"=======================================================")

# Network Architecture
modelo = nn.Sequential(
    nn.Linear(dim_in, 1024), nn.LayerNorm(1024), nn.SiLU(), nn.Dropout(0.2),
    nn.Linear(1024, 512),    nn.LayerNorm(512),  nn.SiLU(), nn.Dropout(0.4),
    nn.Linear(512, 256),     nn.LayerNorm(256),  nn.SiLU(), nn.Dropout(0.3),
    nn.Linear(256, 64),      nn.LayerNorm(64),   nn.SiLU(), nn.Dropout(0.3),
    nn.Linear(64, 9)
)

otimizador = torch.optim.Adam(modelo.parameters(), lr=0.0001)
dataloader_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)

best_val_loss = float('inf')
patience, patience_counter = 400, 0
melhor_modelo_pesos = None

for epoch in range(4000):
    modelo.train()
    
    acum_loss_dados, acum_loss_cpb, acum_loss_swift = 0.0, 0.0, 0.0
    
    for X_batch, Y_batch in dataloader_treino:
        otimizador.zero_grad()
        Y_prev = modelo(X_batch)
        
        # Calculate individual loss components
        loss_dados_crua = nn.functional.mse_loss(Y_prev, Y_batch)
        loss_swift_crua, loss_cpb_crua, loss_n = calcular_losses_pinn_completa(Y_prev, Y_batch, medias_y, desvios_y, eps_tensor)
        
        # Combine losses using defined weights
        loss_total = (W_dados * loss_dados_crua) + (W_cpb * loss_cpb_crua) + (W_swift * loss_swift_crua) + (10.0 * loss_n)
            
        loss_total.backward()
        otimizador.step()
        
        acum_loss_dados += loss_dados_crua.item()
        acum_loss_cpb += loss_cpb_crua.item()
        acum_loss_swift += loss_swift_crua.item()
        
    modelo.eval()
    with torch.no_grad():
        Y_val_prev = modelo(tensors['X_val'])
        val_loss = nn.functional.mse_loss(Y_val_prev, tensors['Y_val']).item()
        r2_val_current = r2_score(tensors['Y_val'].numpy(), Y_val_prev.numpy())
        
    if epoch % 10 == 0:
        nb = len(dataloader_treino)
        print(f"Epoch {epoch:03d} | L_Data: {acum_loss_dados/nb:.4f} | L_CPB: {acum_loss_cpb/nb:.4f} | L_Swift: {acum_loss_swift/nb:.4f} || R² Val: {r2_val_current:.4f}")
        
    # Early Stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        melhor_modelo_pesos = copy.deepcopy(modelo.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early Stopping triggered at epoch {epoch}.")
            break

# ==========================================
# 5. FINAL EVALUATION ON UNSEEN TEST SET
# ==========================================
modelo.load_state_dict(melhor_modelo_pesos)
modelo.eval()
with torch.no_grad():
    Y_test_prev_scaled = modelo(tensors['X_test']).numpy()
    Y_test_prev = scaler_y.inverse_transform(Y_test_prev_scaled)
    
    Y_test_true = tensors['Y_test_real']
    r2_global = r2_score(Y_test_true, Y_test_prev)
    print("\n==========================================")
    print(f"FINAL GLOBAL R-SQUARED (Full PINN): {r2_global:.4f}")
    print("==========================================")
    
    print("\n[R-Squared per Parameter]")
    for i, param in enumerate(colunas_alvo):
        r2_param = r2_score(Y_test_true[:, i], Y_test_prev[:, i])
        print(f" - {param}: {r2_param:.4f}")

# ==========================================
# 6. PARITY PLOTS GENERATION (LATEX)
# ==========================================
print("\n--- Generating Parity Plots for the PINN... ---")

with torch.no_grad():
    Y_tr_prev_scaled = modelo(tensors['X_train']).numpy()
    Y_val_prev_scaled = modelo(tensors['X_val']).numpy()

# Inverse transform to physical space for plotting
y_train_pred = scaler_y.inverse_transform(Y_tr_prev_scaled)
y_val_pred = scaler_y.inverse_transform(Y_val_prev_scaled)

y_train_true = scaler_y.inverse_transform(tensors['Y_train'].numpy())
y_val_true = scaler_y.inverse_transform(tensors['Y_val'].numpy())

labels_map = {
    'C12': r'$C_{12}$', 'C13': r'$C_{13}$', 'C22': r'$C_{22}$',
    'C23': r'$C_{23}$', 'C33': r'$C_{33}$', 'C66': r'$C_{66}$',
    'sigma0': r'$Y_0$', 'k': r'$K$', 'n': r'$n$'
}

plt.rcParams.update({
    "font.family": "serif",  
    "font.serif": ["Times New Roman"],
    "text.usetex": True
})

with plt.rc_context({'font.size': 13}):
    fig, axes = plt.subplots(3, 3, figsize=(18, 21))

    for ax, name in zip(axes.flat, colunas_alvo):
        i = colunas_alvo.index(name)

        true_train = y_train_true[:, i]
        pred_train = y_train_pred[:, i]
        
        true_val = y_val_true[:, i]
        pred_val = y_val_pred[:, i]

        true_test = Y_test_true[:, i]
        pred_test = Y_test_prev[:, i]

        r2_train_i = r2_score(true_train, pred_train)
        r2_val_i = r2_score(true_val, pred_val)
        r2_test_i = r2_score(true_test, pred_test)

        ax.scatter(true_train, pred_train, alpha=0.3, s=8,  color='#d3d3d3', label='Train', zorder=1)
        ax.scatter(true_val,   pred_val,   alpha=0.5, s=8,  color='#808080', label='Validation', zorder=2)
        ax.scatter(true_test,  pred_test,  alpha=0.7, s=15, color='black',   label='Test', zorder=3)
        
        all_vals = np.concatenate([true_train, pred_train, true_val, pred_val, true_test, pred_test])
        lo, hi = all_vals.min(), all_vals.max()
        margem = (hi - lo) * 0.05
        lo, hi = lo - margem, hi + margem
        
        ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=1.5, zorder=4)

        ax.text(0.05, 0.95, labels_map.get(name, name),
                transform=ax.transAxes,
                fontsize=32, fontweight='bold',
                verticalalignment='top', horizontalalignment='left')

        ax.set_title(
            f'$R^2$: train={r2_train_i:.2f} $|$ val={r2_val_i:.2f} $|$ test={r2_test_i:.2f}',
            fontsize=20, pad=10
        )
        
        ax.set_xlabel('True', fontsize=25)
        ax.set_ylabel('Predicted', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.patch.set_alpha(0)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.015), fontsize=25,
               framealpha=0, frameon=True)
    
    plt.tight_layout()
    
    save_path_png = os.path.join(PLOTS_DIR, 'parity_plots_PINN.png')
    save_path_pdf = os.path.join(PLOTS_DIR, 'parity_plots_PINN.pdf')
    
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', transparent=True)
    
    print(f"✅ Plots successfully saved to:\n - {save_path_png}\n - {save_path_pdf}")
