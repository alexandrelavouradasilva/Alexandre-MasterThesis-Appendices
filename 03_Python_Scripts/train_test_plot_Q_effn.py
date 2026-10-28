"""
Training NN Model and Plot for Q_eff,n

This script implements the first final advanced methodology of this research: 
predicting the parameters of the CPB'06 yield criterion within its canonical space. 
It maintains the core PyTorch network architecture, the feature space enrichment 
(merging cruciform and simple shear responses), and the established Optuna 
hyperparameters from previous models. 

However, instead of mapping the mechanical responses to the highly coupled original 
anisotropic matrix, the script actively converts the training targets into their 
canonical variables (q12, q23, q66) prior to training, drastically reducing 
mathematical collinearity. After the network evaluates the unseen test set, a 
custom decoding algorithm rebuilds the original anisotropic parameter matrix (C) 
from the predicted canonical outputs. This inverse transformation enables the 
automatic generation of a new, fully predictable Design of Experiments (DoE), 
which can be directly plugged into the finite element software for the final 
macroscopic validation phase.
"""

# -*- coding: utf-8 -*-
# Importe as suas bibliotecas aqui...
import torch
import pandas as pd
import numpy as np
# ==========================================
# 1. PATHS AND VARIABLES
# ...

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, radians, copysign

# ==========================================
# 1. HYPERPARAMETERS AND PATHS
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

NPZ_FILE = (
    r"C:\AbaqusVM\djcruz\DOE_a2_CorrigidoAgosto"
    r"\data\data_CPB06_a2_corrigidoAgosto.npz"
)

SHEAR_Y_FILE = (
    r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre"
    r"\teseMestradoAlexandre\mlpcp-interp-main"
    r"\data\processedshear\y_compiled_shear.csv"
)

SHEAR_X_FILE = (
    r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre"
    r"\teseMestradoAlexandre\mlpcp-interp-main"
    r"\data\processedshear\x_compiled_shear.csv"
)

DOE_OUTPUT_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\doe"
if not os.path.exists(DOE_OUTPUT_DIR):
    os.makedirs(DOE_OUTPUT_DIR)

colunas_alvo = ['q12', 'q23', 'q66', 'sigma0', 'K', 'n']

# ==========================================
# 2. CANONIZATION AND DECANONIZATION FUNCTIONS
# ==========================================
def compute_H(C, a, k):
    """Computes the H tensor used in the canonical transformation."""
    phi1 = (2*C[0,0] - C[0,1] - C[0,2]) / 3.0
    phi2 = (2*C[1,0] - C[1,1] - C[1,2]) / 3.0
    phi3 = (2*C[2,0] - C[2,1] - C[2,2]) / 3.0
    B = ((abs(phi1) - k*phi1)**a + (abs(phi2) - k*phi2)**a + (abs(phi3) - k*phi3)**a)**(-1.0/a)
    H = (B**2) * (C @ C)
    return H, B

def canonize_targets(params, target_names):
    """Transforms the original anisotropic parameters into the canonical space Q."""
    P = (1.0/3.0) * np.array([[ 2., -1., -1.], [-1.,  2., -1.], [-1., -1.,  2.]])
    idx_map = {name: target_names.index(name) for name in target_names}
    q_targets = np.zeros((len(params), 6))
    
    for i in range(len(params)):
        C = np.array([
            [params[i, idx_map['C11']], params[i, idx_map['C12']], params[i, idx_map['C13']], 0., 0., 0.],
            [params[i, idx_map['C12']], params[i, idx_map['C22']], params[i, idx_map['C23']], 0., 0., 0.],
            [params[i, idx_map['C13']], params[i, idx_map['C23']], params[i, idx_map['C33']], 0., 0., 0.],
            [0., 0., 0., params[i, idx_map['C44']], 0., 0.],
            [0., 0., 0., 0., params[i, idx_map['C55']], 0.],
            [0., 0., 0., 0., 0., params[i, idx_map['C66']]]
        ])
        
        a = params[i, idx_map['a']]
        k = params[i, idx_map['k2006']] if 'k2006' in idx_map else params[i, idx_map.get('k', 0)]
        
        H, _ = compute_H(C, a, k)
        Hn = H[:3, :3]
        Q = P @ Hn @ P
        
        q_targets[i] = [
            Q[0,1], Q[1,2], H[5,5],
            params[i, idx_map['sigma0']], params[i, idx_map['K']], params[i, idx_map['n']]
        ]
    return q_targets

def canonical_C_from_q(q12, q23, q66):
    """Reconstructs the original C parameters from the predicted canonical q variables."""
    q11 = 1.0
    q13 = -1.0 - q12
    q22 = -q12 - q23
    q33 = 1.0 + q12 - q23
    Q = np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]])
    
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.clip(eigvals, 0.0, None)
    C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    C66 = np.sqrt(abs(q66))
    return np.array([C[0,0], C[0,1], C[0,2], C[1,1], C[1,2], C[2,2], C66])

# ==========================================
# 3. ANALYTICAL CPB'06 FUNCTIONS 
# ==========================================
THETAS_DEG = np.arange(0, 91, 1).round(1).tolist()

def cpb2006_2d(sxx, syy, sxy, params, k, a):
    """Computes the CPB'06 yield function and its derivatives in 2D."""
    C11,C22,C33,C66,C12,C13,C23 = params
    X1 = (C13-C12-2*C11+C22+C23)/6*sxx + (C11-C12+C13-2*C22+C23)/6*syy
    A  = (2*C11-3*C12-C13+C22+C23)/6*sxx + (-C11+3*C12-C13-2*C22+C23)/6*syy
    B  = C66*sxy
    disc = sqrt(A**2+B**2) if (A**2+B**2)>0 else 0.
    phi1=X1+disc; phi2=X1-disc
    omega = (C33-2*C13+C23)/3*sxx + (C33+C13-2*C23)/3*syy
    
    def g(p): return (abs(p)+k*p)**a
    def dg(p):
        if abs(p)<1e-15: return 0.
        return a*(copysign(1,p)+k)*(abs(p)+k*p)**(a-1)
        
    F = g(phi1)+g(phi2)+g(omega)
    dX1=[(C13-C12-2*C11+C22+C23)/6,(C11-C12+C13-2*C22+C23)/6]
    dA =[(2*C11-3*C12-C13+C22+C23)/6,(-C11+3*C12-C13-2*C22+C23)/6]
    dom=[(C33-2*C13+C23)/3,(C33+C13-2*C23)/3]
    dF=[0.,0.,0.]
    
    for j in range(3):
        if j<2:
            dd=A*dA[j]/disc if disc>1e-15 else 0.
            dp1=dX1[j]+dd; dp2=dX1[j]-dd; do_=dom[j]
        else:
            dd=B*C66/disc if disc>1e-15 else 0.
            dp1=dd; dp2=-dd; do_=0.
        dF[j]=dg(phi1)*dp1+dg(phi2)*dp2+dg(omega)*do_
    return F, dF
 
def compute_aniso(params, k, a):
    """Calculates directional yield stresses and r-values."""
    F_ref, _ = cpb2006_2d(1., 0., 0., params, k, a)
    D_ref = 2 * F_ref
    if D_ref < 1e-15: return None, None
    
    sy_list, r_list = [], []
    for theta_deg in THETAS_DEG:
        th = radians(theta_deg)
        sxx=cos(th)**2; syy=sin(th)**2; sxy=sin(th)*cos(th)
        F1, dF1 = cpb2006_2d(sxx, syy, sxy, params, k, a)
        F=2*F1; dF_xx=2*dF1[0]; dF_yy=2*dF1[1]; dF_xy=2*dF1[2]
        se = (F/D_ref)**(1./a)
        if se < 1e-15 or not np.isfinite(se): return None, None
        
        sy_list.append(1./se)
        Ezz = -(dF_xx+dF_yy)
        if abs(Ezz) < 1e-10: return None, None
        
        Eyy_p = dF_xx*syy + dF_yy*sxx - dF_xy*sxy
        r = Eyy_p/Ezz
        if not np.isfinite(r): return None, None
        r_list.append(r)
        
    return sy_list, r_list

# ==========================================
# 4. LOAD AND MATCH DATA
# ==========================================
print(f"\n--- 1. LOADING DATA AND MATCHING SETS ---")
data = np.load(NPZ_FILE, allow_pickle=True)
y_train_orig = data['y_train']
x_train_cruci = data['X_train']
y_test_orig = data['y_test']
x_test_cruci = data['X_test']
target_names = list(data['target_names'])

df_y_shear = pd.read_csv(SHEAR_Y_FILE)
df_x_shear = pd.read_csv(SHEAR_X_FILE, header=None)

df_y_shear.columns = df_y_shear.columns.str.strip()
if 'k' in df_y_shear.columns and 'K' not in df_y_shear.columns:
    df_y_shear.rename(columns={'k': 'K'}, inplace=True)

colunas_originais = ['C11', 'C12', 'C13', 'C21', 'C22', 'C23', 'C31', 'C32', 'C33', 'C44', 'C55', 'C66', 'sigma0', 'K', 'n']
idx_map_orig = {name: target_names.index(name) for name in target_names}
indices_orig = [idx_map_orig[c] for c in colunas_originais]

def extrair_matches(y_npz, x_cruci_npz):
    """Matches the cruciform datasets with the shear datasets based on minimum distance."""
    y_npz_subset = y_npz[:, indices_orig]
    y_shear_subset = df_y_shear[colunas_originais].values
    
    scaler = StandardScaler()
    y_shear_scaled = scaler.fit_transform(y_shear_subset)
    y_npz_scaled = scaler.transform(y_npz_subset)
    
    valid_x_cruci = []
    valid_x_shear = []
    valid_y = []
    
    for i in range(len(y_npz_scaled)):
        distancias = np.linalg.norm(y_shear_scaled - y_npz_scaled[i], axis=1)
        best_idx = np.argmin(distancias)
        # Threshold to ensure datasets represent the exact same material
        if distancias[best_idx] < 0.1: 
            valid_x_cruci.append(x_cruci_npz[i])
            valid_x_shear.append(df_x_shear.values[best_idx])
            valid_y.append(y_npz[i])
            
    return np.array(valid_x_cruci), np.array(valid_x_shear), np.array(valid_y)

X_train_cruci_m, X_train_shear_m, y_train_m = extrair_matches(y_train_orig, x_train_cruci)
X_test_cruci_m, X_test_shear_m, y_test_m = extrair_matches(y_test_orig, x_test_cruci)
print(f"Train Matches: {len(y_train_m)} samples | Test Matches: {len(y_test_m)} samples")

# Transform Original Targets into Canonical Space
y_train_canon_m = canonize_targets(y_train_m, target_names)
y_test_canon_m = canonize_targets(y_test_m, target_names)

# Keep the original labels to rebuild the DoE at the end
y_test_orig_m = y_test_m

indices = np.arange(len(y_train_canon_m))
idx_tr, idx_val = train_test_split(indices, test_size=750, random_state=42)

Y_tr, Y_val = y_train_canon_m[idx_tr], y_train_canon_m[idx_val]
X_cruci_tr, X_cruci_val = X_train_cruci_m[idx_tr], X_train_cruci_m[idx_val]
X_shear_tr, X_shear_val = X_train_shear_m[idx_tr], X_train_shear_m[idx_val]

# --- SCALER AND PCA ---
print("\n--- 2. APPLYING SCALER AND PCA ---")
scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_tr)
Y_val_scaled = scaler_y.transform(Y_val)
Y_test_scaled = scaler_y.transform(y_test_canon_m)

scaler_cruci = StandardScaler()
X_cruci_tr_s = scaler_cruci.fit_transform(X_cruci_tr)
X_cruci_val_s = scaler_cruci.transform(X_cruci_val)
X_cruci_te_s = scaler_cruci.transform(X_test_cruci_m)

pca = PCA(n_components=150)
X_cruci_tr_pca = pca.fit_transform(X_cruci_tr_s)
X_cruci_val_pca = pca.transform(X_cruci_val_s)
X_cruci_te_pca = pca.transform(X_cruci_te_s)

scaler_shear = StandardScaler()
X_shear_tr_s = scaler_shear.fit_transform(X_shear_tr)
X_shear_val_s = scaler_shear.transform(X_shear_val)
X_shear_te_s = scaler_shear.transform(X_test_shear_m)

X_final_tr = np.hstack((X_cruci_tr_pca, X_shear_tr_s))
X_final_val = np.hstack((X_cruci_val_pca, X_shear_val_s))
X_final_te = np.hstack((X_cruci_te_pca, X_shear_te_s))

dim_in = X_final_tr.shape[1] 
dim_out = Y_train_scaled.shape[1] 

X_train_tensor = torch.tensor(X_final_tr, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_final_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_final_te, dtype=torch.float32)

dataset_treino = TensorDataset(X_train_tensor, Y_train_tensor)
dataloader_treino = DataLoader(dataset_treino, batch_size=hyperparametros['batch_size'], shuffle=True)

# ==========================================
# 5. PYTORCH NETWORK BUILDING AND TRAINING
# ==========================================
print(f"\n--- 3. BUILDING AND TRAINING NETWORK ({hyperparametros['n_layers']} layers) ---")
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

criterio_loss = nn.MSELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=hyperparametros['lr'])

best_val_loss = float('inf')
melhor_modelo_pesos = None
patience = 250
patience_counter = 0

for epoch in range(4000):
    modelo.train()
    for X_batch, Y_batch in dataloader_treino:
        otimizador.zero_grad()
        loss = criterio_loss(modelo(X_batch), Y_batch)
        loss.backward()
        otimizador.step()
        
    modelo.eval()
    with torch.no_grad():
        Y_train_prev_scaled = modelo(X_train_tensor)
        Y_val_prev_scaled = modelo(X_val_tensor)
        
        train_loss = criterio_loss(Y_train_prev_scaled, Y_train_tensor).item()
        val_loss = criterio_loss(Y_val_prev_scaled, Y_val_tensor).item()
        
        r2_global_tr = r2_score(Y_train_tensor.numpy(), Y_train_prev_scaled.numpy())
        r2_global_va = r2_score(Y_val_tensor.numpy(), Y_val_prev_scaled.numpy())
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Loss Tr: {train_loss:.4f} | Loss Va: {val_loss:.4f} | R2 Tr: {r2_global_tr:.4f} | R2 Va: {r2_global_va:.4f}")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        melhor_modelo_pesos = copy.deepcopy(modelo.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n�� Early Stopping triggered at epoch {epoch+1}.")
            break

modelo.load_state_dict(melhor_modelo_pesos)
print("\n✅ Training complete. Best weights restored!")

# ==========================================
# 6. TEST PARITY PLOTS (PAPER STYLE)
# ==========================================
print("\n--- 4. TEST SET EVALUATION AND PARITY PLOTS ---")
modelo.eval()
with torch.no_grad():
    Y_train_prev = scaler_y.inverse_transform(modelo(X_train_tensor).numpy())
    Y_val_prev = scaler_y.inverse_transform(modelo(X_val_tensor).numpy())
    Y_test_prev = scaler_y.inverse_transform(modelo(X_test_tensor).numpy())

r2_global_final = r2_score(y_test_canon_m, Y_test_prev)
print(f"\n==============================================")
print(f" FINAL GLOBAL R-SQUARED: {r2_global_final:.4f}")
print(f"==============================================\n")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14,
    "mathtext.fontset": "cm", "text.usetex": False,
    "axes.linewidth": 0.8, "axes.facecolor": (1,1,1), "figure.facecolor": (1,1,1)
})

labels_map = {'q12': r'$q_{12}$', 'q23': r'$q_{23}$', 'q66': r'$q_{66}$', 'sigma0': r'$Y_0$', 'K': r'$K$', 'n': r'$n$'}
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, param in enumerate(colunas_alvo):
    ax = axes[i]
    r2_tr = r2_score(Y_tr[:, i], Y_train_prev[:, i])
    r2_va = r2_score(Y_val[:, i], Y_val_prev[:, i])
    r2_te = r2_score(y_test_canon_m[:, i], Y_test_prev[:, i])
    
    all_vals = np.concatenate([Y_tr[:, i], y_test_canon_m[:, i]])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    lo, hi = lo - margin, hi + margin
    
    ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=0.8, zorder=1)
    
    scatter_tr = ax.scatter(Y_tr[:, i], Y_train_prev[:, i], alpha=0.2, s=8, color='#d3d3d3', label='Train', zorder=2)
    scatter_va = ax.scatter(Y_val[:, i], Y_val_prev[:, i], alpha=0.4, s=10, color='#808080', label='Validation', zorder=3)
    scatter_te = ax.scatter(y_test_canon_m[:, i], Y_test_prev[:, i], alpha=0.6, s=12, color='black', label='Test', zorder=4)
    
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(rf"$R^2$: train={r2_tr:.2f} --- val={r2_va:.2f} --- test={r2_te:.2f}", fontsize=12)
    ax.text(0.05, 0.95, labels_map[param], transform=ax.transAxes, fontsize=24, va='top', ha='left')

fig.legend(handles=[scatter_tr, scatter_va, scatter_te], labels=['Train', 'Validation', 'Test'], 
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("parity_plots_canonical.pdf", dpi=300)
print("✅ Parity plots saved as 'parity_plots_canonical.pdf'.")

# ==========================================
# 7. PREDICTED DOE GENERATION
# ==========================================
print(f"\n--- 5. GENERATING PREDICTED DOE IN {DOE_OUTPUT_DIR} ---")

colunas_doe = ['C11', 'C12', 'C13', 'C21', 'C22', 'C23', 'C31', 'C32', 'C33', 'C44', 'C55', 'C66', 'a', 'k2006', 'sigma0', 'K', 'n']
doe_data = []

for i in range(len(y_test_canon_m)):
    q_pred = Y_test_prev[i]
    # Reconstruct original anisotropic matrix C from canonical variables q
    Ccan = canonical_C_from_q(q_pred[0], q_pred[1], q_pred[2])
    
    sigma0_pred = Y_test_prev[i, 3]
    K_pred = Y_test_prev[i, 4]
    n_pred = Y_test_prev[i, 5]
    
    a_val = y_test_orig_m[i, idx_map_orig['a']]
    k_val = y_test_orig_m[i, idx_map_orig['k2006']] if 'k2006' in idx_map_orig else 0.0

    linha = {
        'C11': Ccan[0],
        'C12': Ccan[1],
        'C13': Ccan[2],
        'C21': Ccan[1], 
        'C22': Ccan[3],
        'C23': Ccan[4],
        'C31': Ccan[2], 
        'C32': Ccan[4], 
        'C33': Ccan[5],
        'C44': 1.0,     
        'C55': 1.0,     
        'C66': Ccan[6],
        'a': a_val,
        'k2006': k_val,
        'sigma0': sigma0_pred,
        'K': K_pred,
        'n': n_pred
    }
    doe_data.append(linha)

df_doe = pd.DataFrame(doe_data, columns=colunas_doe)
caminho_csv = os.path.join(DOE_OUTPUT_DIR, "y_predicted_canonical_DoE.csv")
df_doe.to_csv(caminho_csv, sep=';', index=False)
print(f"✅ Predicted DoE file successfully generated: {caminho_csv}")
