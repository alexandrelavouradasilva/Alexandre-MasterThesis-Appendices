"""
Analytical and Numerical Validation of the Final Model

This script implements the final macroscopic validation phase by directly comparing 
the output of the machine learning pipeline against the ground-truth finite element 
simulations. It autonomously scans the directories containing the simulated cruciform 
test datasets, generated respectively from the original parameters, the theoretically 
exact canonical parameters, and the neural network's predictions, and cross-references 
them to evaluate point-by-point macroscopic deviations. 

To highlight the robustness of the canonical transformation, the script isolates 
and scrutinizes three critical boundary cases: the median macroscopic error, the 
worst-case parameter mismatch, and the worst-case anisotropy (r-value) deviation. 
For each selected case, it automatically generates a comprehensive suite of 
comparative plots, visually overlaying the physical yield criteria (yield locus, 
directional stresses, r-values, and Swift hardening) alongside the macroscopic 
responses (reaction forces and nodal strain evolutions), confirming that even the 
pipeline's weakest predictions maintain a high degree of physical and macroscopic 
fidelity.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt, sin, cos, radians, copysign
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PATHS AND CONFIGURATIONS
# ==========================================
CSV_ORIG_DIR = r"C:\AbaqusVM\alexandre.lavoura\alexandre\AbaqusData\data\original_test_data"
CSV_PRED_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\Dados_compare"
PLOTS_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\plots\final_evaluation"

DOE_ORIGINAL_FILE = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\doe\y_test_original_DoE.csv"
DOE_EXPECTED_FILE = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\doe\y_test_canonical_DoE.csv"
DOE_PREDICTED_FILE = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\doe\y_predicted_canonical_DoE_abaqus.csv"

YLDVIZ_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main\data\YldViz-Data_Alexandre"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Visual Formatting - Large Fonts for the Dissertation
plt.rcParams.update({
    "font.family": "serif", 
    "font.serif": ["Times New Roman"], 
    "font.size": 16,           
    "xtick.labelsize": 15,     
    "ytick.labelsize": 15,     
    "text.usetex": False
})

# ==========================================
# 2. MATHEMATICAL FUNCTIONS (SWIFT & CPB06)
# ==========================================
def calcular_curva_swift(sigma0, K, n, max_strain=1.0, points=100):
    """Calculates the theoretical Swift hardening curve."""
    eps_p = np.linspace(0, max_strain, points)
    eps_0 = (sigma0 / K)**(1.0 / n) if (K > 0 and n > 0) else 0
    stress = K * (eps_0 + eps_p)**n
    return eps_p, stress

THETAS_DEG = np.arange(0, 91, 1).round(1).tolist()

def cpb2006_2d(sxx, syy, sxy, params, k_val=0, a_val=2):
    """Computes the 2D CPB06 yield function and its stress derivatives."""
    C11, C22, C33, C66, C12, C13, C23 = params
    X1 = (C13-C12-2*C11+C22+C23)/6*sxx + (C11-C12+C13-2*C22+C23)/6*syy
    A  = (2*C11-3*C12-C13+C22+C23)/6*sxx + (-C11+3*C12-C13-2*C22+C23)/6*syy
    B  = C66*sxy
    disc = sqrt(A**2+B**2) if (A**2+B**2)>0 else 0.
    phi1=X1+disc; phi2=X1-disc
    omega = (C33-2*C13+C23)/3*sxx + (C33+C13-2*C23)/3*syy
    
    def g(p): return (abs(p)+k_val*p)**a_val
    def dg(p):
        if abs(p)<1e-15: return 0.
        return a_val*(copysign(1,p)+k_val)*(abs(p)+k_val*p)**(a_val-1)
    
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

def compute_aniso(params, k_val=0, a_val=2):
    """Calculates directional normalized yield stresses and r-values."""
    F_ref, _ = cpb2006_2d(1., 0., 0., params, k_val, a_val)
    D_ref = 2 * F_ref
    if D_ref < 1e-15: return None, None
    sy_list, r_list = [], []
    for theta_deg in THETAS_DEG:
        th = radians(theta_deg)
        sxx=cos(th)**2; syy=sin(th)**2; sxy=sin(th)*cos(th)
        F1, dF1 = cpb2006_2d(sxx, syy, sxy, params, k_val, a_val)
        F=2*F1; dF_xx=2*dF1[0]; dF_yy=2*dF1[1]; dF_xy=2*dF1[2]
        se = (F/D_ref)**(1./a_val)
        if se < 1e-15 or not np.isfinite(se): return None, None
        sy_list.append(1./se)
        Ezz = -(dF_xx+dF_yy)
        if abs(Ezz) < 1e-10: return None, None
        Eyy_p = dF_xx*syy + dF_yy*sxx - dF_xy*sxy
        r = Eyy_p/Ezz
        if not np.isfinite(r): return None, None
        r_list.append(r)
    return sy_list, r_list

def calc_surface(params, k_val=0, a_val=2):
    """Calculates the bi-axial yield surface boundary."""
    sxx_arr = np.linspace(0, 1.5, 100)
    syy_arr = np.linspace(0, 1.5, 100)
    surf_x, surf_y = [], []
    F_ref, _ = cpb2006_2d(1., 0., 0., params, k_val, a_val)
    D_ref = 2 * F_ref
    for sx in sxx_arr:
        for sy in syy_arr:
            F1, _ = cpb2006_2d(sx, sy, 0., params, k_val, a_val)
            if abs((2*F1/D_ref) - 1.0) < 0.02:
                surf_x.append(sx)
                surf_y.append(sy)
    return surf_x, surf_y

def safe_r2(true_arr, pred_arr):
    """Calculates R-squared safely avoiding division by zero on low variance arrays."""
    true_arr = np.array(true_arr)
    pred_arr = np.array(pred_arr)
    if np.var(true_arr) < 1e-10: return 0.0
    return r2_score(true_arr, pred_arr)

# ==========================================
# 3. READING AND FILTERING SIMULATION CASES
# ==========================================
print("--- 1. READING DOE AND FEA LISTS ---")
df_orig = pd.read_csv(DOE_ORIGINAL_FILE, sep=';')
df_expe = pd.read_csv(DOE_EXPECTED_FILE, sep=';')
df_pred = pd.read_csv(DOE_PREDICTED_FILE, sep=';')

if 'k' in df_orig.columns: df_orig.rename(columns={'k': 'K'}, inplace=True)
if 'k' in df_expe.columns: df_expe.rename(columns={'k': 'K'}, inplace=True)
if 'k' in df_pred.columns: df_pred.rename(columns={'k': 'K'}, inplace=True)

matched_data = []

# Cross-reference the Original and Predicted Datasets to find identical FEA responses
for i in range(min(len(df_orig), len(df_expe), len(df_pred))):
    row_o = df_orig.iloc[i]
    row_e = df_expe.iloc[i]
    row_p = df_pred.iloc[i]
    
    nome_orig = '{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.5f}.csv'.format(
        row_o['C11'], row_o['C12'], row_o['C13'], row_o['C21'], row_o['C22'], row_o['C23'], 
        row_o['C31'], row_o['C32'], row_o['C33'], row_o['C44'], row_o['C55'], row_o['C66'], 
        row_o['a'], row_o.get('k2006', 0.0), row_o['sigma0'], row_o['K'], row_o['n']
    )
    
    nome_pred = '{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.5f}.csv'.format(
        row_p['C11'], row_p['C12'], row_p['C13'], row_p['C21'], row_p['C22'], row_p['C23'], 
        row_p['C31'], row_p['C32'], row_p['C33'], row_p['C44'], row_p['C55'], row_p['C66'], 
        row_p['a'], row_p.get('k2006', 0.0), row_p['sigma0'], row_p['K'], row_p['n']
    )
    
    caminho_orig = r"\\?\{}\{}".format(CSV_ORIG_DIR, nome_orig)
    caminho_pred = r"\\?\{}\{}".format(CSV_PRED_DIR, nome_pred)
    
    if os.path.exists(caminho_orig) and os.path.exists(caminho_pred):
        try:
            val_orig = pd.read_csv(caminho_orig, header=None).values
            val_pred = pd.read_csv(caminho_pred, header=None).values
            
            if val_orig.shape == val_pred.shape:
                r2_fea_global = safe_r2(val_orig.flatten(), val_pred.flatten())
                
                # 1. Canonical Parameter Difference (Expected vs Predicted)
                params_e = [row_e['C11'], row_e['C22'], row_e['C33'], row_e['C66'], row_e['C12'], row_e['C13'], row_e['C23']]
                params_p = [row_p['C11'], row_p['C22'], row_p['C33'], row_p['C66'], row_p['C12'], row_p['C13'], row_p['C23']]
                r2_params = safe_r2(params_e, params_p)
                
                # 2. R-values MSE Difference
                sy_e, r_e = compute_aniso(params_e)
                sy_p, r_p = compute_aniso(params_p)
                mse_r_val = mean_squared_error(r_e, r_p) if (r_e is not None and r_p is not None) else float('inf')
                
                matched_data.append({
                    'idx': i, 'row_o': row_o, 'row_e': row_e, 'row_p': row_p, 
                    'val_orig': val_orig, 'val_pred': val_pred,
                    'r2_fea': r2_fea_global, 'r2_params': r2_params, 'mse_r_val': mse_r_val
                })
        except:
            continue

if not matched_data:
    print("❌ No matching simulation pairs found!")
    exit()

# --- Case Selection ---
# Median FEA Error
matched_data.sort(key=lambda x: x['r2_fea'])
caso_mediano = matched_data[len(matched_data) // 2]

# Worst Parameter Prediction
matched_data.sort(key=lambda x: x['r2_params'])
pior_params = matched_data[0] 

# Worst Analytical R-value (Anisotropy)
matched_data.sort(key=lambda x: x['mse_r_val'], reverse=True)
pior_r_val = matched_data[0] 

casos_selecionados = [
    ("Median_FEA", caso_mediano),
    ("Worst_Params_R2", pior_params),
    ("Worst_R_Values_Anisotropy", pior_r_val)
]

# ==========================================
# 4. UPDATED PLOTTING FUNCTIONS
# ==========================================

def gerar_graficos_constitutivos(nome_caso, set_id, row_o, row_e, row_p):
    """Generates a 2x2 grid with analytical mechanical responses."""
    # YldViz CSV Paths
    file_orig = os.path.join(YLDVIZ_DIR, f"YldViz-Data_Original_{set_id}.csv")
    file_true = os.path.join(YLDVIZ_DIR, f"YldViz-Data_CanonicalExpected_{set_id}.csv")
    file_pred = os.path.join(YLDVIZ_DIR, f"YldViz-Data_CanonicalPredicted_{set_id}.csv")
    
    # Load CSVs
    df_orig = pd.read_csv(file_orig, sep=';').sort_values("aniso_theta")
    df_true = pd.read_csv(file_true, sep=';').sort_values("aniso_theta")
    df_pred = pd.read_csv(file_pred, sep=';').sort_values("aniso_theta")
    
    # Extract Anisotropy Data
    theta_orig, sig_orig, r_orig = df_orig["aniso_theta"].values, df_orig["aniso_sig0"].values, df_orig["aniso_r"].values
    theta_true, sig_true, r_true = df_true["aniso_theta"].values, df_true["aniso_sig0"].values, df_true["aniso_r"].values
    theta_pred, sig_pred, r_pred = df_pred["aniso_theta"].values, df_pred["aniso_sig0"].values, df_pred["aniso_r"].values
    
    # Extract Yield Surface Data
    sigXX_orig, sigYY_orig = df_orig["sigXX-0.0"].values, df_orig["sigYY-0.0"].values
    sigXX_true, sigYY_true = df_true["sigXX-0.0"].values, df_true["sigYY-0.0"].values
    sigXX_pred, sigYY_pred = df_pred["sigXX-0.0"].values, df_pred["sigYY-0.0"].values
    
    # Calculate Analytical Swift Hardening 
    eps_p, swift_o = calcular_curva_swift(row_o['sigma0'], row_o['K'], row_o['n'])
    _, swift_e = calcular_curva_swift(row_e['sigma0'], row_e['K'], row_e['n'])
    _, swift_p = calcular_curva_swift(row_p['sigma0'], row_p['K'], row_p['n'])

    # --- CREATE 2x2 LARGE-FONT PLOTS ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16)) 

    # 1. Normalized Yield Stress
    axes[0, 0].plot(theta_orig, sig_orig, '-k', lw=3, label='Original')
    axes[0, 0].plot(theta_true, sig_true, '--b', lw=3, label='Canonical (Expected)')
    axes[0, 0].plot(theta_pred, sig_pred, ':r', lw=3, label='Canonical (Predicted)')
    axes[0, 0].set_xlabel(r'Loading angle $\theta$ (°)', fontsize=22)
    axes[0, 0].set_ylabel('Normalized yield stress', fontsize=22)
    axes[0, 0].set_xlim(0, 90)
    axes[0, 0].set_ylim(0.5, 1.5)
    axes[0, 0].grid(True); axes[0, 0].legend(fontsize=16)
    axes[0, 0].tick_params(labelsize=18)
    axes[0, 0].set_box_aspect(1) 

    # 2. r-values
    axes[0, 1].plot(theta_orig, r_orig, '-k', lw=3, label='Original')
    axes[0, 1].plot(theta_true, r_true, '--b', lw=3, label='Canonical (Expected)')
    axes[0, 1].plot(theta_pred, r_pred, ':r', lw=3, label='Canonical (Predicted)')
    axes[0, 1].set_xlabel(r'Loading angle $\theta$ (°)', fontsize=22)
    axes[0, 1].set_ylabel(r'$r$-value', fontsize=22)
    axes[0, 1].set_xlim(0, 90)
    axes[0, 1].set_ylim(0, 6)
    axes[0, 1].grid(True); axes[0, 1].legend(fontsize=16)
    axes[0, 1].tick_params(labelsize=18)
    axes[0, 1].set_box_aspect(1) 

    # 3. Yield Surface (SigmaXX vs SigmaYY)
    axes[1, 0].plot(sigXX_orig, sigYY_orig, '-k', lw=3, label='Original')
    axes[1, 0].plot(sigXX_true, sigYY_true, '--b', lw=3, label='Canonical (Expected)')
    axes[1, 0].plot(sigXX_pred, sigYY_pred, ':r', lw=3, label='Canonical (Predicted)')
    axes[1, 0].set_xlabel(r'$\sigma_{xx}$', fontsize=22)
    axes[1, 0].set_ylabel(r'$\sigma_{yy}$', fontsize=22)
    axes[1, 0].set_xlim(0, 2); axes[1, 0].set_ylim(0, 2)
    axes[1, 0].set_aspect('equal', adjustable='box')
    axes[1, 0].grid(True); axes[1, 0].legend(fontsize=16)
    axes[1, 0].tick_params(labelsize=18)
    axes[1, 0].set_box_aspect(1)

    # 4. Swift Hardening
    axes[1, 1].plot(eps_p, swift_o, '-k', lw=3, label='Original')
    axes[1, 1].plot(eps_p, swift_e, '--b', lw=3, label='Canonical (Expected)')
    axes[1, 1].plot(eps_p, swift_p, ':r', lw=3, label='Canonical (Predicted)')
    axes[1, 1].set_xlabel('True Plastic Strain', fontsize=22)
    axes[1, 1].set_ylabel('True Stress (MPa)', fontsize=22)
    axes[1, 1].set_xlim(0, 1.0)
    axes[1, 1].grid(True); axes[1, 1].legend(fontsize=16)
    axes[1, 1].tick_params(labelsize=18)
    axes[1, 1].set_box_aspect(1) 

    plt.tight_layout(pad=3.0) 
    plt.savefig(os.path.join(PLOTS_DIR, f"{nome_caso}_Constitutive.pdf"), dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

def gerar_graficos_fea(nome_caso, val_orig_2D, val_pred_2D, r2_global):
    """Generates the side-by-side FEA correlation plots for Forces and Strains."""
    num_frames = val_orig_2D.shape[0]
    n_nos_abaqus = int((val_orig_2D.shape[1] - 2) / 3)
    deslocamento = np.arange(1, num_frames + 1) * (2.0 / num_frames)
    
    def get_node_cols(node_idx):
        base = 2 + (node_idx - 1) * 3
        return base, base+1, base+2

    REF_NODE = 69 # Updated Center Node
    
    r2_nos = []
    for n in range(1, n_nos_abaqus + 1):
        cols_n = list(get_node_cols(n))
        orig_n = val_orig_2D[:, cols_n].flatten()
        pred_n = val_pred_2D[:, cols_n].flatten()
        r2_nos.append((n, safe_r2(orig_n, pred_n)))
    no_pior_r2, val_pior_r2 = min(r2_nos, key=lambda x: x[1])

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # 1. Plot Reaction Forces
    axes[0].plot(deslocamento, val_orig_2D[:, 0], '-k', lw=2.5, label=r'Original $F_x$')
    axes[0].plot(deslocamento, val_orig_2D[:, 1], '-.k', lw=2.5, label=r'Original $F_y$')
    axes[0].plot(deslocamento, val_pred_2D[:, 0], '--r', lw=2.5, label=r'Predicted $F_x$')
    axes[0].plot(deslocamento, val_pred_2D[:, 1], ':r', lw=2.5, label=r'Predicted $F_y$')
    axes[0].set_xlabel('Displacement (mm)', fontsize=18)
    axes[0].set_ylabel('Reaction Force (N)', fontsize=18)
    axes[0].set_title(f'Forces | Global R2={r2_global:.4f}', fontsize=20, pad=15)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=15)
    axes[0].set_box_aspect(1) 

    def plot_strains_dual_axis(ax, title, node_num):
        cx, cy, cxy = get_node_cols(node_num)
        ax.plot(deslocamento, val_orig_2D[:, cx], '-k', lw=2.5, label=r'Orig $\epsilon_{xx}$')
        ax.plot(deslocamento, val_orig_2D[:, cy], '-.k', lw=2.5, label=r'Orig $\epsilon_{yy}$')
        ax.plot(deslocamento, val_pred_2D[:, cx], '--b', lw=2.5, label=r'Pred $\epsilon_{xx}$')
        ax.plot(deslocamento, val_pred_2D[:, cy], ':b', lw=2.5, label=r'Pred $\epsilon_{yy}$')
        
        # Dual axis for shear strain
        ax2 = ax.twinx()
        ax2.plot(deslocamento, val_orig_2D[:, cxy], '-g', lw=2.0, alpha=0.6, label=r'Orig $\epsilon_{xy}$')
        ax2.plot(deslocamento, val_pred_2D[:, cxy], '--g', lw=2.0, alpha=0.8, label=r'Pred $\epsilon_{xy}$')
        
        ax2.set_ylabel('Shear Strain', color='g', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='g', labelsize=15)
        
        ax.set_xlabel('Displacement (mm)', fontsize=18)
        ax.set_ylabel('Normal Strain', fontsize=18)
        ax.set_title(title, fontsize=20, pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1) 
        
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=14)

    # 2. Node 69 (Reference Center)
    plot_strains_dual_axis(axes[1], f'Reference Center (Node {REF_NODE})', REF_NODE)
    
    # 3. Worst Local Strain Node
    plot_strains_dual_axis(axes[2], f'Worst Local Strain (Node {no_pior_r2}) | R2={val_pior_r2:.2f}', no_pior_r2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{nome_caso}_FEA.pdf"), dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. EXECUTION AND CONSOLE OUTPUT
# ==========================================
params_print = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33', 'C66', 'sigma0', 'K', 'n']

# YldViz Set Mapping
set_map = {
    "Median_FEA": "set1",
    "Worst_Params_R2": "set2",
    "Worst_R_Values_Anisotropy": "set3"
}

print("\n")
for nome, dados in casos_selecionados:
    print(f"==========================================================================")
    print(f" EVALUATING CASE: {nome} (Index: {dados['idx']})")
    print(f"==========================================================================")
    print(f"{'Parameter':<10} | {'Original':<15} | {'Exp. Canonical':<15} | {'Pred. Canonical':<15}")
    print("-" * 65)
    
    for p in params_print:
        v_orig = f"{dados['row_o'][p]:.5f}"
        v_expe = f"{dados['row_e'][p]:.5f}"
        v_pred = f"{dados['row_p'][p]:.5f}"
        print(f"{p:<10} | {v_orig:<15} | {v_expe:<15} | {v_pred:<15}")
    
    print("-" * 65)
    print(f"Global FEA R²       : {dados['r2_fea']:.6f}")
    if nome == "Worst_Params_R2":
        print(f"Parameters C R²     : {dados['r2_params']:.6f}")
    if nome == "Worst_R_Values_Anisotropy":
        print(f"MSE R-Values        : {dados['mse_r_val']:.6f}")
    
    # Retrieve the correct YldViz set matching the evaluation case
    current_set = set_map[nome]
    
    gerar_graficos_constitutivos(nome, current_set, dados['row_o'], dados['row_e'], dados['row_p'])
    gerar_graficos_fea(nome, dados['val_orig'], dados['val_pred'], dados['r2_fea'])
    
    print(f"✅ PDF figures successfully generated in folder: {PLOTS_DIR}\n")
