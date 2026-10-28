"""
Testing XGBoost Model

This script establishes the machine learning testing pipeline, complementing the 
training framework adapted from Marques [1]. It evaluates the generalization 
capability of the previously trained XGBoost model using an unseen testing dataset. 
The script loads the pre-fitted standard scaler and PCA components, applying the 
exact same dimensionality reduction to the test data to maintain mathematical 
consistency. Using the trained model, it generates predictions for the target 
constitutive parameters and isolates the 9 key parameters of interest to calculate 
both global and parameter-specific performance metrics (R-squared, MAE, and MAPE). 
Finally, the script exports the quantitative metrics and the complete array of 
predicted parameters to CSV files for further validation against finite element 
simulations.
"""

import pandas as pd
import os
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)
import time
import joblib

# ==========================================
# 1. PATHS AND VARIABLES
# ==========================================
BASE_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main"
DATA = os.path.join(BASE_DIR, "data", "data_cpb")
MODELS = os.path.join(BASE_DIR, "models")
METRICS = os.path.join(BASE_DIR, "metrics", "testing_ori_metrics.csv")

X_TEST = os.path.join(DATA, "x_test.csv")
Y_TEST = os.path.join(DATA, "y_test.csv")

# ==========================================
# 2. TEST AND EVALUATE FUNCTION
# ==========================================
def test_and_evaluate():
    xgb_model = os.path.join(MODELS, f"xgb_ori.joblib")
    scaler_file = os.path.join(MODELS, f"scaler_ori.joblib")
    pca_file = os.path.join(MODELS, f"pca_ori.joblib")

    # Load feature and target data
    try:
        print(f"Loading data from {X_TEST} and {Y_TEST}")
        X_test = pd.read_csv(X_TEST)
        y_test = pd.read_csv(Y_TEST)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Get number of columns and points
    n_cols = len(X_test.columns)
    points = int((n_cols/20-2)/3)

    # Define columns for X_test
    l=[]
    for x in range(1,21): 
        l.append("Force_x_"+str(x))
        l.append("Force_y_"+str(x))
        for p in range(1, points+1):  
            l.append("Strain_x_"+str(p)+"_"+str(x))
            l.append("Strain_y_"+str(p)+"_"+str(x))
            l.append("Strain_xy_"+str(p)+"_"+str(x))
    
    X_test.columns = l

    # ------------------------------------------
    # PRE-PROCESSING: SCALER
    # ------------------------------------------
    try:
        scaler = joblib.load(scaler_file)
        X_test_scaled = scaler.transform(X_test)
        print("Scaler loaded and applied successfully.")
    except Exception as e:
        print(f"Error loading or applying scaler: {e}")
        return

    # ------------------------------------------
    # PRE-PROCESSING: PCA
    # ------------------------------------------
    try:
        pca = joblib.load(pca_file)
        X_test_pca = pca.transform(X_test_scaled) 
        print("PCA loaded and applied successfully.")
    except Exception as e:
        print(f"Error loading or applying PCA: {e}")
        return

    # ------------------------------------------
    # PREDICTIONS AND METRICS
    # ------------------------------------------
    try:
        print("Loading xgb model...")
        modelo = joblib.load(xgb_model)
        y_test_pred = modelo.predict(X_test_pca)
    except Exception as e:
        print(f"Error predicting values: {e}")
        return
    
    end_time = time.time()

    # 1. DEFINE THE 9 TARGET PARAMETERS (lowercase 'k')
    selected = ['C12', 'C13', 'C22', 'C23', 'C33', 'C66', 'sigma0', 'k', 'n']
    target_names = y_test.columns.tolist()
    
    # 2. ISOLATE ONLY THE 9 ACTUAL AND PREDICTED COLUMNS
    indices = [target_names.index(col) for col in selected]
    y_test_9 = y_test.iloc[:, indices].values
    y_test_pred_9 = y_test_pred[:, indices]

    # 3. CALCULATE GLOBAL METRICS FOR THE 9 PARAMETERS
    try:
        r2_test = r2_score(y_test_9, y_test_pred_9)
        mae_test = mean_absolute_error(y_test_9, y_test_pred_9)
        mape_test = mean_absolute_percentage_error(y_test_9, y_test_pred_9)
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return
    
    print(f'\nR-squared (Global Test - 9 Params): {r2_test:.4f}')
    print(f'MAE (Global Test - 9 Params): {mae_test:.4f}')
    print(f'MAPE (Global Test - 9 Params): {mape_test:.4f}')
    
    print("\n[R-Squared per Parameter - Test]")
    for i, param in enumerate(selected):
        r2_param = r2_score(y_test_9[:, i], y_test_pred_9[:, i])
        print(f" - {param}: {r2_param:.4f}")

    # Save the predicted results to a CSV (maintains the full original structure)
    pred_params_path = os.path.join(DATA, f"y_pred_ori.csv")
    df = pd.DataFrame(y_test_pred, columns=y_test.columns)
    df.to_csv(pred_params_path, mode="w", header=True, index=False)

    return {
        "r2": r2_test,
        "mae": mae_test,
        "mape": mape_test
    }

# ==========================================
# 3. MAIN FUNCTION
# ==========================================
def main():
    start_time = time.time()

    if os.path.exists(METRICS):
        os.remove(METRICS)

    result = test_and_evaluate()
    if result is None:
        return 1
        
    result_df = pd.DataFrame([result])
    write_header = not os.path.exists(METRICS)
    result_df.to_csv(METRICS, mode="a", header=write_header, index=False)
    print(f"\nTesting performance metrics saved to {METRICS}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    print(f"Total elapsed time: {elapsed_minutes}:{elapsed_seconds:02d} minutes.")

if __name__ == "__main__":
    exit(main())
