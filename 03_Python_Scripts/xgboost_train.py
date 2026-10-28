"""
XGBoost Training Pipeline

This script establishes the machine learning training pipeline for the parameter 
identification problem using XGBoost, based on the work by Marques [1]. It automates 
the data pre-processing, model training, and performance evaluation phases. Initially, 
the script standardizes the input features and applies Principal Component Analysis (PCA) 
to reduce the dimensionality of the dataset to 150 principal components. Subsequently, 
it trains a MultiOutput Extreme Gradient Boosting (XGBoost) regression model to map the 
condensed mechanical responses to the target constitutive parameters. Finally, the script 
evaluates the model's accuracy by calculating global and parameter-specific metrics 
(R-squared, MAE, and MAPE) and exports the trained model, the preprocessing scalers, 
and the prediction logs for future validation and visualization.
"""

import pandas as pd
import time
import joblib
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  
import xgboost as xgb
import os

# ==========================================
# 1. PATHS AND VARIABLES
# ==========================================
BASE_DIR = r"C:\Users\alexandre.lavoura\Desktop\teseMestradoAlexandre\teseMestradoAlexandre\mlpcp-interp-main"

DATA = os.path.join(BASE_DIR, "data", "data_cpb") 
MODELS = os.path.join(BASE_DIR, "models")
METRICS = os.path.join(BASE_DIR, "metrics", "training_ori_metrics.csv")

X_TRAIN = os.path.join(DATA, "x_train.csv")
Y_TRAIN = os.path.join(DATA, "y_train.csv")

# ==========================================
# 2. TRAIN AND EVALUATE FUNCTION
# ==========================================
def train_and_evaluate():
    try:
        print(f"Loading data from {X_TRAIN} and {Y_TRAIN}")
        X_train = pd.read_csv(X_TRAIN)
        y_train = pd.read_csv(Y_TRAIN)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    n_cols = len(X_train.columns)
    points = int((n_cols/20-2)/3)

    l = []
    for x in range(1, 21): 
        l.append("Force_x_" + str(x))
        l.append("Force_y_" + str(x))
        for p in range(1, points+1): 
            l.append("Strain_x_" + str(p) + "_" + str(x))
            l.append("Strain_y_" + str(p) + "_" + str(x))
            l.append("Strain_xy_" + str(p) + "_" + str(x))

    X_train.columns = l

    # ------------------------------------------
    # PRE-PROCESSING: SCALER
    # ------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    scaler_file = os.path.join(MODELS, "scaler_ori.joblib")
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved as {scaler_file}")

    # ------------------------------------------
    # PRE-PROCESSING: PCA (150 Components)
    # ------------------------------------------
    print("Applying PCA with 150 components...")
    pca = PCA(n_components=150)
    X_train_pca = pca.fit_transform(X_train_scaled)

    pca_file = os.path.join(MODELS, "pca_ori.joblib")
    joblib.dump(pca, pca_file)
    print(f"PCA model saved as {pca_file}")

    # ------------------------------------------
    # XGBOOST MODEL TRAINING
    # ------------------------------------------
    start_time_training = time.monotonic()

    try:
        print("Starting train...")
        xgb_base = xgb.XGBRegressor(
            learning_rate=0.02, 
            max_depth=3,  # Adjusted to the optimal model configuration
            n_estimators=1000, 
            tree_method="hist", 
            device="cpu"
        )
        modelo = MultiOutputRegressor(xgb_base).fit(X_train_pca, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        return

    end_time_training = time.monotonic()
    training_duration = end_time_training - start_time_training
    print(f"Training duration: {training_duration:.2f} seconds")

    model_filename = os.path.join(MODELS, "xgb_ori.joblib")
    try:
        joblib.dump(modelo, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    # ------------------------------------------
    # PREDICTIONS AND METRICS
    # ------------------------------------------
    try:
        y_train_pred = modelo.predict(X_train_pca)
        
        # SAVE TRAINING PREDICTIONS (Required for dynamic plotting)
        pred_train_file = os.path.join(DATA, "y_pred_train.csv")
        pd.DataFrame(y_train_pred, columns=y_train.columns).to_csv(pred_train_file, index=False)
        print(f"Training predictions saved to {pred_train_file}")
        
    except Exception as e:
        print(f"Error predicting on training data: {e}")
        return

    try:
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # PRINT INDIVIDUAL R-SQUARED PER PARAMETER
        print("\n[R-Squared per Parameter - Training]")
        for i, col in enumerate(y_train.columns):
            r2_col = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            print(f" - {col}: {r2_col:.4f}")
            
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return

    print(f"\n[GLOBAL METRICS]")
    print(f"R-squared: {r2_train:.4f}")
    print(f"MAE: {mae_train:.4f}")
    print(f"MAPE: {mape_train:.4f}")

    return {
        "r2": r2_train,
        "mae": mae_train,
        "mape": mape_train,
        "training_duration": training_duration
    }

# ==========================================
# 3. MAIN FUNCTION
# ==========================================
def main():
    start_time = time.time()

    if not os.path.exists(MODELS):
        os.makedirs(MODELS)
        
    if os.path.exists(METRICS):
        os.remove(METRICS)

    result = train_and_evaluate()
    if result is None:
        return 1
        
    result_df = pd.DataFrame([result])
    write_header = not os.path.exists(METRICS)
    result_df.to_csv(METRICS, mode="a", header=write_header, index=False)
    print(f"Training performance metrics saved to {METRICS}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    print(f"Total elapsed time: {elapsed_minutes}:{elapsed_seconds:02d} minutes.")
    return 0

if __name__ == "__main__":
    exit(main())
