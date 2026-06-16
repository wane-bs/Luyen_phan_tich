import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score, confusion_matrix

def compute_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
    else:
        tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return np.sqrt(tpr * tnr)

def run_monte_carlo_analysis():
    # 1. Load Data
    csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run src/data_pipeline/feature_engineering.py first.")
        return
    df = pd.read_csv(csv_path)

    # 2. Load Model
    model_path = os.path.join("models", "best_default_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train model first.")
        return
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    is_xgb = (model_dict['type'] == 'xgb')
    features = model_dict['features']

    scaler_path = os.path.join("models", "default_scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    X = df[features].copy()
    y = df['default']

    # Suspect features for analysis
    suspect_features = ['credit_score', 'current_age', 'insufficient_balance_rate', 'essential_spend_ratio', 'refund_rate']
    
    # 3. Base performance
    if is_xgb:
        y_pred_base = model.predict(X)
    else:
        X_scaled = scaler.transform(X)
        y_pred_base = model.predict(X_scaled)
        
    base_f1 = f1_score(y, y_pred_base)
    base_gmean = compute_g_mean(y, y_pred_base)
    
    print("=" * 60)
    print("   HIGH-PRECISION MONTE CARLO STABILITY AUDIT")
    print(f"   Baseline F1-Score: {base_f1:.4f}")
    print(f"   Baseline G-Mean:   {base_gmean:.4f}")
    print("=" * 60)

    # Progressive noise steps from 0% to 200% of standard deviation
    noise_multipliers = np.linspace(0.0, 2.0, 21)
    num_simulations = 200
    
    mc_results = {}
    
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    stability_report = []

    X_arr = X.values.copy()
    for i, feat in enumerate(suspect_features):
        if feat not in X.columns:
            continue
            
        feat_idx = features.index(feat)
        std_val = X[feat].std()
        if std_val == 0:
            std_val = 1e-5
            
        f1_means = []
        gmean_means = []
        
        print(f"Simulating noise on: {feat} (std = {std_val:.4f})...")
        
        for alpha in noise_multipliers:
            f1_runs = []
            gmean_runs = []
            
            # Vectorized noise matrix for all simulations
            noise_matrix = np.random.normal(0, alpha * std_val, size=(num_simulations, len(X)))
            
            for sim in range(num_simulations):
                X_noisy_arr = X_arr.copy()
                X_noisy_arr[:, feat_idx] += noise_matrix[sim]
                
                if is_xgb:
                    y_pred = model.predict(X_noisy_arr)
                else:
                    X_noisy_scaled = scaler.transform(X_noisy_arr)
                    y_pred = model.predict(X_noisy_scaled)
                    
                f1_runs.append(f1_score(y, y_pred))
                gmean_runs.append(compute_g_mean(y, y_pred))
                
            f1_means.append(np.mean(f1_runs))
            gmean_means.append(np.mean(gmean_runs))
            
        # Determine threshold where metric drops by 10% and 30%
        drop_10_f1 = np.nan
        drop_30_f1 = np.nan
        drop_10_gmean = np.nan
        drop_30_gmean = np.nan
        
        for idx, alpha in enumerate(noise_multipliers):
            # If F1 drops below 90% of baseline F1
            if np.isnan(drop_10_f1) and f1_means[idx] < 0.90 * base_f1:
                drop_10_f1 = alpha
            if np.isnan(drop_30_f1) and f1_means[idx] < 0.70 * base_f1:
                drop_30_f1 = alpha
                
            # G-Mean drops
            if np.isnan(drop_10_gmean) and gmean_means[idx] < 0.90 * base_gmean:
                drop_10_gmean = alpha
            if np.isnan(drop_30_gmean) and gmean_means[idx] < 0.70 * base_gmean:
                drop_30_gmean = alpha

        # Classify stability rating
        # High: drops by 10% only above 100% noise
        # Moderate: drops by 10% between 40% and 100% noise
        # Low: drops by 10% below 40% noise
        ref_drop = drop_10_f1 if not np.isnan(drop_10_f1) else 2.0
        if ref_drop >= 1.0:
            rating = "🟢 High Stability"
        elif ref_drop >= 0.4:
            rating = "🟡 Moderate Stability"
        else:
            rating = "🔴 Vulnerable (Low Stability)"

        stability_report.append({
            "feature": feat,
            "std": float(std_val),
            "drop_10_f1_noise_pct": float(drop_10_f1) if not np.isnan(drop_10_f1) else None,
            "drop_30_f1_noise_pct": float(drop_30_f1) if not np.isnan(drop_30_f1) else None,
            "drop_10_gmean_noise_pct": float(drop_10_gmean) if not np.isnan(drop_10_gmean) else None,
            "drop_30_gmean_noise_pct": float(drop_30_gmean) if not np.isnan(drop_30_gmean) else None,
            "stability_rating": rating
        })

        # Save to results dict for curves
        mc_results[feat] = {
            "noise_multipliers": list(noise_multipliers),
            "f1_means": list(f1_means),
            "gmean_means": list(gmean_means)
        }

        # Plot curves
        ax1.plot(noise_multipliers * 100, f1_means, label=f"{feat}", color=colors[i % len(colors)], marker='o', markersize=4)
        ax2.plot(noise_multipliers * 100, gmean_means, label=f"{feat}", color=colors[i % len(colors)], marker='x', markersize=4, linestyle='--')

    # Formatting plots
    ax1.axhline(base_f1 * 0.90, color='grey', linestyle=':', label='10% Drop Threshold')
    ax1.set_xlabel('Noise level (% of Feature Std Dev)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score Decay Curves')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    
    ax2.axhline(base_gmean * 0.90, color='grey', linestyle=':', label='10% Drop Threshold')
    ax2.set_xlabel('Noise level (% of Feature Std Dev)')
    ax2.set_ylabel('G-Mean')
    ax2.set_title('G-Mean Decay Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    plt.suptitle('Monte Carlo Progressive Noise Decay Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join("reports", "figures", "monte_carlo_stability_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save metrics JSON
    os.makedirs(os.path.join("data", "outputs"), exist_ok=True)
    with open(os.path.join("data", "outputs", "monte_carlo_stability_report.json"), "w", encoding="utf-8") as f:
        json.dump(stability_report, f, indent=4)
        
    print("\n" + "=" * 60)
    print("   MONTE CARLO STABILITY AUDIT COMPLETE — RESULTS:")
    print("=" * 60)
    print(f"| {'Feature':25s} | {'Std Dev':10s} | {'F1 Drop 10%':12s} | {'F1 Drop 30%':12s} | {'Rating':25s} |")
    print(f"| {'-'*25} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*25} |")
    for rep in stability_report:
        d10 = f"{rep['drop_10_f1_noise_pct']*100:.1f}%" if rep['drop_10_f1_noise_pct'] is not None else ">200%"
        d30 = f"{rep['drop_30_f1_noise_pct']*100:.1f}%" if rep['drop_30_f1_noise_pct'] is not None else ">200%"
        print(f"| {rep['feature']:25s} | {rep['std']:10.4f} | {d10:12s} | {d30:12s} | {rep['stability_rating']:25s} |")
    print("=" * 60)
    print(f"Stability curves plot saved to {plot_path}")

if __name__ == "__main__":
    run_monte_carlo_analysis()
