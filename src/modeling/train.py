import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, f1_score, auc, confusion_matrix
)
from xgboost import XGBClassifier

def compute_g_mean(y_true, y_pred):
    """Calculate Geometric Mean (G-Mean) of sensitivity and specificity."""
    cm = confusion_matrix(y_true, y_pred)
    # If the confusion matrix is empty or 1-dimensional
    if cm.shape != (2, 2):
        # Fallback for edge cases
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
    else:
        tn, fp, fn, tp = cm.ravel()
        
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return np.sqrt(tpr * tnr)

def run_permutation_importance(model, X_eval, y_eval, is_xgb, scaler=None, n_repeats=5):
    """Calculate Permutation Feature Importance using F1-score and G-Mean drop."""
    # Compute baseline scores
    if is_xgb:
        y_pred_base = model.predict(X_eval)
    else:
        X_eval_scaled = scaler.transform(X_eval)
        y_pred_base = model.predict(X_eval_scaled)
        
    base_f1 = f1_score(y_eval, y_pred_base)
    base_gmean = compute_g_mean(y_eval, y_pred_base)
    
    importance_f1 = {}
    importance_gmean = {}
    
    for col in X_eval.columns:
        f1_drops = []
        gmean_drops = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            X_perm[col] = np.random.permutation(X_perm[col])
            
            if is_xgb:
                y_pred_perm = model.predict(X_perm)
            else:
                X_perm_scaled = scaler.transform(X_perm)
                y_pred_perm = model.predict(X_perm_scaled)
                
            perm_f1 = f1_score(y_eval, y_pred_perm)
            perm_gmean = compute_g_mean(y_eval, y_pred_perm)
            
            f1_drops.append(base_f1 - perm_f1)
            gmean_drops.append(base_gmean - perm_gmean)
            
        importance_f1[col] = np.mean(f1_drops)
        importance_gmean[col] = np.mean(gmean_drops)
        
    return importance_f1, importance_gmean

def run_monte_carlo_noise_test(model, X_eval, y_eval, is_xgb, scaler, suspect_features, num_simulations=50):
    """Perform Monte Carlo simulation by adding progressive Gaussian noise to suspect features."""
    noise_levels = np.linspace(0.0, 1.0, 10) # 0% to 100% of std
    feature_stds = {feat: X_eval[feat].std() for feat in suspect_features if feat in X_eval.columns}
    
    results = {}
    for feat in suspect_features:
        if feat not in X_eval.columns:
            continue
        results[feat] = {
            'noise_levels': noise_levels,
            'f1_mean': [], 'f1_std': [],
            'gmean_mean': [], 'gmean_std': []
        }
        
        std_val = feature_stds[feat]
        if std_val == 0:
            std_val = 1e-5 # prevent division by zero or empty noise
            
        for alpha in noise_levels:
            f1_runs = []
            gmean_runs = []
            
            for _ in range(num_simulations):
                X_noisy = X_eval.copy()
                # Inject Gaussian noise
                noise = np.random.normal(0, alpha * std_val, size=len(X_noisy))
                X_noisy[feat] = X_noisy[feat] + noise
                
                if is_xgb:
                    y_pred = model.predict(X_noisy)
                else:
                    X_noisy_scaled = scaler.transform(X_noisy)
                    y_pred = model.predict(X_noisy_scaled)
                    
                f1_runs.append(f1_score(y_eval, y_pred))
                gmean_runs.append(compute_g_mean(y_eval, y_pred))
                
            results[feat]['f1_mean'].append(np.mean(f1_runs))
            results[feat]['f1_std'].append(np.std(f1_runs))
            results[feat]['gmean_mean'].append(np.mean(gmean_runs))
            results[feat]['gmean_std'].append(np.std(gmean_runs))
            
    return results

def inject_train_noise_and_clip(X, columns, config, noise_level=0.05):
    X_noisy = X.copy()
    for col in columns:
        if col in X_noisy.columns:
            std = X_noisy[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            
    # Apply optimal clipping thresholds recommended by LLM or fallback
    if 'credit_score' in X_noisy.columns:
        X_noisy['credit_score'] = np.clip(
            X_noisy['credit_score'], 
            config.get('credit_score_min', 300.0), 
            config.get('credit_score_max', 850.0)
        )
    if 'current_age' in X_noisy.columns:
        X_noisy['current_age'] = np.clip(
            X_noisy['current_age'], 
            config.get('current_age_min', 18.0), 
            config.get('current_age_max', 100.0)
        )
    return X_noisy

def train_and_evaluate():
    # Load dataset
    csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Please run src/data_pipeline/feature_engineering.py first.")
        return

    # Load optimized hyperparameters configuration
    try:
        with open("data/configs/model_config.json", "r") as f:
            model_config = json.load(f)
        print("Successfully loaded model configuration from data/configs/model_config.json.")
    except Exception as e:
        print("Could not load model configuration, using mathematical fallbacks. Error:", e)
        model_config = {
            "credit_score_min": 300.0,
            "credit_score_max": 850.0,
            "current_age_min": 18.0,
            "current_age_max": 100.0,
            "max_depth": 3,
            "min_child_weight": 5.0,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0
        }

    print("Loading feature matrix...")
    df = pd.read_csv(csv_path)

    # Exclude features to prevent target leakage
    default_exclude = ['client_id', 'gender', 'dti', 'max_monthly_cur', 'avg_monthly_cur', 'default', 'fraud', 'security_error_count', 'yearly_income', 'total_debt', 'total_credit_limit', 'net_spend', 'has_spatiotemporal_fraud_signal']
    # fraud_signal_score, security_error_count_heavy, high_online_rate là dẩn xuất trực tiếp của nhãn fraud → phải exclude
    fraud_exclude = ['client_id', 'gender', 'security_error_count', 'default', 'fraud',
                     'fraud_signal_score', 'security_error_count_heavy', 'high_online_rate']

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Suspect features for leakage checks
    suspect_features_def = ['credit_score', 'insufficient_balance_rate', 'essential_spend_ratio', 'refund_rate']
    # Thêm refund_rate và has_spatiotemporal_fraud_signal vào audit fraud model
    suspect_features_frd = ['yearly_income', 'total_debt', 'total_credit_limit', 'net_spend', 'refund_rate', 'has_spatiotemporal_fraud_signal']

    # ----------------------------------------------------
    # PART 1: DEFAULT RISK MODEL
    # ----------------------------------------------------
    print("\n" + "="*60)
    print(" TRAINING DEFAULT RISK MODEL WITH 5-FOLD STRATIFIED K-FOLD ")
    print("="*60)
    
    X_def = df.drop(columns=default_exclude)
    y_def = df['default']
    feature_names_def = list(X_def.columns)
    
    print(f"Features count: {len(feature_names_def)}")
    print(f"Target distribution: {y_def.value_counts().to_dict()}")

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for out-of-fold (OOF) predictions
    lr_oof_probs = np.zeros(len(df))
    lr_oof_preds = np.zeros(len(df))
    xgb_oof_probs = np.zeros(len(df))
    xgb_oof_preds = np.zeros(len(df))
    
    lr_metrics = {'auc': [], 'pr_auc': [], 'f1': [], 'gmean': []}
    xgb_metrics = {'auc': [], 'pr_auc': [], 'f1': [], 'gmean': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_def, y_def)):
        X_train, X_test = X_def.iloc[train_idx], X_def.iloc[test_idx]
        y_train, y_test = y_def.iloc[train_idx], y_def.iloc[test_idx]
        
        # Scale features for LR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale pos weight for XGBoost
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
        
        # 1. Logistic Regression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        y_pred_lr = lr.predict(X_test_scaled)
        
        lr_oof_probs[test_idx] = y_prob_lr
        lr_oof_preds[test_idx] = y_pred_lr
        
        # Metrics LR
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
        pr_auc_lr = auc(rec_lr, prec_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        gmean_lr = compute_g_mean(y_test, y_pred_lr)
        
        lr_metrics['auc'].append(auc_lr)
        lr_metrics['pr_auc'].append(pr_auc_lr)
        lr_metrics['f1'].append(f1_lr)
        lr_metrics['gmean'].append(gmean_lr)
        
        # 2. XGBoost (trains on raw + noisy augmented features with LLM-optimized params/clip bounds)
        X_train_noisy = inject_train_noise_and_clip(X_train, ['credit_score', 'current_age'], model_config)
        X_train_combined = pd.concat([X_train, X_train_noisy], axis=0)
        y_train_combined = pd.concat([y_train, y_train], axis=0)

        xgb = XGBClassifier(
            scale_pos_weight=pos_weight, 
            eval_metric='logloss',
            max_depth=int(model_config.get("max_depth", 3)),
            min_child_weight=float(model_config.get("min_child_weight", 5.0)),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=float(model_config.get("reg_alpha", 1.5)),
            reg_lambda=float(model_config.get("reg_lambda", 3.0)),
            random_state=42
        )
        xgb.fit(X_train_combined, y_train_combined)
        
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
        y_pred_xgb = xgb.predict(X_test)
        
        xgb_oof_probs[test_idx] = y_prob_xgb
        xgb_oof_preds[test_idx] = y_pred_xgb
        
        # Metrics XGB
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
        pr_auc_xgb = auc(rec_xgb, prec_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        gmean_xgb = compute_g_mean(y_test, y_pred_xgb)
        
        xgb_metrics['auc'].append(auc_xgb)
        xgb_metrics['pr_auc'].append(pr_auc_xgb)
        xgb_metrics['f1'].append(f1_xgb)
        xgb_metrics['gmean'].append(gmean_xgb)
        
        print(f"Fold {fold+1} Default Model - LR AUC: {auc_lr:.4f} | XGB AUC: {auc_xgb:.4f}")

    print("\n" + "-"*40)
    print(" SUMMARY METRICS (5-FOLD OOF AVERAGE) ")
    print("-"*40)
    print("Logistic Regression (Default Risk):")
    print(f"  ROC-AUC : {np.mean(lr_metrics['auc']):.4f} +/- {np.std(lr_metrics['auc']):.4f}")
    print(f"  PR-AUC  : {np.mean(lr_metrics['pr_auc']):.4f} +/- {np.std(lr_metrics['pr_auc']):.4f}")
    print(f"  F1-Score: {np.mean(lr_metrics['f1']):.4f} +/- {np.std(lr_metrics['f1']):.4f}")
    print(f"  G-Mean  : {np.mean(lr_metrics['gmean']):.4f} +/- {np.std(lr_metrics['gmean']):.4f}")
    print("XGBoost (Default Risk):")
    print(f"  ROC-AUC : {np.mean(xgb_metrics['auc']):.4f} +/- {np.std(xgb_metrics['auc']):.4f}")
    print(f"  PR-AUC  : {np.mean(xgb_metrics['pr_auc']):.4f} +/- {np.std(xgb_metrics['pr_auc']):.4f}")
    print(f"  F1-Score: {np.mean(xgb_metrics['f1']):.4f} +/- {np.std(xgb_metrics['f1']):.4f}")
    print(f"  G-Mean  : {np.mean(xgb_metrics['gmean']):.4f} +/- {np.std(xgb_metrics['gmean']):.4f}")

    # Plot Combined ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Plot
    fpr_lr, tpr_lr, _ = roc_curve(y_def, lr_oof_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_def, xgb_oof_probs)
    ax1.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_def, lr_oof_probs):.3f})')
    ax1.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_def, xgb_oof_probs):.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve (Default Risk - OOF)')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # PR Plot
    prec_lr_full, rec_lr_full, _ = precision_recall_curve(y_def, lr_oof_probs)
    prec_xgb_full, rec_xgb_full, _ = precision_recall_curve(y_def, xgb_oof_probs)
    ax2.plot(rec_lr_full, prec_lr_full, label=f'Logistic Regression (PR-AUC = {auc(rec_lr_full, prec_lr_full):.3f})')
    ax2.plot(rec_xgb_full, prec_xgb_full, label=f'XGBoost (PR-AUC = {auc(rec_xgb_full, prec_xgb_full):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve (Default Risk - OOF)')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_def_path = os.path.join("reports", "figures", "default_roc.png")
    plt.savefig(plot_def_path)
    plt.close()
    print(f"Saved combined ROC/PR curves plot to {plot_def_path}")

    # Select best architecture by OOF ROC-AUC
    mean_auc_xgb = np.mean(xgb_metrics['auc'])
    mean_auc_lr = np.mean(lr_metrics['auc'])
    if mean_auc_xgb >= mean_auc_lr:
        print("\n--> XGBoost selected as the best Default risk architecture.")
        best_def_type = 'xgb'
    else:
        print("\n--> Logistic Regression selected as the best Default risk architecture.")
        best_def_type = 'lr'

    # Fit final model on the ENTIRE dataset for production
    final_scaler_def = StandardScaler()
    X_def_scaled = final_scaler_def.fit_transform(X_def)
    
    if best_def_type == 'xgb':
        pos_weight_def_full = (len(y_def) - sum(y_def)) / sum(y_def) if sum(y_def) > 0 else 1.0
        X_def_noisy = inject_train_noise_and_clip(X_def, ['credit_score', 'current_age'], model_config)
        X_def_combined = pd.concat([X_def, X_def_noisy], axis=0)
        y_def_combined = pd.concat([y_def, y_def], axis=0)
        
        best_def_model = XGBClassifier(
            scale_pos_weight=pos_weight_def_full, 
            eval_metric='logloss',
            max_depth=int(model_config.get("max_depth", 3)),
            min_child_weight=float(model_config.get("min_child_weight", 5.0)),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=float(model_config.get("reg_alpha", 1.5)),
            reg_lambda=float(model_config.get("reg_lambda", 3.0)),
            random_state=42
        )
        best_def_model.fit(X_def_combined, y_def_combined)
    else:
        best_def_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        best_def_model.fit(X_def_scaled, y_def)

    # Save best model and scaler
    with open(os.path.join("models", "best_default_model.pkl"), "wb") as f:
        pickle.dump({'model': best_def_model, 'type': best_def_type, 'features': feature_names_def}, f)
    with open(os.path.join("models", "default_scaler.pkl"), "wb") as f:
        pickle.dump(final_scaler_def, f)

    # --- Permutation Feature Importance & Monte Carlo for Default Risk ---
    print("\nCalculating Permutation Feature Importance for Default Risk...")
    pfi_f1, pfi_gmean = run_permutation_importance(best_def_model, X_def, y_def, is_xgb=(best_def_type=='xgb'), scaler=final_scaler_def)
    print("\nPermutation Feature Importance (Top Features causing F1 Score Drop):")
    sorted_pfi = sorted(pfi_f1.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_pfi[:7]:
        print(f"  {feat:25s}: {score:.4f}")

    print("\nRunning Monte Carlo Noise Injection Simulator for Default Risk...")
    mc_results_def = run_monte_carlo_noise_test(
        best_def_model, X_def, y_def, is_xgb=(best_def_type=='xgb'), 
        scaler=final_scaler_def, suspect_features=suspect_features_def, num_simulations=50
    )
    
    # Plot Monte Carlo Performance Decay Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for feat, res in mc_results_def.items():
        ax1.errorbar(res['noise_levels'] * 100, res['f1_mean'], yerr=res['f1_std'], label=f"{feat} (F1)", marker='o')
        ax2.errorbar(res['noise_levels'] * 100, res['gmean_mean'], yerr=res['gmean_std'], label=f"{feat} (G-Mean)", marker='x', linestyle='--')
        
    ax1.set_xlabel('Noise level (% of Feature Std Dev)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Default Risk - F1-Score Decay under Noise (Monte Carlo)')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    
    ax2.set_xlabel('Noise level (% of Feature Std Dev)')
    ax2.set_ylabel('G-Mean')
    ax2.set_title('Default Risk - G-Mean Decay under Noise (Monte Carlo)')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    plt.tight_layout()
    mc_plot_path_def = os.path.join("reports", "figures", "default_mc_leakage.png")
    plt.savefig(mc_plot_path_def)
    plt.close()
    print(f"Saved Monte Carlo leakage plot to {mc_plot_path_def}")

    # ----------------------------------------------------
    # PART 2: FRAUD DETECTION MODEL
    # ----------------------------------------------------
    print("\n" + "="*60)
    print(" TRAINING FRAUD DETECTION MODEL WITH 5-FOLD STRATIFIED K-FOLD ")
    print("="*60)
    
    X_frd = df.drop(columns=fraud_exclude)
    y_frd = df['fraud']
    feature_names_frd = list(X_frd.columns)
    
    print(f"Features count: {len(feature_names_frd)}")
    print(f"Target distribution: {y_frd.value_counts().to_dict()}")

    # Storage for OOF predictions
    lr_oof_probs_f = np.zeros(len(df))
    lr_oof_preds_f = np.zeros(len(df))
    xgb_oof_probs_f = np.zeros(len(df))
    xgb_oof_preds_f = np.zeros(len(df))
    
    lr_metrics_f = {'auc': [], 'pr_auc': [], 'f1': [], 'gmean': []}
    xgb_metrics_f = {'auc': [], 'pr_auc': [], 'f1': [], 'gmean': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_frd, y_frd)):
        X_train, X_test = X_frd.iloc[train_idx], X_frd.iloc[test_idx]
        y_train, y_test = y_frd.iloc[train_idx], y_frd.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale pos weight
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
        
        # 1. Logistic Regression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        y_pred_lr = lr.predict(X_test_scaled)
        
        lr_oof_probs_f[test_idx] = y_prob_lr
        lr_oof_preds_f[test_idx] = y_pred_lr
        
        # Metrics LR
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
        pr_auc_lr = auc(rec_lr, prec_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        gmean_lr = compute_g_mean(y_test, y_pred_lr)
        
        lr_metrics_f['auc'].append(auc_lr)
        lr_metrics_f['pr_auc'].append(pr_auc_lr)
        lr_metrics_f['f1'].append(f1_lr)
        lr_metrics_f['gmean'].append(gmean_lr)
        
        # 2. XGBoost
        xgb = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
        y_pred_xgb = xgb.predict(X_test)
        
        xgb_oof_probs_f[test_idx] = y_prob_xgb
        xgb_oof_preds_f[test_idx] = y_pred_xgb
        
        # Metrics XGB
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
        pr_auc_xgb = auc(rec_xgb, prec_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        gmean_xgb = compute_g_mean(y_test, y_pred_xgb)
        
        xgb_metrics_f['auc'].append(auc_xgb)
        xgb_metrics_f['pr_auc'].append(pr_auc_xgb)
        xgb_metrics_f['f1'].append(f1_xgb)
        xgb_metrics_f['gmean'].append(gmean_xgb)
        
        print(f"Fold {fold+1} Fraud Model - LR AUC: {auc_lr:.4f} | XGB AUC: {auc_xgb:.4f}")

    print("\n" + "-"*40)
    print(" SUMMARY METRICS (5-FOLD OOF AVERAGE) ")
    print("-"*40)
    print("Logistic Regression (Fraud Detection):")
    print(f"  ROC-AUC : {np.mean(lr_metrics_f['auc']):.4f} +/- {np.std(lr_metrics_f['auc']):.4f}")
    print(f"  PR-AUC  : {np.mean(lr_metrics_f['pr_auc']):.4f} +/- {np.std(lr_metrics_f['pr_auc']):.4f}")
    print(f"  F1-Score: {np.mean(lr_metrics_f['f1']):.4f} +/- {np.std(lr_metrics_f['f1']):.4f}")
    print(f"  G-Mean  : {np.mean(lr_metrics_f['gmean']):.4f} +/- {np.std(lr_metrics_f['gmean']):.4f}")
    print("XGBoost (Fraud Detection):")
    print(f"  ROC-AUC : {np.mean(xgb_metrics_f['auc']):.4f} +/- {np.std(xgb_metrics_f['auc']):.4f}")
    print(f"  PR-AUC  : {np.mean(xgb_metrics_f['pr_auc']):.4f} +/- {np.std(xgb_metrics_f['pr_auc']):.4f}")
    print(f"  F1-Score: {np.mean(xgb_metrics_f['f1']):.4f} +/- {np.std(xgb_metrics_f['f1']):.4f}")
    print(f"  G-Mean  : {np.mean(xgb_metrics_f['gmean']):.4f} +/- {np.std(xgb_metrics_f['gmean']):.4f}")

    # Plot Combined ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Plot
    fpr_lr_f, tpr_lr_f, _ = roc_curve(y_frd, lr_oof_probs_f)
    fpr_xgb_f, tpr_xgb_f, _ = roc_curve(y_frd, xgb_oof_probs_f)
    ax1.plot(fpr_lr_f, tpr_lr_f, label=f'Logistic Regression (AUC = {roc_auc_score(y_frd, lr_oof_probs_f):.3f})')
    ax1.plot(fpr_xgb_f, tpr_xgb_f, label=f'XGBoost (AUC = {roc_auc_score(y_frd, xgb_oof_probs_f):.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve (Fraud Detection - OOF)')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # PR Plot
    prec_lr_f_full, rec_lr_f_full, _ = precision_recall_curve(y_frd, lr_oof_probs_f)
    prec_xgb_f_full, rec_xgb_f_full, _ = precision_recall_curve(y_frd, xgb_oof_probs_f)
    ax2.plot(rec_lr_f_full, prec_lr_f_full, label=f'Logistic Regression (PR-AUC = {auc(rec_lr_f_full, prec_lr_f_full):.3f})')
    ax2.plot(rec_xgb_f_full, prec_xgb_f_full, label=f'XGBoost (PR-AUC = {auc(rec_xgb_f_full, prec_xgb_f_full):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve (Fraud Detection - OOF)')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_frd_path = os.path.join("reports", "figures", "fraud_roc.png")
    plt.savefig(plot_frd_path)
    plt.close()
    print(f"Saved combined ROC/PR curves plot to {plot_frd_path}")

    # Select best architecture
    mean_auc_xgb_f = np.mean(xgb_metrics_f['auc'])
    mean_auc_lr_f = np.mean(lr_metrics_f['auc'])
    if mean_auc_xgb_f >= mean_auc_lr_f:
        print("\n--> XGBoost selected as the best Fraud detection architecture.")
        best_frd_type = 'xgb'
    else:
        print("\n--> Logistic Regression selected as the best Fraud detection architecture.")
        best_frd_type = 'lr'

    # Fit final model on the ENTIRE dataset for production
    final_scaler_frd = StandardScaler()
    X_frd_scaled = final_scaler_frd.fit_transform(X_frd)
    
    if best_frd_type == 'xgb':
        pos_weight_frd_full = (len(y_frd) - sum(y_frd)) / sum(y_frd) if sum(y_frd) > 0 else 1.0
        best_frd_model = XGBClassifier(scale_pos_weight=pos_weight_frd_full, eval_metric='logloss', random_state=42)
        best_frd_model.fit(X_frd, y_frd)
    else:
        best_frd_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        best_frd_model.fit(X_frd_scaled, y_frd)

    # Save best model and scaler
    with open(os.path.join("models", "best_fraud_model.pkl"), "wb") as f:
        pickle.dump({'model': best_frd_model, 'type': best_frd_type, 'features': feature_names_frd}, f)
    with open(os.path.join("models", "fraud_scaler.pkl"), "wb") as f:
        pickle.dump(final_scaler_frd, f)

    # --- Permutation Feature Importance & Monte Carlo for Fraud Detection ---
    print("\nCalculating Permutation Feature Importance for Fraud Detection...")
    pfi_f1_f, pfi_gmean_f = run_permutation_importance(best_frd_model, X_frd, y_frd, is_xgb=(best_frd_type=='xgb'), scaler=final_scaler_frd)
    print("\nPermutation Feature Importance (Top Features causing F1 Score Drop):")
    sorted_pfi_f = sorted(pfi_f1_f.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_pfi_f[:7]:
        print(f"  {feat:25s}: {score:.4f}")

    print("\nRunning Monte Carlo Noise Injection Simulator for Fraud Detection...")
    mc_results_frd = run_monte_carlo_noise_test(
        best_frd_model, X_frd, y_frd, is_xgb=(best_frd_type=='xgb'), 
        scaler=final_scaler_frd, suspect_features=suspect_features_frd, num_simulations=50
    )
    
    # Plot Monte Carlo Performance Decay Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for feat, res in mc_results_frd.items():
        ax1.errorbar(res['noise_levels'] * 100, res['f1_mean'], yerr=res['f1_std'], label=f"{feat} (F1)", marker='o')
        ax2.errorbar(res['noise_levels'] * 100, res['gmean_mean'], yerr=res['gmean_std'], label=f"{feat} (G-Mean)", marker='x', linestyle='--')
        
    ax1.set_xlabel('Noise level (% of Feature Std Dev)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Fraud Detection - F1-Score Decay under Noise (Monte Carlo)')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    
    ax2.set_xlabel('Noise level (% of Feature Std Dev)')
    ax2.set_ylabel('G-Mean')
    ax2.set_title('Fraud Detection - G-Mean Decay under Noise (Monte Carlo)')
    ax2.legend(loc='lower left')
    ax2.grid(True)
    
    plt.tight_layout()
    mc_plot_path_frd = os.path.join("reports", "figures", "fraud_mc_leakage.png")
    plt.savefig(mc_plot_path_frd)
    plt.close()
    print(f"Saved Monte Carlo leakage plot to {mc_plot_path_frd}")
    # Save validation metrics to JSON
    metrics_data = {
        "default_risk_model": {
            "logistic_regression": {
                "roc_auc_cv": float(np.mean(lr_metrics['auc'])),
                "roc_auc_cv_std": float(np.std(lr_metrics['auc'])),
                "pr_auc_cv": float(np.mean(lr_metrics['pr_auc'])),
                "pr_auc_cv_std": float(np.std(lr_metrics['pr_auc'])),
                "f1_score_cv": float(np.mean(lr_metrics['f1'])),
                "f1_score_cv_std": float(np.std(lr_metrics['f1'])),
                "g_mean_cv": float(np.mean(lr_metrics['gmean'])),
                "g_mean_cv_std": float(np.std(lr_metrics['gmean']))
            },
            "xgboost": {
                "roc_auc_cv": float(np.mean(xgb_metrics['auc'])),
                "roc_auc_cv_std": float(np.std(xgb_metrics['auc'])),
                "pr_auc_cv": float(np.mean(xgb_metrics['pr_auc'])),
                "pr_auc_cv_std": float(np.std(xgb_metrics['pr_auc'])),
                "f1_score_cv": float(np.mean(xgb_metrics['f1'])),
                "f1_score_cv_std": float(np.std(xgb_metrics['f1'])),
                "g_mean_cv": float(np.mean(xgb_metrics['gmean'])),
                "g_mean_cv_std": float(np.std(xgb_metrics['gmean']))
            },
            "selected_model": best_def_type
        },
        "fraud_detection_model": {
            "logistic_regression": {
                "roc_auc_cv": float(np.mean(lr_metrics_f['auc'])),
                "roc_auc_cv_std": float(np.std(lr_metrics_f['auc'])),
                "pr_auc_cv": float(np.mean(lr_metrics_f['pr_auc'])),
                "pr_auc_cv_std": float(np.std(lr_metrics_f['pr_auc'])),
                "f1_score_cv": float(np.mean(lr_metrics_f['f1'])),
                "f1_score_cv_std": float(np.std(lr_metrics_f['f1'])),
                "g_mean_cv": float(np.mean(lr_metrics_f['gmean'])),
                "g_mean_cv_std": float(np.std(lr_metrics_f['gmean']))
            },
            "xgboost": {
                "roc_auc_cv": float(np.mean(xgb_metrics_f['auc'])),
                "roc_auc_cv_std": float(np.std(xgb_metrics_f['auc'])),
                "pr_auc_cv": float(np.mean(xgb_metrics_f['pr_auc'])),
                "pr_auc_cv_std": float(np.std(xgb_metrics_f['pr_auc'])),
                "f1_score_cv": float(np.mean(xgb_metrics_f['f1'])),
                "f1_score_cv_std": float(np.std(xgb_metrics_f['f1'])),
                "g_mean_cv": float(np.mean(xgb_metrics_f['gmean'])),
                "g_mean_cv_std": float(np.std(xgb_metrics_f['gmean']))
            },
            "selected_model": best_frd_type
        }
    }
    metrics_json_path = os.path.join("data", "configs", "training_metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved validation metrics to {metrics_json_path}")

    print("\n" + "="*60)
    print(" MODEL TRAINING, VALIDATION, AND LEAKAGE AUDIT COMPLETED ")
    print("="*60)


if __name__ == "__main__":
    train_and_evaluate()

