import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def calculate_accuracy():
    csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run src/data_pipeline/feature_engineering.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Default Risk Model Accuracy
    def_model_path = os.path.join("models", "best_default_model.pkl")
    if not os.path.exists(def_model_path):
        print(f"Error: {def_model_path} not found. Please train models first.")
        return
        
    with open(def_model_path, "rb") as f:
        def_data = pickle.load(f)
    def_model = def_data['model']
    def_features = def_data['features']
    
    if def_data['type'] == 'lr':
        with open(os.path.join("models", "default_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        X_def = scaler.transform(df[def_features])
    else:
        X_def = df[def_features]
    y_def = df['default']
    y_pred_def = def_model.predict(X_def)
    acc_def = accuracy_score(y_def, y_pred_def)
    cm_def = confusion_matrix(y_def, y_pred_def)
    
    # 2. Fraud Detection Model Accuracy
    frd_model_path = os.path.join("models", "best_fraud_model.pkl")
    if not os.path.exists(frd_model_path):
        print(f"Error: {frd_model_path} not found.")
        return
        
    with open(frd_model_path, "rb") as f:
        frd_data = pickle.load(f)
    frd_model = frd_data['model']
    frd_features = frd_data['features']
    
    if frd_data['type'] == 'lr':
        with open(os.path.join("models", "fraud_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        X_frd = scaler.transform(df[frd_features])
    else:
        X_frd = df[frd_features]
    y_frd = df['fraud']
    y_pred_frd = frd_model.predict(X_frd)
    acc_frd = accuracy_score(y_frd, y_pred_frd)
    cm_frd = confusion_matrix(y_frd, y_pred_frd)
    
    results = {
        "default_risk": {
            "model_type": def_data['type'],
            "accuracy": float(acc_def),
            "confusion_matrix": cm_def.tolist(),
            "target_distribution": y_def.value_counts().to_dict(),
            "classification_report": classification_report(y_def, y_pred_def, output_dict=True)
        },
        "fraud_detection": {
            "model_type": frd_data['type'],
            "accuracy": float(acc_frd),
            "confusion_matrix": cm_frd.tolist(),
            "target_distribution": y_frd.value_counts().to_dict(),
            "classification_report": classification_report(y_frd, y_pred_frd, output_dict=True)
        }
    }
    
    print("CALCULATED METRICS:")
    print(json.dumps(results, indent=2))
    print("=" * 60)
    
    # Run Qwen-Math Validator
    try:
        from ollama import chat
        prompt = f"""
        You are a senior quantitative validator and mathematical model auditor.
        Analyze the following training results and metrics of our credit risk and fraud detection models:
        {json.dumps(results, indent=2)}
        
        Tasks:
        1. Evaluate the global accuracy of both models. Explain the mathematical relationship between the global accuracy and the target class imbalance (majority class dominance).
        2. Assess the quality of the confusion matrix. Explain whether global accuracy is a reliable validator metric for default risk (86 positive vs 1914 negative) and fraud (246 positive vs 1754 negative) under Basel/regulatory frameworks.
        3. Make recommendations for model monitoring.
        
        Provide your audit report in a clear mathematical structure with LaTeX equations where appropriate.
        """
        print("Querying mightykatun/qwen2.5-math:1.5b...")
        response = chat(
            model='mightykatun/qwen2.5-math:1.5b',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.2,
                'num_ctx': 4096
            }
        )
        print("\n=== AUDIT REPORT FROM MIGHTYKATUN/QWEN2.5-MATH:1.5B ===")
        print(response.message.content)
    except Exception as e:
        print("Failed to run Ollama validator:", e)

if __name__ == "__main__":
    calculate_accuracy()
