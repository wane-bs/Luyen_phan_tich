import json
import os
from ollama import chat

# --- Load kết quả từ training_metrics.json nếu có ---
metrics_path = os.path.join("data", "configs", "training_metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            live_metrics = json.load(f)
        frd_model = live_metrics.get("fraud_detection_model", {})
        def_model = live_metrics.get("default_risk_model", {})
        selected_frd = frd_model.get("selected_model", "unknown")
        selected_def = def_model.get("selected_model", "unknown")
    except Exception:
        live_metrics, selected_frd, selected_def = {}, "unknown", "unknown"
else:
    live_metrics, selected_frd, selected_def = {}, "unknown", "unknown"

# --- Load kết quả fraud label validation nếu có ---
validation_path = os.path.join("data", "outputs", "fraud_label_validation.txt")
label_validation_summary = ""
if os.path.exists(validation_path):
    with open(validation_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Chỉ lấy phần Decision để giữ prompt ngắn
    if "## Decision" in raw:
        label_validation_summary = raw[raw.index("## Decision"):raw.index("## Comparison")]

# --- Kết quả mô hình baseline (từ run cũ, dùng làm benchmark so sánh) ---
# Chỉ lấy key metrics từ live_metrics để tránh overflow context window
def _trim_metrics(m):
    if not m:
        return "(Chưa có)"
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            lr = v.get("logistic_regression", {})
            xgb = v.get("xgboost", {})
            out[k] = {
                "LR_AUC": lr.get("roc_auc_cv"), "LR_F1": lr.get("f1_score_cv"), "LR_GMean": lr.get("g_mean_cv"),
                "XGB_AUC": xgb.get("roc_auc_cv"), "XGB_F1": xgb.get("f1_score_cv"),
                "selected": v.get("selected_model")
            }
    return out

results = {
    "default_risk_OLD": {
        "accuracy": 0.6095,
        "confusion_matrix": [[1166, 748], [33, 53]],
        "target_distribution": {"0": 1914, "1": 86}
    },
    "fraud_detection_OLD_label": {
        "note": "Label cũ: security_error_count >= 1 (Label Pollution)",
        "accuracy": 0.9720,
        "confusion_matrix": [[1698, 56], [0, 246]],
        "target_distribution": {"0": 1754, "1": 246}
    },
    "NEW_model_metrics": _trim_metrics(live_metrics)
}

label_section = f"""
## Fraud Label Redefinition Analysis
{label_validation_summary if label_validation_summary else "(Chưa có — chạy validate_fraud_label.py trước)"}
""" if label_validation_summary else ""

prompt = f"""
You are a senior quantitative risk validator. Analyze the following model metrics:
{json.dumps(results, indent=2)}
{label_section}

Audit tasks:
1. State the global accuracy of both models.
2. Evaluate class imbalance. Explain mathematically why global accuracy is highly deceptive for Default Risk (e.g., baseline majority class accuracy is 95.7%) but F1/G-Mean is more reliable.
3. Assess confusion matrix quality (note the 0 false negatives in fraud detection OLD label).
4. LABEL QUALITY ANALYSIS: The old fraud label (security_error_count >= 1) suffered from Label Pollution — it equated "victim of bad PIN entry" with "fraudster". The new multi-signal label requires S_fraud >= theta where S_fraud = 2*(sec_err>=3) + 2*spatiotemporal + 1*(refund_rate>10%) + 1*(online_rate>70%). Mathematically evaluate whether this multi-signal approach improves label precision using Information Value (IV) interpretation: IV > 0.3 = Strong, 0.1-0.3 = Medium.
5. Provide a brief mathematical evaluation using LaTeX for key formulas. Keep it extremely concise and direct.
"""

try:
    print("Querying mightykatun/qwen2.5-math:1.5b...")
    response = chat(
        model='mightykatun/qwen2.5-math:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': 0.1,
            'num_ctx': 2048,
            'num_predict': 600  # Chặn vòng lặp lặp lại output
        }
    )
    content = response.message.content
    import os
    os.makedirs(os.path.join("data", "outputs"), exist_ok=True)
    with open("data/outputs/model_audit_report.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("\n=== AUDIT REPORT SAVED TO data/outputs/model_audit_report.txt ===")
    lines = content.splitlines()
    print("\nPreview of the report:")
    for line in lines[:30]:
        print(line)
    if len(lines) > 30:
        print("...")
except Exception as e:
    print("Failed to query Ollama:", e)
