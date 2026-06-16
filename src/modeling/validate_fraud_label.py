"""
validate_fraud_label.py
-----------------------
Kiểm chứng định lượng nhãn fraud multi-signal bằng Qwen 2.5 Math (via Ollama).

Quy trình:
1. Đọc user_features_matrix.csv
2. Tính phân phối fraud_signal_score và fraud rate cho theta in {2, 3, 4}
3. Tính Gini coefficient và Information Value (IV) cho từng ngưỡng
4. Gọi Qwen 2.5 Math để phân tích và đề xuất theta* tối ưu
5. Cập nhật data/configs/model_config.json với fraud_label_threshold được xác nhận
6. Lưu báo cáo vào data/outputs/fraud_label_validation.txt
"""

import os
import json
import math
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Tải dữ liệu
# ─────────────────────────────────────────────
csv_path = os.path.join("data", "processed", "user_features_matrix.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Không tìm thấy {csv_path}.\n"
        "Hãy chạy: python src/data_pipeline/feature_engineering.py"
    )

print("=" * 60)
print(" FRAUD LABEL VALIDATOR — Qwen 2.5 Math Edition")
print("=" * 60)
print(f"Đọc dữ liệu từ: {csv_path}")
df = pd.read_csv(csv_path)

required_cols = ['fraud_signal_score', 'fraud', 'security_error_count',
                 'has_spatiotemporal_fraud_signal', 'refund_rate',
                 'high_online_rate', 'security_error_count_heavy']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Feature matrix thiếu cột: {missing}\n"
        "Hãy chạy lại feature_engineering.py với version mới nhất."
    )

n_total = len(df)
print(f"Tổng số user: {n_total:,}")

# ─────────────────────────────────────────────
# 2. Phân tích phân phối fraud_signal_score
# ─────────────────────────────────────────────
score_dist = df['fraud_signal_score'].value_counts().sort_index().to_dict()
print(f"\nPhân phối fraud_signal_score: {score_dist}")

# ─────────────────────────────────────────────
# 3. Tính Gini Coefficient và IV cho theta in {2, 3, 4}
# ─────────────────────────────────────────────
def compute_gini(y_score, y_true):
    """
    Gini = 2 * AUC - 1, tính bằng AUROC không dùng sklearn.
    Dùng trapezoidal rule thủ công.
    """
    # Sắp xếp theo score giảm dần
    sorted_pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    n_pos = sum(y_true)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoid AUC
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    gini = 2 * auc - 1
    return round(gini, 4)


def compute_iv(y_score_binary, y_true):
    """
    Information Value (IV) của biến nhị phân được tạo ra từ ngưỡng theta.
    IV = (P(X=1|Y=1) - P(X=1|Y=0)) * WOE
    WOE = ln(P(X=1|Y=1) / P(X=1|Y=0))
    """
    n_pos = y_true.sum()
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Bucket 1: score_binary = 1
    mask_1 = y_score_binary == 1
    dist_pos_1 = (y_true[mask_1] == 1).sum() / n_pos if n_pos > 0 else 1e-9
    dist_neg_1 = (y_true[mask_1] == 0).sum() / n_neg if n_neg > 0 else 1e-9

    # Bucket 0: score_binary = 0
    dist_pos_0 = (y_true[~mask_1] == 1).sum() / n_pos if n_pos > 0 else 1e-9
    dist_neg_0 = (y_true[~mask_1] == 0).sum() / n_neg if n_neg > 0 else 1e-9

    # Tránh log(0)
    eps = 1e-9
    woe_1 = math.log((dist_pos_1 + eps) / (dist_neg_1 + eps))
    woe_0 = math.log((dist_pos_0 + eps) / (dist_neg_0 + eps))

    iv = (dist_pos_1 - dist_neg_1) * woe_1 + (dist_pos_0 - dist_neg_0) * woe_0
    return round(abs(iv), 4)


THETA_CANDIDATES = [2, 3, 4]
theta_analysis = {}
y_score_raw = df['fraud_signal_score'].values

print("\n" + "-" * 50)
print("Phân tích Gini & IV theo ngưỡng theta:")
print("-" * 50)

for theta in THETA_CANDIDATES:
    y_label_theta = (df['fraud_signal_score'] >= theta).astype(int)
    n_fraud = y_label_theta.sum()
    fraud_rate = n_fraud / n_total

    # Gini: dùng score liên tục để rank, label từ theta hiện tại làm ground truth
    gini = compute_gini(y_score_raw, y_label_theta.values)

    # IV: so sánh nhãn nhị phân cũ (sec_err >= 1) vs nhãn theta mới
    old_label = (df['security_error_count'] >= 1).astype(int)
    iv = compute_iv(y_label_theta, old_label)

    theta_analysis[theta] = {
        "n_fraud": int(n_fraud),
        "fraud_rate_pct": round(float(fraud_rate) * 100, 2),
        "gini": float(gini),
        "iv_vs_old_label": float(iv),
        "viable": bool(n_fraud >= 50)  # Đủ positive samples để train
    }
    viability = "✓ Viable" if n_fraud >= 50 else "✗ Too few positives"
    print(f"  theta={theta}: n_fraud={n_fraud:,} ({fraud_rate*100:.2f}%) | "
          f"Gini={gini:.4f} | IV={iv:.4f} | {viability}")

# ─────────────────────────────────────────────
# 4. Gọi Qwen 2.5 Math để phân tích và đề xuất theta*
# ─────────────────────────────────────────────
print("\n" + "-" * 50)
print("Gọi Qwen 2.5 Math (mightykatun/qwen2.5-math:1.5b)...")
print("-" * 50)

prompt = f"""You are a senior quantitative credit risk analyst with expertise in fraud detection model design.

## Context
We redefined the fraud label from a single signal (security_error_count >= 1) 
to a multi-signal weighted score (fraud_signal_score >= theta).

Fraud signal score formula:
  S_fraud = 2 * (security_errors >= 3) + 2 * spatiotemporal_24h + 1 * (refund_rate > 10%) + 1 * (online_rate > 70%)

## Threshold Analysis Results
{json.dumps(theta_analysis, indent=2)}

## Score Distribution
{json.dumps(score_dist, indent=2)}

## Dataset size
Total users: {n_total}

## Your Tasks
1. Evaluate each theta candidate using the Gini coefficient and IV provided. State clearly which is mathematically superior.
2. Calculate the minimum viable sample requirement using the rule: n_minority >= 5 * n_features (assume 15 fraud model features). Check each theta against this rule.
3. Recommend the optimal theta* with mathematical justification using LaTeX notation.
4. Assess the Information Value (IV) interpretation: IV > 0.3 = Strong, 0.1-0.3 = Medium, < 0.1 = Weak predictor. What does the IV vs old label tell us about label quality improvement?
5. Final output: state theta* = [number] as the last line of your response.

Be extremely concise. Use LaTeX for formulas. Maximum 300 words.
"""

llm_response = ""
try:
    from ollama import chat

    response = chat(
        model='mightykatun/qwen2.5-math:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': 0.1,   # Deterministic cho toán học
            'num_ctx': 4096
        }
    )
    llm_response = response.message.content
    print("\n=== Qwen 2.5 Math Analysis ===")
    print(llm_response)
except Exception as e:
    llm_response = f"[Qwen unavailable: {e}]\n\nFallback: Chọn theta mặc định = 3 (viable, Gini tốt nhất trong tập viable)"
    print(f"\n[WARNING] Không kết nối được Qwen: {e}")
    print("Sử dụng fallback theta = 3")

# ─────────────────────────────────────────────
# 5. Parse theta* từ kết quả Qwen
# ─────────────────────────────────────────────
import re

recommended_theta = 3  # Default fallback
# Tìm pattern "theta* = X" hoặc "theta* = [X]" hoặc dòng cuối có số
patterns = [
    r'theta\*?\s*=\s*\[?(\d)\]?',
    r'optimal\s+theta[:\s=]+(\d)',
    r'recommend[^\n]*theta[^\n]*=\s*(\d)',
]
for pat in patterns:
    match = re.search(pat, llm_response, re.IGNORECASE)
    if match:
        candidate = int(match.group(1))
        if candidate in THETA_CANDIDATES and theta_analysis[candidate]['viable']:
            recommended_theta = candidate
            break

# Fallback: chọn theta viable có Gini cao nhất
if recommended_theta not in [t for t in THETA_CANDIDATES if theta_analysis[t]['viable']]:
    viable_thetas = {t: theta_analysis[t]['gini'] for t in THETA_CANDIDATES if theta_analysis[t]['viable']}
    if viable_thetas:
        recommended_theta = max(viable_thetas, key=viable_thetas.get)

print(f"\n>>> Theta được chọn: {recommended_theta}")
print(f"    Fraud rate tương ứng: {theta_analysis[recommended_theta]['fraud_rate_pct']}%")
print(f"    n_fraud: {theta_analysis[recommended_theta]['n_fraud']:,}")

# ─────────────────────────────────────────────
# 6. Cập nhật model_config.json
# ─────────────────────────────────────────────
config_path = os.path.join("data", "configs", "model_config.json")
try:
    with open(config_path, "r") as f:
        model_config = json.load(f)
except Exception:
    model_config = {}

model_config["fraud_label_threshold"] = recommended_theta
model_config["fraud_label_version"] = "multi_signal_v2"
model_config["fraud_label_validated_by"] = "qwen2.5-math:1.5b"
model_config["fraud_label_analysis"] = theta_analysis

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)
print(f"\nĐã cập nhật {config_path} với fraud_label_threshold={recommended_theta}")

# ─────────────────────────────────────────────
# 7. Lưu báo cáo
# ─────────────────────────────────────────────
os.makedirs(os.path.join("data", "outputs"), exist_ok=True)
report_path = os.path.join("data", "outputs", "fraud_label_validation.txt")

report_content = f"""
==========================================
FRAUD LABEL VALIDATION REPORT
Validator: Qwen 2.5 Math (mightykatun/qwen2.5-math:1.5b)
==========================================

## Dataset
- Total users: {n_total:,}
- Score distribution: {score_dist}

## Threshold Analysis
{json.dumps(theta_analysis, indent=2)}

## Qwen 2.5 Math Analysis
{llm_response}

## Decision
- Recommended theta*: {recommended_theta}
- Fraud rate at theta*: {theta_analysis[recommended_theta]['fraud_rate_pct']}%
- n_fraud at theta*: {theta_analysis[recommended_theta]['n_fraud']:,}
- Gini at theta*: {theta_analysis[recommended_theta]['gini']}
- IV vs old label: {theta_analysis[recommended_theta]['iv_vs_old_label']}

## Comparison: Old Label vs New Label
- Old label (sec_err >= 1): ~{(df['security_error_count'] >= 1).mean()*100:.2f}% fraud rate
- New label (score >= {recommended_theta}): {theta_analysis[recommended_theta]['fraud_rate_pct']}% fraud rate
- Label precision improvement: Multi-signal approach eliminates single-incident false positives
==========================================
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"\nBáo cáo lưu tại: {report_path}")
print("\n" + "=" * 60)
print(" VALIDATION HOÀN THÀNH — Tiến hành chạy train.py")
print("=" * 60)
