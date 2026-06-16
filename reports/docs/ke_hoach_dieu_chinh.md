# Kế hoạch điều chỉnh Mô hình Dự báo Vỡ nợ (Default Risk Model Calibration Plan)

**Dự án:** Xóm Bank Credit Underwriting  
**Mục tiêu:** Khắc phục triệt để hiện tượng quá khớp (overfitting) và độ ổn định kém (Low Stability) của hai biến liên tục là Điểm tín dụng (`credit_score`) và Độ tuổi (`current_age`) dưới tác động nhiễu, được phát hiện qua kiểm định Monte Carlo.

---

## 1. Cơ sở khoa học & Nhận diện vấn đề
Qua phân tích Monte Carlo progressive noise, chỉ số F1-Score của mô hình XGBoost Default Risk sụt giảm đột ngột **10%** chỉ khi bổ sung mức nhiễu siêu nhỏ **10%** của độ lệch chuẩn ($\sigma$):
*   `credit_score` ($\sigma = 67.22$): Bị sụt giảm hiệu năng nghiêm trọng khi có sai số nhiễu $\pm 6.7$ điểm FICO.
*   `current_age` ($\sigma = 18.41$): Bị ảnh hưởng mạnh khi có sai lệch tuổi $\pm 1.8$ năm.

**Nguyên nhân:** Mô hình XGBoost được huấn luyện với tham số mặc định (`max_depth = 6`) kết hợp với trọng số lớp mất cân đối cao (`scale_pos_weight = 22.25`) đã tạo dựng các điểm phân nhánh quá sâu, bám chặt vào vị trí chính xác của từng quan sát vỡ nợ (Positive class) trên không gian đặc trưng liên tục.

---

## 2. Giải pháp điều chỉnh chi tiết

### A. Kiểm soát độ phức tạp cây quyết định (Model Complexity & Regularization)
Áp dụng các tham số giới hạn cấu trúc cây để buộc XGBoost xây dựng các quyết định phân lớp mang tính tổng quát hóa cao hơn:
1.  **`max_depth`:** Giảm từ $6$ xuống $3$ (hoặc tối đa là $4$). Cây nông hơn sẽ hạn chế các tương tác đặc trưng phi tuyến quá mức.
2.  **`min_child_weight`:** Tăng từ $1.0$ (mặc định) lên $5.0$. Buộc mỗi nút lá của cây quyết định phải chứa tổng trọng số mẫu tối thiểu tương đương ít nhất 5 quan sát thực tế, loại bỏ việc tối ưu hóa cho các điểm dữ liệu cá lẻ.
3.  **`subsample` và `colsample_bytree`:** Thiết lập ở mức $0.8$ để tạo tính ngẫu nhiên trên các tập mẫu và tập đặc trưng khi xây dựng từng cây con.
4.  **Regularization L1/L2 (`reg_alpha` / `reg_lambda`):** Đặt `reg_alpha = 1.5` và `reg_lambda = 3.0` để áp hình phạt lên các trọng số lá lớn.

### B. Huấn luyện đối kháng bằng cách bổ sung nhiễu (Adversarial Training via Noise Injection)
Chủ động đưa thêm sai số ngẫu nhiên vào dữ liệu huấn luyện để san phẳng ranh giới quyết định (decision boundaries):
1.  Trước khi huấn luyện mô hình trên từng Fold của Stratified K-Fold, nhân bản tập dữ liệu huấn luyện (Data Augmentation) và cộng nhiễu Gaussian ngẫu nhiên $N(0, \eta \cdot \sigma)$ vào hai biến `credit_score` và `current_age` với $\eta = 5\%$.
2.  Mô hình được huấn luyện đồng thời trên cả dữ liệu sạch và dữ liệu bị nhiễu sẽ học được cách bỏ qua các dao động nhỏ cục bộ của điểm số FICO và độ tuổi.

### C. Tối ưu hóa tham số thích ứng qua LLM (Pre-training LLM Optimization)
Sử dụng mô hình ngôn ngữ toán học `mightykatun/qwen2.5-math:1.5b` để phân tích các thống kê mô tả phân phối dữ liệu, từ đó tự động hóa việc tính toán:
1.  **Cận cắt biên logic (Clipping bounds):** Đề xuất giá trị chặn dưới và chặn trên tối ưu dựa trên phân phối thực tế của `credit_score` và `current_age`.
2.  **Siêu tham số XGBoost (Regularization Parameters):** Đề xuất cấu hình hyperparameter tối ưu nhằm tránh overfitting dưới dữ liệu nhiễu.

---

## 3. Các bước thực thi mã nguồn

### Bước 0: Tạo tập lệnh tối ưu hóa `src/optimize_hyperparameters.py`
Sử dụng `mightykatun/qwen2.5-math:1.5b` thông qua thư viện `ollama` để sinh file cấu hình `data/model_config.json`:
```python
import os
import json
import pandas as pd
from ollama import chat

def run_pre_training_optimization():
    print("Executing pre-training optimization using qwen2.5-math:1.5b...")
    df = pd.read_csv("data/user_features_matrix.csv")
    
    stats = {}
    for col in ['credit_score', 'current_age']:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
        
    prompt = f"""
    You are an expert mathematical optimization system for risk scoring models. 
    Analyze the following feature statistics from our credit underwriting database:
    {json.dumps(stats, indent=2)}
    
    Task:
    1. Calculate logical clipping thresholds to handle Gaussian noise without losing critical feature distributions.
    2. Propose optimized regularized parameters for XGBoost (max_depth, min_child_weight, reg_alpha, reg_lambda).
    
    Your output MUST be a single raw JSON block with the following keys, and nothing else (no reasoning, no extra text):
    {{
        "credit_score_min": <float>,
        "credit_score_max": <float>,
        "current_age_min": <float>,
        "current_age_max": <float>,
        "max_depth": <int>,
        "min_child_weight": <float>,
        "reg_alpha": <float>,
        "reg_lambda": <float>
    }}
    """
    
    response = chat(
        model='mightykatun/qwen2.5-math:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
    )
    
    raw_content = response.message.content.strip()
    # Clean possible markdown fencing
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    raw_content = raw_content.strip()
        
    try:
        config = json.loads(raw_content)
        os.makedirs("data", exist_ok=True)
        with open("data/model_config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Generated data/model_config.json successfully.")
    except Exception as e:
        print("Fallback to standard mathematical rules due to parse error:", e)
        fallback = {
            "credit_score_min": 300.0,
            "credit_score_max": 850.0,
            "current_age_min": 18.0,
            "current_age_max": 100.0,
            "max_depth": 3,
            "min_child_weight": 5.0,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0
        }
        with open("data/model_config.json", "w") as f:
            json.dump(fallback, f, indent=4)

if __name__ == "__main__":
    run_pre_training_optimization()
```


### Bước 1: Cập nhật hàm huấn luyện trong `src/train.py`
Thay đổi khai báo khởi tạo XGBoost trong vòng lặp huấn luyện chéo và huấn luyện sản xuất để đọc các siêu tham số từ `data/model_config.json`:
```python
# Tải cấu hình tối ưu từ pre-training step
try:
    with open("data/model_config.json", "r") as f:
        config = json.load(f)
except Exception:
    config = {
        "credit_score_min": 300.0,
        "credit_score_max": 850.0,
        "current_age_min": 18.0,
        "current_age_max": 100.0,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "reg_alpha": 1.5,
        "reg_lambda": 3.0
    }

# Sau điều chỉnh (Nạp cấu hình động đề xuất từ LLM):
xgb = XGBClassifier(
    scale_pos_weight=pos_weight,
    eval_metric='logloss',
    max_depth=int(config.get("max_depth", 3)),
    min_child_weight=float(config.get("min_child_weight", 5.0)),
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=float(config.get("reg_alpha", 1.5)),
    reg_lambda=float(config.get("reg_lambda", 3.0)),
    random_state=42
)
```

### Bước 2: Viết hàm bổ sung nhiễu huấn luyện kèm cắt biên (Train-time Augmentation with Clipping)
Tạo generator hỗ trợ sinh nhiễu đối kháng và thực hiện cắt biên (clipping) tự động dựa trên cấu hình:
```python
def inject_train_noise_and_clip(X, columns, config, noise_level=0.05):
    X_noisy = X.copy()
    for col in columns:
        if col in X_noisy.columns:
            std = X_noisy[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            
    # Áp dụng giới hạn clipping tối ưu đề xuất từ LLM
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
```
Áp dụng `inject_train_noise_and_clip` để ghép thêm (concat) dữ liệu nhiễu vào `X_train` trước khi chạy `xgb.fit()`.


---

## 4. Kế hoạch nghiệm thu & Thẩm định lại hiệu năng (Verification Plan)

Sau khi chạy mã nguồn điều chỉnh, hệ thống cần đáp ứng các tiêu chuẩn kỹ thuật sau:
1.  **Hiệu năng Cross-Validation ($ROC\text{-}AUC_{CV}$):** Đạt tối thiểu $0.5700$ (cho phép giảm nhẹ so với $0.5884$ của mô hình cũ để đổi lấy tính ổn định).
2.  **Chỉ số Monte Carlo Stability Rating:** Cả hai biến `credit_score` và `current_age` phải nâng hạng đánh giá từ `🔴 Yếu` lên **`🟢 Rất cao`** hoặc **`🟡 Trung bình`** (F1-score chỉ bắt đầu suy giảm $10\%$ khi nhiễu vượt qua ngưỡng $50\%$ độ lệch chuẩn).
3.  **Tương thích tích hợp:** Mô hình đã lưu (`best_default_model.pkl`) phải load và chạy giả lập trơn tru trên Streamlit Dashboard (`src/app.py`).
