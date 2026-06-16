# Xóm Bank Credit Underwriting System 💳

Hệ thống chấm điểm tín dụng nội bộ, đánh giá rủi ro vỡ nợ (Default Risk), phát hiện gian lận (Fraud Detection) và phê duyệt hạn mức tín dụng tự động cho **Xóm Bank**.

Dự án tích hợp các mô hình máy học (Logistic Regression, XGBoost), kiểm định độ ổn định Monte Carlo, hệ thống kiểm chứng nhãn định lượng bằng **Qwen 2.5 Math** và giao diện giả lập phê duyệt thời gian thực (Streamlit Dashboard).

---

### ⚠️ Hướng dẫn Bảo mật & Tuyên bố miễn trừ trách nhiệm (Security Guide & Disclaimer)

*   **Mục đích**: Dự án được xây dựng phục vụ nghiên cứu, giả lập và kiểm thử nghiệp vụ tín dụng nội bộ của Xóm Bank. Mọi quy tắc định mức và thuật toán cần được thẩm định bởi Hội đồng rủi ro chuyên trách trước khi đưa vào thực tế.
*   **Nguồn dữ liệu**: Dữ liệu giao dịch ngân hàng giả lập được lấy theo cấu trúc bảng của [Xomdata Banking Schema Dataset](https://dataset.xomdata.com/datasets/schema/banking). Dữ liệu thật được bảo vệ nghiêm ngặt và không đẩy lên Git.
*   **Mô hình sử dụng**:
    *   **Default Risk**: Huấn luyện bằng XGBoost.
    *   **Fraud Detection**: Huấn luyện bằng Logistic Regression dựa trên nhãn hành vi Weighted Multi-Signal v2.
    *   **Label Validation & Hyperparameter Optimization**: Sử dụng mô hình toán học chuyên biệt `mightykatun/qwen2.5-math:1.5b` chạy cục bộ (offline) qua Ollama.
*   **Chính sách bảo mật (Zero-Trust)**:
    *   *Local Data Isolation*: Toàn bộ cơ sở dữ liệu SQLite cục bộ, ma trận đặc trưng (`data/processed/`), các file nhị phân mô hình (`models/*.pkl`) được cấu hình loại bỏ nghiêm ngặt tại [.gitignore](./.gitignore).
    *   *Credential Protection*: Không ghi cứng mật khẩu hay khóa truy cập trong mã nguồn. Thông tin kết nối SQL Server được tải qua tệp cấu hình `.env` cục bộ.
    *   *Local AI Execution*: Xử lý AI chạy hoàn toàn offline trên máy trạm của doanh nghiệp qua Ollama, bảo vệ thông tin khách hàng.
*   **Disclaimer**: Hiệu năng mô hình thực tế có thể biến động do thay đổi trong hành vi của đối tượng gian lận (Concept Drift). Tác giả không chịu trách nhiệm pháp lý cho các tổn thất tài chính hoặc rò rỉ dữ liệu phát sinh do cấu hình bảo mật lỏng lẻo từ phía người vận hành.

---

## 📊 Kiến trúc Hệ thống & Quy trình Hoạt động

Hệ thống được thiết kế theo mô hình Modular Pipeline bao gồm 5 giai đoạn cốt lõi:

```text
[Dữ liệu Thô] ──> [Feature Engineering] ──> [Label Validation] ──> [Training] ──> [Dashboard]
 (SQL Server)       (fraud_signal_score)      (Qwen 2.5 Math)     (XGB / LR)    (Streamlit)
```

1. **Data Pipeline**: Fork dữ liệu từ nguồn SQL Server cá nhân sang SQLite nội bộ, làm sạch và xây dựng ma trận đặc trưng ở cấp độ User (User-level Profile Matrix), bao gồm **Fraud Signal Score** multi-tín hiệu.
2. **Fraud Label Validation**: Kiểm chứng định lượng ngưỡng nhãn fraud bằng `mightykatun/qwen2.5-math:1.5b` — tính Gini coefficient và Information Value (IV) cho $\theta \in \{2,3,4\}$, tự động chọn $\theta^*$ tối ưu.
3. **Modeling**:
   - Tối ưu hóa siêu tham số thích ứng thông qua mô hình ngôn ngữ toán học `qwen2.5-math:1.5b`.
   - Huấn luyện mô hình XGBoost và Logistic Regression bằng phương pháp 5-Fold Stratified Cross-Validation.
   - Kiểm định đối kháng chống quá khớp (Overfitting) bằng phương pháp bổ sung nhiễu Gaussian (Noise Injection) kèm giới hạn biên logic (Clipping bounds).
4. **Validation (Monte Carlo)**: Kiểm định độ ổn định của các biến đặc trưng liên tục (`credit_score`, `current_age`,...) dưới tác động nhiễu progressive để đánh giá khả năng chống suy giảm hiệu năng ($F_1$-Score & $G$-Mean).
5. **Cash Flow-Based Underwriting**: Áp dụng quy tắc quản trị rủi ro dòng tiền (CFADS), chốt chặn đòn bẩy (Leverage Cap), chiết khấu rủi ro AI (AI Haircuts) và bộ lọc chốt chặn cưỡng chế (Circuit Breakers) để ra quyết định phê duyệt hạn mức tín dụng tối ưu.

---

## 🏷️ Fraud Label — Multi-Signal v2

Nhãn `fraud` được tái kiến trúc từ đơn tín hiệu (`security_error_count >= 1`) sang hệ thống **Weighted Multi-Signal Scoring** được Qwen 2.5 Math kiểm chứng:

$$S_{\text{fraud}}(u) = 2\cdot\mathbb{1}[\text{sec-err} \geq 3] + 2\cdot\mathbb{1}[\text{spatiotemporal}=1] + \mathbb{1}[\text{refund-rate}>0.10] + \mathbb{1}[\text{online-rate}>0.70]$$

$$\text{fraud}(u) = \mathbb{1}[S_{\text{fraud}}(u) \geq \theta^{*}], \quad \theta^{*} = 2 \quad \text{(Qwen validated, IV = 8.18 Strong)}$$

| Phiên bản | Fraud Rate | F1-Score | G-Mean | Ý nghĩa |
|:---|:---:|:---:|:---:|:---|
| Nhãn cũ (`sec_err≥1`) | 12.3% | ~1.0 (giả tạo) | N/A | Label Pollution |
| **Nhãn mới (θ*=2)** | **4.95%** | **0.6012** | **0.9379** | Behavioral patterns thực sự |

---

## 📂 Cấu trúc Thư mục Dự án (Standardized)

Dự án được tổ chức theo cấu trúc chuẩn hóa dành cho các ứng dụng Data Science & Machine Learning:

```text
f:\data_project\
├── data\
│   ├── raw\                    # Dữ liệu gốc (SQLite, CSV thô)
│   │   └── local_replica.db
│   ├── processed\              # Dữ liệu sau xử lý / Ma trận đặc trưng
│   │   └── user_features_matrix.csv
│   ├── configs\                # Cấu hình siêu tham số và siêu dữ liệu mô hình
│   │   ├── model_config.json   # fraud_label_threshold=2 (Qwen validated)
│   │   └── training_metrics.json
│   └── outputs\                # Báo cáo đánh giá dạng văn bản, JSON
│       ├── fraud_label_validation.txt   # Báo cáo Gini/IV từ Qwen 2.5 Math
│       ├── model_audit_report.txt
│       └── monte_carlo_stability_report.json
├── models\                     # Lưu trữ các file nhị phân mô hình và scaler (.pkl)
├── reports\                    # Tài liệu báo cáo dự án
│   ├── figures\                # Các biểu đồ kết quả (ROC, MC Stability Curves)
│   ├── markdown\               # Các báo cáo tiến trình (4 phần)
│   └── docs\                   # Tài liệu hướng dẫn & khung thực thi
├── src\                        # Mã nguồn Python
│   ├── data_pipeline\          # Module xử lý dữ liệu và trích xuất đặc trưng
│   │   ├── feature_engineering.py   # Multi-signal fraud score + dynamic theta
│   │   ├── analyze.py
│   │   ├── process.py
│   │   └── replicate.py
│   ├── modeling\               # Module huấn luyện, tối ưu và kiểm định
│   │   ├── validate_fraud_label.py  # [MỚI] Qwen 2.5 Math Label Validator
│   │   ├── train.py
│   │   ├── calculate_accuracy.py
│   │   ├── monte_carlo_analysis.py
│   │   ├── optimize_hyperparameters.py
│   │   └── run_simple_audit.py
│   └── app.py                  # Ứng dụng Streamlit Dashboard chính
├── .env.example
├── .gitignore
├── config.json
└── requirements.txt
```

---

## 🛠️ Hướng dẫn Cài đặt & Sử dụng

### 1. Yêu cầu Hệ thống
- Python >= 3.10
- **Ollama** với model `mightykatun/qwen2.5-math:1.5b` (bắt buộc cho Label Validation & Hyperparameter Optimization)

```bash
ollama pull mightykatun/qwen2.5-math:1.5b
```

### 2. Cài đặt Thư viện
Khởi tạo môi trường ảo và cài đặt các thư viện phụ thuộc:
```bash
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Cấu hình Môi trường
Sao chép tệp cấu hình mẫu và điền thông số kết nối cơ sở dữ liệu của bạn:
```bash
cp .env.example .env
```
Các biến môi trường cần thiết:
```env
DB_SERVER=your_server_host
DB_DATABASE=your_database_name
DB_USE_WINDOWS_AUTH=True
DB_LIBRARY=pymssql
```

---

## 🚀 Hướng dẫn Thực thi

Thực hiện tuần tự các bước dưới đây tại thư mục gốc của dự án:

### Bước 1: Trích xuất và Tiền xử lý Dữ liệu
1. Đồng bộ dữ liệu từ SQL Server về SQLite local:
   ```bash
   python src/data_pipeline/replicate.py
   ```
2. Kiểm tra tính nhất quán dữ liệu thô:
   ```bash
   python src/data_pipeline/analyze.py
   ```
3. Chạy pipeline tiền xử lý và trích xuất đặc trưng rủi ro tài chính:
   ```bash
   python src/data_pipeline/feature_engineering.py
   ```
   *Kết quả*: Ma trận đặc trưng (bao gồm `fraud_signal_score`) lưu tại `data/processed/user_features_matrix.csv`.

### Bước 2: Kiểm chứng nhãn Fraud bằng Qwen 2.5 Math *(MỚI)*
```bash
python src/modeling/validate_fraud_label.py
```
*Kết quả*:
- Báo cáo Gini/IV tại `data/outputs/fraud_label_validation.txt`
- `data/configs/model_config.json` được cập nhật với `fraud_label_threshold` tối ưu ($\theta^* = 2$)
- Tự động re-run `feature_engineering.py` với $\theta^*$ mới nếu ngưỡng thay đổi

### Bước 3: Tối ưu Siêu tham số & Huấn luyện Mô hình
1. (Tùy chọn) Chạy tối ưu hóa siêu tham số qua Ollama LLM (`qwen2.5-math:1.5b`):
   ```bash
   python src/modeling/optimize_hyperparameters.py
   ```
2. Huấn luyện mô hình đánh giá rủi ro (Default Risk & Fraud Detection):
   ```bash
   python src/modeling/train.py
   ```
   *Kết quả*: 
   - Các mô hình nhị phân và scaler lưu tại `models/`.
   - Các biểu đồ đường cong ROC lưu tại `reports/figures/`.
   - Metrics đánh giá lưu tại `data/configs/training_metrics.json`.

### Bước 4: Thẩm định & Phân tích Độ ổn định
1. Tính toán hiệu năng chính xác tổng quát:
   ```bash
   python src/modeling/calculate_accuracy.py
   ```
2. Kiểm định độ ổn định Monte Carlo dưới nhiễu progressive:
   ```bash
   python src/modeling/monte_carlo_analysis.py
   ```
3. Chạy báo cáo kiểm định toán học tự động (Qwen 2.5 Math):
   ```bash
   python src/modeling/run_simple_audit.py
   ```
   *Kết quả*: Báo cáo lưu tại `data/outputs/model_audit_report.txt`.

### Bước 5: Khởi chạy Streamlit Dashboard
```bash
streamlit run src/app.py
```
Giao diện sẽ tự động mở tại `http://localhost:8501`. Cho phép bạn:
- Xem tổng quan phân phối rủi ro và tín dụng toàn hệ thống.
- So sánh hiệu năng giữa XGBoost và Logistic Regression.
- Sử dụng **Credit Sandbox Simulator** để giả lập phê duyệt thẻ và hạn mức tín dụng thời gian thực, bao gồm panel **Phân tích Fraud Signal Score** hiển thị breakdown từng tín hiệu gian lận.

---

## 📈 Kết quả Mô hình (Phiên bản hiện tại)

| Mô hình | Thuật toán | ROC-AUC | F1-Score | G-Mean |
|:---|:---:|:---:|:---:|:---:|
| Default Risk | XGBoost | 0.6268 ± 0.1051 | 0.1381 ± 0.0768 | 0.3443 ± 0.1027 |
| **Fraud Detection** | **Logistic Regression** | **0.9800 ± 0.0106** | **0.6012 ± 0.0526** | **0.9379 ± 0.0172** |

> *Fraud Model sử dụng Multi-Signal Label v2 ($\theta^*=2$, IV=8.18, validated by Qwen 2.5 Math)*

---

## 📚 Tài liệu tham khảo (References)
- Chi tiết về Khung toán học định mức, thiết kế quy trình chấm điểm và các nguồn lực nghiên cứu (bao gồm tập dữ liệu và mô hình Ollama) được lưu trữ tại [Tài liệu Tham khảo](./reports/docs/tham_khao.md).



---

## 📜 Giấy phép (License)
Dự án được cấp phép theo các điều khoản của [MIT License](./LICENSE).
