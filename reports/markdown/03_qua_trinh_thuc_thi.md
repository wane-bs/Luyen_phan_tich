# Báo cáo Dự án Xóm Bank Credit Underwriting
## Phần 3: Quá trình thực thi (Execution Process)

---

Quá trình triển khai kỹ thuật của dự án được thực hiện một cách hệ thống qua 4 giai đoạn cốt lõi, từ khai thác dữ liệu thô cho tới đóng gói ứng dụng.

### 1. Chi tiết các giai đoạn thực thi

#### Giai đoạn 1: Khảo sát và Tiền xử lý dữ liệu (Data Cleaning & Exploration)
* **Replication**: Sao chép dữ liệu từ database gốc về bản sao SQLite cục bộ tại `data/local_replica.db` thông qua script `src/replicate.py` nhằm đảm bảo an toàn dữ liệu và tối ưu tốc độ truy vấn.
* **Cleaning**: Khảo sát cấu trúc bảng giao dịch `transactions`. Xử lý các giao dịch âm bằng cách tách biệt chúng thành biến `refund_amount = |amount|` thay vì để số tiền âm triệt tiêu trực tiếp giá trị chi tiêu thực tế, tránh bóp méo dòng tiền chi tiêu của người dùng.
* **Time-series parsing**: Chuyển đổi định dạng ngày giao dịch sang dạng chuỗi thời gian để phân nhóm theo tháng nhằm phục vụ việc tính toán biến động hạn mức sử dụng (CUR).

#### Giai đoạn 2: Trích xuất đặc trưng & Tạo dữ liệu huấn luyện (Feature Engineering & Dataset Creation)
* Thực thi script `src/feature_engineering.py` để thực hiện phép gom nhóm (`groupby`) trên 1.3 triệu dòng giao dịch về cấp độ người dùng (`client_id`).
* Tính toán các tỷ lệ hành vi (Rate Features): tỷ lệ giao dịch trực tuyến (`online_tx_rate`), tỷ lệ giao dịch lỗi số dư (`insufficient_balance_rate`), tỷ lệ chi tiêu nhu yếu phẩm (`essential_spend_ratio`).
* Tính toán **Fraud Signal Score** (`fraud_signal_score`) tổng hợp 4 tín hiệu với trọng số khác nhau, cùng với 2 tín hiệu thành phần mới: `security_error_count_heavy` và `high_online_rate`.
* Áp dụng logic gán nhãn: `default` (DTI > 3.0 hoặc max CUR > 0.20) và `fraud` (đọc ngưỡng $\theta^*$ từ `data/configs/model_config.json`, được Qwen 2.5 Math xác nhận).
* Lưu trữ ma trận đặc trưng cuối cùng gồm thông tin của tất cả người dùng vào bảng `user_features_matrix` trong SQLite và xuất ra file `data/processed/user_features_matrix.csv`.

#### Giai đoạn 2.5: Kiểm chứng nhãn Fraud bằng Qwen 2.5 Math (Fraud Label Validation)
* Thực thi script `src/modeling/validate_fraud_label.py` để phân tích định lượng nhãn.
* Tính Gini coefficient và Information Value (IV) cho từng ngưỡng $\theta \in \{2, 3, 4\}$ theo công thức:
  $$IV = \sum_{i} \left(P(X_i|Y=1) - P(X_i|Y=0)\right) \cdot \ln\frac{P(X_i|Y=1)}{P(X_i|Y=0)}$$
* Gọi `mightykatun/qwen2.5-math:1.5b` để phân tích và đề xuất $\theta^*$ tối ưu, tự động cập nhật `model_config.json`.
* Kết quả: $\theta^* = 2$, fraud rate = 4.95%, IV = 8.18 (Strong).

#### Giai đoạn 3: Huấn luyện và Đánh giá Mô hình (Model Training & Evaluation)
* **Kiểm định chéo phân tầng (Stratified K-Fold Cross-Validation)**: Thay thế việc phân tách đơn lẻ bằng kiểm định chéo 5-Fold Stratified ($K = 5$) nhằm loại bỏ sự ngẫu nhiên và đảm bảo tỷ lệ phân phối nhãn đồng đều trên mỗi fold.
* **Chuẩn hóa chống rò rỉ (Leakage-free Scaling)**: Khởi tạo và khớp `StandardScaler` riêng biệt trên tập huấn luyện của mỗi fold, sau đó biến đổi cho tập kiểm thử tương ứng. Phương pháp này ngăn chặn triệt để sự rò rỉ thông tin phân phối (Data Leakage) từ tập kiểm thử ngược về tập huấn luyện.
* **Hệ thống chỉ số tối ưu cho nhãn lệch**: Bổ sung Precision-Recall Curve (PRC), F1-Score và chỉ số G-Mean ($G\text{-Mean} = \sqrt{\text{Recall} \times \text{Specificity}}$) để đánh giá toàn diện hiệu năng của thuật toán trên các nhóm nhãn rủi ro thiểu số.
* **Huấn luyện và Lưu trữ**: Đóng gói mô hình tối ưu nhất dưới dạng file `.pkl` trong thư mục `models/` sau khi huấn luyện lại trên toàn bộ tập dữ liệu (Full Dataset Fit) bằng cấu hình siêu tham số tối ưu đã chọn lọc qua kiểm định chéo.

#### Giai đoạn 4: Đóng gói và Xây dựng Sandbox (Dashboarding & Integration)
* Triển khai ứng dụng giao diện trực quan bằng thư viện Streamlit tại `src/app.py`.
* Tích hợp bảng thống kê tổng quan (System Overview), biểu đồ so sánh mô hình (hiển thị đồng thời cả ROC Curve và Precision-Recall Curve side-by-side) và phân hệ **Credit Sandbox Simulator** để chạy phê duyệt tín dụng thực nghiệm.

---

### 2. Các vấn đề phát sinh và giải pháp khắc phục (Troubleshooting)

Trong quá trình thực thi, đội ngũ phát triển đã xử lý thành công các lỗi kỹ thuật và nghiệp vụ nghiêm trọng:

#### A. Xác thực và Đo lường Rò rỉ dữ liệu mục tiêu (Target Leakage) bằng Monte Carlo & PFI
* **Triệu chứng**: Trong phiên bản thử nghiệm đầu tiên, mô hình dự báo vỡ nợ (Default Model) đạt điểm số ROC-AUC bằng **1.00** tuyệt đối.
* **Nguyên nhân**: Cột `dti` và `max_monthly_cur` được đưa vào tập đặc trưng huấn luyện ($X$), trong khi nhãn mục tiêu ($y$) lại được định nghĩa trực tiếp từ hai cột này.
* **Giải pháp khắc phục & Kiểm định nâng cao (Phase 3)**:
  1. **Loại bỏ đặc trưng rò rỉ (Feature Exclusion)**: Loại bỏ hoàn toàn 4 đặc trưng tài chính thô là `yearly_income`, `total_debt`, `total_credit_limit`, và `net_spend` khỏi tập dữ liệu huấn luyện của mô hình Default Risk, chỉ giữ lại các biến hành vi giao dịch nguyên bản và điểm FICO (`credit_score`).
  2. **Tái kiểm định hoán vị đặc trưng (PFI)**: Sau khi loại bỏ đặc trưng rò rỉ, điểm số quan trọng nhất dịch chuyển về các thuộc tính hành vi sạch: Điểm FICO (`credit_score` gây giảm **0.5918** điểm F1) và Tuổi (`current_age` gây giảm **0.5352** điểm F1). Các đặc trưng thô không còn gây nhiễu cho mô hình.
  3. **Tái kiểm định Monte Carlo**: Thực hiện bơm nhiễu Gaussian tăng dần ($0\% \to 100\%$ độ lệch chuẩn) vào các đặc trưng hành vi mới trong mô hình. Đồ thị `default_mc_leakage.png` ghi nhận các đường cong hiệu năng sụt giảm rất thoai thoải và đều đặn (F1-score và G-Mean duy trì tính ổn định, không sụp đổ đột ngột), chứng minh mô hình không còn bị rò rỉ mục tiêu và đạt tính bền vững cao dưới tác động nhiễu.


#### B. Lỗi không tương thích định dạng đầu vào trong Sandbox (KeyError & Shape Mismatch)
* **Triệu chứng**: Khi người dùng nhấn nút chạy giả lập trên giao diện Sandbox Streamlit, hệ thống báo lỗi `KeyError` hoặc crash do thiếu các trường đặc trưng như `refund_tx_count`, `insufficient_balance_count`, v.v.
* **Nguyên nhân**: Mô hình đã được huấn luyện với các cột đếm số lượng giao dịch tuyệt đối (`refund_tx_count`, `insufficient_balance_count`, `online_tx_count`, `essential_spend`). Tuy nhiên, trên giao diện Sandbox, hệ thống yêu cầu nhập các chỉ số tỷ lệ phần trăm (như `refund_rate`, `insufficient_balance_rate`).
* **Giải pháp**: Viết thêm một lớp xử lý trung gian trong `src/app.py`. Khi nhận thông tin từ form nhập liệu, hệ thống tự động quy đổi các tỷ lệ phần trăm ngược lại thành số lượng tuyệt đối dựa trên tổng số giao dịch (`total_tx_count`):
  * `refund_tx_count = total_tx_count * refund_rate`
  * `insufficient_balance_count = total_tx_count * insufficient_balance_rate`
  * `online_tx_count = total_tx_count * online_tx_rate`
  * `essential_spend = net_spend * essential_spend_ratio`
  Dữ liệu sau quy đổi khớp hoàn toàn với cấu trúc đặc trưng huấn luyện, giúp Sandbox vận hành trơn tru và chính xác.

#### C. Định dạng đầu ra không hợp lệ từ LLM & Hậu xử lý cưỡng chế (LLM Output Formatting & Regulatory Constraints)
*   **Triệu chứng**: Mô hình ngôn ngữ lớn `mightykatun/qwen2.5-math:1.5b` trong các lần gọi ngẫu nhiên có thể trả về văn bản chứa mã Markdown hoặc định dạng chuỗi từ điển kiểu Python (sử dụng dấu nháy đơn `'` thay vì dấu nháy kép `"` tiêu chuẩn của JSON, ví dụ: `{'credit_score_min': 480.0}`). Ngoài ra, mô hình đề xuất ngưỡng tuổi tối thiểu phi thực tế là `8.15` tuổi.
*   **Nguyên nhân**: Sự không nhất quán trong định dạng đầu ra của mô hình ngôn ngữ nhỏ (1.5B) khi nhận chỉ thị xuất dữ liệu dạng cấu trúc, và sự thiếu hụt kiến thức nghiệp vụ thực tế về luật cấp thẻ tín dụng cho trẻ em.
*   **Giải pháp**:
    1.  **Bộ lọc regex & Thư viện AST**: Trong script [optimize_hyperparameters.py](file:///f:/data_project/src/optimize_hyperparameters.py), hệ thống áp dụng regex để loại bỏ các ký tự thừa và sử dụng hàm `ast.literal_eval` để biên dịch an toàn các chuỗi định dạng Python thành kiểu dữ liệu Dictionary của Python.
    2.  **Bộ lọc chốt chặn nghiệp vụ (Business Rule Fallbacks)**: Tích hợp hàm kiểm soát và ép buộc giá trị (cắt biên cứng):
        $$\text{current\_age\_min} = \max(\text{value}_{\text{LLM}}, 18.0)$$
        $$\text{credit\_score\_min} = \max(\text{value}_{\text{LLM}}, 480.0)$$
        Đảm bảo các siêu tham số đầu ra luôn nằm trong tầm kiểm soát an toàn của bộ phận Quản trị rủi ro.

#### D. Lỗi lặp đầu ra của LLM khi xử lý prompt dài (LLM Output Repetition Loop)
*   **Triệu chứng**: Model `mightykatun/qwen2.5-math:1.5b` được gọi trong `run_simple_audit.py` phát sinh hiện tượng lặp lại cùng một đoạn văn hàng chục lần liên tục trong output, gây ra file báo cáo kích thước bất thường (~40KB cho nội dung thực chất ~2KB).
*   **Nguyên nhân**: Prompt quá dài (đưa toàn bộ `training_metrics.json` vào context 4096 tokens) khiến mô hình nhỏ (1.5B tham số) mất kiểm soát stop condition, rơi vào vòng lặp sinh văn bản vô hạn.
*   **Giải pháp**:
    1.  **Cắt ngắn dữ liệu đầu vào**: Hàm `_trim_metrics()` chỉ trích xuất các cột AUC, F1, G-Mean trước khi đưa vào prompt, giảm context window từ 4096 xuống 2048 tokens.
    2.  **Giới hạn token đầu ra**: Thêm tham số `num_predict=600` trong Ollama options để bắt buộc dừng sinh văn bản sau 600 tokens.


