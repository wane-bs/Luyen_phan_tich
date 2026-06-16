Dưới đây là khung phương án thực thi (Framework) được thiết kế chuyên biệt cho cấu trúc dữ liệu hiện tại của Xóm Bank:

---
2 bài toán quan trọng cần phải giải quyết trước khi cấp tín dụng: Quản trị rủi ro vỡ nợ (Default Risk) và Phát hiện gian lận (Fraud Detection)

## 1. Tái cấu trúc dữ liệu phục vụ mô hình (Feature Engineering)

Từ 3 bảng dữ liệu thô (`users`, `cards`, `transactions`), chúng ta cần biến đổi thành các biến số định lượng mang đặc trưng rủi ro tài chính:

### Khả năng tài chính và Nghĩa vụ nợ (Bảng Users)

* 
**Debt-to-Income (DTI) Ratio**: Tỷ lệ Tổng nợ / Thu nhập năm (`total_debt` / `yearly_income`). Đây là biến số quan trọng nhất phản ánh áp lực đòn bẩy tài chính cá nhân.


* **Thu nhập khả dụng ước tính (Proxy)**: Mặc dù chưa có chi phí sinh hoạt cụ thể, có thể ước tính dựa trên mức chi tiêu trung bình hàng tháng trích xuất từ dữ liệu giao dịch.

### Hành vi chi tiêu và Thanh toán (Bảng Transactions & Cards)

* 
**Tần suất rủi ro giao dịch lỗi**: Tỷ lệ giao dịch thất bại do thiếu số dư (`Insufficient Balance`) hoặc sai mã PIN (`Bad PIN`) trên tổng số giao dịch. Tần suất `Insufficient Balance` cao là tín hiệu cảnh báo sớm về thanh khoản (Liquidity distress).


* **Tốc độ và Xu hướng chi tiêu (Velocity Features)**:
* Tổng giá trị giao dịch (Volume) và số lượng giao dịch (Count) theo tháng.
* Tỷ lệ chi tiêu cho các danh mục thiết yếu (Grocery Stores - MCC 5411) so với danh mục xa xỉ nhằm xác định tính bền vững trong dòng tiền của khách hàng.




* 
**Phương thức giao dịch**: Tỷ lệ giao dịch trực tuyến (`Online Transaction`) và quẹt thẻ (`Swipe Transaction`). Các hình thức này thường có rủi ro gian lận (Fraud) cao hơn giao dịch chíp (`Chip Transaction`).



### Hiệu suất sử dụng hạn mức (Credit Utilization)

* 
**Hạn mức thực tế hiện tại**: Tổng hạn mức của nhóm thẻ tín dụng (`Credit`) thuộc sở hữu của từng user.


* 
**Credit Utilization Rate (CUR)**: Tỷ lệ tổng chi tiêu trong tháng trên tổng hạn mức được cấp.


* *Lưu ý*: CUR > 30% bắt đầu kích hoạt cảnh báo rủi ro; CUR sát 100% thể hiện khách hàng đang phụ thuộc quá lớn vào tín dụng và có nguy cơ vỡ nợ (Default) cao.





---

## 2. Thiết kế cấu trúc Thuật toán Đánh giá & Cấp tín dụng

Thuật toán sẽ vận hành qua 3 cấu phần độc lập nhưng tuần tự:

### Bước 1: Bộ lọc điều kiện tiên quyết (Hard Cut-off Rules)

Loại bỏ ngay lập tức các hồ sơ dưới chuẩn để tối ưu chi phí vận hành thuật toán:

* 
**Credit Score**: `credit_score` < 580 (Ngưỡng rủi ro cao dựa trên phân phối dữ liệu hiện tại, với Q25 là 681).


* **DTI Ratio**: `total_debt` / `yearly_income` > Ngưỡng quy định của ngân hàng (thông thường là > 4-5 lần thu nhập năm).
* 
**Fraud Trigger**: Khách hàng có chuỗi giao dịch lỗi liên tiếp tại các vị trí địa lý khác nhau (Dấu hiệu thẻ bị đánh cắp - Stolen Card).



### Bước 2: Mô hình chấm điểm tín dụng nội bộ (Internal Credit Scoring)

Kết hợp điểm số truyền thống (`credit_score`) với điểm hành vi từ Xóm Bank (Behavioral Score):


$$\text{Internal Score} = w_1 \times \text{Scaled Credit Score} + w_2 \times \text{Repayment Behavior Score} - w_3 \times \text{Risk Signal Score}$$

* 
**Repayment Behavior Score**: Tính toán dựa trên độ ổn định của dòng tiền chi trả và tỷ lệ CUR duy trì ở mức an toàn (10% - 30%).


* 
**Risk Signal Score**: Tỷ lệ thuận với số lần xuất hiện lỗi `Insufficient Balance` và tốc độ chi tiêu bất thường.



### Bước 3: Mô hình xác định hạn mức (Credit Limit Allocation)

Hạn mức cấp mới hoặc điều chỉnh sửa đổi ($CL_{new}$) sẽ là hàm số của Thu nhập, Điểm số nội bộ và Rủi ro vỡ nợ:


$$CL_{new} = \text{Base Limit} \times f(\text{Internal Score}) \times (1 - \text{CUR}_{\text{average}})$$

* 
**Đối với khách hàng mới**: Hạn mức dựa trên `yearly_income` và `credit_score`.


* 
**Đối với khách hàng hiện tại (Tái cấp hạn mức)**: Đánh giá qua lịch sử giao dịch historical (~1.3M dòng). Nếu CUR cao nhưng lịch sử thanh toán sạch (không lỗi số dư), xem xét tăng hạn mức. Nếu xuất hiện dấu hiệu rủi ro, tiến hành đóng băng hoặc giảm hạn mức để quản trị rủi ro nợ xấu.



---

## 3. Kế hoạch Triển khai Kỹ thuật (Implementation Roadmap)

Để hiện thực hóa bài toán này, quy trình xử lý dữ liệu và xây dựng mô hình bằng Python sẽ trải qua 4 giai đoạn cốt lõi:

```
[Khảo sát & Làm sạch] ──> [Trích xuất Đặc trưng] ──> [Huấn luyện Mô hình] ──> [Đóng gói & Giám sát]
  (Xử lý AMT âm/dương)     (Tính toán DTI, CUR)       (Log. Regression/XGBoost)    (Dashboard Dashboard)

```

1. **Giai đoạn 1: Tiền xử lý dữ liệu (Data Preprocessing)**
* Tách biệt tập dữ liệu giao dịch: Xử lý các giá trị âm (`amount < 0`) thành biến refund riêng biệt, không gộp chung vào chi tiêu ròng.


* Chuẩn hóa thời gian (`Transaction Date Range` từ 2022 đến 2024) thành các chuỗi thời gian (Time-series) theo tháng để tính toán xu hướng.




2. **Giai đoạn 2: Trích xuất đặc trưng (Feature Engineering)**
* Sử dụng Python để kết hợp bảng `users` và `transactions` qua khóa `client_id`.


* Xây dựng ma trận đặc trưng ở cấp độ User (User-level profile Matrix).


3. **Giai đoạn 3: Huấn luyện mô hình (Model Training)**
* Sử dụng phương pháp **Logistic Regression** (để đảm bảo tính giải thích cao cho các phòng ban quản trị rủi ro - CRO) hoặc **XGBoost** để phân loại nhóm khách hàng có khả năng trả nợ kém.


4. **Giai đoạn 4: Trực quan hóa & Giám sát (Dashboarding)**
* Xây dựng hệ thống giám sát thời gian thực (Real-time monitoring) tập trung vào tỷ lệ thẻ đóng băng (`Dormant cards`), tỷ lệ active và biến động của chỉ số CUR toàn hệ thống để báo cáo cho Head of Cards Product và Head of Retail Banking.





---
