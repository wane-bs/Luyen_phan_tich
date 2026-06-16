# Báo cáo Dự án Xóm Bank Credit Underwriting
## Phần 1: Bối cảnh và Đặt vấn đề

---

### 1. Bối cảnh dự án (Project Context)
Trong xu thế chuyển đổi số tài chính tiêu dùng, **Xóm Bank** hướng tới xây dựng hệ thống tự động hóa phê duyệt và cấp tín dụng (Automatic Credit Underwriting) thông qua dữ liệu lớn. Mục tiêu cốt lõi là ra quyết định cấp thẻ tín dụng nhanh chóng, cá nhân hóa hạn mức dựa trên năng lực tài chính, đồng thời kiểm soát rủi ro ở mức tối thiểu.

Hệ thống cần giải quyết hai bài toán phòng vệ rủi ro cốt lõi trước khi quyết định cấp tín dụng:
1. **Quản trị rủi ro vỡ nợ (Default Risk Management)**: Dự báo khả năng khách hàng không thể hoàn trả các khoản nợ thẻ khi đến hạn.
2. **Phát hiện gian lận (Fraud Detection)**: Nhận diện các hành vi giao dịch bất thường hoặc dấu hiệu tài khoản/thẻ bị đánh cắp trước khi gây ra thiệt hại tài chính.

#### Cấu trúc dữ liệu hiện có
Dự án được triển khai trên nền tảng cơ sở dữ liệu giao dịch nội bộ lưu trữ tại SQLite (`data/local_replica.db`), bao gồm 3 bảng chính:
* **Bảng `users` (Thông tin khách hàng)**: Chứa thông tin cơ bản về nhân khẩu học (`current_age`, `gender`), thu nhập hàng năm (`yearly_income`), dư nợ hiện tại (`total_debt`) và điểm tín dụng truyền thống FICO (`credit_score`).
* **Bảng `cards` (Thông tin thẻ)**: Lưu trữ các loại thẻ mà người dùng đang sở hữu (Credit hoặc Debit), cùng hạn mức tín dụng được cấp (`credit_limit`).
* **Bảng `transactions` (Lịch sử giao dịch)**: Ghi nhận chi tiết từng giao dịch (~1.3 triệu bản ghi) bao gồm số tiền (`amount`), hình thức sử dụng (`use_chip` - Online/Chip/Swipe), mã danh mục cửa hàng (`mcc`), thời gian giao dịch (`date`), và các lỗi phát sinh nếu có (`errors` - ví dụ: Insufficient Balance, Bad PIN, Bad CVV, v.v.).

---

### 2. Đặt vấn đề (Problem Statement)
Mặc dù sở hữu kho dữ liệu giao dịch phong phú, Xóm Bank gặp phải các rào cản kỹ thuật và nghiệp vụ nghiêm trọng sau:

#### A. Khoảng cách giữa dữ liệu thô và chỉ báo rủi ro (Feature Engineering Gap)
Dữ liệu giao dịch ở dạng nhật ký (log) chuỗi thời gian thô không thể đưa trực tiếp vào các mô hình học máy. Hệ thống cần chuyển hóa các giao dịch đơn lẻ thành những biến số định lượng phản ánh:
* Áp lực nợ nần và đòn bẩy tài chính cá nhân.
* Hiệu suất sử dụng hạn mức tín dụng theo tháng.
* Thói quen thanh toán và các tín hiệu mất thanh khoản sớm (ví dụ: liên tục quẹt thẻ lỗi thiếu số dư).
* Tỷ lệ chi tiêu cho các danh mục thiết yếu so với xa xỉ phẩm nhằm đánh giá độ bền vững dòng tiền.

#### B. Thách thức mất cân bằng nhãn cực đoan (Extreme Class Imbalance)
Trong các bộ dữ liệu thực tế của ngân hàng, tỷ lệ khách hàng thực sự vỡ nợ hoặc gian lận thường rất nhỏ. Ban đầu, nếu áp dụng các ngưỡng lý thuyết nghiêm ngặt:
* **Default**: Định nghĩa khi tỷ lệ nợ trên thu nhập (DTI) > 5.0 hoặc tỷ lệ sử dụng hạn mức (CUR) > 90%. Kết quả thống kê cho thấy tỷ lệ vỡ nợ thực tế là **0%** (không có mẫu dương nào).
* **Fraud**: Định nghĩa dựa trên số lần gặp lỗi bảo mật nghiêm trọng.

Tỷ lệ nhãn vỡ nợ bằng 0% khiến mô hình học máy hoàn toàn mất khả năng học các đặc trưng rủi ro (zero variance). Do đó, dự án đặt ra bài toán **phải tái định nghĩa các ngưỡng nhãn thực tế** để tạo ra phân phối dữ liệu cân bằng hơn cho huấn luyện mô hình, mà vẫn đảm bảo tính phản ánh đúng bản chất rủi ro kinh doanh.

#### C. Tự động hóa ra quyết định phê duyệt và cấp hạn mức (Decisioning and Limit Allocation)
Khi có xác suất rủi ro từ mô hình, ngân hàng cần một cơ chế phê duyệt tự động kết hợp:
1. **Quy tắc chặn cứng (Hard Cut-off Rules)** để loại bỏ lập tức các hồ sơ dưới chuẩn, tiết kiệm tài nguyên tính toán.
2. **Hệ thống điểm số nội bộ (Internal Scoring)** kết hợp giữa điểm tín dụng truyền thống (FICO) và hành vi chi tiêu thực tế tại Xóm Bank.
3. **Cơ chế cấp hạn mức tự động (Limit Allocation)** điều chỉnh linh hoạt theo thu nhập khả dụng và rủi ro của từng đối tượng khách hàng.

#### D. Sự bất ổn định dưới tác động nhiễu và nhu cầu hiệu chỉnh thông minh (Noise Sensitivity & Intelligent Calibration)
Các đặc trưng liên tục như điểm tín dụng (`credit_score`) và độ tuổi (`current_age`) rất dễ bị nhiễu do lỗi nhập liệu hoặc biến động ngắn hạn. Thử nghiệm Monte Carlo ban đầu cho thấy mô hình dễ bị quá khớp (overfitting), hiệu năng giảm sâu ngay cả khi chỉ có nhiễu nhỏ ($\pm 10\% \sigma$).
Do đó, dự án cần tích hợp thêm **bộ hiệu chỉnh siêu tham số và giới hạn logic toán học tự động** thông qua mô hình ngôn ngữ lớn chuyên toán (`qwen2.5-math:1.5b`) nhằm xác lập các ngưỡng cắt biên (clipping bounds) an toàn và tối ưu hóa hệ số điều hòa chống quá khớp cho XGBoost.

#### E. Ô nhiễm nhãn gian lận (Fraud Label Pollution) — Vấn đề phát sinh sau kiểm định
Sau khi triển khai hệ thống phiên bản đầu, quá trình kiểm định nghiệp vụ phát hiện một lỗi cơ bản trong định nghĩa nhãn `fraud`: logic gán nhãn `security_error_count >= 1` **không phân biệt được** "nạn nhân nhập sai PIN một lần" với "thủ phạm thực hiện gian lận có chủ đích". Hệ quả là:

| Thực tế nghiệp vụ | Nhãn cũ |
|:---|:---:|
| Khách nhập sai PIN 1 lần | `fraud = 1` ❌ |
| Thẻ bị kẻ gian thử (không thành công) | `fraud = 1` ❌ |
| Gian lận thực sự với pattern đa tín hiệu | `fraud = 1` ✓ |

Chốt chặn `P_Fraud >= 50%` trở nên **vô nghĩa về mặt kinh doanh** vì mô hình học từ nhãn không phản ánh hành vi gian lận thực sự. Dự án đặt ra yêu cầu **tái kiến trúc nhãn fraud** theo hướng multi-signal AND logic với kiểm chứng định lượng bằng Qwen 2.5 Math.
