# Báo cáo Dự án Xóm Bank Credit Underwriting
## Phần 2: Phương án giải quyết (Methodology & Design)

---

Để vượt qua các thách thức đã nêu ở Phần 1, dự án đã thiết kế và triển khai một phương án toàn diện từ kỹ nghệ đặc trưng, tái cấu trúc nhãn, đến thiết kế thuật toán phê duyệt đồng thuận (consensus-based).

### 1. Phương án Kỹ nghệ Đặc trưng (Feature Engineering)
Hệ thống kết hợp dữ liệu từ 3 bảng thông qua khóa liên kết `client_id` để xây dựng ma trận đặc trưng cấp độ khách hàng (User-level profile Matrix):

#### A. Khả năng tài chính và Đòn bẩy
* **Debt-to-Income (DTI) Ratio**:
  $$DTI = \frac{\text{total\_debt}}{\text{yearly\_income}}$$
  Chỉ số phản ánh áp lực trả nợ. Nếu thu nhập bằng 0, giá trị này được chuẩn hóa về 0 để tránh lỗi chia cho 0.
* **Tổng hạn mức cũ đang có (`total_credit_limit`)**: Tổng hạn mức của tất cả các thẻ tín dụng (`Credit`) mà khách hàng đang sở hữu.

#### B. Hiệu suất sử dụng hạn mức (Credit Utilization Rate - CUR)
Để tính toán CUR động theo thời gian:
1. Lọc toàn bộ giao dịch phát sinh từ thẻ tín dụng (`card_type = 'Credit'`).
2. Nhóm chi tiêu thực tế (`amount > 0`) theo từng người dùng và từng tháng.
3. Tính toán tỷ lệ sử dụng hạn mức hàng tháng:
   $$CUR_{monthly} = \frac{\text{Tổng chi tiêu trong tháng}}{\text{total\_credit\_limit}}$$
4. Trích xuất hai chỉ số rủi ro quan trọng:
   * **`max_monthly_cur`**: Mức CUR lớn nhất trong lịch sử (chỉ báo căng thẳng tài chính cực đại).
   * **`avg_monthly_cur`**: Mức CUR trung bình (chỉ báo thói quen chi tiêu thường nhật).

#### C. Hành vi chi tiêu và tín hiệu thanh khoản
* **Tách biệt chi tiêu và hoàn tiền**: Các giao dịch âm (`amount < 0`) được tách thành biến hoàn tiền riêng (`refund_amount = |amount|`), tránh làm nhiễu dòng tiền chi tiêu thực tế (`spend_amount`).
* **Tỷ lệ giao dịch lỗi số dư (`insufficient_balance_rate`)**:
  $$\text{insufficient\_balance\_rate} = \frac{\text{Số giao dịch bị lỗi Insufficient Balance}}{\text{Tổng số giao dịch}}$$
  Tần suất lỗi thiếu số dư cao là dấu hiệu trực tiếp cảnh báo dòng tiền cạn kiệt (Liquidity distress).
* **Tỷ lệ chi tiêu thiết yếu (`essential_spend_ratio`)**:
  $$\text{essential\_spend\_ratio} = \frac{\text{Chi tiêu cho danh mục thiết yếu}}{\text{Tổng chi tiêu ròng}}$$
  Danh mục thiết yếu được định nghĩa qua danh sách mã MCC tiêu chuẩn: Grocery Stores (5411), Utilities (4900), Telecom (4814), Gas Stations (5541), Drug Stores (5912), v.v. Khách hàng chi tiêu phần lớn vào nhu yếu phẩm thường có tính cam kết trả nợ ổn định hơn nhóm chi tiêu xa xỉ.
* **Tỷ lệ giao dịch trực tuyến (`online_tx_rate`)**: Số giao dịch trực tuyến chia cho tổng số giao dịch (giao dịch online thường đi kèm tỷ lệ gian lận cao hơn).

---

### 2. Phương án định nghĩa nhãn thực tế (Label Redefinition)
Nhằm khắc phục tình trạng mất cân bằng nhãn tuyệt đối (0% mặc định), dự án đã nghiên cứu và thống nhất các ngưỡng nhãn thực tế dựa trên phân phối dữ liệu thực nghiệm:

* **Nhãn Vỡ nợ (Default)**: Khách hàng được coi là vỡ nợ nếu:
  $$DTI > 3.0 \quad \text{HOẶC} \quad max\_monthly\_cur > 0.20$$
  *Ý nghĩa nghiệp vụ*: Áp lực nợ vượt quá 3 lần thu nhập năm hoặc từng chi tiêu vượt quá 20% tổng hạn mức thẻ trong tháng (trong điều kiện dòng tiền yếu). Ngưỡng này giúp phát hiện sớm nhóm khách hàng có dấu hiệu rủi ro trung-cao. Tỷ lệ nhãn Default sau điều chỉnh đạt **4.45%** (phân phối lý tưởng cho mô hình phân loại).

* **Nhãn Gian lận (Fraud) — Multi-Signal v2**: Được tái kiến trúc hoàn toàn so với phiên bản đơn tín hiệu ban đầu. Khách hàng được coi là có rủi ro gian lận khi **Fraud Signal Score** vượt ngưỡng $\theta^*$ được xác nhận bởi Qwen 2.5 Math:

  $$S_{fraud}(u) = 2\cdot\mathbb{1}[\text{sec\_err} \geq 3] + 2\cdot\mathbb{1}[\text{spatiotemporal}=1] + \mathbb{1}[\text{refund\_rate}>0.10] + \mathbb{1}[\text{online\_rate}>0.70]$$

  $$\text{fraud}(u) = \mathbb{1}[S_{fraud}(u) \geq \theta^*], \quad \theta^* = 2$$

  | Tín hiệu | Trọng số | Điều kiện kích hoạt | Ý nghĩa nghiệp vụ |
  |:---|:---:|:---|:---|
  | Lỗi bảo mật nặng | w=2 | `security_error_count >= 3` | Gian lận có chủ đích (lặp lại) |
  | Đa địa lý 24h | w=2 | ≥ 5 bang khác nhau trong 24h | Clone thẻ / cardnapping |
  | Hoàn tiền bất thường | w=1 | `refund_rate > 10%` | Gian lận hoàn tiền |
  | Online quá cao | w=1 | `online_tx_rate > 70%` | Card-not-present fraud |

  Tỷ lệ nhãn Fraud sau tái định nghĩa đạt **4.95%** (giảm từ 12.3% của phiên bản cũ). IV = 8.18 (Strong) xác nhận chất lượng nhãn mới.

---

### 3. Thiết kế cấu trúc Thuật toán Đánh giá & Cấp tín dụng Dòng tiền Đa tầng (Phase 2)
Quy trình phê duyệt tự động được nâng cấp toàn diện dựa trên triết lý **Thẩm định theo Dòng tiền khả dụng thực tế (Cash Flow-based Underwriting)** kết hợp các chốt chặn và hệ số đệm thích ứng:

```
[Thông số tài chính & hành vi]
              │
              ▼
┌──────────────────────────────────────────────┐
│ BƯỚC 1: Tính Dòng tiền ròng khả dụng (CFADS) │ ── (CFADS_monthly <= 0) ──> ❌ Từ chối ngay
└──────────────────────────────────────────────┘
              │ (CFADS > 0)
              ▼
┌──────────────────────────────────────────────┐
│ BƯỚC 2: Tính Đệm rủi ro động (C_target)      │ ── (Cộng phạt FICO, CUR, Liquidity, AI)
└──────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│ BƯỚC 3: Công thức Niên kim Hạn mức Cơ sở     │ ── (Annuity Formula quy đổi PMT_max)
└──────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│ BƯỚC 4: Chốt chặn & Khấu trừ rủi ro đa tầng  │ ── (Circuit Breakers / AI Haircuts)
└──────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│ BƯỚC 5: Chốt chặn Đòn bẩy (Leverage Cap)     │ ── (Giới hạn nợ mới + cũ <= 50% Thu nhập)
└──────────────────────────────────────────────┘
              │
              ▼
    [Hạn mức phê duyệt cuối cùng CL_new]
```

#### Bước 1: Thẩm định Dòng tiền khả dụng hàng tháng (CFADS_monthly)
Đo lường năng lực tài chính thặng dư thực tế sau khi trừ đi toàn bộ nghĩa vụ nợ hiện tại phân bổ trong 12 tháng:
$$CFADS_{\text{monthly}} = \max\left(0, \frac{\text{yearly\_income} - \text{total\_debt}}{12}\right)$$
*Luật chốt chặn cưỡng chế*: Nếu $CFADS_{\text{monthly}} \le 0$ (tương ứng chỉ số đòn bẩy ban đầu $DTI \ge 1.0$), hệ thống kích hoạt **Cash Flow Breaker** lập tức từ chối hồ sơ ($CL_{new} = 0.0$).

#### Bước 2: Thiết lập Đệm rủi ro thích ứng ($C_{\text{target}}$)
Hệ số an toàn được cấu trúc động dựa trên hành vi và rủi ro tích hợp:
$$C_{\text{target}} = C_{\text{base}} + \Delta C_{\text{FICO}} + \Delta C_{\text{CUR}} + \Delta C_{\text{Liquidity}} + \Delta C_{\text{AI}}$$
*   **$C_{\text{base}}$ (Cushion nền):** Mặc định ở mức $1.25$.
*   **$\Delta C_{\text{FICO}}$ (Phạt điểm tín dụng yếu):** Cộng thêm **$0.30$** nếu $FICO < 620$; cộng **$0.15$** nếu $620 \le FICO < 680$.
*   **$\Delta C_{\text{CUR}}$ (Phạt tiêu quá hạn mức cũ):** Cộng thêm **$0.25$** nếu $max\_monthly\_cur > 30\%$; cộng **$0.15$** nếu $avg\_monthly\_cur > 15\%$.
*   **$\Delta C_{\text{Liquidity}}$ (Phạt thâm hụt thanh khoản):** Cộng thêm **$0.30$** nếu $insufficient\_balance\_rate > 5\%$; cộng **$0.15$** nếu $insufficient\_balance\_rate > 0\%$.
*   **$\Delta C_{\text{AI}}$ (Phạt xác suất AI):** Trung bình cộng xác suất vỡ nợ và gian lận dự báo từ mô hình:
    $$\Delta C_{\text{AI}} = \frac{P_{\text{Default}} + P_{\text{Fraud}}}{2}$$

#### Bước 3: Xác định Khả năng chi trả tối đa ($PMT_{\text{max}}$) & Hạn mức Cơ sở ($L_{\text{base}}$)
*   **Nghĩa vụ trả nợ tối đa mỗi tháng**:
    $$PMT_{\text{max}} = \frac{CFADS_{\text{monthly}}}{C_{\text{target}}}$$
*   **Hiện giá niên kim quy đổi hạn mức ($L_{\text{base}}$)**: Áp dụng lãi suất giả định $r = 1.5\%$/tháng và kỳ hạn $n = 12$ tháng:
    $$L_{\text{base}} = PMT_{\text{max}} \times \left[ \frac{1 - (1 + r)^{-12}}{r} \right] \approx PMT_{\text{max}} \times 10.9075$$

#### Bước 4: Chốt chặn Cưỡng chế & Khấu trừ rủi ro AI (AI Haircuts)
*   **Circuit Breakers**: Từ chối cho vay lập tức nếu vi phạm bất kỳ tiêu chí nào sau:
    *   Xác suất rủi ro AI quá cao ($P_{\text{Default}} \ge 0.50$ hoặc $P_{\text{Fraud}} \ge 0.50$).
    *   Điểm FICO dưới chuẩn tối thiểu ($FICO < 580$).
    *   Tỷ lệ lỗi vượt số dư nghiêm trọng ($insufficient\_balance\_rate > 20\%$).
*   **AI Haircuts (Chiết khấu rủi ro)**: Khấu trừ hạn mức nếu rơi vào nhóm cảnh báo:
    *   *Warning Level 2 (Watch)*: Chiết khấu $15\%$ hạn mức ($L_{\text{base}} \times 0.85$) nếu xác suất rủi ro AI cận biên $\ge 35\%$.
    *   *Warning Level 3 (Stress)*: Chiết khấu $40\%$ hạn mức ($L_{\text{base}} \times 0.60$) nếu $620 \le FICO < 650$ hoặc $avg\_monthly\_cur > 40\%$.
    *   *Quy tắc áp dụng*: Nếu trùng nhiều điều kiện, hệ thống chọn mức chiết khấu cao nhất (tối đa 40%).

#### Bước 5: Chốt chặn Đòn bẩy Bảng Cân đối (Leverage Cap)
Đảm bảo tổng nợ của khách hàng sau khi cấp thẻ mới không vượt quá $50\%$ thu nhập năm:
$$\frac{Total\ Debt + L_{final}}{Yearly\ Income} \le 0.50 \implies L_{final} \le 0.50 \times Yearly\ Income - Total\ Debt$$
$$Leverage\ Cap = \max\left(0.0, 0.50 \times Yearly\ Income - Total\ Debt\right)$$

Nếu hạn mức sau chiết khấu lớn hơn $Leverage\ Cap$, hệ thống sẽ cưỡng chế cắt giảm hạn mức về đúng bằng $Leverage\ Cap$ và chuyển trạng thái phê duyệt sang dạng giới hạn đòn bẩy.

---

### 4. Phương án hiệu chỉnh siêu tham số và giới hạn logic bằng LLM (LLM-based Calibration & Optimization Method)
Nhằm giải quyết triệt để rủi ro quá khớp trên hai đặc trưng liên tục nhạy cảm (`credit_score` và `current_age`), dự án xây dựng quy trình tự động hóa hiệu chỉnh thông minh:

#### A. Khai thác LLM chuyên Toán (qwen2.5-math:1.5b)
Trước khi huấn luyện mô hình, hệ thống tính toán các chỉ số thống kê mô tả đầu vào ($\mu$, $\sigma$, $\min$, $\max$) và gửi yêu cầu tới mô hình ngôn ngữ lớn cục bộ để:
1.  **Tính toán cận cắt biên toán học (Mathematical Clipping Bounds)**: Xác lập các cận trên/dưới tối ưu đối với biến liên tục nhằm ngăn chặn các giá trị nhiễu đột biến phá vỡ không gian quyết định.
2.  **Đề xuất siêu tham số điều hòa (Regularization parameters)**: Định cấu hình cây quyết định (`max_depth`, `min_child_weight`) và hình phạt $\mathcal{L}_1/\mathcal{L}_2$ (`reg_alpha`/`reg_lambda`) giúp san phẳng ranh giới phân tách lớp.

#### B. Áp dụng chốt chặn an toàn nghiệp vụ (Business Regulatory Clipping)
Để đảm bảo tính tuân thủ pháp lý và quản trị rủi ro kinh doanh thực tế, kết quả do LLM đề xuất phải đi qua bộ lọc hậu xử lý bắt buộc (Post-processing check):
*   **Ngưỡng tuổi tối thiểu**: Cưỡng chế chặn dưới tuổi ở mức $18.0$ tuổi (đảm bảo năng lực hành vi dân sự đầy đủ theo luật định) ngay cả khi LLM đề xuất ngưỡng thấp hơn (ví dụ: $8.15$ tuổi).
*   **Ngưỡng FICO tối thiểu**: Đảm bảo điểm FICO được giới hạn tối thiểu ở mức $480.0$ để tránh rủi ro cho vay dưới chuẩn quá sâu.
*   **Huấn luyện đối kháng (Adversarial Training)**: Bơm nhiễu Gauss ngẫu nhiên $\epsilon \sim \mathcal{N}(0, 0.05 \sigma^2)$ trực tiếp vào dữ liệu huấn luyện để tăng sức đề kháng cho mô hình.### 5. Phương án kiểm chứng định lượng nhãn Fraud (Fraud Label Quantitative Validation)
Sau khi tái định nghĩa nhãn `fraud`, dự án xây dựng quy trình kiểm chứng tự động bằng Qwen 2.5 Math (`mightykatun/qwen2.5-math:1.5b`) để xác định ngưỡng $\theta^*$ tối ưu:

```
[user_features_matrix.csv]
           │
           ▼
┌──────────────────────────────┐
│ Tính Gini & IV cho θ∈{2,3,4}│  ── Gini=1.0 cho θ=2 và θ=3
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Gọi Qwen 2.5 Math Validator │  ── Phân tích định lượng
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Parse θ* + Min Sample Check  │  ── n_fraud ≥ 5×n_features
└──────────────────────────────┘
           │
           ▼
     [Cập nhật model_config.json: fraud_label_threshold=θ*]
```

**Kết quả kiểm chứng:**

| $\theta$ | $n_{fraud}$ | Fraud Rate | Gini | IV | Viable? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | **99** | **4.95%** | 1.0000 | **8.18** (Strong) | ✅ |
| 3 | 10 | 0.50% | 1.0000 | 0.71 | ✗ (n<50) |
| 4 | 0 | 0% | 0 | 0 | ✗ |

$\theta^* = 2$ được Qwen 2.5 Math xác nhận là ngưỡng tối ưu duy nhất đủ điều kiện viable ($n_{fraud} \geq 75 = 5 \times 15$ features).


