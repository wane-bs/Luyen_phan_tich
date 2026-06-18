import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Config
st.set_page_config(page_title="Xóm Bank - Credit Underwriting", layout="wide", page_icon="💳")

# Title
st.title("💳 Xóm Bank - Hệ thống Phê duyệt & Cấp tín dụng")

# Load models and scalers
@st.cache_resource
def load_models():
    try:
        with open("models/best_default_model.pkl", "rb") as f:
            default_dict = pickle.load(f)
        with open("models/default_scaler.pkl", "rb") as f:
            default_scaler = pickle.load(f)
            
        with open("models/best_fraud_model.pkl", "rb") as f:
            fraud_dict = pickle.load(f)
        with open("models/fraud_scaler.pkl", "rb") as f:
            fraud_scaler = pickle.load(f)
            
        return default_dict, default_scaler, fraud_dict, fraud_scaler
    except Exception as e:
        return None, None, None, None

default_dict, default_scaler, fraud_dict, fraud_scaler = load_models()

# Load Data
@st.cache_data
def load_data():
    csv_path = "data/processed/user_features_matrix.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_data()

# Navigation
tabs = st.tabs(["📊 Tổng quan hệ thống", "📈 Giám sát rủi ro vĩ mô", "⚙️ So sánh mô hình", "🧪 Credit Sandbox Simulator"])

# --- TAB 1: TỔNG QUAN HỆ THỐNG ---
with tabs[0]:
    st.header("1. Phân phối Rủi ro & Tín dụng Hệ thống")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số Khách hàng", f"{len(df):,}")
        col2.metric("Tỷ lệ Vỡ nợ (Default)", f"{df['default'].mean()*100:.2f}%")
        col3.metric("Tỷ lệ Gian lận (Fraud)", f"{df['fraud'].mean()*100:.2f}%")
        col4.metric("Thu nhập TB / Năm", f"${df['yearly_income'].mean():,.0f}")
        
        st.write("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Phân phối DTI (Debt-to-Income)")
            st.bar_chart(np.histogram(df['dti'], bins=20)[0])
            st.caption("Trục Y: Số lượng User | Trục X: Bins DTI (Max: 4.98)")
            
        with col2:
            st.subheader("Phân phối CUR (Credit Utilization Rate lớn nhất)")
            st.bar_chart(np.histogram(df['max_monthly_cur'], bins=20)[0])
            st.caption("Trục Y: Số lượng User | Trục X: Bins CUR (Max: 0.62)")
    else:
        st.warning("Vui lòng chạy `src/data_pipeline/feature_engineering.py` để tạo dữ liệu.")
        
# --- TAB 2: GIÁM SÁT RỦI RO VĨ MÔ ---
with tabs[1]:
    st.header("2. Giám sát Rủi ro Vĩ mô Toàn hệ thống")
    st.write("Cung cấp cái nhìn tổng quan về trạng thái hoạt động của thẻ và rủi ro sử dụng hạn mức thực tế.")
    
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trạng thái hoạt động của thẻ (Active vs Dormant)")
            avg_dormant_ratio = df['dormant_card_ratio'].mean()
            active_ratio = 1.0 - avg_dormant_ratio
            
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            ax_pie.pie([active_ratio, avg_dormant_ratio], labels=['Active Cards', 'Dormant Cards'], 
                       autopct='%1.1f%%', startangle=90, colors=['#2ca02c', '#d62728'])
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            st.caption("Tỷ lệ thẻ Dormant: Thẻ Credit mở > 2 năm trước 2024-10-31 và chưa từng giao dịch.")
            
        with col2:
            st.subheader("Phân phối CUR động 30 ngày (Dynamic Credit Utilization)")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(df['cur_30d'], bins=20, color='#1f77b4', edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel('Credit Utilization Rate (30 Days)')
            ax_hist.set_ylabel('Số lượng khách hàng')
            ax_hist.set_title('Phân phối CUR 30 ngày gần nhất')
            st.pyplot(fig_hist)
            st.caption("Chỉ số CUR động tính toán dựa trên tổng chi tiêu 30 ngày qua trên tổng hạn mức active.")
    else:
        st.warning("Vui lòng chạy `src/data_pipeline/feature_engineering.py` để tạo dữ liệu.")

# --- TAB 3: SO SÁNH MÔ HÌNH ---
with tabs[2]:
    st.header("2. Đánh giá Hiệu năng Mô hình (XGBoost vs Logistic Regression)")
    st.write("Cấu trúc điểm số nội bộ và phê duyệt tín dụng của ngân hàng phụ thuộc vào độ chính xác của 2 mô hình máy học.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mô hình Dự báo Vỡ nợ (Default Risk)")
        st.info("Nhãn Default: DTI > 3.0 hoặc CUR max > 20%")
        if os.path.exists("reports/figures/default_roc.png"):
            img_def = Image.open("reports/figures/default_roc.png")
            st.image(img_def, use_container_width=True)
            st.success("Mô hình tốt nhất: **XGBoost** (Được chọn để triển khai)")
        else:
            st.warning("Chưa có biểu đồ đánh giá. Hãy chạy `src/modeling/train.py`.")
            
    with col2:
        st.subheader("Mô hình Phát hiện Gian lận (Fraud Detection)")
        st.info("Nhãn Fraud (Multi-Signal v2): S_fraud = 2×(Lỗi bảo mật ≥3) + 2×(Đa bang 24h) + 1×(Hoàn tiền>10%) + 1×(Online>70%) ≥ θ")
        if os.path.exists("reports/figures/fraud_roc.png"):
            img_frd = Image.open("reports/figures/fraud_roc.png")
            st.image(img_frd, use_container_width=True)
            st.success("Mô hình tốt nhất: **Logistic Regression** (Được chọn để triển khai)")
        else:
            st.warning("Chưa có biểu đồ đánh giá. Hãy chạy `src/modeling/train.py`.")

# --- TAB 4: CREDIT SANDBOX SIMULATOR ---
with tabs[3]:
    st.header("3. Giả lập Phê duyệt Hạn mức Thời gian thực")
    st.write("Nhập thông tin giả định của một khách hàng để xem hệ thống ra quyết định duyệt thẻ và cấp hạn mức như thế nào.")
    
    if default_dict is None or fraud_dict is None:
        st.error("Chưa tìm thấy Mô hình máy học! Vui lòng chạy `train.py`.")
    else:
        # User input form
        with st.form("sandbox_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Hồ sơ cơ bản")
                current_age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
                yearly_income = st.number_input("Thu nhập năm ($)", min_value=1000, value=50000, step=1000)
                total_debt = st.number_input("Tổng nợ hiện tại ($)", min_value=0, value=20000, step=1000)
                credit_score = st.slider("Điểm tín dụng (FICO)", min_value=480, max_value=850, value=700)
                gender_encoded = st.selectbox("Giới tính", options=[("Nữ", 1), ("Nam", 0)], format_func=lambda x: x[0])[1]
                
            with col2:
                st.markdown("#### Lịch sử chi tiêu")
                total_credit_limit = st.number_input("Tổng hạn mức cũ đang có ($)", min_value=0, value=5000)
                net_spend = st.number_input("Tổng chi tiêu ròng ($)", min_value=0, value=25000)
                total_tx_count = st.number_input("Tổng số lượt giao dịch", min_value=1, value=500)
                max_monthly_cur = st.slider("CUR lớn nhất tháng (%)", 0.0, 1.0, 0.15)
                avg_monthly_cur = st.slider("CUR trung bình tháng (%)", 0.0, 1.0, 0.05)
                cur_30d = st.slider("CUR 30 ngày qua (%)", 0.0, 1.0, 0.05)
                dormant_card_ratio = st.slider("Tỷ lệ thẻ dormant (%)", 0.0, 1.0, 0.0)
                
            with col3:
                st.markdown("#### Hành vi & Lỗi giao dịch")
                insufficient_balance_rate = st.slider("Tỷ lệ lỗi Insufficient Balance (%)", 0.0, 1.0, 0.0)
                online_tx_rate = st.slider("Tỷ lệ giao dịch Online (%)", 0.0, 1.0, 0.2)
                essential_spend_ratio = st.slider("Tỷ lệ chi tiêu Thiết yếu (%)", 0.0, 1.0, 0.4)
                total_refund = st.number_input("Tổng tiền hoàn ($)", min_value=0, value=0)
                refund_rate = st.slider("Tỷ lệ giao dịch Hoàn tiền (%)", 0.0, 1.0, 0.0)
                has_spatiotemporal_fraud_signal = st.selectbox("GD đa địa lý trong 24h?", options=[("Không", 0), ("Có", 1)], format_func=lambda x: x[0])[1]
                security_error_count = st.number_input("Số lần lỗi bảo mật (Bad PIN/CVV)", min_value=0, max_value=100, value=0, step=1, help="Nhập >= 3 kết hợp tín hiệu khác để kích hoạt Fraud Signal")
                
            submit_btn = st.form_submit_button("Chạy Đánh giá Tín dụng", type="primary")
            
        if submit_btn:
            st.markdown("---")
            st.subheader("📊 Kết quả Phê duyệt Tín dụng")
            
            dti = total_debt / yearly_income if yearly_income > 0 else 0
            
            # Tính các đặc trưng phái sinh của Fraud phục vụ cho Default Model và hiển thị
            _sec_heavy = 1 if security_error_count >= 3 else 0
            _high_online = 1 if online_tx_rate > 0.70 else 0
            fraud_signal_score_display = (
                2 * _sec_heavy +
                2 * has_spatiotemporal_fraud_signal +
                1 * (1 if refund_rate > 0.10 else 0) +
                1 * _high_online
            )

            input_def = {
                'current_age': current_age,
                'yearly_income': yearly_income,
                'total_debt': total_debt,
                'credit_score': credit_score,
                'gender_encoded': gender_encoded,
                'total_credit_limit': total_credit_limit,
                'total_tx_count': total_tx_count,
                'net_spend': net_spend,
                'total_refund': total_refund,
                'refund_tx_count': total_tx_count * refund_rate,
                'insufficient_balance_count': total_tx_count * insufficient_balance_rate,
                'online_tx_count': total_tx_count * online_tx_rate,
                'essential_spend': net_spend * essential_spend_ratio,
                'refund_rate': refund_rate,
                'insufficient_balance_rate': insufficient_balance_rate,
                'online_tx_rate': online_tx_rate,
                'essential_spend_ratio': essential_spend_ratio,
                'cur_30d': cur_30d,
                'dormant_card_ratio': dormant_card_ratio,
                'security_error_count_heavy': _sec_heavy,
                'high_online_rate': _high_online,
                'fraud_signal_score': fraud_signal_score_display
            }
            df_def = pd.DataFrame([input_def])[default_dict['features']]
            X_def_scaled = default_scaler.transform(df_def)
            if default_dict['type'] in ['xgb', 'brf']:
                prob_def = default_dict['model'].predict_proba(df_def)[0, 1]
            else:
                prob_def = default_dict['model'].predict_proba(X_def_scaled)[0, 1]
                
            # Prepare input data for Fraud Model
            input_frd = input_def.copy()
            input_frd['dti'] = dti
            input_frd['max_monthly_cur'] = max_monthly_cur
            input_frd['avg_monthly_cur'] = avg_monthly_cur
            input_frd['has_spatiotemporal_fraud_signal'] = has_spatiotemporal_fraud_signal
            
            df_frd = pd.DataFrame([input_frd])[fraud_dict['features']]
            X_frd_scaled = fraud_scaler.transform(df_frd)
            if fraud_dict['type'] in ['xgb', 'brf']:
                prob_frd = fraud_dict['model'].predict_proba(df_frd)[0, 1]
            else:
                prob_frd = fraud_dict['model'].predict_proba(X_frd_scaled)[0, 1]

            # --- PHASE 2: CASH FLOW-BASED UNDERWRITING LOGIC ---
            # Step 1: Compute CFADS (monthly)
            cfads_monthly = max(0.0, (yearly_income - total_debt) / 12.0)
            
            # Step 2: Compute Dynamic Target Risk Cushion (C_target)
            c_base = 1.25
            
            # 2.1. FICO penalty
            if credit_score < 620:
                delta_fico = 0.30
            elif credit_score < 680:
                delta_fico = 0.15
            else:
                delta_fico = 0.0
                
            # 2.2. CUR penalty
            if max_monthly_cur > 0.30:
                delta_cur = 0.25
            elif avg_monthly_cur > 0.15:
                delta_cur = 0.15
            else:
                delta_cur = 0.0
                
            # 2.3. Liquidity penalty
            if insufficient_balance_rate > 0.05:
                delta_liquidity = 0.30
            elif insufficient_balance_rate > 0.0:
                delta_liquidity = 0.15
            else:
                delta_liquidity = 0.0
                
            # 2.4. AI risk penalty
            delta_ai = (prob_def + prob_frd) / 2.0
            
            c_target = c_base + delta_fico + delta_cur + delta_liquidity + delta_ai
            
            # Step 3: Max Monthly Payment (PMT_max)
            pmt_max = cfads_monthly / c_target if c_target > 0 else 0.0
            
            # Step 4: Base Limit (L_base) via Annuity Formula
            r_monthly = 0.015  # Lãi suất giả định 1.5%/tháng (18%/năm)
            annuity_factor = (1 - (1 + r_monthly)**(-12)) / r_monthly  # ~10.9075
            l_base = pmt_max * annuity_factor
            
            # Step 5: Circuit Breakers (Bộ chốt chặn cưỡng chế)
            cb_triggered = []
            if prob_def >= 0.50:
                cb_triggered.append("AI Default Risk Breaker (P_Default >= 50%)")
            if prob_frd >= 0.50:
                cb_triggered.append("AI Fraud Risk Breaker (P_Fraud >= 50%)")
            if credit_score < 580:
                cb_triggered.append("FICO Breaker (FICO < 580)")
            if cfads_monthly <= 0:
                cb_triggered.append("Cash Flow Breaker (CFADS <= 0, DTI >= 1.0)")
            if insufficient_balance_rate > 0.20:
                cb_triggered.append("Liquidity Distress Breaker (Tỷ lệ lỗi số dư > 20%)")
            if has_spatiotemporal_fraud_signal == 1:
                cb_triggered.append("Spatiotemporal Fraud Breaker (Giao dịch đa bang trong 24h)")
                
            # Step 6: AI Haircuts (Chiết khấu rủi ro)
            haircut_applied = 0.0
            haircut_reason = "Không áp dụng"
            if (0.35 <= prob_def < 0.50) or (0.35 <= prob_frd < 0.50):
                haircut_applied = 0.15
                haircut_reason = "Level 2 (Watch) - Rủi ro AI tăng cao (xác suất >= 35%)"
            if (620 <= credit_score < 650) or (avg_monthly_cur > 0.40):
                if 0.40 > haircut_applied:
                    haircut_applied = 0.40
                    haircut_reason = "Level 3 (Stress) - FICO cận biên (< 650) hoặc CUR TB cao (> 40%)"
            
            l_after_haircut = l_base * (1.0 - haircut_applied)
            
            # Step 7: Leverage Cap (Chốt chặn đòn bẩy)
            leverage_cap = max(0.0, 0.50 * yearly_income - total_debt)
            
            # Final decision and limit
            if len(cb_triggered) > 0:
                cl_new = 0.0
                decision = "❌ TỪ CHỐI CẤP TÍN DỤNG"
                color = "error"
            else:
                if l_after_haircut > leverage_cap:
                    cl_new = leverage_cap
                    decision = "⚠️ PHÊ DUYỆT (CÓ GIỚI HẠN ĐÒN BẨY)"
                    color = "warning"
                else:
                    cl_new = l_after_haircut
                    decision = "✅ PHÊ DUYỆT THÀNH CÔNG"
                    color = "success"

            # Display results
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Xác suất Vỡ nợ (P_Default)", f"{prob_def*100:.1f}%", delta="Rủi ro Cao" if prob_def >= 0.5 else "An toàn", delta_color="inverse")
            with col_b:
                st.metric("Xác suất Gian lận (P_Fraud)", f"{prob_frd*100:.1f}%", delta="Rủi ro Cao" if prob_frd >= 0.5 else "An toàn", delta_color="inverse")
            with col_c:
                st.metric("Hệ số đệm C_target", f"{c_target:.3f}", delta=f"+{(c_target - c_base):.2f} (Phạt rủi ro)", delta_color="inverse")

            st.write("---")
            if color == "success":
                st.success(f"### Quyết định: {decision}")
                st.success(f"### Hạn mức đề xuất (CL_new): ${cl_new:,.2f}")
            elif color == "warning":
                st.warning(f"### Quyết định: {decision}")
                st.warning(f"### Hạn mức đề xuất (CL_new): ${cl_new:,.2f} *(Hạn mức cơ sở ${l_after_haircut:,.2f} đã bị cắt giảm để tuân thủ chốt chặn đòn bẩy)*")
            else:
                st.error(f"### Quyết định: {decision}")
                for cb in cb_triggered:
                    st.error(f"- Vi phạm chốt chặn: **{cb}**")

            # Expandable Step-by-Step Underwriting Audit Report
            with st.expander("🔍 Báo cáo chi tiết thẩm định dòng tiền và quản trị rủi ro"):
                col_i, col_j = st.columns(2)
                with col_i:
                    st.markdown("#### 1. Thẩm định Dòng tiền thực tế")
                    st.write(f"- Thu nhập hàng tháng: `${yearly_income/12:,.2f}`")
                    st.write(f"- Nghĩa vụ nợ cũ hàng tháng (Phân bổ 12 tháng): `${total_debt/12:,.2f}`")
                    st.markdown(f"- **CFADS Cá nhân (CFADS_monthly)**: `${cfads_monthly:,.2f}`")
                    st.write(f"- Khả năng chi trả gốc lãi hàng tháng tối đa ($PMT_{{max}}$): `${pmt_max:,.2f}`")
                    st.write(f"- Quy đổi hạn mức cơ sở ($L_{{base}}$ - Hiện giá Niên kim): `${l_base:,.2f}`")
                    
                with col_j:
                    st.markdown("#### 2. Phân tích Biên đệm Rủi ro & Chiết khấu")
                    st.write(f"- Hệ số đệm nền ($C_{{base}}$): `1.25`")
                    st.write(f"- Phạt điểm FICO ($\\Delta C_{{FICO}}$): `+{delta_fico:.2f}`")
                    st.write(f"- Phạt sử dụng thẻ ($\\Delta C_{{CUR}}$): `+{delta_cur:.2f}`")
                    st.write(f"- Phạt thanh khoản ($\\Delta C_{{Liquidity}}$): `+{delta_liquidity:.2f}`")
                    st.write(f"- Phạt mô hình AI ($\\Delta C_{{AI}}$): `+{delta_ai:.2f}`")
                    st.write(f"- **Chiết khấu rủi ro AI (Haircut)**: `{haircut_applied*100:.1f}%` ({haircut_reason})")
                    st.write(f"- Hạn mức sau chiết khấu: `${l_after_haircut:,.2f}`")
                    st.write(f"- **Giới hạn đòn bẩy tối đa (Leverage Cap)**: `${leverage_cap:,.2f}` *(Hạn mức tối đa để nợ mới + cũ <= 50% thu nhập năm)*")

                with st.container():
                    st.markdown("#### 3. Phân tích Fraud Signal Score")
                    st.write(f"- Lỗi bảo mật nặng (≥3): `{'✓' if security_error_count >= 3 else '✗'}` | `w=2` → `{2 if security_error_count >= 3 else 0} điểm`")
                    st.write(f"- Giao dịch đa bang 24h: `{'✓' if has_spatiotemporal_fraud_signal else '✗'}` | `w=2` → `{2 if has_spatiotemporal_fraud_signal else 0} điểm`")
                    st.write(f"- Hoàn tiền bất thường (>10%): `{'✓' if refund_rate > 0.10 else '✗'}` | `w=1` → `{1 if refund_rate > 0.10 else 0} điểm`")
                    st.write(f"- Giao dịch online cao (>70%): `{'✓' if online_tx_rate > 0.70 else '✗'}` | `w=1` → `{1 if online_tx_rate > 0.70 else 0} điểm`")
                    st.metric("Fraud Signal Score (S_fraud)", fraud_signal_score_display, delta=f"{'≥ θ=3 → FRAUD' if fraud_signal_score_display >= 3 else '< θ=3 → Bình thường'}", delta_color="inverse" if fraud_signal_score_display >= 3 else "normal")


