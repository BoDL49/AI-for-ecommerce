import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import json
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(
    page_title="Hệ thống dự báo doanh thu",
    page_icon="🚀",
    layout="wide"
)

#Thông báo
def show_custom_toast(message, icon, duration_ms=5000):
    st.toast(message, icon=icon)
    toast_css = f"""
        <style>
            @keyframes progress-bar-countdown {{
                from {{ width: 100%; }}
                to {{ width: 0%; }}
            }}

            [data-testid="stToast"]:last-of-type {{
                position: relative; 
                overflow: hidden;   
            }}

            [data-testid="stToast"]:last-of-type::after {{
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 5px;
                background-color: #00B084; 
                animation: progress-bar-countdown {duration_ms / 1000}s linear forwards;
            }}
        </style>
        """
    st.markdown(toast_css, unsafe_allow_html=True)


@st.cache_resource
def load_olist_resources(category):
    safe_category_name = category.replace(' ', '_').replace('&', 'and')
    model_path = os.path.join('olist_models', f'lgbm_model_{safe_category_name}.pkl')
    encoder_path = os.path.join('olist_models', 'state_encoder_GLOBAL.pkl')
    try:
        model = joblib.load(model_path); state_encoder = joblib.load(encoder_path)
        return model, state_encoder
    except FileNotFoundError: return None, None

@st.cache_resource
def load_rossmann_resources():
    model_path = os.path.join('rossmann_models', 'lgbm_model_rossmann.pkl')
    features_path = os.path.join('rossmann_models', 'model_features.json')
    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f: features = json.load(f)
        return model, features
    except FileNotFoundError: return None, None

st.title("🚀 Hệ thống dự báo doanh thu")

selected_dataset = st.sidebar.selectbox("Chọn bộ dữ liệu muốn phân tích:", ["Olist E-commerce", "Rossmann Store Sales"])

#Olist E-commerce
if selected_dataset == "Olist E-commerce":
    st.header("Phân tích dự báo doanh thu Olist")
    #st.markdown("Tải lên file dữ liệu lịch sử Olist, chọn danh mục và khoảng thời gian để so sánh dự đoán với thực tế.")
    
    categories = ['bed_bath_table', 'health_beauty', 'sports_leisure', 'furniture_decor', 'computers_accessories']
    selected_category = st.sidebar.selectbox("Chọn Danh mục Sản phẩm:", categories)
    
    with st.spinner(f"Đang tải mô hình Olist cho '{selected_category}'..."):
        model, state_encoder = load_olist_resources(selected_category)

    if not model or not state_encoder:
        st.error(f"Lỗi: Không tìm thấy file mô hình Olist cho danh mục '{selected_category}'.")
    else:
        if 'olist_model_loaded' not in st.session_state or st.session_state.olist_model_loaded != selected_category:
            show_custom_toast(f"Đã tải mô hình cho '{selected_category}'", icon="✅")
            st.session_state.olist_model_loaded = selected_category
            
        #uploaded_file = st.file_uploader("File Olist (phải chứa 'order_purchase_timestamp' và 'revenue')", type=['csv', 'xlsx'], key="olist_uploader")
        uploaded_file = st.file_uploader("",type=['csv', 'xlsx'], key="olist_uploader")
        if uploaded_file is not None:
            try:
                if 'df_olist' not in st.session_state or st.session_state.get('file_name_olist') != uploaded_file.name:
                    with st.spinner('Đang đọc và xử lý file...'):
                        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, parse_dates=['order_purchase_timestamp'])
                        else: df = pd.read_excel(uploaded_file, parse_dates=['order_purchase_timestamp'])
                        st.session_state['df_olist'] = df
                        st.session_state['file_name_olist'] = uploaded_file.name
                        show_custom_toast("Tải file thành công!", icon="📄")
                df = st.session_state['df_olist']
                
                st.subheader("Chọn khoảng thời gian để phân tích")
                period_type = st.selectbox("Chọn loại chu kỳ:", ["Ngày", "Tuần", "Tháng", "Năm"], key="olist_period")
                min_date = df['order_purchase_timestamp'].min().date(); max_date = df['order_purchase_timestamp'].max().date()

                if period_type == "Ngày": start_date = end_date = st.date_input("Chọn ngày:", value=max_date, min_value=min_date, max_value=max_date, key="olist_date")
                elif period_type == "Tuần": selected_day = st.date_input("Chọn một ngày trong tuần:", value=max_date, min_value=min_date, max_value=max_date, key="olist_week"); start_date = selected_day - timedelta(days=selected_day.weekday()); end_date = start_date + timedelta(days=6)
                elif period_type == "Tháng": 
                    year = st.selectbox("Chọn năm:", options=sorted(df['order_purchase_timestamp'].dt.year.unique(), reverse=True), key="olist_year"); month = st.selectbox("Chọn tháng:", options=range(1, 13), key="olist_month")
                    start_date = datetime(year, month, 1).date(); next_m = (start_date.replace(day=28) + timedelta(days=4)); end_date = (next_m - timedelta(days=next_m.day))
                else:
                    year = st.selectbox("Chọn năm:", options=sorted(df['order_purchase_timestamp'].dt.year.unique(), reverse=True), key="olist_year_full")
                    start_date = datetime(year, 1, 1).date(); end_date = datetime(year, 12, 31).date()

                if st.button("Phân tích & so sánh Olist", type="primary", use_container_width=True, key="olist_button"):
                    df_period = df[(df['order_purchase_timestamp'].dt.date >= start_date) & (df['order_purchase_timestamp'].dt.date <= end_date) & (df['category'] == selected_category)].copy()
                    if df_period.empty: show_custom_toast(f"Không tìm thấy dữ liệu cho '{selected_category}'", icon="⚠️")
                    else:
                        with st.spinner('Đang xử lý và dự đoán Olist...'):
                            start_time = time.perf_counter()
                            df_period['purchase_month'] = df_period['order_purchase_timestamp'].dt.month; df_period['purchase_dayofweek'] = df_period['order_purchase_timestamp'].dt.dayofweek
                            y_actual = df_period['revenue']; X_features = df_period.drop(columns=['revenue', 'order_purchase_timestamp', 'order_delivered_customer_date', 'category'])
                            X_features['customer_state'] = state_encoder.transform(X_features['customer_state']); X_ready = X_features[model.feature_name_]
                            predictions = model.predict(X_ready)
                            end_time = time.perf_counter(); processing_time = end_time - start_time
                        
                        show_custom_toast("Phân tích hoàn tất!", icon="🎉")
                        st.header("Kết quả phân tích Olist")
                        total_actual = y_actual.sum(); total_predicted = predictions.sum(); mape = mean_absolute_percentage_error(y_actual, predictions) * 100
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Doanh thu Thực tế", f"{total_actual:,.2f} BRL"); col2.metric("Doanh thu Dự đoán", f"{total_predicted:,.2f} BRL", f"{(total_predicted-total_actual)/total_actual:.2%}"); col3.metric("Độ sai số (MAPE)", f"{mape:.2f}%"); col4.metric("Độ chính xác", f"{100 - mape:.2f}%"); col5.metric("Thời gian Xử lý", f"{processing_time:.4f} giây")
                        st.subheader("Biểu đồ So sánh Doanh thu"); chart_data = pd.DataFrame({'Ngày': df_period['order_purchase_timestamp'], 'Doanh thu Thực tế': y_actual.values, 'Doanh thu Dự đoán': predictions}).set_index('Ngày')
                        if period_type == "Năm": st.line_chart(chart_data.resample('M').sum())
                        elif len(chart_data) > 1 and chart_data.index.to_series().dt.date.nunique() > 1: st.line_chart(chart_data.resample('D').sum())
                        else: st.bar_chart(chart_data)
                        st.subheader("Bảng Dữ liệu Chi tiết"); st.dataframe(chart_data)
            except Exception as e: show_custom_toast(f"Đã xảy ra lỗi: {e}", icon="❌")

#Rossmann Store Sales
elif selected_dataset == "Rossmann Store Sales":
    st.header("Phân tích dự báo doanh số Rossmann")
    #st.markdown("Tải lên file dữ liệu lịch sử của Rossmann để so sánh dự đoán doanh số với thực tế.")
    
    with st.spinner("Đang tải mô hình Rossmann..."):
        model, model_features = load_rossmann_resources()
        
    if not model or not model_features:
        st.error("Lỗi: Không tìm thấy file mô hình Rossmann.")
    else:
        if 'rossmann_model_loaded' not in st.session_state:
            show_custom_toast("Đã tải thành công mô hình Rossmann.", icon="✅")
            st.session_state.rossmann_model_loaded = True
            
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx'], key="rossmann_uploader")
        
        if uploaded_file is not None:
            try:
                if 'df_rossmann' not in st.session_state or st.session_state.get('file_name_rossmann') != uploaded_file.name:
                    with st.spinner('Đang đọc và xử lý file...'):
                        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                        else: df = pd.read_excel(uploaded_file, parse_dates=['Date'])
                        st.session_state['df_rossmann'] = df
                        st.session_state['file_name_rossmann'] = uploaded_file.name
                        show_custom_toast("Tải file thành công!", icon="📄")
                df = st.session_state['df_rossmann']
                
                st.subheader("Chọn Cửa hàng và khoảng thời gian")
                store_id = st.selectbox("Chọn ID Cửa hàng:", sorted(df['Store'].unique()), key="rossmann_store")
                df_store = df[df['Store'] == store_id].copy()
                min_date = df_store['Date'].min().date(); max_date = df_store['Date'].max().date()
                period_type = st.selectbox("Chọn loại chu kỳ:", ["Ngày", "Tuần", "Tháng", "Năm"], key="rossmann_period")

                if period_type == "Ngày": start_date = end_date = st.date_input("Chọn ngày:", value=max_date, min_value=min_date, max_value=max_date, key="rossmann_date")
                elif period_type == "Tuần": selected_day = st.date_input("Chọn một ngày trong tuần:", value=max_date, min_value=min_date, max_value=max_date, key="rossmann_week"); start_date = selected_day - timedelta(days=selected_day.weekday()); end_date = start_date + timedelta(days=6)
                elif period_type == "Tháng": 
                    year = st.selectbox("Chọn năm:", options=sorted(df_store['Date'].dt.year.unique(), reverse=True), key="rossmann_year"); month = st.selectbox("Chọn tháng:", options=range(1, 13), key="rossmann_month")
                    start_date = datetime(year, month, 1).date(); next_m = (start_date.replace(day=28) + timedelta(days=4)); end_date = (next_m - timedelta(days=next_m.day))
                else: 
                    year = st.selectbox("Chọn năm:", options=sorted(df_store['Date'].dt.year.unique(), reverse=True), key="rossmann_year_full")
                    start_date = datetime(year, 1, 1).date(); end_date = datetime(year, 12, 31).date()

                if st.button("Phân tích & So sánh Rossmann", type="primary", use_container_width=True, key="rossmann_button"):
                    df_period = df_store[(df_store['Date'].dt.date >= start_date) & (df_store['Date'].dt.date <= end_date)].copy()
                    if df_period.empty: show_custom_toast("Không tìm thấy dữ liệu cho cửa hàng và khoảng thời gian đã chọn.", icon="⚠️")
                    else:
                        with st.spinner('Đang xử lý và dự đoán Rossmann...'):
                            start_time = time.perf_counter()
                            y_actual = df_period['Sales']; X_features = df_period.drop(columns=['Sales', 'Date'])
                            for col in model_features:
                                if col not in X_features.columns: X_features[col] = 0
                            X_ready = X_features[model_features]
                            predictions = model.predict(X_ready); predictions[predictions < 0] = 0
                            end_time = time.perf_counter(); processing_time = end_time - start_time
                        
                        show_custom_toast("Phân tích hoàn tất!", icon="🎉")
                        st.header(f"Kết quả Phân tích cho Cửa hàng {store_id}")
                        total_actual = y_actual.sum(); total_predicted = predictions.sum(); mape = mean_absolute_percentage_error(y_actual, predictions) * 100
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Doanh số Thực tế", f"{total_actual:,.0f}"); col2.metric("Doanh số Dự đoán", f"{total_predicted:,.0f}", f"{(total_predicted-total_actual)/total_actual:.2%}"); col3.metric("Độ sai số (MAPE)", f"{mape:.2f}%"); col4.metric("Độ chính xác", f"{100 - mape:.2f}%"); col5.metric("Thời gian Xử lý", f"{processing_time:.4f} giây")
                        st.subheader("Biểu đồ So sánh Doanh số"); chart_data = pd.DataFrame({'Ngày': df_period['Date'], 'Doanh số Thực tế': y_actual.values, 'Doanh số Dự đoán': predictions}).set_index('Ngày')
                        if period_type == "Năm": st.line_chart(chart_data.resample('W').sum())
                        elif period_type in ["Tháng", "Tuần"]: st.line_chart(chart_data.resample('D').sum())
                        else: st.bar_chart(chart_data)
                        st.subheader("Bảng Dữ liệu Chi tiết"); st.dataframe(chart_data)
            except Exception as e: show_custom_toast(f"Đã xảy ra lỗi: {e}", icon="❌")