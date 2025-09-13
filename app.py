import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kebangkrutan Perusahaan",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fungsi untuk memuat CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS '{file_name}' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan path-nya benar.")
        return None

# Memuat custom styles dan model
load_css('style.css')
model = load_model('model/final_xgb_model.joblib')

# Daftar fitur final yang dibutuhkan model
FINAL_FEATURES = [
    'operating gross margin', 'persistent eps in the last four seasons',
    'operating profit per person', 'net income to total assets',
    'net income to stockholder\'s equity', 'cash flow rate',
    'cash flow per share', 'current ratio', 'quick ratio',
    'working capital to total assets', 'quick assets/total assets',
    'cash/total assets', 'cash flow to total assets', 'cfo to assets',
    'cash flow to equity', 'current liability to current assets',
    'total debt/total net worth', 'long-term fund suitability ratio (a)',
    'borrowing dependency', 'current liability to assets',
    'liability-assets flag', 'equity to liability', 'total asset turnover',
    'total expense/assets', 'after-tax net profit growth rate',
    'net value growth rate', 'total asset return growth rate ratio',
    'net value per share (a)', 'working capital/equity'
]


# Header Aplikasi
st.markdown("""
<div class="main-header">
    <div class="main-title">Prediksi Kebangkrutan Perusahaan</div>
    <div class="main-subtitle">Sistem Analisis Keuangan dengan Bantuan Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Membuat form 
with st.form(key='financial_data_form'):
    # Membuat tab untuk mengelompokkan fitur
    tab_profit, tab_liquid, tab_leverage, tab_activity, tab_growth, tab_market = st.tabs([
        "📊 Profitabilitas", "💧 Likuiditas", "⚖️ Solvabilitas", "🔄 Efisiensi", "📈 Pertumbuhan", "🏢 Nilai Pasar"
    ])

    # Dictionary untuk menampung semua input
    inputs = {}

    with tab_profit:
        st.markdown('<div class="section-title">Rasio Profitabilitas</div>', unsafe_allow_html=True)
        inputs['operating gross margin'] = st.number_input('Operating Gross Margin', value=0.6, format="%.4f")
        inputs['persistent eps in the last four seasons'] = st.number_input('Persistent EPS in the Last Four Seasons', value=0.2, format="%.4f")
        inputs['operating profit per person'] = st.number_input('Operating Profit per Person', value=0.4, format="%.4f")
        inputs['net income to total assets'] = st.number_input('Net Income to Total Assets', value=0.8, format="%.4f")
        inputs['net income to stockholder\'s equity'] = st.number_input('Net Income to Stockholder\'s Equity (ROE)', value=0.8, format="%.4f")

    with tab_liquid:
        st.markdown('<div class="section-title">Rasio Likuiditas</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inputs['cash flow rate'] = st.number_input('Cash Flow Rate', value=0.46, format="%.4f")
            inputs['cash flow per share'] = st.number_input('Cash Flow Per Share', value=0.31, format="%.4f")
            inputs['current ratio'] = st.number_input('Current Ratio', value=0.01, format="%.4f")
            inputs['quick ratio'] = st.number_input('Quick Ratio', value=0.007, format="%.4f")
            inputs['working capital to total assets'] = st.number_input('Working Capital to Total Assets', value=0.8, format="%.4f")
            inputs['quick assets/total assets'] = st.number_input('Quick Assets / Total Assets', value=0.4, format="%.4f")
        with c2:
            inputs['cash/total assets'] = st.number_input('Cash / Total Assets', value=0.07, format="%.4f")
            inputs['cash flow to total assets'] = st.number_input('Cash Flow to Total Assets', value=0.6, format="%.4f")
            inputs['cfo to assets'] = st.number_input('CFO to Assets', value=0.5, format="%.4f")
            inputs['cash flow to equity'] = st.number_input('Cash Flow to Equity', value=0.3, format="%.4f")
            inputs['current liability to current assets'] = st.number_input('Current Liability to Current Assets', value=0.03, format="%.4f")

    with tab_leverage:
        st.markdown('<div class="section-title">Rasio Solvabilitas (Leverage)</div>', unsafe_allow_html=True)
        inputs['total debt/total net worth'] = st.number_input('Total Debt / Total Net Worth', value=0.005, format="%.4f")
        inputs['long-term fund suitability ratio (a)'] = st.number_input('Long-term Fund Suitability Ratio (A)', value=0.005, format="%.4f")
        inputs['borrowing dependency'] = st.number_input('Borrowing Dependency', value=0.3, format="%.4f")
        inputs['current liability to assets'] = st.number_input('Current Liability to Assets', value=0.08, format="%.4f")
        inputs['liability-assets flag'] = st.selectbox('Liability-Assets Flag (1 jika Utang > Aset)', [0, 1])
        inputs['equity to liability'] = st.number_input('Equity to Liability', value=0.04, format="%.4f")

    with tab_activity:
        st.markdown('<div class="section-title">Rasio Efisiensi & Aktivitas</div>', unsafe_allow_html=True)
        inputs['total asset turnover'] = st.number_input('Total Asset Turnover', value=0.2, format="%.4f")
        inputs['total expense/assets'] = st.number_input('Total Expense / Assets', value=0.03, format="%.4f")

    with tab_growth:
        st.markdown('<div class="section-title">Rasio Pertumbuhan</div>', unsafe_allow_html=True)
        inputs['after-tax net profit growth rate'] = st.number_input('After-tax Net Profit Growth Rate', value=0.6, format="%.4f")
        inputs['net value growth rate'] = st.number_input('Net Value Growth Rate', value=0.0004, format="%.4f")
        inputs['total asset return growth rate ratio'] = st.number_input('Total Asset Return Growth Rate Ratio', value=0.2, format="%.4f")

    with tab_market:
        st.markdown('<div class="section-title">Rasio Nilai Pasar</div>', unsafe_allow_html=True)
        inputs['net value per share (a)'] = st.number_input('Net Value Per Share (A)', value=0.1, format="%.4f")
        inputs['working capital/equity'] = st.number_input('Working Capital / Equity', value=0.7, format="%.4f")

    # Tombol submit di dalam form
    submitted = st.form_submit_button("🔍 Analisis Risiko Kebangkrutan")


# create logic
if submitted:
    if model:
        # get data
        input_data = pd.DataFrame([inputs], columns=FINAL_FEATURES)

        # get prediction
        prediction = model.predict(input_data)[0]
        proba_bangkrut = model.predict_proba(input_data)[0][1]
        
        # Output 1 untuk Teks Kesimpulan Prediksi Biner
        if prediction == 1:
            prediction_text = "Berdasarkan model prediksi dari kondisi keuangan yang dimasukkan, perusahaan berpotensi mengalami kebangkrutan."
        else:
            prediction_text = "Berdasarkan model prediksi dari kondisi keuangan yang dimasukkan, perusahaan tidak berpotensi mengalami kebangkrutan."

        # Output 2 untuk Level Risiko Berdasarkan Probabilitas
        if proba_bangkrut >= 0.7:
            risk_level = "RISIKO TINGGI"
            risk_color = "#ff4757" # Merah
            risk_class = "risk-high"
        elif proba_bangkrut >= 0.5: # Rentang 0.5 s/d 0.7
            risk_level = "RISIKO SEDANG"
            risk_color = "#ffa502" # Oranye
            risk_class = "risk-medium"
        else: # Di bawah 0.5
            risk_level = "RISIKO RENDAH"
            risk_color = "#2ed573" # Hijau
            risk_class = "risk-low"
        
        # Output 3 untuk Skor Kesehatan Finansial
        health_score = (1 - proba_bangkrut) * 100

        # Menampilkan hasil gabungan
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <h3 style="color: {risk_color}; margin-bottom: 1rem;">{risk_level}</h3>
            <div style="font-size: 3.5rem; margin: 1rem 0; font-weight: bold; color: {risk_color};">{proba_bangkrut:.2f}</div>
            <div style="color: #666;">Skor Kesehatan Finansial: {health_score:.2f}</div>
            <hr style="margin: 1.5rem 0;">
            <p style="color: #333; line-height: 1.6;"><strong>Kesimpulan Prediksi:</strong><br>{prediction_text}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Model tidak berhasil dimuat. Prediksi tidak dapat dilakukan.")

# Footer
st.markdown("""
<div style="text-align: center; color: white; margin-top: 3rem; opacity: 0.8;">
    <small>🏢 Sistem Prediksi Kebangkrutan Perusahaan | Project Data Science</small>
</div>
""", unsafe_allow_html=True)