import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
from datetime import datetime, timedelta
import base64

st.set_page_config(page_title="yFinance Veri İndirici", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 720px; padding-top: 2rem;}
    .stDownloadButton > button {width: 100%; background-color: #0d6efd; color: white; font-weight: 600;}
    .info-box {background: #f0f2f6; border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("📊 yFinance Veri İndirici")

# --- 1. Sembol Girişi ---
symbol = st.text_input("Sembol", placeholder="Örn: THYAO.IS, AAPL, BTC-USD")

if symbol:
    ticker = yf.Ticker(symbol)

    # En eski veriyi bul
    try:
        hist_max = ticker.history(period="max")
        if hist_max.empty:
            st.error(f"'{symbol}' için veri bulunamadı. Sembolü kontrol edin.")
            st.stop()
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

    hist_max.index = hist_max.index.tz_localize(None)
    oldest_date = hist_max.index.min().date()
    newest_date = hist_max.index.max().date()
    total_days = len(hist_max)

    st.markdown(f"""
    <div class="info-box">
        <b>En eski veri tarihi:</b> {oldest_date}<br>
        <b>En yeni veri tarihi:</b> {newest_date}<br>
        <b>Toplam veri günü:</b> {total_days:,}
    </div>
    """, unsafe_allow_html=True)

    # --- 2. Tarih Aralığı ---
    st.subheader("Tarih Aralığı")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Başlangıç", value=oldest_date, min_value=oldest_date, max_value=newest_date)
    with col2:
        end_date = st.date_input("Bitiş", value=newest_date, min_value=oldest_date, max_value=newest_date)

    if start_date > end_date:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        st.stop()

    # Tarih aralığına göre filtrele
    mask = (hist_max.index.date >= start_date) & (hist_max.index.date <= end_date)
    df = hist_max.loc[mask].copy()

    if df.empty:
        st.warning("Seçilen tarih aralığında veri yok.")
        st.stop()

    # --- 3. Boş/0 değer sayısı ---
    zero_or_null = int((df.isnull().sum().sum()) + (df == 0).sum().sum())
    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki veri günü:</b> {len(df):,}<br>
        <b>Boş veya 0 değer taşıyan hücre sayısı:</b> {zero_or_null:,}
    </div>
    """, unsafe_allow_html=True)

    # --- 4. Mevcut sütunlar ve seçim ---
    st.subheader("Veri Seçimi")
    available_cols = [c for c in df.columns]

    st.caption("İndirmek istediğiniz verileri seçin:")
    selected_cols = []
    cols_per_row = 4
    rows = [available_cols[i:i+cols_per_row] for i in range(0, len(available_cols), cols_per_row)]
    for row in rows:
        checkbox_cols = st.columns(len(row))
        for i, col_name in enumerate(row):
            with checkbox_cols[i]:
                if st.checkbox(col_name, value=True, key=f"cb_{col_name}"):
                    selected_cols.append(col_name)

    if not selected_cols:
        st.info("En az bir sütun seçmelisiniz.")
        st.stop()

    # --- 5. Kapanış Grafiği ---
    if "Close" in df.columns:
        st.subheader("Kapanış Grafiği")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Close"], color="#0d6efd", linewidth=1.2)
        ax.set_title(f"{symbol} - Kapanış Fiyatı", fontsize=13, fontweight="bold")
        ax.set_xlabel("Tarih", fontsize=10)
        ax.set_ylabel("Kapanış", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Görseli göster
        st.pyplot(fig)

        # Görseli kopyalanabilir hale getir (PNG olarak base64)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        st.markdown("*Görsele sağ tıklayıp kopyalayabilirsiniz:*")
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" style="width:100%; border-radius:8px; cursor:pointer;" />',
            unsafe_allow_html=True
        )

    # --- 6. Excel İndirme ---
    st.subheader("İndir")
    export_df = df[selected_cols].copy()
    export_df.index.name = "Date"
    export_df = export_df.reset_index()

    # Excel oluştur
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Data")
    excel_buf.seek(0)

    file_name = f"{symbol.replace('.', '_')}_{start_date}_{end_date}.xlsx"

    st.download_button(
        label=f"📥 Excel İndir ({len(export_df):,} satır)",
        data=excel_buf.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
