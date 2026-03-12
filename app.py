import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
from datetime import datetime, timedelta
import base64

st.set_page_config(page_title="yfinance veri indirici", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 720px; padding-top: 2rem;}
    .stDownloadButton > button {width: 100%; background-color: #0d6efd; color: white; font-weight: 600;}
    .info-box {background: #f0f2f6; border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("📊 yfinance veri indirici")
st.caption("Varlık sembolünü bilmiyorsanız Gemini'ye 'yfinance ...... tickerı nedir' yazın.")

# ============================================================
# Teknik İndikatör Fonksiyonları
# ============================================================

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    bbw = (bb_upper - bb_lower) / sma
    return bb_upper, bb_lower, bbw

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1 = up (bullish), -1 = down (bearish)

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend

# ============================================================
# Ana Uygulama
# ============================================================

# --- 1. Sembol Girişi ---
symbol = st.text_input("Sembol", placeholder="Örn: THYAO.IS, AAPL, BTC-USD")

if symbol:
    ticker = yf.Ticker(symbol)

    # En eski veriyi bul
    try:
        hist_max = ticker.history(period="max", actions=False)
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

    # --- Teknik İndikatörleri Hesapla ---
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA_20"] = calc_ema(close, 20)
    df["EMA_50"] = calc_ema(close, 50)
    df["EMA_200"] = calc_ema(close, 200)

    df["RSI"] = calc_rsi(close)

    macd_line, macd_signal, macd_hist = calc_macd(close)
    df["MACD"] = macd_line
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    df["ATR"] = calc_atr(high, low, close)

    bb_upper, bb_lower, bbw = calc_bollinger(close)
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    df["BBW"] = bbw

    df["Supertrend"] = calc_supertrend(high, low, close)

    # --- 3. Boş/0 değer sayısı (sadece OHLCV) ---
    check_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    zero_or_null = int(((df[check_cols].isnull().any(axis=1)) | (df[check_cols] == 0).any(axis=1)).sum())

    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki veri günü:</b> {len(df):,}<br>
        <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zero_or_null:,}
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
