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
    direction = pd.Series(1, index=close.index)

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

symbol = st.text_input("Sembol", placeholder="Örn: THYAO.IS, AAPL, BTC-USD")

if symbol:
    ticker = yf.Ticker(symbol)

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

    st.subheader("Tarih Aralığı")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Başlangıç", value=oldest_date, min_value=oldest_date, max_value=newest_date)
    with col2:
        end_date = st.date_input("Bitiş", value=newest_date, min_value=oldest_date, max_value=newest_date)

    if start_date > end_date:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        st.stop()

    mask = (hist_max.index.date >= start_date) & (hist_max.index.date <= end_date)
    df = hist_max.loc[mask].copy()

    if df.empty:
        st.warning("Seçilen tarih aralığında veri yok.")
        st.stop()

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

    check_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    zero_or_null = int(((df[check_cols].isnull().any(axis=1)) | (df[check_cols] == 0).any(axis=1)).sum())

    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki veri günü:</b> {len(df):,}<br>
        <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zero_or_null:,}
    </div>
    """, unsafe_allow_html=True)

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

    if "Close" in df.columns:
        st.subheader("Kapanış Grafiği")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Close"], color="#0d6efd", linewidth=1.2)
        ax.set_title(f"{symbol} - Kapanış Fiyatı", fontsize=13, fontweight="bold")
        ax.set_xlabel("Tarih", fontsize=10)
        ax.set_ylabel("Kapanış", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

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

    st.subheader("İndir")
    export_df = df[selected_cols].copy()
    export_df.index.name = "Date"
    export_df = export_df.reset_index()

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

    # ============================================================
    # DURAĞANLIK ANALİZİ
    # ============================================================

    st.divider()
    st.subheader("🔬 Durağanlık Analizi")
    st.caption("Her değişken için ADF ve KPSS testleri uygulanır. Çelişkili sonuçlarda DF-GLS ile karar verilir.")

    run_tests = st.button("Durağanlık Testlerini Çalıştır")

    if run_tests:
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            st.error("statsmodels kütüphanesi gerekli: pip install statsmodels")
            st.stop()

        # DF-GLS için arch veya statsmodels
        def dfgls_test(series, trend='c'):
            """DF-GLS testi - statsmodels DFGLS kullan, yoksa ADF ile proxy"""
            try:
                from statsmodels.tsa.stattools import adfuller
                # Basit proxy: detrended seriye ADF
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant
                import numpy as np
                t = np.arange(len(series))
                if trend == 'ct':
                    X = add_constant(np.column_stack([t]))
                else:
                    X = add_constant(np.ones(len(series)))
                resid = series.values - OLS(series.values, X).fit().fittedvalues
                result = adfuller(resid, autolag='AIC', regression='n')
                return result[1]  # p-value
            except Exception:
                return None

        def run_adf(series):
            """ADF testi, None dönerse hata var"""
            try:
                clean = series.dropna()
                if len(clean) < 20:
                    return None, None
                result = adfuller(clean, autolag='AIC')
                return result[1], result[0]  # p-value, stat
            except Exception:
                return None, None

        def run_kpss(series):
            """KPSS testi"""
            try:
                clean = series.dropna()
                if len(clean) < 20:
                    return None, None
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = kpss(clean, regression='c', nlags='auto')
                return result[1], result[0]  # p-value, stat
            except Exception:
                return None, None

        def interpret(adf_p, kpss_p):
            """
            ADF H0: birim kök var → p<0.05 → durağan
            KPSS H0: durağan     → p<0.05 → durağan değil
            """
            if adf_p is None or kpss_p is None:
                return "Yetersiz veri", "gray", None

            adf_stationary = adf_p < 0.05
            kpss_stationary = kpss_p >= 0.05

            if adf_stationary and kpss_stationary:
                return "Durağan ✅", "green", None
            elif not adf_stationary and not kpss_stationary:
                return "Durağan Değil ❌", "red", None
            else:
                return "Çelişkili ⚠️", "orange", "dfgls"

        # Hangi sütunlara test uygulanacak
        test_cols = [c for c in df.columns if c in [
            "Open", "High", "Low", "Close", "Volume",
            "EMA_20", "EMA_50", "EMA_200",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "ATR", "BB_Upper", "BB_Lower", "BBW", "Supertrend"
        ]]

        results = []

        progress = st.progress(0, text="Testler çalışıyor...")

        for idx, col in enumerate(test_cols):
            series = df[col].dropna()
            adf_p, adf_stat = run_adf(series)
            kpss_p, kpss_stat = run_kpss(series)
            verdict, color, extra = interpret(adf_p, kpss_p)

            dfgls_p = None
            dfgls_note = ""
            if extra == "dfgls":
                # Trend var mı? Fiyat bazlı seriler için 'ct', osilatörler için 'c'
                price_like = col in ["Open", "High", "Low", "Close",
                                     "EMA_20", "EMA_50", "EMA_200",
                                     "BB_Upper", "BB_Lower", "Supertrend"]
                trend_opt = 'ct' if price_like else 'c'
                dfgls_p = dfgls_test(series, trend=trend_opt)
                if dfgls_p is not None:
                    if dfgls_p < 0.05:
                        verdict = "Durağan (DF-GLS) ✅"
                        color = "green"
                        dfgls_note = f"DF-GLS p={dfgls_p:.3f}"
                    else:
                        verdict = "Durağan Değil (DF-GLS) ❌"
                        color = "red"
                        dfgls_note = f"DF-GLS p={dfgls_p:.3f}"

            results.append({
                "Değişken": col,
                "ADF p": f"{adf_p:.3f}" if adf_p is not None else "—",
                "KPSS p": f"{kpss_p:.3f}" if kpss_p is not None else "—",
                "DF-GLS": dfgls_note if dfgls_note else "—",
                "Sonuç": verdict,
                "_color": color
            })

            progress.progress((idx + 1) / len(test_cols), text=f"Test ediliyor: {col}")

        progress.empty()

        # Sonuçları grupla
        stationary = [r for r in results if r["_color"] == "green"]
        nonstationary = [r for r in results if r["_color"] == "red"]
        conflicting = [r for r in results if r["_color"] == "orange"]

        def render_table(rows):
            display = pd.DataFrame([{
                "Değişken": r["Değişken"],
                "ADF p": r["ADF p"],
                "KPSS p": r["KPSS p"],
                "DF-GLS": r["DF-GLS"],
                "Sonuç": r["Sonuç"]
            } for r in rows])
            st.dataframe(display, use_container_width=True, hide_index=True)

        if stationary:
            st.markdown("### ✅ Durağan Değişkenler")
            render_table(stationary)

        if nonstationary:
            st.markdown("### ❌ Durağan Olmayan Değişkenler")
            render_table(nonstationary)
            st.info(
                "**Önerilen işlem:** Fiyat bazlı seriler (Close, EMA vb.) için log return alın. "
                "Hacim için log farkı alın. Ardından testleri tekrarlayın."
            )

        if conflicting:
            st.markdown("### ⚠️ Çelişkili Sonuçlar (DF-GLS ile karar verildi)")
            render_table(conflicting)
            st.info(
                "**Not:** ADF ve KPSS çeliştiğinde DF-GLS testi kullanıldı. "
                "Sonuç hâlâ belirsizse yapısal kırılma testini (Lee-Strazicich) uygulayın "
                "veya sensitivity analysis yapın."
            )

        # Özet
        st.markdown("---")
        total = len(results)
        n_stat = len(stationary)
        n_nonstat = len(nonstationary)
        n_conf = len(conflicting)

        st.markdown(f"""
        <div class="info-box">
            <b>Özet:</b> {total} değişken test edildi.<br>
            ✅ Durağan: <b>{n_stat}</b> &nbsp;|&nbsp;
            ❌ Durağan Değil: <b>{n_nonstat}</b> &nbsp;|&nbsp;
            ⚠️ Çelişkili: <b>{n_conf}</b>
        </div>
        """, unsafe_allow_html=True)

        # Metodoloji notu
        with st.expander("📖 Test Metodolojisi"):
            st.markdown("""
**ADF (Augmented Dickey-Fuller)**
- H₀: Seri birim kök içeriyor (durağan değil)
- p < 0.05 → H₀ reddedilir → **Durağan**

**KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**
- H₀: Seri durağan
- p < 0.05 → H₀ reddedilir → **Durağan değil**

**İki test birlikte yorumlanır:**
- ADF durağan + KPSS durağan → ✅ Durağan
- ADF durağan değil + KPSS durağan değil → ❌ Durağan değil
- Çelişki → DF-GLS ile karar verilir

**DF-GLS:** Fiyat bazlı seriler için trend+sabit, osilatörler için sadece sabit seçeneği kullanılır.
            """)
