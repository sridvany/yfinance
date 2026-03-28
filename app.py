import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from datetime import datetime
import base64
from scipy import stats
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import mutual_info_regression

st.set_page_config(page_title="yfinance veri indirici", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 720px; padding-top: 2rem;}
    .stDownloadButton > button {width: 100%; background-color: #0d6efd; color: white; font-weight: 600;}
    .info-box {background: #f0f2f6; border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("📊 yfinance veri indirici")
st.caption("Garbage In, Garbage Out")

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
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_bollinger(close, period=20, std_dev=2):
    sma      = close.rolling(window=period).mean()
    std      = close.rolling(window=period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    return bb_upper, bb_lower, (bb_upper - bb_lower) / sma

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    atr        = calc_atr(high, low, close, period)
    hl2        = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = pd.Series(np.nan, index=close.index)
    direction  = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if   close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] ==  1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]
        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
    return supertrend

# ============================================================
# YENİ İNDİKATÖR FONKSİYONLARI
# ============================================================

def calc_roc(close, period=10):
    """Rate of Change — fiyat değişim hızı (%)"""
    return ((close - close.shift(period)) / close.shift(period)) * 100

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K ve %D"""
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calc_adx(high, low, close, period=14):
    """Average Directional Index — trend gücü"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm,  index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    atr_s     = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_s
    minus_di  = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_s

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx

def calc_williams_r(high, low, close, period=14):
    """Williams %R — aşırı alım/satım"""
    highest_high = high.rolling(window=period).max()
    lowest_low   = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def calc_cci(high, low, close, period=20):
    """Commodity Channel Index — ortalamadan sapma"""
    typical_price = (high + low + close) / 3
    sma_tp  = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_dev)

def calc_obv(close, volume):
    """On Balance Volume — hacim-fiyat uyumu"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow — para akışı yönü"""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def calc_volume_roc(volume, period=10):
    """Volume Rate of Change — hacim ivmesi (%)"""
    return ((volume - volume.shift(period)) / volume.shift(period)) * 100

def calc_mfi(high, low, close, volume, period=14):
    """Money Flow Index — RSI'nin hacim ağırlıklı versiyonu"""
    typical_price  = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction      = typical_price.diff()
    pos_mf         = raw_money_flow.where(direction > 0, 0.0)
    neg_mf         = raw_money_flow.where(direction < 0, 0.0)
    pos_sum        = pos_mf.rolling(window=period).sum()
    neg_sum        = neg_mf.rolling(window=period).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum))

def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI — RSI'ya Stochastic formülü uygulanır"""
    rsi        = calc_rsi(close, rsi_period)
    min_rsi    = rsi.rolling(window=stoch_period).min()
    max_rsi    = rsi.rolling(window=stoch_period).max()
    stoch_rsi  = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_k = stoch_rsi.rolling(window=k_smooth).mean() * 100
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()
    return stoch_rsi_k, stoch_rsi_d


# ============================================================
# Ana Uygulama
# ============================================================

symbol = st.text_input("Sembol", placeholder="Örn: THYAO.IS, AAPL, BTC-USD")

INTERVAL_OPTIONS = {
    "1 Dakika":  "1m",
    "2 Dakika":  "2m",
    "5 Dakika":  "5m",
    "15 Dakika": "15m",
    "30 Dakika": "30m",
    "1 Saat":    "1h",
    "1 Gün":     "1d",
    "1 Hafta":   "1wk",
    "1 Ay":      "1mo",
}
INTERVAL_MAX_DAYS = {
    "1m": 30, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": None, "1wk": None, "1mo": None,
}

selected_interval_label = st.selectbox("Zaman Dilimi", list(INTERVAL_OPTIONS.keys()), index=6)
interval    = INTERVAL_OPTIONS[selected_interval_label]
is_intraday = interval in ("1m", "2m", "5m", "15m", "30m", "1h")
max_days    = INTERVAL_MAX_DAYS.get(interval)

if is_intraday and max_days:
    st.info(f"⏱ **{selected_interval_label}** verisi için yfinance en fazla **son {max_days} günlük** veri sunuyor.")

if symbol:
    ticker = yf.Ticker(symbol)

    try:
        if is_intraday:
            hist_max = ticker.history(period=f"{max_days}d", interval=interval, actions=False)
        else:
            hist_max = ticker.history(period="max", interval=interval, actions=False)

        if hist_max.empty:
            st.error(f"'{symbol}' için '{selected_interval_label}' verisinde veri bulunamadı.")
            st.stop()
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

    if hist_max.index.tz is not None:
        hist_max.index = hist_max.index.tz_localize(None)

    oldest_date = hist_max.index.min().date()
    newest_date = hist_max.index.max().date()
    bar_label   = "bar" if is_intraday else "gün"

    st.markdown(f"""
    <div class="info-box">
        <b>En eski veri tarihi:</b> {oldest_date}<br>
        <b>En yeni veri tarihi:</b> {newest_date}<br>
        <b>Toplam {bar_label} sayısı:</b> {len(hist_max):,}
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
    df   = hist_max.loc[mask].copy()
    if df.empty:
        st.warning("Seçilen tarih aralığında veri yok.")
        st.stop()

    close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]

    # Mevcut indikatörler
    df["EMA_20"]     = calc_ema(close, 20)
    df["EMA_50"]     = calc_ema(close, 50)
    df["EMA_200"]    = calc_ema(close, 200)
    df["RSI"]        = calc_rsi(close)
    df["MACD"]       = calc_macd(close)[0]
    df["ATR"]        = calc_atr(high, low, close)
    df["BB_Upper"], df["BB_Lower"], df["BBW"] = calc_bollinger(close)
    df["Supertrend"] = calc_supertrend(high, low, close)
    df["Return"]     = np.log(close).diff()

    # Yeni indikatörler
    df["ROC"]        = calc_roc(close)
    df["Stoch_K"], df["Stoch_D"] = calc_stochastic(high, low, close)
    df["ADX"]        = calc_adx(high, low, close)
    df["Williams_R"] = calc_williams_r(high, low, close)
    df["CCI"]        = calc_cci(high, low, close)
    df["OBV"]        = calc_obv(close, volume)
    df["CMF"]        = calc_cmf(high, low, close, volume)
    df["Volume_ROC"] = calc_volume_roc(volume)
    df["MFI"]          = calc_mfi(high, low, close, volume)
    df["StochRSI_K"], df["StochRSI_D"] = calc_stoch_rsi(close)

    check_cols   = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    zero_or_null = int(((df[check_cols].isnull().any(axis=1)) | (df[check_cols] == 0).any(axis=1)).sum())
    consec_dupes = int((df[check_cols].eq(df[check_cols].shift(1)).all(axis=1)).sum())
    null_only    = int(df[check_cols].isnull().any(axis=1).sum())
    ohlc_same    = int(((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])).sum())

    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki {bar_label} sayısı:</b> {len(df):,}<br>
        <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zero_or_null:,}<br>
        <b>Arka arkaya aynı OHLCV satır sayısı:</b> {consec_dupes:,}<br>
        <b>Boş hücresi olan satır sayısı (0 hariç):</b> {null_only:,}<br>
        <b>Open=High=Low=Close olan satır sayısı:</b> {ohlc_same:,}
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # Veri Seçimi
    # ============================================================

    st.subheader("Veri Seçimi")
    st.caption("İndirmek istediğiniz verileri seçin:")

    CATEGORIES = {
        "📊 Ham Veri":    (["Open", "High", "Low", "Close", "Volume"],        "fiyat ve hacim verisinin ham hali"),
        "📈 Trend":       (["EMA_20", "EMA_50", "EMA_200", "MACD", "Supertrend", "ADX"], "fiyatın hangi yönde hareket ettiğini gösterir"),
        "⚡ Momentum":    (["RSI", "ROC", "CCI", "Williams_R", "Stoch_K", "Stoch_D", "StochRSI_K", "StochRSI_D"], "fiyat hareketinin hızını ve gücünü ölçer"),
        "🌊 Volatilite":  (["ATR", "BB_Upper", "BB_Lower", "BBW"],            "fiyatın ne kadar oynadığını ölçer"),
        "📦 Hacim":       (["OBV", "CMF", "MFI", "Volume_ROC"],               "alım-satım hacminin yönünü ve gücünü gösterir"),
        "💹 Fiyat":       (["Return"],                                         "logaritmik günlük getiri"),
    }

    selected_cols = []
    available_set = set(df.columns)

    for cat_label, (cat_cols, cat_desc) in CATEGORIES.items():
        existing = [c for c in cat_cols if c in available_set]
        if not existing:
            continue
        st.markdown(f"**{cat_label}** *({cat_desc})*")
        cb_cols = st.columns(4)
        for i, col_name in enumerate(existing):
            with cb_cols[i % 4]:
                if st.checkbox(col_name, value=True, key=f"cb_{col_name}"):
                    selected_cols.append(col_name)

    # Kategoride olmayan sütunlar varsa "Diğer" altında göster
    categorized = {c for cols, _ in CATEGORIES.values() for c in cols}
    other_cols  = [c for c in df.columns if c not in categorized]
    if other_cols:
        st.markdown("**📎 Diğer**")
        cb_cols = st.columns(4)
        for i, col_name in enumerate(other_cols):
            with cb_cols[i % 4]:
                if st.checkbox(col_name, value=True, key=f"cb_{col_name}"):
                    selected_cols.append(col_name)

    if not selected_cols:
        st.info("En az bir sütun seçmelisiniz.")
        st.stop()

    # ============================================================
    # Kapanış Grafiği
    # ============================================================

    if "Close" in df.columns:
        st.subheader("Kapanış Grafiği")
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines",
            line=dict(color="#16a34a", width=1.5),
            name="Close"
        ))
        fig_px.update_layout(
            title="",
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=True, thickness=0.07),
                rangeselector=dict(
                    buttons=[
                        dict(count=1,  label="1M",  step="month", stepmode="backward"),
                        dict(count=3,  label="3M",  step="month", stepmode="backward"),
                        dict(count=6,  label="6M",  step="month", stepmode="backward"),
                        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#f0f2f6", activecolor="#f0f2f6",
                )
            ),
            yaxis=dict(title="Close", fixedrange=False),
            dragmode="pan",
            hovermode="x unified",
            height=420,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_px, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})

    # ============================================================
    # Excel İndir
    # ============================================================

    st.subheader("İndir")
    export_df = df[selected_cols].copy()
    export_df.index.name = "Datetime" if is_intraday else "Date"
    export_df = export_df.reset_index()
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Data")
    excel_buf.seek(0)
    file_name = f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}.xlsx"
    st.download_button(
        label=f"📥 Excel İndir ({len(export_df):,} satır)",
        data=excel_buf.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # OHLC eşit satır filtreleyip indikatörleri yeniden hesapla
    ohlc_mask   = ~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))
    df_clean    = df[ohlc_mask][["Open", "High", "Low", "Close", "Volume"]].copy()
    removed_cnt = len(df) - len(df_clean)

    if not df_clean.empty:
        _c = df_clean["Close"]; _h = df_clean["High"]; _l = df_clean["Low"]; _v = df_clean["Volume"]

        # Mevcut indikatörler
        df_clean["EMA_20"]     = calc_ema(_c, 20)
        df_clean["EMA_50"]     = calc_ema(_c, 50)
        df_clean["EMA_200"]    = calc_ema(_c, 200)
        df_clean["RSI"]        = calc_rsi(_c)
        df_clean["MACD"]       = calc_macd(_c)[0]
        df_clean["ATR"]        = calc_atr(_h, _l, _c)
        df_clean["BB_Upper"], df_clean["BB_Lower"], df_clean["BBW"] = calc_bollinger(_c)
        df_clean["Supertrend"] = calc_supertrend(_h, _l, _c)
        df_clean["Return"]     = np.log(_c).diff()

        # Yeni indikatörler
        df_clean["ROC"]        = calc_roc(_c)
        df_clean["Stoch_K"], df_clean["Stoch_D"] = calc_stochastic(_h, _l, _c)
        df_clean["ADX"]        = calc_adx(_h, _l, _c)
        df_clean["Williams_R"] = calc_williams_r(_h, _l, _c)
        df_clean["CCI"]        = calc_cci(_h, _l, _c)
        df_clean["OBV"]        = calc_obv(_c, _v)
        df_clean["CMF"]        = calc_cmf(_h, _l, _c, _v)
        df_clean["Volume_ROC"] = calc_volume_roc(_v)
        df_clean["MFI"]          = calc_mfi(_h, _l, _c, _v)
        df_clean["StochRSI_K"], df_clean["StochRSI_D"] = calc_stoch_rsi(_c)

        clean_selected = [c for c in selected_cols if c in df_clean.columns]
        export_clean   = df_clean[clean_selected].copy()
        export_clean.index.name = "Datetime" if is_intraday else "Date"
        export_clean   = export_clean.reset_index()

        excel_clean = BytesIO()
        with pd.ExcelWriter(excel_clean, engine="openpyxl") as writer:
            export_clean.to_excel(writer, index=False, sheet_name="Data")
        excel_clean.seek(0)

        file_name_clean = f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}_cleaned.xlsx"
        st.download_button(
            label=f"📥 Excel İndir — OHLC Eşit Satırlar Çıkarılmış ({len(export_clean):,} satır, {removed_cnt:,} satır silindi)",
            data=excel_clean.getvalue(),
            file_name=file_name_clean,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 3. Excel: boş hücreli satırlar da çıkarılmış
        df_clean2     = df_clean[clean_selected].dropna()
        removed_nan   = len(df_clean) - len(df_clean2)
        export_clean2 = df_clean2.copy()
        export_clean2.index.name = "Datetime" if is_intraday else "Date"
        export_clean2 = export_clean2.reset_index()

        excel_clean2 = BytesIO()
        with pd.ExcelWriter(excel_clean2, engine="openpyxl") as writer:
            export_clean2.to_excel(writer, index=False, sheet_name="Data")
        excel_clean2.seek(0)

        file_name_clean2 = f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}_fully_cleaned.xlsx"
        st.download_button(
            label=f"📥 Excel İndir — OHLC Eşit + Boş Hücreli Satırlar Çıkarılmış ({len(export_clean2):,} satır, {removed_nan:,} satır daha silindi)",
            data=excel_clean2.getvalue(),
            file_name=file_name_clean2,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ============================================================
        # SPEARMAN KORELASYON ANALİZİ
        # ============================================================

        st.divider()
        st.subheader("🔗 Spearman Korelasyon Analizi")
        st.caption(
            "Seçilen sayısal sütunlar arasında Spearman sıra korelasyonu ve p-değerleri hesaplanır. "
            "Parametrik olmayan bu yöntem, lineer olmayan ilişkileri de yakalar."
        )

        numeric_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df[c])]

        if len(numeric_cols) < 2:
            st.info("Spearman analizi için en az 2 sayısal sütun seçmelisiniz.")
        else:
            k     = len(numeric_cols)
            alpha = st.slider(
                "Anlamlılık Eşiği (α)",
                min_value=0.01, max_value=0.10, value=0.05, step=0.01,
                help="Bu değerin altındaki p-değerleri istatistiksel olarak anlamlı kabul edilir."
            )

            if st.button("Spearman Korelasyonunu Hesapla"):
                sub = df[numeric_cols].dropna()
                n   = len(sub)

                rho_mat = pd.DataFrame(np.nan, index=numeric_cols, columns=numeric_cols)
                p_mat   = pd.DataFrame(np.nan, index=numeric_cols, columns=numeric_cols)

                for i, c1 in enumerate(numeric_cols):
                    for j, c2 in enumerate(numeric_cols):
                        if i == j:
                            rho_mat.loc[c1, c2] = 1.0
                            p_mat.loc[c1, c2]   = 0.0
                        elif i < j:
                            rho, pval = stats.spearmanr(sub[c1], sub[c2])
                            rho_mat.loc[c1, c2] = rho_mat.loc[c2, c1] = round(rho, 4)
                            p_mat.loc[c1, c2]   = p_mat.loc[c2, c1]   = round(pval, 4)

                st.session_state["spearman_rho"]   = rho_mat
                st.session_state["spearman_p"]     = p_mat
                st.session_state["spearman_n"]     = n
                st.session_state["spearman_alpha"] = alpha
                st.session_state["spearman_cols"]  = numeric_cols

            if "spearman_rho" in st.session_state:
                rho_mat      = st.session_state["spearman_rho"]
                p_mat        = st.session_state["spearman_p"]
                n            = st.session_state["spearman_n"]
                alpha        = st.session_state["spearman_alpha"]
                numeric_cols = st.session_state["spearman_cols"]
                k            = len(numeric_cols)

                st.markdown("### Korelasyon Isı Haritası")
                fig2, ax2 = plt.subplots(figsize=(max(6, k * 0.9), max(5, k * 0.8)))
                cmap = plt.cm.RdYlGn
                im   = ax2.imshow(rho_mat.values.astype(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
                plt.colorbar(im, ax=ax2, shrink=0.8, label="Spearman ρ")
                ax2.set_xticks(range(k)); ax2.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
                ax2.set_yticks(range(k)); ax2.set_yticklabels(numeric_cols, fontsize=9)
                ax2.set_title(f"{symbol} — Spearman Korelasyon Matrisi (n={n})", fontsize=11, fontweight="bold")

                for i in range(k):
                    for j in range(k):
                        val = rho_mat.values[i, j]
                        if not np.isnan(val):
                            sig       = (i != j) and (p_mat.values[i, j] < alpha)
                            txt_color = "black" if abs(val) < 0.5 else "white"
                            marker    = "*" if sig else ""
                            ax2.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                                     fontsize=7.5, color=txt_color, fontweight="bold" if sig else "normal")

                fig2.tight_layout()
                buf2 = BytesIO()
                fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
                buf2.seek(0)
                img_b64_2 = base64.b64encode(buf2.read()).decode()
                plt.close(fig2)
                st.markdown("*  = α düzeyinde istatistiksel olarak anlamlı &nbsp;|&nbsp; Görsele sağ tıklayıp kopyalayabilirsiniz.*")
                st.markdown(
                    f'<img src="data:image/png;base64,{img_b64_2}" style="width:100%; border-radius:8px;" />',
                    unsafe_allow_html=True
                )

                st.markdown(f"### Anlamlı Korelasyonlar (p < {alpha})")
                pairs = []
                for i, c1 in enumerate(numeric_cols):
                    for j, c2 in enumerate(numeric_cols):
                        if i >= j:
                            continue
                        rho  = rho_mat.loc[c1, c2]
                        pval = p_mat.loc[c1, c2]
                        if not np.isnan(rho) and pval < alpha:
                            pairs.append({
                                "Değişken 1": c1,
                                "Değişken 2": c2,
                                "ρ":          rho,
                                "p-değeri":   pval,
                                "Yön":        "Pozitif ↑" if rho > 0 else "Negatif ↓",
                                "Güç":        (
                                    "Çok Güçlü" if abs(rho) >= 0.80 else
                                    "Güçlü"     if abs(rho) >= 0.60 else
                                    "Orta"      if abs(rho) >= 0.40 else
                                    "Zayıf"
                                ),
                            })

                if pairs:
                    pairs_df = pd.DataFrame(pairs).sort_values("ρ", key=abs, ascending=False)

                    def color_rho(val):
                        if not isinstance(val, float): return ""
                        c     = plt.cm.RdYlGn((val + 1) / 2)
                        hex_c = mcolors.to_hex(c)
                        text  = "black" if abs(val) < 0.5 else "white"
                        return f"background-color: {hex_c}; color: {text}; font-weight: bold"

                    st.dataframe(
                        pairs_df.style.map(color_rho, subset=["ρ"]),
                        use_container_width=True, hide_index=True
                    )

                    excel_corr = BytesIO()
                    with pd.ExcelWriter(excel_corr, engine="openpyxl") as w:
                        rho_mat.to_excel(w, sheet_name="Rho Matrisi")
                        p_mat.to_excel(w, sheet_name="P Matrisi")
                        pairs_df.to_excel(w, sheet_name="Anlamlı Çiftler", index=False)
                    excel_corr.seek(0)
                    st.download_button(
                        label="📥 Korelasyon Sonuçlarını İndir (Excel)",
                        data=excel_corr.getvalue(),
                        file_name=f"{symbol.replace('.', '_')}_spearman_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info(f"α = {alpha} düzeyinde anlamlı korelasyon bulunamadı.")

                total_pairs = k * (k - 1) // 2
                sig_count   = len(pairs) if pairs else 0
                st.markdown(f"""
                <div class="info-box">
                    <b>Gözlem sayısı:</b> {n:,} &nbsp;|&nbsp;
                    <b>Test edilen çift:</b> {total_pairs} &nbsp;|&nbsp;
                    <b>Anlamlı çift (p &lt; {alpha}):</b> {sig_count}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📖 Spearman Korelasyonu Hakkında"):
                    st.markdown(f"""
**Spearman ρ (rho):** Pearson'ın aksine ham değerler yerine sıralar üzerinden hesaplanır;
doğrusal olmayan monoton ilişkileri ve aykırı değerlere karşı dayanıklılığı sayesinde
finansal zaman serilerinde tercih edilir.

| |ρ| | Güç |
|------|------|
| 0.80 – 1.00 | Çok Güçlü |
| 0.60 – 0.79 | Güçlü |
| 0.40 – 0.59 | Orta |
| 0.00 – 0.39 | Zayıf |

**Not:** Yıldız (*) işaretli hücreler α = {alpha} düzeyinde anlamlıdır.
Korelasyon nedensellik anlamına gelmez.
                    """)

        # ============================================================
        # FEATURE SEÇİM ANALİZİ
        # ============================================================

        st.divider()
        st.subheader("🔬 Feature Seçim Analizi")
        st.caption(
            "Fully cleaned veri üzerinde sırasıyla 4 analiz uygulanır. "
            "Her adımda çıkarılması önerilen değişkenler listelenir. "
            "Adımlar zincir şeklinde: her adım öncekinden hayatta kalanlar üzerinde çalışır."
        )

        fs_df         = df_clean2.copy()
        target        = "Close"
        date_col_name = "Datetime" if is_intraday else "Date"

        all_candidates = [
            c for c in fs_df.columns
            if pd.api.types.is_numeric_dtype(fs_df[c]) and c != target
        ]

        fs_key = f"{symbol}_{start_date}_{end_date}"
        if st.session_state.get("fs_key") != fs_key:
            for k_ in [
                "fs_after_corr", "fs_corr_remove", "fs_corr_low", "fs_corr_high", "fs_corr_table",
                "fs_after_vif", "fs_vif_df", "fs_vif_remove",
                "fs_after_mi", "fs_mi_df", "fs_mi_remove", "fs_mi_thr",
                "fs_adf_df", "fs_non_stationary",
                "fs_s1", "fs_s2", "fs_s3", "fs_s4",
            ]:
                st.session_state.pop(k_, None)
            st.session_state["fs_key"]        = fs_key
            st.session_state["fs_after_corr"] = all_candidates.copy()
            st.session_state["fs_after_vif"]  = all_candidates.copy()
            st.session_state["fs_after_mi"]   = all_candidates.copy()

        st.markdown(
            f"**Başlangıç feature seti ({len(all_candidates)}):** "
            f"`{'`, `'.join(all_candidates)}`"
        )

        # ── ADIM 1: KORELASYON ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 1️⃣ Korelasyon Filtresi")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            corr_low_thr = st.slider(
                "Düşük eşik — |r| < bu değer → çıkar",
                0.05, 0.30, 0.15, 0.01, key="corr_low_thr"
            )
        with col_c2:
            corr_high_thr = st.slider(
                "Yüksek eşik — |r| > bu değer → multicollinearity riski",
                0.900, 0.999, 0.995, 0.001, format="%.3f", key="corr_high_thr"
            )

        if st.button("▶ Korelasyon Analizini Çalıştır", key="run_corr"):
            sub       = fs_df[all_candidates + [target]].dropna()
            corr_vals = sub[all_candidates].corrwith(sub[target]).abs()
            low_list  = corr_vals[corr_vals < corr_low_thr].index.tolist()

            fm    = sub[all_candidates].corr().abs()
            upper = fm.where(np.triu(np.ones(fm.shape), k=1).astype(bool))
            high_list = []
            for col in upper.columns:
                partners = upper.index[upper[col] > corr_high_thr].tolist()
                for p in partners:
                    drop = p if corr_vals.get(p, 0) <= corr_vals.get(col, 0) else col
                    if drop not in high_list and drop not in low_list:
                        high_list.append(drop)

            remove  = list(set(low_list + high_list))
            survive = [f for f in all_candidates if f not in remove]

            tbl = corr_vals.reset_index()
            tbl.columns = ["Feature", "|r| ile Close"]
            tbl = tbl.sort_values("|r| ile Close", ascending=False)

            st.session_state.update({
                "fs_corr_table":  tbl,
                "fs_corr_low":    low_list,
                "fs_corr_high":   high_list,
                "fs_corr_remove": remove,
                "fs_after_corr":  survive,
                "fs_s1": True,
            })

        if st.session_state.get("fs_s1"):
            st.dataframe(
                st.session_state["fs_corr_table"].style.format({"|r| ile Close": "{:.4f}"}),
                use_container_width=True, hide_index=True
            )
            low_r   = st.session_state["fs_corr_low"]
            high_r  = st.session_state["fs_corr_high"]
            survive = st.session_state["fs_after_corr"]
            ca, cb  = st.columns(2)
            with ca:
                st.error(f"**Düşük |r| → çıkar:** {low_r if low_r else 'Yok'}")
                st.warning(f"**Multicollinearity → çıkar:** {high_r if high_r else 'Yok'}")
            with cb:
                st.success(f"**Kalan ({len(survive)}):** {survive}")

        # ── ADIM 2: VIF ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 2️⃣ VIF Analizi")
        st.caption("Korelasyon adımından hayatta kalanlar üzerinde çalışır.")

        vif_thr = st.slider(
            "VIF eşiği — bu değerin üzerindekiler çıkarılır",
            5.0, 20.0, 10.0, 0.5, key="vif_thr"
        )

        if st.button("▶ VIF Analizini Çalıştır", key="run_vif"):
            after_corr = st.session_state.get("fs_after_corr", all_candidates)
            sub = fs_df[after_corr].dropna()
            X   = sub.values.astype(float)

            rows = []
            for i, col in enumerate(after_corr):
                try:
                    v = variance_inflation_factor(X, i)
                except Exception:
                    v = np.nan
                rows.append({"Feature": col, "VIF": round(v, 2)})

            vif_df  = pd.DataFrame(rows).sort_values("VIF", ascending=False)
            vif_rem = vif_df[vif_df["VIF"] > vif_thr]["Feature"].tolist()
            survive = [f for f in after_corr if f not in vif_rem]

            st.session_state.update({
                "fs_vif_df":     vif_df,
                "fs_vif_remove": vif_rem,
                "fs_after_vif":  survive,
                "fs_s2": True,
            })

        if st.session_state.get("fs_s2"):
            vif_df   = st.session_state["fs_vif_df"]
            vif_rem  = st.session_state["fs_vif_remove"]
            survive  = st.session_state["fs_after_vif"]
            _vif_thr = vif_thr

            def _vif_color(val):
                if not isinstance(val, (int, float)): return ""
                if val > _vif_thr:            return "background-color:#f8d7da; color:#842029"
                if val > _vif_thr * 0.7:      return "background-color:#fff3cd; color:#664d03"
                return                                "background-color:#d1e7dd; color:#0a3622"

            st.dataframe(
                vif_df.style.map(_vif_color, subset=["VIF"]),
                use_container_width=True, hide_index=True
            )
            ca, cb = st.columns(2)
            with ca:
                st.error(f"**VIF > {vif_thr} → çıkar:** {vif_rem if vif_rem else 'Yok'}")
            with cb:
                st.success(f"**Kalan ({len(survive)}):** {survive}")

        # ── ADIM 3: MUTUAL INFORMATION ───────────────────────────
        st.markdown("---")
        st.markdown("### 3️⃣ Mutual Information")
        st.caption("VIF adımından hayatta kalanlar üzerinde çalışır. Non-lineer ilişkileri de yakalar.")

        mi_pct = st.slider(
            "Alt yüzdelik dilim eşiği — bu dilimin altındakiler çıkarılır (%)",
            5, 40, 20, 5, key="mi_pct"
        )

        if st.button("▶ MI Analizini Çalıştır", key="run_mi"):
            after_vif = st.session_state.get("fs_after_vif", all_candidates)
            sub       = fs_df[after_vif + [target]].replace([np.inf, -np.inf], np.nan).dropna()
            X_mi      = sub[after_vif].values
            y_mi      = sub[target].values

            scores  = mutual_info_regression(X_mi, y_mi, random_state=42)
            mi_df   = pd.DataFrame({"Feature": after_vif, "MI Skoru": scores.round(4)})
            mi_df   = mi_df.sort_values("MI Skoru", ascending=False)

            thr_val = np.percentile(scores, mi_pct)
            mi_rem  = mi_df[mi_df["MI Skoru"] < thr_val]["Feature"].tolist()
            survive = [f for f in after_vif if f not in mi_rem]

            st.session_state.update({
                "fs_mi_df":     mi_df,
                "fs_mi_remove": mi_rem,
                "fs_after_mi":  survive,
                "fs_mi_thr":    thr_val,
                "fs_s3": True,
            })

        if st.session_state.get("fs_s3"):
            mi_df   = st.session_state["fs_mi_df"]
            mi_rem  = st.session_state["fs_mi_remove"]
            survive = st.session_state["fs_after_mi"]
            thr_val = st.session_state.get("fs_mi_thr", 0)

            colors_mi = ["#dc3545" if f in mi_rem else "#198754" for f in mi_df["Feature"]]
            fig_mi, ax_mi = plt.subplots(figsize=(8, max(3, len(mi_df) * 0.45)))
            ax_mi.barh(mi_df["Feature"], mi_df["MI Skoru"], color=colors_mi)
            ax_mi.axvline(thr_val, color="#ffc107", linestyle="--", linewidth=1.2, label=f"Eşik: {thr_val:.4f}")
            ax_mi.legend(fontsize=9)
            ax_mi.set_xlabel("MI Skoru")
            ax_mi.set_title("Mutual Information Skorları")
            ax_mi.invert_yaxis()
            fig_mi.tight_layout()
            buf_mi = BytesIO()
            fig_mi.savefig(buf_mi, format="png", dpi=130, bbox_inches="tight")
            buf_mi.seek(0)
            st.image(buf_mi)
            plt.close(fig_mi)

            ca, cb = st.columns(2)
            with ca:
                st.error(f"**Alt %{mi_pct} → çıkar:** {mi_rem if mi_rem else 'Yok'}")
            with cb:
                st.success(f"**Kalan ({len(survive)}):** {survive}")

        # ── ADIM 4: ADF DURAĞANLIK ───────────────────────────────
        st.markdown("---")
        st.markdown("### 4️⃣ ADF Durağanlık Testi")
        st.caption(
            "MI adımından hayatta kalanlar + Close üzerinde çalışır. "
            "p < 0.05 → durağan  |  p ≥ 0.05 → durağan değil."
        )

        if st.button("▶ ADF Testini Çalıştır", key="run_adf"):
            after_mi  = st.session_state.get("fs_after_mi", all_candidates)
            test_cols = after_mi + [target]
            rows = []
            for col in test_cols:
                series = fs_df[col].dropna()
                try:
                    stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
                    stationary = pval < 0.05
                    rows.append({
                        "Feature":         col,
                        "ADF İstatistiği": round(stat, 4),
                        "p-değeri":        round(pval, 4),
                        "Kritik (%5)":     round(crit["5%"], 4),
                        "Durum":           "✅ Durağan" if stationary else "❌ Durağan Değil",
                    })
                except Exception:
                    rows.append({
                        "Feature": col, "ADF İstatistiği": np.nan,
                        "p-değeri": np.nan, "Kritik (%5)": np.nan, "Durum": "⚠️ Hata",
                    })

            adf_df   = pd.DataFrame(rows)
            non_stat = adf_df[~adf_df["Durum"].str.startswith("✅")]["Feature"].tolist()

            st.session_state.update({
                "fs_adf_df":         adf_df,
                "fs_non_stationary": non_stat,
                "fs_s4": True,
            })

        if st.session_state.get("fs_s4"):
            adf_df   = st.session_state["fs_adf_df"]
            non_stat = st.session_state.get("fs_non_stationary", [])

            def _adf_color(val):
                if not isinstance(val, str): return ""
                if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                return ""

            st.dataframe(
                adf_df.style.map(_adf_color, subset=["Durum"]),
                use_container_width=True, hide_index=True
            )

            non_feat = [f for f in non_stat if f != target]
            if non_feat:
                st.warning(
                    f"**Durağan Olmayan Feature'lar:** `{'`, `'.join(non_feat)}` — "
                    "log veya diff dönüşümü önerilebilir. LSTM için zorunlu değildir."
                )
            if target in non_stat:
                st.warning(
                    "**Close durağan değil** — tahmin hedefi olduğu için çıkarılmaz; "
                    "gerekirse log(Close) türetebilirsiniz."
                )
            if not non_stat:
                st.success("Tüm değişkenler durağan.")

        # ── SONUÇ & EXCEL İNDİR ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏁 Sonuç — Seçilen Feature'lar")

        final_features = st.session_state.get("fs_after_mi", all_candidates)
        st.info(
            f"**Hayatta kalan feature'lar ({len(final_features)}):** "
            f"`{'`, `'.join(final_features)}`"
        )

        # Checkbox seçim konsolu
        st.caption("İndirilecek sütunları seçin (Close her zaman dahildir):")
        export_candidates = [target] + [f for f in all_candidates if f in fs_df.columns]
        final_selected = [target]  # Close her zaman dahil
        fs_cols_per_row = 4
        fs_rows = [export_candidates[i:i+fs_cols_per_row] for i in range(0, len(export_candidates), fs_cols_per_row)]
        for fs_row in fs_rows:
            fs_cb_cols = st.columns(len(fs_row))
            for i, col_name in enumerate(fs_row):
                with fs_cb_cols[i]:
                    disabled = col_name == target
                    checked  = st.checkbox(col_name, value=True, key=f"fs_cb_{col_name}", disabled=disabled)
                    if checked and col_name not in final_selected:
                        final_selected.append(col_name)

        export_final = fs_df[final_selected].copy()
        export_final.index.name = date_col_name
        export_final = export_final.reset_index()

        buf_final = BytesIO()
        with pd.ExcelWriter(buf_final, engine="openpyxl") as writer:
            export_final.to_excel(writer, index=False, sheet_name="Data")
        buf_final.seek(0)

        st.download_button(
            label=f"📥 Seçili Feature'ları İndir — {len(final_selected)} sütun, {len(export_final):,} satır",
            data=buf_final.getvalue(),
            file_name=f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}_selected_features.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        st.warning("Filtreleme sonrası veri kalmadı.")
