import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from datetime import datetime, date
import base64
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, linear_reset
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="veri indir", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 720px; padding-top: 2rem;}
    .stDownloadButton > button {width: 100%; background-color: #0d6efd; color: white; font-weight: 600;}
    .info-box {background: #f0f2f6; border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("📊 yfinance veri indir")
st.caption("yfinance sembolünü öğrenmek için finance.yahoo.com/lookup/")

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

def calc_roc(close, period=10):
    return ((close - close.shift(period)) / close.shift(period)) * 100

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calc_adx(high, low, close, period=14):
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
    highest_high = high.rolling(window=period).max()
    lowest_low   = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def calc_cci(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma_tp  = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_dev)

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_cmf(high, low, close, volume, period=20):
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def calc_volume_roc(volume, period=10):
    return ((volume - volume.shift(period)) / volume.shift(period)) * 100

def calc_mfi(high, low, close, volume, period=14):
    typical_price  = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction      = typical_price.diff()
    pos_mf         = raw_money_flow.where(direction > 0, 0.0)
    neg_mf         = raw_money_flow.where(direction < 0, 0.0)
    pos_sum        = pos_mf.rolling(window=period).sum()
    neg_sum        = neg_mf.rolling(window=period).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum))

def calc_amihud(close, volume):
    ret   = np.log(close).diff().abs()
    return ret / volume.replace(0, np.nan)

def calc_mec(close, window=63):
    T          = 6
    ret_long   = np.log(close / close.shift(30))
    ret_short  = np.log(close / close.shift(5))
    var_long   = ret_long.rolling(window=window).var()
    var_short  = ret_short.rolling(window=window).var()
    return var_long / (T * var_short)

def calc_corwin_schultz(high, low):
    sqrt2   = np.sqrt(2)
    denom   = 3 - 2 * sqrt2
    log_hl      = np.log(high / low)
    log_hl2     = log_hl ** 2
    log_hl_prev = np.log(high.shift(1) / low.shift(1)) ** 2
    beta  = log_hl2 + log_hl_prev
    h2    = pd.concat([high.shift(1), high], axis=1).max(axis=1)
    l2    = pd.concat([low.shift(1),  low],  axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread

def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
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
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": None, "1wk": None, "1mo": None,
}
INTERVAL_STL_PERIOD = {
    "1m": 390, "2m": 195, "5m": 78, "15m": 26,
    "30m": 13, "1h": 7, "1d": 252, "1wk": 52, "1mo": 12,
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
        default_start = min(max(oldest_date, date(2006, 1, 1)), newest_date)
        start_date = st.date_input("Başlangıç", value=default_start, min_value=oldest_date, max_value=newest_date)
    with col2:
        default_end = max(min(newest_date, date(2026, 1, 1)), oldest_date)
        end_date = st.date_input("Bitiş", value=default_end, min_value=oldest_date, max_value=newest_date)

    if start_date > end_date:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        st.stop()

    mask = (hist_max.index.date >= start_date) & (hist_max.index.date <= end_date)
    df   = hist_max.loc[mask].copy()
    if df.empty:
        st.warning("Seçilen tarih aralığında veri yok.")
        st.stop()

    close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]

    df["EMA_20"]     = calc_ema(close, 20)
    df["EMA_50"]     = calc_ema(close, 50)
    df["EMA_200"]    = calc_ema(close, 200)
    df["RSI"]        = calc_rsi(close)
    df["MACD"]       = calc_macd(close)[0]
    df["ATR"]        = calc_atr(high, low, close)
    df["BB_Upper"], df["BB_Lower"], df["BBW"] = calc_bollinger(close)
    df["Supertrend"] = calc_supertrend(high, low, close)
    df["Return"]     = close.pct_change()
    df["ROC"]        = calc_roc(close)
    df["Stoch_K"], df["Stoch_D"] = calc_stochastic(high, low, close)
    df["ADX"]        = calc_adx(high, low, close)
    df["Williams_R"] = calc_williams_r(high, low, close)
    df["CCI"]        = calc_cci(high, low, close)
    df["OBV"]        = calc_obv(close, volume)
    df["CMF"]        = calc_cmf(high, low, close, volume)
    df["Volume_ROC"] = calc_volume_roc(volume)
    df["MFI"]        = calc_mfi(high, low, close, volume)
    df["StochRSI_K"], df["StochRSI_D"] = calc_stoch_rsi(close)
    df["Amihud"]     = calc_amihud(close, volume)
    df["MEC"]        = calc_mec(close)
    df["CS_Spread"]  = calc_corwin_schultz(high, low)
    df["Daily_Range"] = high - low

    check_cols   = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    zero_or_null = int(((df[check_cols].isnull().any(axis=1)) | (df[check_cols] == 0).any(axis=1)).sum())
    consec_dupes = int((df[check_cols].eq(df[check_cols].shift(1)).all(axis=1)).sum())
    null_only    = int(df[check_cols].isnull().any(axis=1).sum())
    ohlc_same    = int(((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])).sum())

    loss_mask = (
        df[check_cols].isnull().any(axis=1)
        | (df[check_cols] == 0).any(axis=1)
        | df[check_cols].eq(df[check_cols].shift(1)).all(axis=1)
        | ((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))
    )
    n_loss   = int(loss_mask.sum())
    loss_pct = (n_loss / len(df) * 100) if len(df) > 0 else 0.0

    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki {bar_label} sayısı:</b> {len(df):,}<br>
        <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zero_or_null:,}<br>
        <b>Arka arkaya aynı OHLCV satır sayısı:</b> {consec_dupes:,}<br>
        <b>Boş hücresi olan satır sayısı (0 hariç):</b> {null_only:,}<br>
        <b>Open=High=Low=Close olan satır sayısı:</b> {ohlc_same:,}<br>
        <b>Kayıp Veri:</b> {n_loss:,} satır (%{loss_pct:.2f})
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Kayıp veri nasıl hesaplanır?", expanded=False):
        st.markdown("""
Yukarıdaki 4 koşuldan **herhangi birini** sağlayan satırlar sayılır. Birden fazla koşula uyan satır tek sayılır (net kayıp):

1. OHLCV'de boş hücre (NaN) içeren satırlar
2. OHLCV'de sıfır değer içeren satırlar
3. Bir önceki satırla tamamen aynı OHLCV'ye sahip satırlar
4. Open = High = Low = Close olan satırlar

Formül:
`loss_pct = (loss_mask.sum() / len(df)) × 100`

Mantıksal **OR** ile birleştirildiği için aynı satır birden fazla koşulu sağlasa bile yalnızca bir kez sayılır.
        """)

    if zero_or_null > 0 or null_only > 0:
        st.markdown("")
        with st.expander("🔍 Boş ve Sıfır Değerlerin Değişken Dağılımı", expanded=False):
            detail_rows = []
            for col in check_cols:
                n_null = int(df[col].isnull().sum())
                n_zero = int((df[col] == 0).sum())
                if n_null > 0 or n_zero > 0:
                    detail_rows.append({
                        "Değişken": col,
                        "Boş (NaN)": n_null,
                        "Sıfır (0)": n_zero,
                        "Toplam Sorunlu": n_null + n_zero,
                        "Oran (%)": round((n_null + n_zero) / len(df) * 100, 2),
                    })
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Tüm değişkenler temiz.")

    n30        = max(1, int(len(df) * 0.30))
    df_last30  = df.iloc[-n30:]
    zn_last    = int(((df_last30[check_cols].isnull().any(axis=1)) | (df_last30[check_cols] == 0).any(axis=1)).sum())
    cd_last    = int((df_last30[check_cols].eq(df_last30[check_cols].shift(1)).all(axis=1)).sum())
    nl_last    = int(df_last30[check_cols].isnull().any(axis=1).sum())
    os_last    = int(((df_last30["Open"] == df_last30["High"]) & (df_last30["High"] == df_last30["Low"]) & (df_last30["Low"] == df_last30["Close"])).sum())

    with st.expander("📅 Son %30'luk Dilim Özeti", expanded=False):
        st.markdown(f"""
<div class="info-box">
    <b>Seçilen aralıktaki gün sayısı:</b> {n30:,}<br>
    <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zn_last:,}<br>
    <b>Arka arkaya aynı OHLCV satır sayısı:</b> {cd_last:,}<br>
    <b>Boş hücresi olan satır sayısı (0 hariç):</b> {nl_last:,}<br>
    <b>Open=High=Low=Close olan satır sayısı:</b> {os_last:,}
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Makro Varlıklarla Korelasyon
    # ============================================================
    st.subheader("Makro Faktör Duyarlılığı")
    st.caption("Seçilen tarih aralığında, ortak günler üzerinden hesaplanır.")

    MACRO_ASSETS = {
        "Altın (GC=F)":     "GC=F",
        "BITCOIN":       "BTC-USD",
        "EUR/USD (EURUSD=X)":       "EURUSD=X",
        "S&P 500 (^GSPC)":  "^GSPC",
        "Dolar Endeksi (DX-Y.NYB)": "DX-Y.NYB",
        "Brent Petrol (BZ=F)":      "BZ=F",
        "BIST100":       "XU100.IS",
        "Korku Endeksi":       "^VIX",
        "MSCI Dünya":       "^990100-USD-STRD",
        "ABD 10Y Tahvil (^TNX)": "^TNX",
        
        
    }

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_macro_close(sym, start, end, intv):
        try:
            t = yf.Ticker(sym)
            h = t.history(start=start, end=end, interval=intv, actions=False)
            if h.empty:
                return None
            if h.index.tz is not None:
                h.index = h.index.tz_localize(None)
            return h["Close"]
        except Exception:
            return None

    with st.spinner("Makro veriler çekiliyor..."):
        corr_rows = []
        base_close = df["Close"]
        # Bitiş tarihine +1 gün ekle (yfinance end exclusive)
        end_fetch = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()

        for label, sym in MACRO_ASSETS.items():
            macro_close = fetch_macro_close(sym, start_date, end_fetch, interval)
            if macro_close is None or macro_close.empty:
                corr_rows.append({
                    "Varlık": label, "Sembol": sym,
                    "Pearson": np.nan, "Spearman": np.nan,
                    "Ortak Bar": 0,
                })
                continue

            joined = pd.concat([base_close, macro_close], axis=1, join="inner").dropna()
            joined.columns = ["base", "macro"]
            n_common = len(joined)

            if n_common < 5:
                pearson_v = np.nan
                spearman_v = np.nan
            else:
                pearson_v = joined["base"].corr(joined["macro"], method="pearson")
                spearman_v = joined["base"].corr(joined["macro"], method="spearman")

            corr_rows.append({
                "Varlık": label, "Sembol": sym,
                "Pearson": round(pearson_v, 4) if not pd.isna(pearson_v) else np.nan,
                "Spearman": round(spearman_v, 4) if not pd.isna(spearman_v) else np.nan,
                "Ortak Bar": n_common,
            })

    corr_df = pd.DataFrame(corr_rows)

    def _corr_color(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return "color: #888888"
        if val >= 0.7:   return "background-color:#d1e7dd; color:#0a3622; font-weight:600"
        if val >= 0.3:   return "background-color:#fff3cd; color:#664d03"
        if val <= -0.7:  return "background-color:#f8d7da; color:#842029; font-weight:600"
        if val <= -0.3:  return "background-color:#ffe5d0; color:#8a3a00"
        return ""

    st.dataframe(
        corr_df.style
            .format({"Pearson": "{:.4f}", "Spearman": "{:.4f}"}, na_rep="—")
            .map(_corr_color, subset=["Pearson", "Spearman"]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Pearson** \"X %1 artarsa Y % kaç artar\" sorusunu, "
        "**Spearman** \"X artarsa Y de artıyor mu\" sorusunu cevaplar. "
        "Pearson büyüklüğe, Spearman yöne bakar. "
        "|ρ| ≥ 0.7 güçlü, 0.3–0.7 orta, < 0.3 zayıf kabul edilir."
    )

    # ============================================================
    # Veri Seçimi
    # ============================================================

    st.subheader("Veri Seçimi")
    st.caption("İndirmek istediğiniz verileri seçin:")

    CATEGORIES = {
        "📊 Ham Veri":    (["Open", "High", "Low", "Close", "Volume", "Return"], "borsadan gelen ham fiyat verisi ve günlük getiri"),
        "📈 Trend":       (["EMA_20", "EMA_50", "EMA_200", "MACD", "Supertrend", "ADX"], "fiyatın hangi yönde gittiğini ve trendin ne kadar güçlü olduğunu gösterir"),
        "⚡ Momentum":    (["RSI", "ROC", "CCI", "Williams_R", "Stoch_K", "Stoch_D", "StochRSI_K", "StochRSI_D"], "fiyat hareketinin hızını ve gücünü ölçer, aşırı alım/satım bölgelerini gösterir"),
        "🌊 Volatilite":  (["ATR", "BB_Upper", "BB_Lower", "BBW"],            "fiyatın ne kadar sert ve geniş hareket ettiğini ölçer"),
        "📦 Hacim":       (["OBV", "CMF", "MFI", "Volume_ROC"],               "alım-satım hacminin yönünü, gücünü ve para akışını gösterir"),
        "💧 Likidite":    (["Amihud", "MEC", "CS_Spread", "Daily_Range", "Volume"], "piyasanın ne kadar derin ve verimli işlem gördüğünü ölçer"),
    }

    LIQUIDITY_DIMS = {
        "CS_Spread":   "Sıkılık",
        "Daily_Range": "Anlıklık",
        "Volume":      "Derinlik",
        "Amihud":      "Genişlik",
        "MEC":         "Esneklik",
    }

    selected_cols = []
    available_set = set(df.columns)

    for cat_label, (cat_cols, cat_desc) in CATEGORIES.items():
        existing = [c for c in cat_cols if c in available_set]
        if not existing:
            continue
        is_ham_veri  = cat_label == "📊 Ham Veri"
        is_likidite  = cat_label == "💧 Likidite"
        group_active = st.checkbox(
            f"**{cat_label}** *({cat_desc})*",
            value=is_ham_veri,
            key=f"grp_{cat_label}",
        )
        if group_active:
            cb_cols = st.columns(4)
            for i, col_name in enumerate(existing):
                with cb_cols[i % 4]:
                    dim          = LIQUIDITY_DIMS.get(col_name) if is_likidite else None
                    label        = f"{col_name} — {dim}" if dim else col_name
                    item_default = col_name != "Return"
                    if st.checkbox(label, value=item_default, key=f"cb_{cat_label}_{col_name}"):
                        selected_cols.append(col_name)

    categorized = {c for cols, _ in CATEGORIES.values() for c in cols}
    other_cols  = [c for c in df.columns if c not in categorized]
    if other_cols:
        st.markdown("**📎 Diğer**")
        cb_cols = st.columns(4)
        for i, col_name in enumerate(other_cols):
            with cb_cols[i % 4]:
                if st.checkbox(col_name, value=True, key=f"cb_other_{col_name}"):
                    selected_cols.append(col_name)

    selected_cols = list(dict.fromkeys(selected_cols))

    if not selected_cols:
        st.info("En az bir sütun seçmelisiniz.")
        st.stop()

    # ============================================================
    # SEKMELER
    # ============================================================

    clean_selected = []
    df_clean2      = pd.DataFrame()
    dl_df          = pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["📥 Veri İndir", "🔬 Feature Analizi", "🔍 Örüntü Analizi"])

    with tab1:

        # ── Kapanış Grafiği ──────────────────────────────────────
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

        # ── STL Ayrışım Grafiği ───────────────────────────────────
        st.subheader("STL — Seasonal-Trend Decomposition using Loess")

        # ACF ile otomatik periyot tahmini
        @st.cache_data
        def acf_period_estimate(values, max_lag, fallback):
            try:
                from statsmodels.tsa.stattools import acf
                n_lags = min(max_lag, len(values) // 2 - 1)
                if n_lags < 4:
                    return fallback, None
                acf_vals = acf(values, nlags=n_lags, fft=True)
                # lag=0 hariç ilk belirgin tepe: komşularından büyük olan laglar
                peaks = [
                    i for i in range(2, len(acf_vals) - 1)
                    if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0.05
                ]
                if peaks:
                    return int(peaks[0]), acf_vals
                return fallback, acf_vals
            except Exception:
                return fallback, None

        fallback_period  = INTERVAL_STL_PERIOD.get(interval, 12)
        max_acf_lag      = min(fallback_period * 3, len(close.dropna()) // 2 - 1, 600)
        acf_period, _    = acf_period_estimate(np.log(close.dropna()).values, max_acf_lag, fallback_period)

        if acf_period != fallback_period:
            st.success(f"📐 ACF ile tahmin edilen periyot: **{acf_period}** (akademik varsayılan: {fallback_period})")
        else:
            st.error(f"📐 ACF belirgin tepe bulamadı, akademik varsayılan kullanılıyor: **{fallback_period}**")

        stl_period = st.number_input(
            "STL Periyodu",
            min_value=2, max_value=1000,
            value=acf_period, step=1,
            help=(
                "Akademik varsayılanlar: "
                "1d → 252 (işlem günü/yıl), "
                "1wk → 52, "
                "1mo → 12, "
                "1h → 7 (günlük döngü), "
                "30m → 13, "
                "15m → 26, "
                "5m → 78, "
                "2m → 195, "
                "1m → 390"
            ),
        )
        close_clean = close.dropna()
        if len(close_clean) > stl_period * 2:
            try:
                log_close = np.log(close_clean)
                stl_res = STL(log_close, period=stl_period, robust=True).fit()

                panels = [
                    ("Observed", close_clean.values, "#bfdbfe", "#1d4ed8", "line"),
                    ("Trend",         stl_res.trend,       "#bbf7d0", "#15803d", "line"),
                    ("Seasonal",     stl_res.seasonal,    "#fed7aa", "#c2410c", "line"),
                    ("Residual",         stl_res.resid,       "#e9d5ff", "#7e22ce", "bar"),
                ]

                fig_stl = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.25, 0.25, 0.25, 0.25],
                )

                # Renkli arka plan şeritleri (paper koordinatları)
                bg_y = [1.0, 0.72, 0.49, 0.26]   # her panel üst kenarı (yaklaşık)
                bg_h = [0.28, 0.23, 0.23, 0.26]

                shapes = []
                for idx, (_, _, bg_color, _, _) in enumerate(panels):
                    shapes.append(dict(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=0, x1=1,
                        y0=bg_y[idx] - bg_h[idx],
                        y1=bg_y[idx],
                        fillcolor=bg_color,
                        opacity=0.4,
                        line_width=0,
                        layer="below",
                    ))

                for i, (label, values, bg_color, line_color, chart_type) in enumerate(panels, start=1):
                    if chart_type == "bar":
                        fig_stl.add_trace(
                            go.Bar(
                                x=close_clean.index, y=values,
                                marker_color=line_color,
                                marker_opacity=0.7,
                                name=label,
                            ),
                            row=i, col=1,
                        )
                    else:
                        fig_stl.add_trace(
                            go.Scatter(
                                x=close_clean.index, y=values,
                                mode="lines",
                                line=dict(color=line_color, width=1.2),
                                name=label,
                            ),
                            row=i, col=1,
                        )

                    # Panel başlığını sol üst köşeye annotation olarak ekle
                    fig_stl.add_annotation(
                        text=f"<b>{label}</b>",
                        xref="paper", yref="paper",
                        x=0.01,
                        y=bg_y[i - 1] - 0.01,
                        showarrow=False,
                        font=dict(size=12, color=line_color),
                        xanchor="left", yanchor="top",
                    )

                fig_stl.update_layout(
                    height=720,
                    width=720,
                    showlegend=False,
                    margin=dict(l=60, r=30, t=20, b=40),
                    hovermode="x unified",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    bargap=0,
                )
                fig_stl.update_xaxes(showgrid=False)
                fig_stl.update_yaxes(showgrid=True, gridcolor="#e5e7eb", gridwidth=0.5)

                st.plotly_chart(fig_stl, use_container_width=True, config={"scrollZoom": True})

                # ── MEVSİMSELLİK YORUMU ───────────────────────────
                with st.expander("🌊 Mevsimsellik Analizi", expanded=False):
                    seasonal_s = pd.Series(stl_res.seasonal, index=close_clean.index)
                    amp        = seasonal_s.max() - seasonal_s.min()
                    amp_pct    = (np.exp(seasonal_s.abs().mean()) - 1) * 100
                    s_max_date = seasonal_s.idxmax()
                    s_min_date = seasonal_s.idxmin()
                    roll_std   = seasonal_s.rolling(window=stl_period).std()

                    st.markdown("**📋 Genel Özet**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Salınım Genişliği (log)", f"{amp:.4f}")
                    c2.metric("Ort. Fiyat Etkisi (yaklaşık)", f"%{amp_pct:.1f}")
                    c3.metric("Periyot", f"{stl_period} bar")

                    st.markdown(f"""
- **En güçlü pozitif mevsimsel etki:** {s_max_date.strftime("%Y-%m-%d")} → log değer: `{seasonal_s.max():.4f}`
- **En güçlü negatif mevsimsel etki:** {s_min_date.strftime("%Y-%m-%d")} → log değer: `{seasonal_s.min():.4f}`
- **Ortalama mutlak mevsimsel etki:** `{seasonal_s.abs().mean():.4f}` (log birim)
- **Yorum:** Ortalama mevsimsel fiyat etkisi yaklaşık **%{amp_pct:.1f}**. {"Bu yüksek bir oran — döngüsel alım/satım stratejileri için kullanılabilir." if amp_pct > 5 else "Bu düşük bir oran — mevsimsellik bu varlık için belirleyici değil."}
                    """)

                    st.markdown("**📊 Periyot Dilimine Göre Ortalama Mevsimsel Etki**")
                    if interval in ("1d", "1wk"):
                        group_label = "Ay"
                        group_key   = seasonal_s.index.month
                        tick_labels = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
                    elif interval in ("1h", "30m", "15m", "5m", "2m", "1m"):
                        group_label = "Haftanın Günü"
                        group_key   = seasonal_s.index.dayofweek
                        tick_labels = ["Pzt","Sal","Çar","Per","Cum"]
                    else:
                        group_label = "Çeyrek"
                        group_key   = seasonal_s.index.quarter
                        tick_labels = ["Q1","Q2","Q3","Q4"]

                    group_mean  = seasonal_s.groupby(group_key).mean()
                    bar_colors  = ["#15803d" if v >= 0 else "#dc2626" for v in group_mean.values]
                    fig_seas_bar = go.Figure(go.Bar(
                        x=list(range(len(group_mean))), y=group_mean.values,
                        marker_color=bar_colors,
                        text=[f"{v:.4f}" for v in group_mean.values],
                        textposition="outside",
                    ))
                    fig_seas_bar.add_hline(y=0, line_color="black", line_width=0.8)
                    fig_seas_bar.update_layout(
                        height=300,
                        xaxis=dict(tickvals=list(range(len(group_mean))), ticktext=tick_labels[:len(group_mean)], title=group_label),
                        yaxis=dict(title="Ort. Mevsimsel Etki (log)"),
                        margin=dict(l=40, r=20, t=20, b=40),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_seas_bar, use_container_width=True)

                    st.markdown("**📈 Mevsimsel Gücün Zaman İçindeki Değişimi** *(rolling std)*")
                    fig_roll = go.Figure(go.Scatter(
                        x=roll_std.index, y=roll_std.values,
                        mode="lines", line=dict(color="#c2410c", width=1.2),
                    ))
                    fig_roll.update_layout(
                        height=220, margin=dict(l=40, r=20, t=10, b=30),
                        yaxis_title="Std (log)", plot_bgcolor="white", hovermode="x unified",
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)
                    _rd = roll_std.dropna()
                    trend_dir = "güçleniyor 📈" if _rd.iloc[-1] > _rd.iloc[len(_rd)//2] else "zayıflıyor 📉"
                    st.caption(f"Mevsimsel etki son dönemde **{trend_dir}**.")

                # ── ARTIK YORUMU ──────────────────────────────────
                with st.expander("⚡ Artık (Residual) Analizi", expanded=False):
                    resid_s    = pd.Series(stl_res.resid, index=close_clean.index)
                    r_mean     = resid_s.mean()

                    # Rolling MAD: her dönem kendi yerel bazeline göre normalize edilir
                    roll_win   = min(126, len(resid_s) // 4)
                    roll_mad   = (resid_s - resid_s.rolling(roll_win, center=True, min_periods=roll_win//2).median())                                      .abs()                                      .rolling(roll_win, center=True, min_periods=roll_win//2).median()
                    roll_std   = (roll_mad * 1.4826).fillna(roll_mad.median() * 1.4826)

                    # z-score: her gün kendi yerel std ile normalize
                    z_score    = resid_s / roll_std.replace(0, np.nan).ffill().bfill()
                    sigma2     = z_score[z_score.abs() > 2]
                    sigma3     = z_score[z_score.abs() > 3]
                    pos_shocks = int((z_score > 2).sum())
                    neg_shocks = int((z_score < -2).sum())

                    # Global robust std (gösterim için)
                    g_mad  = (resid_s - resid_s.median()).abs().median()
                    r_std_global = g_mad * 1.4826

                    st.markdown("**📋 Genel Özet**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Ortalama", f"{r_mean:.5f}")
                    c2.metric("Medyan |Artık|", f"{resid_s.abs().median():.4f}")
                    c3.metric("±2σ Şok Sayısı", len(sigma2))
                    c4.metric("±3σ Şok Sayısı", len(sigma3))

                    st.markdown(f"""
- **Pozitif şok (2σ üzeri):** {pos_shocks} adet — beklenmedik sert yükseliş
- **Negatif şok (2σ altı):** {neg_shocks} adet — beklenmedik sert düşüş
- **Yorum:** {"Artık bileşen büyük ölçüde sıfır etrafında — model seriyi iyi açıklıyor." if (len(sigma2) / len(resid_s)) < 0.05 else "Artıkta belirgin aşırı değerler var — açıklanamayan şoklar mevcut, anomali tespiti için kullanılabilir."}
- **Not:** Sigma bantları **rolling MAD** ile hesaplanmıştır — her dönem kendi yerel bazeline göre normalize edilir, yapısal kırılmalar diğer şokları bastırmaz.
                    """)

                    st.markdown("**📊 Artık Serisi — Yerel ±2σ ve ±3σ Bantları**")
                    upper2 = roll_std * 2
                    lower2 = -roll_std * 2
                    upper3 = roll_std * 3
                    lower3 = -roll_std * 3

                    fig_resid = go.Figure()
                    # Bantlar çizgi olarak (dinamik, rolling)
                    fig_resid.add_trace(go.Scatter(x=resid_s.index, y=upper2.values, mode="lines",
                        line=dict(color="#16a34a", dash="dash", width=0.8), name="+2σ"))
                    fig_resid.add_trace(go.Scatter(x=resid_s.index, y=lower2.values, mode="lines",
                        line=dict(color="#16a34a", dash="dash", width=0.8), name="-2σ"))
                    fig_resid.add_trace(go.Scatter(x=resid_s.index, y=upper3.values, mode="lines",
                        line=dict(color="#dc2626", dash="dash", width=0.8), name="+3σ"))
                    fig_resid.add_trace(go.Scatter(x=resid_s.index, y=lower3.values, mode="lines",
                        line=dict(color="#dc2626", dash="dash", width=0.8), name="-3σ"))
                    # Artık barlar: yerel z-score'a göre renklendirme
                    bar_colors = ["#dc2626" if z > 2 else "#15803d" if z < -2 else "#7e22ce"
                                  for z in z_score.values]
                    fig_resid.add_trace(go.Bar(
                        x=resid_s.index, y=resid_s.values,
                        marker_color=bar_colors, marker_opacity=0.7, name="Residual",
                    ))
                    fig_resid.add_hline(y=0, line_color="black", line_width=0.5)
                    fig_resid.update_layout(
                        height=320, bargap=0,
                        margin=dict(l=40, r=20, t=10, b=30),
                        plot_bgcolor="white", hovermode="x unified",
                        legend=dict(orientation="h", y=1.08),
                    )
                    st.plotly_chart(fig_resid, use_container_width=True)

                    st.markdown("**🔴 Yıl Bazında En Büyük Şoklar (yerel z-score)**")
                    # Her yıldan en büyük 1 şok — temporal çeşitlilik garantili
                    z_abs = z_score.abs()
                    yearly_top = z_abs.groupby(z_abs.index.year).idxmax()
                    yearly_top = yearly_top.dropna()
                    shock_vals = resid_s[yearly_top]
                    shock_z    = z_score[yearly_top]
                    shock_df = pd.DataFrame({
                        "Yıl":         yearly_top.index,
                        "Tarih":       pd.DatetimeIndex(yearly_top.values).strftime("%Y-%m-%d"),
                        "Residual":       shock_vals.values.round(5),
                        "Yön":         ["🔴 Negatif Şok" if v < 0 else "🟢 Pozitif Şok" for v in shock_vals.values],
                        "Yerel Z":     shock_z.values.round(2),
                    }).sort_values("Yıl", ascending=False).reset_index(drop=True)
                    st.dataframe(shock_df, use_container_width=True, hide_index=True)

                # ── DECOMPOSİTİON TEYİDİ ─────────────────────────
                st.subheader("🔎 Decomposition Teyidi")

                # --- 1. Kruskal-Wallis ---
                with st.expander("1️⃣ Kruskal-Wallis Testi — Mevsimsellik İstatistiksel Olarak Anlamlı mı?", expanded=False):
                    from scipy.stats import kruskal
                    try:
                        seasonal_s2 = pd.Series(stl_res.seasonal, index=close_clean.index)
                        if interval in ("1d", "1wk"):
                            groups = [seasonal_s2[seasonal_s2.index.month == m].values for m in range(1, 13)]
                            group_names = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
                            group_label = "ay"
                        elif interval in ("1h", "30m", "15m", "5m", "2m", "1m"):
                            groups = [seasonal_s2[seasonal_s2.index.dayofweek == d].values for d in range(5)]
                            group_names = ["Pzt","Sal","Çar","Per","Cum"]
                            group_label = "gün"
                        else:
                            groups = [seasonal_s2[seasonal_s2.index.quarter == q].values for q in range(1, 5)]
                            group_names = ["Q1","Q2","Q3","Q4"]
                            group_label = "çeyrek"

                        groups = [g for g in groups if len(g) > 1]
                        stat, pval = kruskal(*groups)

                        if pval < 0.01:
                            yorum = f"✅ **Güçlü istatistiksel kanıt** (p={pval:.4f} < 0.01) — {group_label} grupları arasında anlamlı fark var. Mevsimsellik tesadüf değil."
                        elif pval < 0.05:
                            yorum = f"⚠️ **Zayıf istatistiksel kanıt** (p={pval:.4f}, 0.01–0.05 arası) — mevsimsellik muhtemelen gerçek ama güven sınırda."
                        else:
                            yorum = f"❌ **İstatistiksel kanıt yok** (p={pval:.4f} > 0.05) — {group_label} grupları arasında anlamlı fark bulunamadı. Mevsimsel kalıp tesadüf olabilir."

                        # En güçlü ve zayıf ay/gün
                        group_means = {group_names[i]: groups[i].mean() for i in range(len(groups))}
                        en_guclu = max(group_means, key=group_means.get)
                        en_zayif = min(group_means, key=group_means.get)

                        st.markdown(f"""
**Kruskal-Wallis H İstatistiği:** `{stat:.4f}`
**p-değeri:** `{pval:.6f}`

{yorum}

**En güçlü mevsimsel {group_label}:** {en_guclu} (ort. `{group_means[en_guclu]:.4f}`)
**En zayıf mevsimsel {group_label}:** {en_zayif} (ort. `{group_means[en_zayif]:.4f}`)

> Kruskal-Wallis parametrik olmayan bir testtir — normal dağılım varsayımı gerektirmez, finansal veri için uygundur.
                        """)
                    except Exception as e:
                        st.warning(f"Kruskal-Wallis hesaplanamadı: {e}")

                # --- 2. Çoklu Sembol Karşılaştırma ---
                with st.expander("2️⃣ Çoklu Sembol Karşılaştırma — Şoklar Sistematik mi, Spesifik mi?", expanded=False):
                    comp_symbols_raw = st.text_input(
                        "Karşılaştırılacak semboller (virgülle ayır)",
                        placeholder="Örn: PGSUS.IS, XU100.IS, USD/TRY",
                        key="comp_symbols_input"
                    )
                    if st.button("▶ Karşılaştır", key="comp_run"):
                        comp_symbols = [s.strip().upper() for s in comp_symbols_raw.split(",") if s.strip()]
                        if not comp_symbols:
                            st.warning("En az bir sembol girin.")
                        else:
                            # Ana sembolün artık şok tarihleri (|z| > 2)
                            shock_dates = set(z_score[z_score.abs() > 2].index.normalize())

                            results_text = []
                            for csym in comp_symbols:
                                try:
                                    cticker = yf.Ticker(csym)
                                    if is_intraday:
                                        chist = cticker.history(period=f"{max_days}d", interval=interval, actions=False)
                                    else:
                                        chist = cticker.history(period="max", interval=interval, actions=False)
                                    if chist.index.tz is not None:
                                        chist.index = chist.index.tz_localize(None)
                                    cmask = (chist.index.date >= start_date) & (chist.index.date <= end_date)
                                    cclose = chist.loc[cmask, "Close"].dropna()
                                    if len(cclose) < stl_period * 2:
                                        results_text.append(f"**{csym}:** Yetersiz veri ({len(cclose)} bar)")
                                        continue
                                    clog = np.log(cclose)
                                    cstl = STL(clog, period=stl_period, robust=True).fit()
                                    cresid = pd.Series(cstl.resid, index=cclose.index)
                                    croll_mad = (cresid - cresid.rolling(126, center=True, min_periods=63).median()).abs().rolling(126, center=True, min_periods=63).median()
                                    croll_std = (croll_mad * 1.4826).fillna(croll_mad.median() * 1.4826)
                                    cz = cresid / croll_std.replace(0, np.nan).ffill().bfill()
                                    cshock_dates = set(cz[cz.abs() > 2].index.normalize())

                                    overlap = shock_dates & cshock_dates
                                    if len(shock_dates) > 0:
                                        overlap_pct = len(overlap) / len(shock_dates) * 100
                                    else:
                                        overlap_pct = 0

                                    if overlap_pct > 50:
                                        yorum_c = "🔴 **Yüksek örtüşme** — şoklar büyük ölçüde sistematik (piyasa geneli olay)"
                                    elif overlap_pct > 25:
                                        yorum_c = "🟡 **Orta örtüşme** — hem sistematik hem varlığa özgü etkenler var"
                                    else:
                                        yorum_c = "🟢 **Düşük örtüşme** — şoklar büyük ölçüde bu varlığa özgü"

                                    results_text.append(f"""
**{csym}** — Örtüşme: `%{overlap_pct:.1f}` ({len(overlap)} ortak şok / {len(shock_dates)} toplam şok)
{yorum_c}
""")
                                except Exception as e:
                                    results_text.append(f"**{csym}:** Hata — {e}")

                            for r in results_text:
                                st.markdown(r)
                                st.divider()

                # --- 3. Farklı Periyot Testi ---
                with st.expander("3️⃣ Farklı Periyot Testi — Mevsimsel Kalıp Tutarlı mı?", expanded=False):
                    if st.button("▶ Periyot Testi Çalıştır", key="period_test_run"):
                        test_periods = [p for p in [63, 126, 252, 504] if len(close_clean) > p * 2]
                        if len(test_periods) < 2:
                            st.warning("Yeterli veri yok — en az 2 periyot test edilebilmeli.")
                        else:
                            period_results = []
                            for tp in test_periods:
                                try:
                                    tp_stl = STL(np.log(close_clean), period=tp, robust=True).fit()
                                    tp_seas = pd.Series(tp_stl.seasonal, index=close_clean.index)
                                    tp_amp  = tp_seas.max() - tp_seas.min()
                                    tp_pct  = (np.exp(tp_seas.abs().mean()) - 1) * 100

                                    if interval in ("1d", "1wk"):
                                        tp_group = tp_seas.groupby(tp_seas.index.month).mean()
                                        tp_peak  = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"][tp_group.idxmax() - 1]
                                        tp_trough = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"][tp_group.idxmin() - 1]
                                    else:
                                        tp_peak  = str(tp_seas.idxmax().date())
                                        tp_trough = str(tp_seas.idxmin().date())

                                    period_results.append({
                                        "periyot": tp,
                                        "amp": tp_amp,
                                        "pct": tp_pct,
                                        "peak": tp_peak,
                                        "trough": tp_trough,
                                    })
                                except Exception:
                                    pass

                            if len(period_results) < 2:
                                st.warning("Yeterli sonuç üretilemedi.")
                            else:
                                # Tutarlılık kontrolü
                                peaks   = [r["peak"] for r in period_results]
                                troughs = [r["trough"] for r in period_results]
                                peak_consistent   = len(set(peaks)) == 1
                                trough_consistent = len(set(troughs)) == 1
                                amps = [r["amp"] for r in period_results]
                                amp_cv = np.std(amps) / np.mean(amps) if np.mean(amps) > 0 else 0

                                lines = []
                                for r in period_results:
                                    lines.append(
                                        f"- **Periyot {r['periyot']}:** Amplitüd=`{r['amp']:.4f}`, "
                                        f"Ort. Etki≈`%{r['pct']:.1f}`, "
                                        f"Zirve={r['peak']}, Dip={r['trough']}"
                                    )
                                st.markdown("\n".join(lines))
                                st.markdown("---")

                                # Çoğunluk bazlı tutarlılık — tek aykırı periyot sonucu bozmaz
                                from collections import Counter
                                peak_counts   = Counter(peaks)
                                trough_counts = Counter(troughs)
                                peak_majority   = peak_counts.most_common(1)[0][1]    # en çok tekrar eden zirvenin sayısı
                                trough_majority = trough_counts.most_common(1)[0][1]
                                n_periods = len(period_results)
                                dominant_peak   = peak_counts.most_common(1)[0][0]
                                dominant_trough = trough_counts.most_common(1)[0][0]

                                if peak_majority >= n_periods * 0.75 and trough_majority >= n_periods * 0.75:
                                    tutarlilik = f"✅ **Yüksek tutarlılık** — periyotların büyük çoğunluğu aynı zirve ({dominant_peak}) ve dip ({dominant_trough}) dönemini gösteriyor. Mevsimsel kalıp kararlı."
                                elif peak_majority >= n_periods * 0.5 or trough_majority >= n_periods * 0.5:
                                    tutarlilik = f"⚠️ **Kısmi tutarlılık** — baskın kalıp: zirve={dominant_peak}, dip={dominant_trough}. Uzun periyotlar (252+) örtüşüyorsa bu kalıba güvenilebilir; kısa periyotlar (63) gürültüye duyarlıdır."
                                else:
                                    tutarlilik = "❌ **Düşük tutarlılık** — hiçbir periyotta ortak kalıp oluşmuyor. STL bu veri için mevsimselliği tutarlı bulamıyor."

                                # Kısa periyot uyarısı
                                if 63 in [r["periyot"] for r in period_results]:
                                    tutarlilik += "\n\n> ⚠️ **Not:** 63 bar periyodu gürültüye çok duyarlıdır — kısa dönem dalgalanmaları mevsimsel kalıp olarak algılanabilir. 252+ periyot sonuçlarına daha fazla ağırlık verin."

                                if amp_cv < 0.2:
                                    amp_yorum = f"Amplitüd varyasyonu düşük (CV=`{amp_cv:.2f}`) — güçlü sinyal."
                                elif amp_cv < 0.5:
                                    amp_yorum = f"Amplitüd varyasyonu orta (CV=`{amp_cv:.2f}`) — periyot seçimi sonucu etkiliyor."
                                else:
                                    amp_yorum = f"Amplitüd varyasyonu yüksek (CV=`{amp_cv:.2f}`) — periyot seçimine çok duyarlı, dikkatli yorumla."

                                st.markdown(f"{tutarlilik}\n\n{amp_yorum}")

            except Exception as e:
                st.warning(f"STL hesaplanamadı: {e}")
        else:
            st.info(f"STL için en az {stl_period * 2} satır gerekli, mevcut: {len(close_clean)}")

        # ── Excel İndir ───────────────────────────────────────────
        st.subheader("İndir - Seçili Veriler")
        export_df = df[selected_cols].copy()
        export_df.index.name = "Datetime" if is_intraday else "Date"
        export_df = export_df.reset_index()
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Data")
        excel_buf.seek(0)
        file_name = f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}.xlsx"
        st.download_button(
            label=f"📥 Ham Veriler — {symbol.upper()} ({len(export_df):,} satır)",
            data=excel_buf.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ── Makro Faktör Verileri Excel İndir ─────────────────────
        macro_frames = {}
        for label, sym in MACRO_ASSETS.items():
            m_close = fetch_macro_close(sym, start_date, end_fetch, interval)
            if m_close is not None and not m_close.empty:
                macro_frames[label] = m_close

        if macro_frames:
            macro_df = pd.concat(macro_frames, axis=1)
            macro_df.index.name = "Datetime" if is_intraday else "Date"
            macro_df = macro_df.reset_index()

            excel_macro = BytesIO()
            with pd.ExcelWriter(excel_macro, engine="openpyxl") as writer:
                macro_df.to_excel(writer, index=False, sheet_name="Macro")
            excel_macro.seek(0)
            macro_file = f"makro_faktorler_{interval}_{start_date}_{end_date}.xlsx"
            st.download_button(
                label=f"📥 Makro Faktör Verileri ({len(macro_df):,} satır × {len(macro_frames)} varlık)",
                data=excel_macro.getvalue(),
                file_name=macro_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        ohlc_mask   = ~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))
        df_clean    = df[ohlc_mask][["Open", "High", "Low", "Close", "Volume"]].copy()
        removed_cnt = len(df) - len(df_clean)

        if not df_clean.empty:
            _c = df_clean["Close"]; _h = df_clean["High"]; _l = df_clean["Low"]; _v = df_clean["Volume"]

            df_clean["EMA_20"]     = calc_ema(_c, 20)
            df_clean["EMA_50"]     = calc_ema(_c, 50)
            df_clean["EMA_200"]    = calc_ema(_c, 200)
            df_clean["RSI"]        = calc_rsi(_c)
            df_clean["MACD"]       = calc_macd(_c)[0]
            df_clean["ATR"]        = calc_atr(_h, _l, _c)
            df_clean["BB_Upper"], df_clean["BB_Lower"], df_clean["BBW"] = calc_bollinger(_c)
            df_clean["Supertrend"] = calc_supertrend(_h, _l, _c)
            df_clean["Return"]     = _c.pct_change()
            df_clean["ROC"]        = calc_roc(_c)
            df_clean["Stoch_K"], df_clean["Stoch_D"] = calc_stochastic(_h, _l, _c)
            df_clean["ADX"]        = calc_adx(_h, _l, _c)
            df_clean["Williams_R"] = calc_williams_r(_h, _l, _c)
            df_clean["CCI"]        = calc_cci(_h, _l, _c)
            df_clean["OBV"]        = calc_obv(_c, _v)
            df_clean["CMF"]        = calc_cmf(_h, _l, _c, _v)
            df_clean["Volume_ROC"] = calc_volume_roc(_v)
            df_clean["MFI"]        = calc_mfi(_h, _l, _c, _v)
            df_clean["StochRSI_K"], df_clean["StochRSI_D"] = calc_stoch_rsi(_c)
            df_clean["Amihud"]     = calc_amihud(_c, _v)
            df_clean["MEC"]        = calc_mec(_c)
            df_clean["CS_Spread"]  = calc_corwin_schultz(_h, _l)
            df_clean["Daily_Range"] = _h - _l

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
                label=f"📥 OHLC Eşit Satırlar Çıkarılmış ({len(export_clean):,} satır, {removed_cnt:,} satır silindi)",
                data=excel_clean.getvalue(),
                file_name=file_name_clean,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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
                label=f"📥 OHLC Eşit + Boş Hücreli Satırlar Çıkarılmış ({len(export_clean2):,} satır, {removed_nan:,} satır daha silindi)",
                data=excel_clean2.getvalue(),
                file_name=file_name_clean2,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            dl_df   = df_clean2[clean_selected].copy()
            epsilon = 1e-10

            if "OBV" in dl_df.columns:
                obv_diff = dl_df["OBV"].diff()
                dl_df["OBV"] = np.log1p(obv_diff.abs()) * np.sign(obv_diff)

            if "Amihud" in dl_df.columns:
                dl_df["Amihud"] = dl_df["Amihud"].replace(0, epsilon)
                dl_df["Amihud"] = np.log1p(dl_df["Amihud"] * 1e9)

            if "Volume" in dl_df.columns:
                dl_df["Volume"] = np.log1p(dl_df["Volume"])

            if "Volume_ROC" in dl_df.columns:
                dl_df["Volume_ROC"] = np.log1p(dl_df["Volume_ROC"].abs()) * np.sign(dl_df["Volume_ROC"])

            if "CMF" in dl_df.columns:
                dl_df["CMF"] = dl_df["CMF"].where(dl_df["CMF"] > -0.9999, np.nan).ffill()

            if "CS_Spread" in dl_df.columns:
                dl_df["CS_Spread"] = dl_df["CS_Spread"].replace(0, np.nan).ffill()

            dl_df = dl_df.replace([np.inf, -np.inf], np.nan).dropna()

            export_dl = dl_df.copy()
            export_dl.index.name = "Datetime" if is_intraday else "Date"
            export_dl = export_dl.reset_index()

            buf_dl = BytesIO()
            with pd.ExcelWriter(buf_dl, engine="openpyxl") as writer:
                export_dl.to_excel(writer, index=False, sheet_name="Data")
            buf_dl.seek(0)

            st.download_button(
                label=f"📥 Temizlenmiş ve Dönüştürülmüş Veri Seti — {len(export_dl.columns)-1} sütun, {len(export_dl):,} satır",
                data=buf_dl.getvalue(),
                file_name=f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}_temizlenmis_ve_transforme_edilmis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            with st.expander("📋 Standart Dönüşüm Uygulamaları"):
                st.markdown("""
| Değişken | Uygulanan Dönüşüm | Gerekçe |
|---|---|---|
| **OBV** | `diff()` → `log1p(abs) × sign` | Kümülatif serinin farkı alınır; büyük değerler log ile sıkıştırılır, yön korunur |
| **Amihud** | `replace(0, ε=1e-10)` → `log1p(x × 1e9)` | 1e-8 mertebesindeki çok küçük sayılar pozitif bölgeye taşınır, log1p ile ölçeklenir |
| **Volume** | `log1p(x)` | Hacim dağılımı sağa çarpık; log dönüşümü ölçeği dengeler |
| **Volume_ROC** | `log1p(abs) × sign` | Yüzde değişim serisi çok büyük değerler alabilir; yön korunarak sıkıştırılır |
| **CMF** | `–0.9999` sınırındaki değerler `NaN` → `ffill` | –1 sınırında sıkışan uç değerler ileri doldurma ile giderilir |
| **CS_Spread** | `0` → `NaN` → `ffill` | Sıfır spread değerleri (hesaplanamayan günler) ileri doldurma ile giderilir |
| **Diğer tüm değişkenler** | Ham değer (dönüşüm yok) | Zaten uygun ölçekte; MinMax scaling öncesi ek işlem gerektirmez |
| **Sonsuz / NaN satırlar** | `replace(±inf, NaN)` → `dropna()` | Hesaplama kaynaklı bozuk satırlar tamamen çıkarılır |

> **Not:** Bu adımlar MinMax ölçekleme öncesinde uygulanır. Sıkı klipleme (winsorization) kullanılmaz; değer aralığı korunarak sıkıştırılır.
                """)

    with tab2:

        if not clean_selected:
            st.info("Önce **Veri İndir** sekmesinde veri yükleyin.")
        else:
            st.caption(
                "Seçilen veri setleri üzerinde Spearman Korelasyon → VIF → ADF → Ljung-Box → ARCH analizleri çalıştırılır. "
                "Sonuçlar hem ayrı ayrı hem karşılaştırmalı gösterilir."
            )

            target        = "Close"
            date_col_name = "Datetime" if is_intraday else "Date"

            DATASETS = {
                "1 — Ham Veri":              df[[c for c in clean_selected if c in df.columns]].dropna(),
                "2 — OHLC Temizlenmiş":      df_clean[clean_selected].dropna(),
                "3 — Tam Temizlenmiş":       df_clean2[clean_selected],
                "4 — Temizlenmiş ve Transforme Edilmiş": dl_df,
            }

            numeric_cols  = [c for c in clean_selected if pd.api.types.is_numeric_dtype(df[c])]
            default_idx   = numeric_cols.index("Close") if "Close" in numeric_cols else 0
            target        = st.selectbox("🎯 Hedef Değişken (Target)", numeric_cols, index=default_idx, key="fs_target")

            dataset_options   = list(DATASETS.keys())
            selected_datasets = st.multiselect(
                "Analiz edilecek veri setlerini seçin:",
                dataset_options,
                default=dataset_options,
                key="fs_ds_select",
            )

            if not selected_datasets:
                st.info("En az bir veri seti seçin.")
            else:
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    corr_low_thr  = st.slider("Düşük |ρ| eşiği — altı çıkar",  0.05, 0.30,  0.15, 0.01,   key="fs_corr_low")
                with col_p2:
                    corr_high_thr = st.slider("Yüksek |ρ| eşiği — üstü çıkar", 0.900, 0.999, 0.995, 0.001, key="fs_corr_high", format="%.3f")
                with col_p3:
                    vif_thr       = st.slider("VIF eşiği — üstü çıkar",         5.0, 20.0,  10.0, 0.5,    key="fs_vif_thr")

                if st.button("▶ Tümünü Çalıştır", key="fs_run_all"):
                    fs_results = {}
                    for ds_name in selected_datasets:
                        ds = DATASETS[ds_name].copy()
                        if target not in ds.columns:
                            continue
                        candidates = [
                            c for c in ds.columns
                            if pd.api.types.is_numeric_dtype(ds[c]) and c != target
                        ]
                        sub = ds[candidates + [target]].dropna()

                        corr_vals = sub[candidates].apply(
                            lambda col: stats.spearmanr(col, sub[target])[0]
                        ).abs()
                        low_list  = corr_vals[corr_vals < corr_low_thr].index.tolist()
                        fm        = sub[candidates].corr(method="spearman").abs()
                        upper     = fm.where(np.triu(np.ones(fm.shape), k=1).astype(bool))
                        high_list = []
                        for col in upper.columns:
                            partners = upper.index[upper[col] > corr_high_thr].tolist()
                            for p in partners:
                                drop = p if corr_vals.get(p, 0) <= corr_vals.get(col, 0) else col
                                if drop not in high_list and drop not in low_list:
                                    high_list.append(drop)
                        corr_remove = list(set(low_list + high_list))
                        after_corr  = [f for f in candidates if f not in corr_remove]
                        corr_tbl    = corr_vals.reset_index()
                        corr_tbl.columns = ["Feature", "|ρ| ile Close"]
                        corr_tbl = corr_tbl.sort_values("|ρ| ile Close", ascending=False)

                        remaining = after_corr.copy()
                        vif_rem   = []
                        while True:
                            sub_vif = sub[remaining].dropna()
                            X       = sub_vif.values.astype(float)
                            vif_vals = {}
                            for i, col in enumerate(remaining):
                                try:
                                    vif_vals[col] = variance_inflation_factor(X, i)
                                except Exception:
                                    vif_vals[col] = np.nan
                            max_col = max(vif_vals, key=lambda c: vif_vals[c] if not np.isnan(vif_vals[c]) else 0)
                            if vif_vals[max_col] > vif_thr:
                                vif_rem.append(max_col)
                                remaining.remove(max_col)
                            else:
                                break
                        sub_vif = sub[remaining].dropna()
                        X       = sub_vif.values.astype(float)
                        vif_rows = []
                        for i, col in enumerate(remaining):
                            try:
                                v = variance_inflation_factor(X, i)
                            except Exception:
                                v = np.nan
                            vif_rows.append({"Feature": col, "VIF": round(v, 2)})
                        vif_df    = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
                        after_vif = remaining

                        adf_rows = []
                        for col in after_vif + [target]:
                            series = ds[col].dropna()
                            try:
                                stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
                                stationary = pval < 0.05
                                adf_rows.append({
                                    "Feature":         col,
                                    "ADF İstatistiği": round(stat, 4),
                                    "p-değeri":        round(pval, 4),
                                    "Kritik (%5)":     round(crit["5%"], 4),
                                    "Durum":           "✅ Durağan" if stationary else "❌ Durağan Değil",
                                })
                            except Exception:
                                adf_rows.append({
                                    "Feature": col, "ADF İstatistiği": np.nan,
                                    "p-değeri": np.nan, "Kritik (%5)": np.nan, "Durum": "⚠️ Hata",
                                })
                        adf_df   = pd.DataFrame(adf_rows)
                        non_stat = adf_df[~adf_df["Durum"].str.startswith("✅")]["Feature"].tolist()

                        lb_rows = []
                        for col in after_vif + [target]:
                            series = ds[col].dropna()
                            try:
                                lb_result = acorr_ljungbox(series, lags=[10], return_df=True)
                                pval      = float(lb_result["lb_pvalue"].iloc[0])
                                has_ac    = pval < 0.05
                                lb_rows.append({
                                    "Feature":     col,
                                    "LB p-değeri": round(pval, 4),
                                    "Durum":       "❌ Otokorelasyon Var" if has_ac else "✅ Otokorelasyon Yok",
                                })
                            except Exception:
                                lb_rows.append({"Feature": col, "LB p-değeri": np.nan, "Durum": "⚠️ Hata"})
                        lb_df      = pd.DataFrame(lb_rows)
                        lb_problem = lb_df[lb_df["Durum"].str.startswith("❌")]["Feature"].tolist()

                        arch_rows = []
                        for col in after_vif + [target]:
                            series = ds[col].dropna()
                            try:
                                _, pval, _, _ = het_arch(series, nlags=5)
                                has_arch      = pval < 0.05
                                arch_rows.append({
                                    "Feature":       col,
                                    "ARCH p-değeri": round(pval, 4),
                                    "Durum":         "❌ ARCH Etkisi Var" if has_arch else "✅ ARCH Etkisi Yok",
                                })
                            except Exception:
                                arch_rows.append({"Feature": col, "ARCH p-değeri": np.nan, "Durum": "⚠️ Hata"})
                        arch_df      = pd.DataFrame(arch_rows)
                        arch_problem = arch_df[arch_df["Durum"].str.startswith("❌")]["Feature"].tolist()

                        jb_rows = []
                        try:
                            sub_jb   = ds[after_vif + [target]].dropna()
                            X_jb     = add_constant(sub_jb[after_vif].values.astype(float))
                            model_jb = OLS(sub_jb[target].values, X_jb).fit()
                            jb_stat, jb_pval, skew, kurt = jarque_bera(model_jb.resid)
                            non_normal = jb_pval < 0.05
                            jb_rows.append({
                                "Model":        f"OLS ({len(after_vif)} feature → {target})",
                                "JB p-değeri":  round(jb_pval, 4),
                                "Çarpıklık":    round(skew, 4),
                                "Basıklık":     round(kurt, 4),
                                "Durum":        "❌ Normal Değil" if non_normal else "✅ Normal",
                            })
                            jb_problem = [target] if non_normal else []
                        except Exception:
                            jb_rows.append({"Model": f"OLS ({len(after_vif)} feature → {target})", "JB p-değeri": np.nan, "Çarpıklık": np.nan, "Basıklık": np.nan, "Durum": "⚠️ Hata"})
                            jb_problem = []
                        jb_df = pd.DataFrame(jb_rows)

                        reset_rows = []
                        sub_reset  = ds[after_vif + [target]].dropna()
                        for col in after_vif:
                            try:
                                X_r    = add_constant(sub_reset[[col]])
                                model  = OLS(sub_reset[target], X_r).fit()
                                rst    = linear_reset(model, power=2, use_f=True)
                                pval   = rst.pvalue
                                nonlin = pval < 0.05
                                reset_rows.append({
                                    "Feature":         col,
                                    "RESET p-değeri":  round(pval, 4),
                                    "Durum":           "❌ Doğrusal Değil" if nonlin else "✅ Doğrusal",
                                })
                            except Exception:
                                reset_rows.append({"Feature": col, "RESET p-değeri": np.nan, "Durum": "⚠️ Hata"})
                        reset_df      = pd.DataFrame(reset_rows)
                        reset_problem = reset_df[reset_df["Durum"].str.startswith("❌")]["Feature"].tolist()

                        cusum_rows = []
                        for col in after_vif + [target]:
                            series = ds[col].dropna()
                            try:
                                X_c      = add_constant(np.arange(len(series)))
                                model_c  = OLS(series.values, X_c).fit()
                                _, pval, _ = breaks_cusumolsresid(model_c.resid)
                                has_break  = pval < 0.05
                                cusum_rows.append({
                                    "Feature":        col,
                                    "CUSUM p-değeri": round(pval, 4),
                                    "Durum":          "❌ Yapısal Kırılma Var" if has_break else "✅ Stabil",
                                })
                            except Exception:
                                cusum_rows.append({"Feature": col, "CUSUM p-değeri": np.nan, "Durum": "⚠️ Hata"})
                        cusum_df      = pd.DataFrame(cusum_rows)
                        cusum_problem = cusum_df[cusum_df["Durum"].str.startswith("❌")]["Feature"].tolist()

                        fs_results[ds_name] = {
                            "n":            len(sub),
                            "candidates":   candidates,
                            "corr_tbl":     corr_tbl,
                            "corr_low":     low_list,
                            "corr_high":    high_list,
                            "after_corr":   after_corr,
                            "vif_df":       vif_df,
                            "vif_rem":      vif_rem,
                            "after_vif":    after_vif,
                            "adf_df":       adf_df,
                            "non_stat":     non_stat,
                            "lb_df":        lb_df,
                            "lb_problem":   lb_problem,
                            "arch_df":      arch_df,
                            "arch_problem": arch_problem,
                            "jb_df":        jb_df,
                            "jb_problem":   jb_problem,
                            "reset_df":     reset_df,
                            "reset_problem": reset_problem,
                            "cusum_df":     cusum_df,
                            "cusum_problem": cusum_problem,
                        }

                    st.session_state["fs_results"]      = fs_results
                    st.session_state["fs_vif_thr_used"] = vif_thr

                if "fs_results" in st.session_state:
                    fs_results = st.session_state["fs_results"]
                    _vif_thr   = st.session_state.get("fs_vif_thr_used", 10.0)

                    for ds_name, res in fs_results.items():
                        with st.expander(f"📊 {ds_name}  —  n={res['n']:,}", expanded=False):

                            st.markdown("**1️⃣ Spearman Korelasyon Filtresi**")
                            st.dataframe(
                                res["corr_tbl"].style.format({"|ρ| ile Close": "{:.4f}"}),
                                use_container_width=True, hide_index=True,
                            )
                            ca, cb = st.columns(2)
                            with ca:
                                st.error(f"Düşük |ρ| → çıkar: {res['corr_low'] if res['corr_low'] else 'Yok'}")
                                st.warning(f"Multicollinearity → çıkar: {res['corr_high'] if res['corr_high'] else 'Yok'}")
                            with cb:
                                st.success(f"Kalan ({len(res['after_corr'])}): {res['after_corr']}")

                            st.markdown("**2️⃣ VIF Analizi**")
                            _t = _vif_thr
                            def _vc(val, t=_t):
                                if not isinstance(val, (int, float)): return ""
                                if val > t:         return "background-color:#f8d7da; color:#842029"
                                if val > t * 0.7:   return "background-color:#fff3cd; color:#664d03"
                                return                     "background-color:#d1e7dd; color:#0a3622"
                            st.dataframe(
                                res["vif_df"].style.map(_vc, subset=["VIF"]),
                                use_container_width=True, hide_index=True,
                            )
                            ca, cb = st.columns(2)
                            with ca:
                                st.error(f"VIF > {_vif_thr} → çıkar: {res['vif_rem'] if res['vif_rem'] else 'Yok'}")
                            with cb:
                                st.success(f"Kalan ({len(res['after_vif'])}): {res['after_vif']}")

                            st.markdown("**3️⃣ ADF Durağanlık Testi**")
                            def _ac(val):
                                if not isinstance(val, str): return ""
                                if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                return ""
                            st.dataframe(
                                res["adf_df"].style.map(_ac, subset=["Durum"]),
                                use_container_width=True, hide_index=True,
                            )
                            non_feat = [f for f in res["non_stat"] if f != target]
                            if non_feat:
                                st.warning(f"Durağan Olmayan: `{'`, `'.join(non_feat)}`")
                            if not res["non_stat"]:
                                st.success("Tüm değişkenler durağan.")

                            st.markdown("**4️⃣ Ljung-Box Otokorelasyon Testi** *(lag=10)*")
                            def _lbc(val):
                                if not isinstance(val, str): return ""
                                if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                return ""
                            st.dataframe(
                                res["lb_df"].style.map(_lbc, subset=["Durum"]),
                                use_container_width=True, hide_index=True,
                            )
                            lb_feat = [f for f in res["lb_problem"] if f != target]
                            if lb_feat:
                                st.warning(f"Otokorelasyon Tespit Edildi: `{'`, `'.join(lb_feat)}` — GLS/HAC standart hata veya fark alma önerilir.")
                            else:
                                st.success("Otokorelasyon tespit edilmedi.")

                            st.markdown("**5️⃣ ARCH Heteroskedasticity Testi** *(lag=5)*")
                            def _archc(val):
                                if not isinstance(val, str): return ""
                                if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                return ""
                            st.dataframe(
                                res["arch_df"].style.map(_archc, subset=["Durum"]),
                                use_container_width=True, hide_index=True,
                            )
                            arch_feat = [f for f in res["arch_problem"] if f != target]
                            if arch_feat:
                                st.warning(f"ARCH Etkisi Tespit Edildi: `{'`, `'.join(arch_feat)}` — Volatilite kümelenmesi var, GARCH modelleme düşünülebilir.")
                            else:
                                st.success("ARCH etkisi tespit edilmedi.")

                            if "jb_df" not in res:
                                st.info("6️⃣-8️⃣ testler için analizi yeniden çalıştırın.")
                            else:
                                st.markdown("**6️⃣ Jarque-Bera Normallik Testi** *(model artıkları)*")
                                def _jbc(val):
                                    if not isinstance(val, str): return ""
                                    if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                    if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                    return ""
                                st.dataframe(
                                    res["jb_df"].style.map(_jbc, subset=["Durum"]),
                                    use_container_width=True, hide_index=True,
                                )
                                if res.get("jb_problem"):
                                    st.warning("Model artıkları normal dağılmıyor — Durağan/zayıf bağımlı serilerde CLT geçerlidir; ADF/ARCH sorunları mevcutsa HAC olmadan normallik varsayımına dayanılamaz.")
                                else:
                                    st.success("Model artıkları normal dağılıyor.")

                                st.markdown("**7️⃣ RESET Doğrusallık Testi**")
                                def _rc(val):
                                    if not isinstance(val, str): return ""
                                    if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                    if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                    return ""
                                st.dataframe(
                                    res["reset_df"].style.map(_rc, subset=["Durum"]),
                                    use_container_width=True, hide_index=True,
                                )
                                reset_feat = res["reset_problem"]
                                if reset_feat:
                                    st.warning(f"Doğrusal Olmayan İlişki: `{'`, `'.join(reset_feat)}` — Finansal zaman serilerinde doğrusallık nadiren sağlanır; OLS katsayıları yaklaşık yorumlanmalıdır. Rejim değişikliği şüphesi varsa TAR/Markov Switching düşünülebilir.")
                                else:
                                    st.success("Tüm feature-target ilişkileri doğrusal.")

                                st.markdown("**8️⃣ CUSUM Yapısal Kırılma Testi**")
                                def _cc(val):
                                    if not isinstance(val, str): return ""
                                    if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                                    if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
                                    return ""
                                st.dataframe(
                                    res["cusum_df"].style.map(_cc, subset=["Durum"]),
                                    use_container_width=True, hide_index=True,
                                )
                                cusum_feat = [f for f in res["cusum_problem"] if f != target]
                                if cusum_feat:
                                    st.warning(f"Yapısal Kırılma Tespit Edildi: `{'`, `'.join(cusum_feat)}` — Zaman serisi ikiye bölünüp ayrı model kurulabilir veya rolling window kullanılabilir.")
                                else:
                                    st.success("Yapısal kırılma tespit edilmedi.")

                    st.markdown("---")
                    st.markdown("### 📊 Karşılaştırma")

                    comp_rows = []
                    for ds_name, res in fs_results.items():
                        adf_pass   = res["adf_df"]["Durum"].str.startswith("✅").sum()
                        adf_total  = len(res["adf_df"])
                        lb_pass    = res["lb_df"]["Durum"].str.startswith("✅").sum()
                        arch_pass  = res["arch_df"]["Durum"].str.startswith("✅").sum()
                        jb_pass    = res["jb_df"]["Durum"].str.startswith("✅").sum() if "jb_df" in res else "-"
                        jb_total   = len(res["jb_df"]) if "jb_df" in res else "-"
                        reset_pass = res["reset_df"]["Durum"].str.startswith("✅").sum() if "reset_df" in res else "-"
                        reset_tot  = len(res["reset_df"]) if "reset_df" in res else "-"
                        cusum_pass = res["cusum_df"]["Durum"].str.startswith("✅").sum() if "cusum_df" in res else "-"
                        comp_rows.append({
                            "Veri Seti":          ds_name,
                            "Başlangıç":          len(res["candidates"]),
                            "Korelasyon Sonrası": len(res["after_corr"]),
                            "VIF Sonrası":        len(res["after_vif"]),
                            "ADF Geçen":          f"{adf_pass}/{adf_total}",
                            "LB Geçen":           f"{lb_pass}/{adf_total}",
                            "ARCH Geçen":         f"{arch_pass}/{adf_total}",
                            "JB Geçen":           f"{jb_pass}/{jb_total}" if jb_pass != "-" else "-",
                            "RESET Geçen":        f"{reset_pass}/{reset_tot}" if reset_pass != "-" else "-",
                            "CUSUM Geçen":        f"{cusum_pass}/{adf_total}" if cusum_pass != "-" else "-",
                            "Hayatta Kalanlar":   ", ".join(res["after_vif"]),
                        })

                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                    if len(fs_results) > 1:
                        all_survivors = [set(res["after_vif"]) for res in fs_results.values()]
                        common        = set.intersection(*all_survivors)
                        if common:
                            st.success(f"**Tüm veri setlerinde ortak hayatta kalanlar ({len(common)}):** `{'`, `'.join(sorted(common))}`")
                        else:
                            st.warning("Tüm veri setlerinde ortak hayatta kalan feature yok.")

                    advice_key = st.selectbox(
                        "Tavsiye için veri seti seçin:",
                        list(fs_results.keys()),
                        index=len(fs_results) - 1,
                        key="fs_advice_key",
                    )
                    advice_res  = fs_results[advice_key]
                    any_ns      = len(advice_res["non_stat"]) > 0
                    any_lb      = len(advice_res["lb_problem"]) > 0
                    any_arch    = len(advice_res["arch_problem"]) > 0
                    any_jb      = len(advice_res.get("jb_problem", [])) > 0
                    any_reset   = len(advice_res.get("reset_problem", [])) > 0
                    any_cusum   = len(advice_res.get("cusum_problem", [])) > 0

                    with st.expander("💡 Test Sonuçlarına Göre Model Tavsiyesi", expanded=True):
                        st.markdown("""
| Test | Sonuç | Regresyon Etkisi | Tavsiye |
|---|---|---|---|
| **ADF (Durağanlık)** | {} | Durağan olmayan seri sahte regresyon üretir | {} |
| **Ljung-Box (Otokorelasyon)** | {} | Standart hatalar yanlış, t-istatistikleri güvenilmez | {} |
| **ARCH (Heteroskedasticity)** | {} | Varyans sabit değil, OLS verimsiz kalır | {} |
| **Jarque-Bera (Normallik)** | {} | Küçük örneklemde katsayı testi güvenilmezleşir | {} |
| **RESET (Doğrusallık)** | {} | Lineer model ilişkiyi eksik yakalar | {} |
| **CUSUM (Yapısal Kırılma)** | {} | Katsayılar zaman içinde değişiyor, model kararsız | {} |
""".format(
                            "❌ Sorun var" if any_ns    else "✅ Temiz",
                            "Fark alma (`diff`) veya log-return kullan" if any_ns else "İşlem gerekmez",
                            "❌ Sorun var" if any_lb    else "✅ Temiz",
                            "OLS + **HAC/Newey-West** standart hata kullan" if any_lb else "Standart OLS uygulanabilir",
                            "❌ Sorun var" if any_arch  else "✅ Temiz",
                            "Volatilite tahmini → **GARCH**; getiri tahmini → OLS+HAC yeterli" if any_arch else "OLS varyans tahmini güvenilir",
                            "❌ Sorun var" if any_jb    else "✅ Temiz",
                            "Durağan/zayıf bağımlı serilerde CLT geçerlidir; bu veri setinde ADF/ARCH sorunları da mevcutsa OLS + HAC kullan" if any_jb else "Normallik varsayımı sağlanıyor",
                            "❌ Sorun var" if any_reset else "✅ Temiz",
                            "Finansal zaman serilerinde doğrusallık nadiren sağlanır; OLS katsayıları yaklaşık yorumla. Rejim değişikliği şüphesi varsa TAR/Markov Switching düşünülebilir" if any_reset else "Doğrusal model yeterli",
                            "❌ Sorun var" if any_cusum else "✅ Temiz",
                            "Rolling window veya zaman dilimine göre ayrı model kur" if any_cusum else "Katsayılar zaman içinde stabil",
                        ))

                        st.markdown("---")
                        st.markdown("#### 🎯 Önerilen Yaklaşım")

                        problems = sum([any_ns, any_lb, any_arch, any_jb, any_reset, any_cusum])

                        if problems == 0:
                            st.success("""
**Tüm testler temiz.**

- Klasik OLS regresyon doğrudan uygulanabilir
- ML / Derin öğrenme doğrudan kullanılabilir — normallik, doğrusallık ve multicollinearity varsayımları zaten geçerli değil; durağanlık ve yapısal kararlılık da sağlandığından concept drift riski düşük
                            """)
                        else:
                            msg = "**Tespit edilen sorunlar ve öneriler:**\n\n"
                            if any_ns:
                                msg += "- **Durağan değil** → Fark alma veya log-return ile dönüştür\n"
                            if any_lb:
                                msg += "- **Otokorelasyon var** → OLS + HAC (Newey-West) standart hata kullan\n"
                            if any_arch:
                                msg += "- **ARCH etkisi var** → Volatilite tahmini için GARCH; getiri tahmini için OLS+HAC yeterli\n"
                            if any_jb:
                                msg += "- **Normal dağılmıyor** → Durağan/zayıf bağımlı serilerde CLT geçerlidir; bu veri setinde ADF/ARCH sorunları da mevcutsa OLS + HAC kullan\n"
                            if any_reset:
                                msg += "- **Doğrusal değil** → Finansal zaman serilerinde doğrusallık nadiren sağlanır; OLS katsayıları yaklaşık yorumla. Rejim değişikliği şüphesi varsa TAR/Markov Switching düşünülebilir\n"
                            if any_cusum:
                                msg += "- **Yapısal kırılma var** → Rolling window veya zaman dilimine göre ayrı model kur\n"
                            msg += "\n**ML / Derin öğrenme kullanacaksan:**\n"
                            msg += "- Normallik (JB), doğrusallık (RESET) ve multicollinearity (VIF) varsayımları geçerli değil — bu testleri görmezden gelebilirsin\n"
                            if any_ns:
                                msg += "- ⚠️ **Durağanlık (ADF) ML/DL için de kritik** — durağan olmayan veri concept drift ve overfitting riskini artırır; log-return almayı düşün\n"
                            if any_cusum:
                                msg += "- ⚠️ **Yapısal kırılma (CUSUM) ML/DL için de kritik** — eğitim/test dağılımı farklılaşır; rolling window ile eğit\n"
                            if any_arch:
                                msg += "- ℹ️ ARCH etkisi LSTM/Transformer gibi sequence modellerinde window boyutu seçimini etkileyebilir\n"
                            st.warning(msg)

                    st.markdown("---")
                    st.markdown("### 🏁 Sonuç — Seçilen Feature'lar")

                    dl_key = st.selectbox(
                        "İndirilecek veri setini seçin:",
                        list(fs_results.keys()),
                        index=len(fs_results) - 1,
                        key="fs_dl_key",
                    )
                    ref_res        = fs_results[dl_key]
                    final_features = ref_res["after_vif"]
                    fs_df          = DATASETS[dl_key]
                    all_candidates = ref_res["candidates"]

                    st.info(
                        f"**Hayatta kalan feature'lar — {dl_key}"
                        f" ({len(final_features)}):** `{'`, `'.join(final_features)}`"
                    )

                    st.caption("İndirilecek sütunları seçin (Close her zaman dahildir):")
                    export_candidates = [target] + [f for f in all_candidates if f in fs_df.columns]
                    final_selected    = [target]
                    fs_cols_per_row   = 4
                    fs_rows = [export_candidates[i:i+fs_cols_per_row] for i in range(0, len(export_candidates), fs_cols_per_row)]
                    for fs_row in fs_rows:
                        fs_cb_cols = st.columns(len(fs_row))
                        for i, col_name in enumerate(fs_row):
                            with fs_cb_cols[i]:
                                disabled = col_name == target
                                checked  = st.checkbox(col_name, value=disabled, key=f"fs_cb_{col_name}", disabled=disabled)
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
                        label=f"📥 Temizlenmiş ve Transforme Edilmiş Veri Seti — Seçili Feature'lar ({len(final_selected)} sütun, {len(export_final):,} satır)",
                        data=buf_final.getvalue(),
                        file_name=f"{symbol.replace('.', '_')}_{interval}_{start_date}_{end_date}_{dl_key.split('—')[-1].strip().replace(' ', '_').lower()}_secili.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    with tab3:

        st.caption(
            "Bugünkü formasyona en benzeyen geçmiş dönemleri DTW ile bulur. "
            "z-score normalize fiyat üzerinde çalışır; tekil tahmin değil, "
            "benzer dönemlerin **sonrasındaki getiri dağılımı** raporlanır."
        )

        try:
            from dtaidistance import dtw
        except ImportError:
            st.error(
                "`dtaidistance` kurulu değil. requirements.txt'e `dtaidistance` "
                "ekleyip `pip install dtaidistance` çalıştırın."
            )
            st.stop()

        if "Close" not in df.columns:
            st.info("Bu analiz için **Close** sütunu gerekli.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                pa_window = st.number_input("Pencere (gün)", min_value=10, max_value=400, value=60, step=5, key="pa_window")
            with c2:
                pa_topk = st.number_input("Top-K eşleşme", min_value=3, max_value=30, value=10, step=1, key="pa_topk")
            with c3:
                pa_h2 = st.number_input("Uzun ufuk (gün)", min_value=5, max_value=250, value=60, step=5, key="pa_h2")
            pa_h1 = 20
            horizons = sorted({pa_h1, int(pa_h2)})

            run_pa = st.button("Örüntü Analizini Çalıştır", key="pa_run")

            def _z(a):
                s = a.std()
                return np.zeros_like(a) if s == 0 else (a - a.mean()) / s

            def _shape_desc(z):
                """z-score pencereyi sözel forma çevir: eğim + dip/tepe konumu."""
                w = len(z)
                third = w // 3
                slope = z[-1] - z[0]
                trend = ("yükseliş" if slope > 0.5 else
                         "düşüş" if slope < -0.5 else "yatay")
                imin, imax = int(np.argmin(z)), int(np.argmax(z))
                def _loc(i):
                    return ("başında" if i < third else
                            "ortasında" if i < 2 * third else "sonunda")
                return trend, f"dip pencerenin {_loc(imin)}, tepe {_loc(imax)}"

            if run_pa:
                prices = df["Close"].dropna()
                values = prices.values.astype(np.float64)
                dates  = prices.index
                n      = len(values)
                window = int(pa_window)
                top_k  = int(pa_topk)
                max_h  = max(horizons)
                min_gap = window // 2

                if n < window + max_h + 5:
                    st.warning(f"Yetersiz veri: {n} gün. En az {window + max_h + 5} gerekli.")
                    st.session_state.pop("pa_result", None)
                else:
                    query = _z(values[-window:])
                    last_start = n - window - max_h

                    with st.spinner("DTW taraması yapılıyor..."):
                        cands = []
                        for start in range(0, last_start + 1):
                            d = dtw.distance_fast(query, _z(values[start:start + window]))
                            cands.append((d, start))
                        cands.sort(key=lambda x: x[0])

                        selected = []
                        for d, start in cands:
                            if all(abs(start - s) >= min_gap for _, s in selected):
                                selected.append((d, start))
                            if len(selected) == top_k:
                                break

                    # DTW -> benzerlik %: en kötü adaya göre normalize, 0..100
                    worst = cands[-1][0] if cands else 1.0
                    worst = worst if worst > 0 else 1.0

                    matches = []
                    for rank, (d, start) in enumerate(selected, 1):
                        end = start + window
                        end_price = values[end - 1]
                        fwd = {h: round((values[end - 1 + h] / end_price - 1) * 100, 2)
                               for h in horizons}
                        matches.append({
                            "rank": rank,
                            "start": int(start),
                            "dtw": float(d),
                            "sim": round((1 - d / worst) * 100, 1),
                            "start_date": str(dates[start].date()),
                            "end_date": str(dates[end - 1].date()),
                            "fwd": fwd,
                        })

                    st.session_state["pa_result"] = {
                        "values": values, "query": query, "window": window,
                        "horizons": horizons, "matches": matches,
                        "dates": [str(dt.date()) for dt in dates],
                    }

            # ── Sonuçları session_state'ten render et (buton sonrası da kalsın) ──
            res = st.session_state.get("pa_result")
            if res:
                values  = res["values"]
                query   = res["query"]
                window  = res["window"]
                horizons = res["horizons"]
                matches = res["matches"]

                # Tablo
                rows = []
                for m in matches:
                    row = {
                        "#":         m["rank"],
                        "Benzerlik %": m["sim"],
                        "Başlangıç": m["start_date"],
                        "Bitiş":     m["end_date"],
                        "DTW":       round(m["dtw"], 3),
                    }
                    for h in horizons:
                        row[f"+{h}g %"] = m["fwd"][h]
                    rows.append(row)
                table = pd.DataFrame(rows)

                st.subheader("Overlay — bugün vs en benzer dönemler")
                fig_ov = go.Figure()
                x = list(range(window))
                for m in matches:
                    s0 = m["start"]
                    fig_ov.add_trace(go.Scatter(
                        x=x, y=_z(values[s0:s0 + window]),
                        mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=1),
                        showlegend=False, hovertemplate=f"#{m['rank']} ({m['start_date']})<extra></extra>"
                    ))
                fig_ov.add_trace(go.Scatter(
                    x=x, y=query, mode="lines",
                    line=dict(color="#dc2626", width=3), name="Bugün"
                ))
                fig_ov.update_layout(
                    xaxis_title="Pencere içi gün", yaxis_title="z-score fiyat",
                    height=420, margin=dict(l=50, r=20, t=20, b=40), hovermode="x unified"
                )
                st.plotly_chart(fig_ov, use_container_width=True)

                st.subheader("En benzer dönemler")
                st.dataframe(table, use_container_width=True, hide_index=True)

                # ── Tek eşleşme detayı + neden benzer açıklaması ──
                st.subheader("Eşleşme detayı")
                labels = [f"#{m['rank']} — {m['start_date']} → {m['end_date']} "
                          f"(%{m['sim']} benzer)" for m in matches]
                sel_idx = st.selectbox(
                    "İncelenecek dönem:", range(len(matches)),
                    format_func=lambda i: labels[i], key="pa_match_sel"
                )
                m = matches[sel_idx]
                s0 = m["start"]
                z_match = _z(values[s0:s0 + window])

                # Eşleşen dönemin SONRASI (~3 ay = 63 işgünü), pencere
                # istatistiğiyle aynı z-score uzayında devam ettir.
                FWD_DAYS = 63
                win_slice = values[s0:s0 + window]
                w_mean, w_std = win_slice.mean(), win_slice.std()
                w_std = w_std if w_std != 0 else 1.0
                post_end = min(s0 + window + FWD_DAYS, len(values))
                post_raw = values[s0 + window:post_end]
                z_post = (post_raw - w_mean) / w_std
                x_post = list(range(window, window + len(z_post)))

                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=x, y=query, mode="lines",
                    line=dict(color="#dc2626", width=2.5), name="Bugün"
                ))
                fig_d.add_trace(go.Scatter(
                    x=x, y=z_match, mode="lines",
                    line=dict(color="#2563eb", width=2.5),
                    name=f"#{m['rank']} ({m['start_date']})"
                ))
                if len(z_post):
                    # pencere sonu ile sonrasını görsel olarak bağla
                    fig_d.add_trace(go.Scatter(
                        x=[window - 1] + x_post, y=[z_match[-1]] + list(z_post),
                        mode="lines", line=dict(color="#93c5fd", width=2.5, dash="dot"),
                        name="↳ sonraki ~3 ay"
                    ))
                # pencere sonunu işaretle
                fig_d.add_vline(x=window - 1, line=dict(color="gray", width=1, dash="dash"))
                fig_d.update_layout(
                    xaxis_title="Pencere içi gün (kesikli çizgi = formasyon sonu)",
                    yaxis_title="z-score fiyat",
                    height=380, margin=dict(l=50, r=20, t=20, b=40), hovermode="x unified"
                )
                st.plotly_chart(fig_d, use_container_width=True)

                # Neden benzer — sözel karşılaştırma
                qt, qd = _shape_desc(query)
                mt, md = _shape_desc(z_match)
                corr = float(np.corrcoef(query, z_match)[0, 1])
                fwd_txt = " · ".join(f"+{h}g: %{m['fwd'][h]}" for h in horizons)

                if qt == mt:
                    trend_line = f"İkisi de genel olarak **{qt}** eğiliminde."
                else:
                    trend_line = (f"Bugün **{qt}**, bu dönem **{mt}** eğiliminde "
                                  "— eğilim farkı var, DTW şekli zaman kaydırarak hizalamış.")

                st.markdown(
                    f"**Neden benzer?**\n\n"
                    f"- {trend_line}\n"
                    f"- Bugünün formu: {qd}. Bu dönemin formu: {md}.\n"
                    f"- Nokta-nokta korelasyon: **{corr:.2f}** "
                    f"(DTW benzerlik skoru: **%{m['sim']}**).\n"
                    f"- Bu dönemin sonrasındaki getiri → {fwd_txt}"
                )

                st.subheader("Forward getiri dağılımı")
                summ = []
                for h in horizons:
                    s = table[f"+{h}g %"]
                    summ.append({
                        "Ufuk":       f"+{h}g",
                        "Ortalama %": round(s.mean(), 2),
                        "Medyan %":   round(s.median(), 2),
                        "Std %":      round(s.std(), 2),
                        "Min %":      round(s.min(), 2),
                        "Max %":      round(s.max(), 2),
                        "Pozitif %":  round((s > 0).mean() * 100, 0),
                    })
                st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True)

                st.warning(
                    "**Uyarı:** 'En benzer' geçmiş dönem, geleceğin aynı olacağı "
                    "anlamına gelmez. Std yüksek ve pozitif oran %50'ye yakınsa "
                    "sinyal zayıftır; tekil eşleşmeye değil dağılıma bakın."
                )

else:
    st.warning("Filtreleme sonrası veri kalmadı.")
