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
# IG.com — ticker → market sayfası eşleştirme + snapshot çekme
# ============================================================

# yfinance ticker -> IG hafta sonu market sayfası göreli yolu (bölge kodu hariç)
# Yalnızca IG'de HAFTA SONU işlem gören enstrümanlar listelenir.
# Doğrulanmış: weekend-gold, weekend-wall-street, weekend-us-tech-100-e1
IG_MARKET_MAP = {
    # Altın
    "GC=F": "indices/markets-indices/weekend-gold",
    "GLD":  "indices/markets-indices/weekend-gold",
    "XAUUSD=X": "indices/markets-indices/weekend-gold",
    # Endeksler
    "^DJI":  "indices/markets-indices/weekend-wall-street",
    "YM=F":  "indices/markets-indices/weekend-wall-street",
    "^IXIC": "indices/markets-indices/weekend-us-tech-100-e1",
    "NQ=F":  "indices/markets-indices/weekend-us-tech-100-e1",
    "^FTSE": "indices/markets-indices/weekend-uk-100",
    "^GDAXI":"indices/markets-indices/weekend-germany-40",
    "^HSI":  "indices/markets-indices/weekend-hong-kong-hs50",
    # Forex (hafta sonu sadece 3 majör)
    "GBPUSD=X": "forex/markets-forex/weekend-gbp-usd",
    "EURUSD=X": "forex/markets-forex/weekend-eur-usd",
    "USDJPY=X": "forex/markets-forex/weekend-usd-jpy",
    # Kripto (zaten 7/24)
    "BTC-USD": "forex/markets-forex/bitcoin-1",
    "ETH-USD": "forex/markets-forex/ether",
}

def fetch_ig_snapshot(rel_path, region="en"):
    """IG market sayfasından anlık BUY/SELL/değişim/High/Low/sentiment çeker.
    Dönüş: dict(name, code, sell, buy, change, change_pct, high, low,
                long_pct, short_pct, url) veya hata için dict(error=...)."""
    import re
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.ig.com/{region}/{rel_path}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code} — IG sayfası alınamadı.", "url": url}
    except Exception as e:
        return {"error": f"Bağlantı hatası: {e}", "url": url}

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    # Non-breaking space ve çoklu boşlukları normalize et (IG &nbsp; kullanır)
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    text = re.sub(r"\s+", " ", text)

    def _num(s):
        if s is None:
            return None
        s = s.strip()
        # Hem nokta hem virgül varsa: virgül binlik ayraç -> sil
        if "," in s and "." in s:
            s = s.replace(",", "")
        # Sadece virgül varsa: ondalık ayraç -> noktaya çevir
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    out = {"url": url, "name": None, "code": None,
           "sell": None, "buy": None, "change": None, "change_pct": None,
           "high": None, "low": None, "long_pct": None, "short_pct": None}

    # SELL / BUY — sayfada "SELL<fiyat>" / "BUY<fiyat>" şeklinde geçiyor
    m = re.search(r"SELL\s*([\d,]+\.?\d*)", text)
    if m: out["sell"] = _num(m.group(1))
    m = re.search(r"BUY\s*([\d,]+\.?\d*)", text)
    if m: out["buy"] = _num(m.group(1))

    # Değişim — birincil: "-9.2(-0.22%)" / "-9,2 (-0,22%)" / "66.19(1.55%)"
    m = re.search(r"(-?[\d.,]+)\s*\(\s*(-?[\d.,]+)\s*%\s*\)", text)
    if m:
        out["change"] = _num(m.group(1))
        out["change_pct"] = _num(m.group(2))
    else:
        # Yedek: değişim ve yüzde ayrı; High'tan hemen önceki sayı + (..%)
        m_pct = re.search(r"\(\s*(-?[\d.,]+)\s*%\s*\)", text)
        if m_pct:
            out["change_pct"] = _num(m_pct.group(1))
        # change'i BUY fiyatı ile High arasındaki ilk işaretli sayı say
        seg = re.search(r"BUY\s*[\d.,]+\s*(-?[\d.,]+)", text)
        if seg:
            out["change"] = _num(seg.group(1))

    m = re.search(r"High:\s*([\d,]+\.?\d*)", text)
    if m: out["high"] = _num(m.group(1))
    m = re.search(r"Low:\s*([\d,]+\.?\d*)", text)
    if m: out["low"] = _num(m.group(1))

    # Sentiment — birincil: "76% of client accounts are long on this market"
    out["long_pct"] = None
    out["short_pct"] = None
    m = re.search(r"(\d+)\s*%\s*of\s+client\s+accounts\s+are\s+(long|short)",
                  text, re.IGNORECASE)
    if m:
        pct = int(m.group(1))
        if m.group(2).lower() == "long":
            out["long_pct"], out["short_pct"] = pct, 100 - pct
        else:
            out["short_pct"], out["long_pct"] = pct, 100 - pct
    else:
        # Yedek: "Long Short 76% 24%" bloğu (etiketlerden sonra iki yüzde)
        m2 = re.search(r"Long\s+Short\s+(\d+)\s*%\s+(\d+)\s*%", text, re.IGNORECASE)
        if m2:
            out["long_pct"] = int(m2.group(1))
            out["short_pct"] = int(m2.group(2))

    # Enstrüman adı: ilk H1
    h1 = soup.find("h1")
    if h1: out["name"] = h1.get_text(strip=True)

    return out


def fetch_ig_auto(rel_path, regions=("za", "en", "ae")):
    """Bölgeleri sırayla dener; FİYATI dolu gelen İLK bölgede durur ve
    o bölgenin tüm verisini (sentiment dahil) döndürür. Bölgeler arası
    veri karıştırılmaz — fiyat bir bölgeden, sentiment başkasından gelmez."""
    last_err = None
    for reg in regions:
        snap = fetch_ig_snapshot(rel_path, reg)
        if snap.get("error"):
            last_err = snap
            continue
        if snap.get("sell") is not None:   # bu bölge gerçek veri verdi
            snap["region_used"] = reg
            return snap
    return last_err or {"error": "Hiçbir bölgeden veri alınamadı.", "url": ""}


# ============================================================
# Google Sheets — IG snapshot geçmişi (kalıcı kayıt)
# ============================================================

IG_HISTORY_COLUMNS = [
    "timestamp", "ticker", "ig_name", "sell", "buy",
    "long_pct", "short_pct", "change", "change_pct", "direction",
    "high", "low",
]

def _direction_from_change(chg):
    """IG günlük değişimine göre yön etiketi."""
    if chg is None:
        return ""
    if chg > 0:
        return "up"
    if chg < 0:
        return "down"
    return "flat"

def _get_gsheets_conn():
    """st.connection ile gsheets bağlantısı döndürür; yoksa None."""
    try:
        from streamlit_gsheets import GSheetsConnection
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception:
        return None

def save_ig_snapshot(snap, ticker):
    """Snapshot'ı Google Sheets'e yeni satır olarak ekler.
    Dönüş: (ok: bool, mesaj: str)."""
    conn = _get_gsheets_conn()
    if conn is None:
        return False, "Google Sheets bağlantısı kurulu değil (secrets/paket eksik)."
    try:
        try:
            df = conn.read(ttl=0)
            df = df.dropna(how="all")
        except Exception:
            df = pd.DataFrame(columns=IG_HISTORY_COLUMNS)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "ig_name": snap.get("name") or "",
            "sell": snap.get("sell"), "buy": snap.get("buy"),
            "change": snap.get("change"), "change_pct": snap.get("change_pct"),
            "high": snap.get("high"), "low": snap.get("low"),
            "long_pct": snap.get("long_pct"), "short_pct": snap.get("short_pct"),
            "direction": _direction_from_change(snap.get("change")),
        }
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        # Kolonları sabit sıraya diz (eksikler boş eklenir)
        for c in IG_HISTORY_COLUMNS:
            if c not in new_df.columns:
                new_df[c] = None
        new_df = new_df[IG_HISTORY_COLUMNS]
        conn.update(data=new_df)
        return True, "Kayıt eklendi."
    except Exception as e:
        return False, f"Kayıt hatası: {e}"

def load_ig_history(ticker):
    """Verilen ticker'ın tüm geçmiş kayıtlarını DataFrame olarak döndürür."""
    conn = _get_gsheets_conn()
    if conn is None:
        return None
    try:
        df = conn.read(ttl=0).dropna(how="all")
        if df.empty or "ticker" not in df.columns:
            return pd.DataFrame(columns=IG_HISTORY_COLUMNS)
        df = df[df["ticker"] == ticker].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.sort_values("timestamp")
    except Exception:
        return None

def delete_ig_row(ticker, timestamp_str):
    """ticker + timestamp eşleşen satır(lar)ı Sheets'ten siler.
    Dönüş: (ok, mesaj)."""
    conn = _get_gsheets_conn()
    if conn is None:
        return False, "Google Sheets bağlantısı kurulu değil."
    try:
        df = conn.read(ttl=0).dropna(how="all")
        if df.empty:
            return False, "Silinecek kayıt yok."
        before = len(df)
        mask = ~((df["ticker"] == ticker) &
                 (df["timestamp"].astype(str) == timestamp_str))
        df2 = df[mask]
        if len(df2) == before:
            return False, "Eşleşen kayıt bulunamadı."
        conn.update(data=df2)
        return True, "Kayıt silindi."
    except Exception as e:
        return False, f"Silme hatası: {e}"


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

    tab1, tab3, tab4 = st.tabs(
        ["📥 Veri İndir", "🔍 Örüntü Analizi", "🟡 IG Hafta Sonu"]
    )

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

    with tab3:

        st.caption(
            "Bugünkü formasyona en benzeyen geçmiş dönemleri DTW ile bulur. "
            "z-score normalize fiyat üzerinde grafik/şekil benzerliğine göre çalışır."
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
            c1, c2 = st.columns(2)
            with c1:
                pa_window = st.selectbox(
                    "Pencere (gün)", options=[30, 60, 90, 120, 240, 360], index=1, key="pa_window",
                    help="Formasyonun uzunluğu. Bugünün son kaç günlük hareketini bir kalıp "
                         "sayıp geçmişte arayacağı. 60 ≈ son 3 ay. Küçük değer kısa vadeli "
                         "kalıpları, büyük değer uzun trend formasyonlarını yakalar."
                )
            with c2:
                pa_topk = st.number_input(
                    "Top-K eşleşme", min_value=3, max_value=30, value=5, step=1, key="pa_topk",
                    help="Kaç benzer dönem listelensin. En benzeyen ilk K dönem gösterilir. "
                         "Çok büyütmek alttaki zayıf eşleşmeleri de katar."
                )
            st.caption(
                "**Pencere** = formasyonun uzunluğu · **Top-K** = kaç benzer dönem gösterilsin. "
                "Kutuların yanındaki **?** işaretinde detay var."
            )

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
                min_gap = window // 2

                if n < window + 5:
                    st.warning(f"Yetersiz veri: {n} gün. En az {window + 5} gerekli.")
                    st.session_state.pop("pa_result", None)
                else:
                    query = _z(values[-window:])
                    query_start = n - window
                    # Sorgu penceresinin kendisini ve onunla örtüşen
                    # (min_gap'ten yakın) son pencereleri tarama dışı bırak.
                    last_start = query_start - min_gap

                    if last_start < 0:
                        st.warning("Pencere veriye göre çok uzun; daha kısa pencere seçin.")
                        st.session_state.pop("pa_result", None)
                        st.stop()

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
                        matches.append({
                            "rank": rank,
                            "start": int(start),
                            "dtw": float(d),
                            "sim": round((1 - d / worst) * 100, 1),
                            "start_date": str(dates[start].date()),
                            "end_date": str(dates[end - 1].date()),
                        })

                    st.session_state["pa_result"] = {
                        "values": values, "query": query, "window": window,
                        "matches": matches,
                        "dates": [str(dt.date()) for dt in dates],
                    }

            # ── Sonuçları session_state'ten render et (buton sonrası da kalsın) ──
            res = st.session_state.get("pa_result")
            if res:
                values  = res["values"]
                query   = res["query"]
                window  = res["window"]
                matches = res["matches"]

                # Tablo
                rows = []
                for m in matches:
                    rows.append({
                        "#":         m["rank"],
                        "Benzerlik %": m["sim"],
                        "Başlangıç": m["start_date"],
                        "Bitiş":     m["end_date"],
                        "DTW":       round(m["dtw"], 3),
                    })
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
                    f"(DTW benzerlik skoru: **%{m['sim']}**)."
                )

                st.warning(
                    "**Uyarı:** 'En benzer' geçmiş dönem, geleceğin aynı olacağı "
                    "anlamına gelmez. Görsel benzerlik yüksek olsa bile sonrasının "
                    "yönü farklı olabilir."
                )

    with tab4:
        st.subheader("🟡 IG.com anlık piyasa bilgisi")

        with st.expander("ℹ️ IG'de hafta sonu hangi piyasalar açık?"):
            st.markdown(
                "IG hafta sonu işlemi sınırlı bir enstrüman setinde sunar "
                "(düşük likidite, ayrı fiyatlama). Saatler UK saatine göredir:\n\n"
                "**Endeksler & Altın** — Cmt 16:00 → Pzt sabahı\n"
                "- Weekend Gold\n"
                "- Weekend Wall Street (Dow)\n"
                "- Weekend US Tech 100 (Nasdaq)\n"
                "- Weekend UK 100 (FTSE)\n"
                "- Weekend Germany 40 (DAX)\n"
                "- Weekend Hong Kong HS50\n\n"
                "**Forex (yalnızca 3 majör)** — Cmt 16:00 → Pzt sabahı\n"
                "- GBP/USD, EUR/USD, USD/JPY\n\n"
                "**Kripto** — 7/24 (Cmt 06:00–16:00 arası ~10 saat hariç)\n"
                "- Bitcoin, Ether ve diğer majör coinler, Crypto 10 endeksi\n\n"
                "**Açık olmayanlar:** tekil hisseler (Apple vb.), çoğu emtia, "
                "diğer forex pariteleri hafta sonu kapalıdır — son kapanışı gösterir."
            )

        rel = IG_MARKET_MAP.get(symbol.strip().upper())
        if rel is None:
            st.warning(
                f"⚠️ **{symbol}** IG'de hafta sonu işlemine sahip değil.\n\n"
                "IG hafta sonu yalnızca altın, bazı endeksler (Wall Street, "
                "US Tech 100, UK 100, Germany 40, Hong Kong HS50), 3 majör "
                "forex paritesi (GBP/USD, EUR/USD, USD/JPY) ve kriptolarda açık."
            )
            with st.expander("Hafta sonu desteklenen tickerlar"):
                st.write(", ".join(sorted(IG_MARKET_MAP.keys())))
        else:
            region = st.selectbox(
                "IG bölgesi", ["Otomatik", "za", "en", "ae"], index=0, key="ig_region"
            )
            if st.button("IG verisini çek", key="ig_fetch"):
                with st.spinner("IG sayfası çekiliyor..."):
                    if region == "Otomatik":
                        snap = fetch_ig_auto(rel)
                    else:
                        snap = fetch_ig_snapshot(rel, region)
                st.session_state["ig_snap"] = snap
                # Başarılı snapshot'ı kalıcı geçmişe kaydet
                if snap and not snap.get("error"):
                    ok, msg = save_ig_snapshot(snap, symbol.strip().upper())
                    st.session_state["ig_save_msg"] = (ok, msg)

            snap = st.session_state.get("ig_snap")
            if snap:
                if snap.get("error"):
                    st.error(snap["error"])
                    st.caption(f"Denenen URL: {snap.get('url','')}")
                else:
                    reg_used = snap.get("region_used")
                    reg_tag = f" · bölge: `{reg_used}`" if reg_used else ""
                    st.markdown(
                        f"**{snap.get('name') or symbol}** — "
                        f"[IG sayfası]({snap['url']}){reg_tag}"
                    )

                    # BUY / SELL
                    c1, c2 = st.columns(2)
                    c1.metric("SELL", snap["sell"] if snap["sell"] is not None else "—")
                    c2.metric("BUY",  snap["buy"]  if snap["buy"]  is not None else "—")

                    # Değişim — yeşil/kırmızı + belirgin yön etiketi
                    chg, pct = snap.get("change"), snap.get("change_pct")
                    if chg is not None:
                        if chg > 0:
                            color, sign, label = "#16a34a", "▲", "📈 YÜKSELİŞTE"
                        elif chg < 0:
                            color, sign, label = "#dc2626", "▼", "📉 DÜŞÜŞTE"
                        else:
                            color, sign, label = "#6b7280", "■", "➖ DEĞİŞİM YOK"
                        pct_s = f" ({pct:+.2f}%)" if pct is not None else ""
                        st.markdown(
                            f"<div style='font-size:1.3em;font-weight:700;color:{color}'>"
                            f"{sign} {chg:+.2f}{pct_s} &nbsp;·&nbsp; {label}</div>",
                            unsafe_allow_html=True,
                        )

                    # High / Low
                    h, l = snap.get("high"), snap.get("low")
                    if h is not None or l is not None:
                        st.caption(f"High: {h if h is not None else '—'}  |  "
                                   f"Low: {l if l is not None else '—'}")

                    # Sentiment — long/short
                    lp, sp = snap.get("long_pct"), snap.get("short_pct")
                    if lp is not None and sp is not None:
                        st.markdown("**Müşteri pozisyonları**")
                        st.markdown(
                            f"<div style='display:flex;border-radius:6px;overflow:hidden;"
                            f"font-size:0.9em;font-weight:600;color:white'>"
                            f"<div style='flex:{lp};background:#16a34a;padding:6px;"
                            f"text-align:center'>Long %{lp}</div>"
                            f"<div style='flex:{sp};background:#dc2626;padding:6px;"
                            f"text-align:center'>Short %{sp}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"IG müşteri hesaplarının %{lp}'i **long**, %{sp}'i **short**."
                        )
                    else:
                        st.caption("Sentiment (long/short) verisi bu sayfada bulunamadı.")

                    st.warning(
                        "IG fiyatları sayfa yüklendiği andaki snapshot'tır, canlı tick "
                        "değildir. Hafta içi kapalı enstrümanlarda değer eski olabilir."
                    )

                    # Kayıt durumu mesajı
                    save_msg = st.session_state.get("ig_save_msg")
                    if save_msg:
                        ok, msg = save_msg
                        if ok:
                            st.success(f"💾 {msg} Bu snapshot geçmişe eklendi.")
                        else:
                            st.info(f"💾 Kaydedilemedi: {msg}")

            # ── Geçmiş kayıtlar (tüm zaman) ──────────────────────────
            st.divider()
            st.markdown("### 📈 Geçmiş kayıtlar")
            hist = load_ig_history(symbol.strip().upper())

            if hist is None:
                st.info(
                    "Google Sheets bağlantısı bulunamadı. Geçmiş için secrets "
                    "ayarlarını ve `st-gsheets-connection` paketini kontrol et."
                )
            elif hist.empty:
                st.caption(
                    "Bu enstrüman için henüz kayıt yok. 'IG verisini çek' "
                    "butonuna bastıkça geçmiş birikir."
                )
            else:
                # Long/Short zaman serisi grafiği
                h2 = hist.dropna(subset=["long_pct", "short_pct"])
                if not h2.empty:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=h2["timestamp"], y=h2["long_pct"], mode="lines+markers",
                        name="Long %", line=dict(color="#16a34a", width=2)
                    ))
                    fig_h.add_trace(go.Scatter(
                        x=h2["timestamp"], y=h2["short_pct"], mode="lines+markers",
                        name="Short %", line=dict(color="#dc2626", width=2)
                    ))
                    fig_h.update_layout(
                        yaxis_title="% müşteri pozisyonu", xaxis_title="Tarih",
                        height=320, margin=dict(l=50, r=20, t=20, b=40),
                        hovermode="x unified", yaxis=dict(range=[0, 100]),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                    # En eski kaydı vurgula ("o zaman şöyleydi")
                    first = h2.iloc[0]
                    st.caption(
                        f"İlk kayıt ({first['timestamp']:%d.%m.%Y}): "
                        f"%{int(first['long_pct'])} long, %{int(first['short_pct'])} short."
                    )

                # Fiyat geçmişi (varsa)
                hp = hist.dropna(subset=["sell"])
                if not hp.empty:
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(
                        x=hp["timestamp"], y=hp["sell"], mode="lines+markers",
                        name="SELL", line=dict(color="#2563eb", width=2)
                    ))
                    fig_p.update_layout(
                        yaxis_title="Fiyat (SELL)", xaxis_title="Tarih",
                        height=280, margin=dict(l=50, r=20, t=20, b=40),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                # Tam tablo
                st.markdown("**Tüm kayıtlar**")
                show = hist.copy()
                show["timestamp"] = show["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                if "direction" in show.columns:
                    _dmap = {"up": "📈 yükseliş", "down": "📉 düşüş", "flat": "➖ sabit"}
                    show["direction"] = show["direction"].map(
                        lambda v: _dmap.get(str(v), "")
                    )
                cols = [c for c in IG_HISTORY_COLUMNS if c in show.columns]
                st.dataframe(
                    show[cols], use_container_width=True, hide_index=True
                )

                # ── Kayıt silme (PIN korumalı) ──
                with st.expander("🗑️ Kayıt sil"):
                    ts_options = hist["timestamp"].astype(str).tolist()
                    to_del = st.selectbox(
                        "Silinecek kayıt (timestamp):", ts_options, key="ig_del_sel"
                    )
                    pin_in = st.text_input(
                        "Silme PIN'i", type="password", key="ig_del_pin",
                        help="Silme işlemi için yetki gerekir."
                    )
                    if st.button("Seçili kaydı sil", key="ig_del_btn", type="secondary"):
                        real_pin = st.secrets.get("ig_delete_pin")
                        if not real_pin:
                            st.error("Silme PIN'i tanımlı değil (secrets: ig_delete_pin).")
                        elif pin_in != real_pin:
                            st.error("PIN hatalı. Silme yetkin yok.")
                        else:
                            ok, msg = delete_ig_row(symbol.strip().upper(), to_del)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

else:
    st.warning("Filtreleme sonrası veri kalmadı.")
