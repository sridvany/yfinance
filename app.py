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
import warnings

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
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    return bb_upper, bb_lower, (bb_upper - bb_lower) / sma

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = pd.Series(np.nan, index=close.index)
    direction  = pd.Series(1, index=close.index)
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
            st.error(f"'{symbol}' için veri bulunamadı.")
            st.stop()
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

    hist_max.index = hist_max.index.tz_localize(None)
    oldest_date = hist_max.index.min().date()
    newest_date = hist_max.index.max().date()

    st.markdown(f"""
    <div class="info-box">
        <b>En eski veri tarihi:</b> {oldest_date}<br>
        <b>En yeni veri tarihi:</b> {newest_date}<br>
        <b>Toplam veri günü:</b> {len(hist_max):,}
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

    close = df["Close"]; high = df["High"]; low = df["Low"]
    df["EMA_20"] = calc_ema(close, 20)
    df["EMA_50"] = calc_ema(close, 50)
    df["EMA_200"] = calc_ema(close, 200)
    df["RSI"] = calc_rsi(close)
    df["MACD"] = calc_macd(close)[0]
    df["ATR"] = calc_atr(high, low, close)
    df["BB_Upper"], df["BB_Lower"], df["BBW"] = calc_bollinger(close)
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
    available_cols = list(df.columns)
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
        ax.set_xlabel("Tarih"); ax.set_ylabel("Kapanış"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        st.markdown("*Görsele sağ tıklayıp kopyalayabilirsiniz:*")
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" style="width:100%; border-radius:8px;" />',
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
    # GRUP / FONKSİYON TANIMLARI (durağanlık + VIF için ortak)
    # ============================================================

    PRICE_LIKE  = {"Open", "High", "Low", "Close",
                   "EMA_20", "EMA_50", "EMA_200",
                   "BB_Upper", "BB_Lower", "Supertrend"}
    VOLUME_LIKE = {"Volume"}
    VOL_MEASURE = {"ATR", "BBW"}
    OSCILLATORS = {"RSI", "MACD"}
    ALL_TEST    = PRICE_LIKE | VOLUME_LIKE | VOL_MEASURE | OSCILLATORS

    from statsmodels.tsa.stattools import adfuller, kpss

    def run_adf(series):
        try:
            clean = series.dropna()
            if len(clean) < 20: return None, None
            r = adfuller(clean, autolag='AIC')
            return r[1], r[0]
        except: return None, None

    def run_kpss(series):
        try:
            clean = series.dropna()
            if len(clean) < 20: return None, None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = kpss(clean, regression='c', nlags='auto')
            return r[1], r[0]
        except: return None, None

    def run_dfgls(series, price_like=False):
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools import add_constant
            t = np.arange(len(series))
            X = add_constant(np.column_stack([t])) if price_like else add_constant(np.ones(len(series)))
            resid = series.values - OLS(series.values, X).fit().fittedvalues
            r = adfuller(resid, autolag='AIC', regression='n')
            return r[1]
        except: return None

    def run_pp(series):
        try:
            from statsmodels.tsa.stattools import PhillipsPerron
            clean = series.dropna()
            if len(clean) < 20: return None
            return PhillipsPerron(clean).pvalue
        except:
            try:
                return adfuller(series.dropna(), autolag='AIC', regression='ct')[1]
            except: return None

    def is_stat(adf_p, kpss_p):
        if adf_p is None or kpss_p is None: return None
        if adf_p < 0.05 and kpss_p >= 0.05: return True
        if adf_p >= 0.05 and kpss_p < 0.05: return False
        return None

    def get_transform(col, series):
        if col in PRICE_LIKE:
            return np.log(series.replace(0, np.nan)).diff().dropna(), "Log Return [ln(Pt/Pt-1)]"
        elif col in VOLUME_LIKE:
            return np.log(series.replace(0, np.nan)).diff().dropna(), "Log Fark"
        elif col in VOL_MEASURE:
            log_s = np.log(series.replace(0, np.nan)).dropna()
            if is_stat(*run_adf(log_s)[:1] + run_kpss(log_s)[:1]):
                return log_s, "Log Dönüşümü"
            return log_s.diff().dropna(), "Log + Fark"
        else:
            return series.diff().dropna(), "Birinci Fark"

    def resolve_stat(series, price_like_flag):
        """ADF+KPSS → DF-GLS → PP → muhafazakar hiyerarşisiyle karar ver."""
        adf_p, _ = run_adf(series)
        kpss_p, _ = run_kpss(series)
        stat = is_stat(adf_p, kpss_p)
        dfgls_note = pp_note = "—"
        karar = "ADF+KPSS"

        if stat is None:
            p = run_dfgls(series, price_like=price_like_flag)
            if p is not None:
                dfgls_note = f"p={p:.3f}"
                stat = p < 0.05
                karar = "DF-GLS"

        if stat is None:
            p = run_pp(series)
            if p is not None:
                pp_note = f"p={p:.3f}"
                stat = p < 0.05
                karar = "PP"

        if stat is None:
            stat = False
            karar = "Muhafazakar"

        return stat, adf_p, kpss_p, dfgls_note, pp_note, karar

    test_cols = [c for c in df.columns if c in ALL_TEST and c in selected_cols]

    # ============================================================
    # DURAĞANLIK ANALİZİ
    # ============================================================

    st.divider()
    st.subheader("🔬 Durağanlık Analizi")
    st.caption(
        "Her değişken için ADF → KPSS → DF-GLS → PP hiyerarşisi uygulanır. "
        "Durağan olmayan seriler için otomatik dönüşüm yapılır."
    )

    if st.button("Durağanlık Testlerini Çalıştır"):
        results = []
        progress = st.progress(0, text="Testler çalışıyor...")

        for idx, col in enumerate(test_cols):
            series = df[col].dropna()
            price_like_flag = col in PRICE_LIKE
            stat, adf_p, kpss_p, dfgls_note, pp_note, karar = resolve_stat(series, price_like_flag)

            transform_label = transform_verdict = "—"
            if not stat:
                transformed, transform_label = get_transform(col, series)
                t_stat, t_adf_p, t_kpss_p, _, _, _ = resolve_stat(transformed, False)
                transform_verdict = "Durağan ✅" if t_stat else "Hâlâ Durağan Değil ⚠️"
                ham_sonuc = "Durağan Değil ❌"; ham_color = "red"
            else:
                ham_sonuc = "Durağan ✅"; ham_color = "green"

            results.append({
                "Değişken":          col,
                "ADF p":             f"{adf_p:.3f}" if adf_p is not None else "—",
                "KPSS p":            f"{kpss_p:.3f}" if kpss_p is not None else "—",
                "DF-GLS p":          dfgls_note,
                "PP p":              pp_note,
                "Karar Dayanağı":    karar,
                "Ham Sonuç":         ham_sonuc,
                "Önerilen Dönüşüm":  transform_label,
                "Dönüşüm Sonrası":   transform_verdict,
                "_color":            ham_color,
            })
            progress.progress((idx + 1) / len(test_cols), text=f"Test ediliyor: {col}")

        progress.empty()
        st.session_state["stat_results"]   = results
        st.session_state["stat_test_cols"] = test_cols

    # Sonuçları göster (session_state'ten)
    if "stat_results" in st.session_state:
        results   = st.session_state["stat_results"]
        test_cols = st.session_state["stat_test_cols"]

        stationary    = [r for r in results if r["_color"] == "green"]
        nonstationary = [r for r in results if r["_color"] == "red"]

        if stationary:
            st.markdown("### ✅ Ham Haliyle Durağan")
            st.dataframe(pd.DataFrame([{
                "Değişken": r["Değişken"], "ADF p": r["ADF p"], "KPSS p": r["KPSS p"],
                "DF-GLS p": r["DF-GLS p"], "PP p": r["PP p"],
                "Karar Dayanağı": r["Karar Dayanağı"], "Sonuç": r["Ham Sonuç"],
            } for r in stationary]), use_container_width=True, hide_index=True)

        if nonstationary:
            st.markdown("### ❌ Durağan Değil → Dönüşüm Uygulandı")
            st.dataframe(pd.DataFrame([{
                "Değişken": r["Değişken"], "Karar Dayanağı": r["Karar Dayanağı"],
                "Önerilen Dönüşüm": r["Önerilen Dönüşüm"], "Dönüşüm Sonrası": r["Dönüşüm Sonrası"],
            } for r in nonstationary]), use_container_width=True, hide_index=True)
            still_bad = [r["Değişken"] for r in nonstationary if "Hâlâ" in r["Dönüşüm Sonrası"]]
            if still_bad:
                st.warning(f"**{', '.join(still_bad)}** dönüşüm sonrası hâlâ durağan değil. "
                           "Lee-Strazicich veya ikinci fark önerilebilir.")

        n_cons = sum(1 for r in results if r["Karar Dayanağı"] == "Muhafazakar")
        n_fixed = sum(1 for r in nonstationary if r["Dönüşüm Sonrası"] == "Durağan ✅")
        st.markdown(f"""
        <div class="info-box">
            <b>Özet:</b> {len(results)} değişken test edildi. &nbsp;
            ✅ Ham durağan: <b>{len(stationary)}</b> &nbsp;|&nbsp;
            ❌ Durağan değil: <b>{len(nonstationary)}</b> (dönüşümle düzeltilen: <b>{n_fixed}</b>)
            {"&nbsp;|&nbsp; ⚠️ Muhafazakar karar: <b>" + str(n_cons) + "</b>" if n_cons > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 Test Metodolojisi & Karar Hiyerarşisi"):
            st.markdown("""
**Karar hiyerarşisi:** ADF+KPSS → DF-GLS → PP → Muhafazakar (durağan değil say)

| Grup | Dönüşüm |
|------|---------|
| Fiyat bazlı (Close, EMA'lar vb.) | Log Return |
| Hacim | Log Fark |
| Volatilite (ATR, BBW) | Log → yetmezse Log+Fark |
| Osilatörler (RSI, MACD vb.) | Birinci Fark |
            """)

        # ============================================================
        # VIF ANALİZİ
        # ============================================================

        st.divider()
        st.subheader("📐 VIF Analizi (Çoklu Doğrusallık)")
        st.caption("Ham ve dönüştürülmüş seriler için ayrı ayrı VIF hesaplanır.")

        vif_threshold = st.slider("VIF Eşiği", min_value=2, max_value=20, value=10, step=1,
                                  help="Bu eşiğin üzerindeki değişkenler çoklu doğrusallık içeriyor.")

        if st.button("VIF Analizini Çalıştır"):
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            def calc_vif(data_df):
                clean = data_df.dropna()
                if clean.shape[0] < clean.shape[1] + 1:
                    return None
                rows_out = []
                for i in range(clean.shape[1]):
                    try:    vif_val = variance_inflation_factor(clean.values, i)
                    except: vif_val = np.nan
                    rows_out.append({"Değişken": clean.columns[i], "VIF": round(vif_val, 2)})
                return pd.DataFrame(rows_out)

            def style_vif(val):
                if not isinstance(val, float): return ""
                if val > vif_threshold:         return "color: red; font-weight: bold"
                if val > vif_threshold / 2:     return "color: orange"
                return ""

            

            # Dönüştürülmüş seriler
            st.markdown("### Dönüştürülmüş (Durağan) Seriler")
            trans_dict = {}
            for col in test_cols:
                r = next((x for x in results if x["Değişken"] == col), None)
                if r is None: continue
                series = df[col].dropna()
                if r["Önerilen Dönüşüm"] == "—":
                    trans_dict[col] = series
                else:
                    t_series, _ = get_transform(col, series)
                    trans_dict[col] = t_series

            vif_trans = calc_vif(pd.DataFrame(trans_dict))
            if vif_trans is not None:
                high = vif_trans[vif_trans["VIF"] > vif_threshold]
                st.dataframe(vif_trans.style.applymap(style_vif, subset=["VIF"]),
                             use_container_width=True, hide_index=True)
                if not high.empty:
                    st.warning(f"VIF > {vif_threshold}: **{', '.join(high['Değişken'].tolist())}** "
                               "— dönüşüm sonrası hâlâ yüksek çoklu doğrusallık. PCA önerilebilir.")
                else:
                    st.success(f"Dönüştürülmüş serilerde tüm VIF ≤ {vif_threshold}.")
            else:
                st.warning("Yeterli gözlem yok.")

            with st.expander("📖 VIF Yorumlama Rehberi"):
                st.markdown(f"""
| VIF | Yorum |
|-----|-------|
| 1 | Çoklu doğrusallık yok |
| 1 – {vif_threshold//2} | Kabul edilebilir |
| {vif_threshold//2} – {vif_threshold} | Orta düzey, dikkat |
| > {vif_threshold} | Yüksek — sorun var |

**Çözüm:** Yüksek korelasyonlu değişkeni çıkar · PCA uygula · Ridge/Lasso kullan
                """)
