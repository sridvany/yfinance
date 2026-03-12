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
    st.caption(
        "Her değişken için ADF ve KPSS testleri uygulanır. "
        "Çelişkili sonuçlarda DF-GLS ile karar verilir. "
        "Durağan olmayan seriler için otomatik dönüşüm uygulanır ve sonuç doğrulanır."
    )

    run_tests = st.button("Durağanlık Testlerini Çalıştır")

    if run_tests:
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            st.error("statsmodels kütüphanesi gerekli: pip install statsmodels")
            st.stop()

        import warnings

        # ----------------------------------------------------------
        # Yardımcı Test Fonksiyonları
        # ----------------------------------------------------------

        def run_adf(series):
            try:
                clean = series.dropna()
                if len(clean) < 20:
                    return None, None
                result = adfuller(clean, autolag='AIC')
                return result[1], result[0]
            except Exception:
                return None, None

        def run_kpss(series):
            try:
                clean = series.dropna()
                if len(clean) < 20:
                    return None, None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = kpss(clean, regression='c', nlags='auto')
                return result[1], result[0]
            except Exception:
                return None, None

        def dfgls_test(series, price_like=False):
            """GLS-detrended seriye ADF uygular (DF-GLS proxy)"""
            try:
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant
                t = np.arange(len(series))
                if price_like:
                    X = add_constant(np.column_stack([t]))
                else:
                    X = add_constant(np.ones(len(series)))
                resid = series.values - OLS(series.values, X).fit().fittedvalues
                result = adfuller(resid, autolag='AIC', regression='n')
                return result[1]
            except Exception:
                return None

        def run_pp(series):
            """Phillips-Perron testi"""
            try:
                from statsmodels.tsa.stattools import PhillipsPerron
                clean = series.dropna()
                if len(clean) < 20:
                    return None
                result = PhillipsPerron(clean)
                return result.pvalue
            except Exception:
                try:
                    clean = series.dropna()
                    result = adfuller(clean, autolag='AIC', regression='ct')
                    return result[1]
                except Exception:
                    return None

        def is_stationary(adf_p, kpss_p):
            """True=durağan, False=değil, None=çelişkili"""
            if adf_p is None or kpss_p is None:
                return None
            adf_ok = adf_p < 0.05
            kpss_ok = kpss_p >= 0.05
            if adf_ok and kpss_ok:
                return True
            if not adf_ok and not kpss_ok:
                return False
            return None  # çelişkili

        # ----------------------------------------------------------
        # Dönüşüm Kuralları (değişken tipine göre)
        # ----------------------------------------------------------

        # Grup tanımları
        PRICE_LIKE   = {"Open", "High", "Low", "Close",
                        "EMA_20", "EMA_50", "EMA_200",
                        "BB_Upper", "BB_Lower", "Supertrend"}
        VOLUME_LIKE  = {"Volume"}
        VOL_MEASURE  = {"ATR", "BBW"}          # volatilite ölçüleri
        OSCILLATORS  = {"RSI", "MACD", "MACD_Signal", "MACD_Hist"}

        def get_transform(col, series):
            """
            Durağan olmayan seri için dönüşüm uygular.
            (col_name, transformed_series, transform_label) döndürür.
            """
            if col in PRICE_LIKE:
                # log return
                log_s = np.log(series.replace(0, np.nan))
                transformed = log_s.diff().dropna()
                label = "Log Return [ln(Pt/Pt-1)]"
            elif col in VOLUME_LIKE:
                # log fark
                log_s = np.log(series.replace(0, np.nan))
                transformed = log_s.diff().dropna()
                label = "Log Fark"
            elif col in VOL_MEASURE:
                # önce log; hâlâ birim kök varsa fark al
                log_s = np.log(series.replace(0, np.nan)).dropna()
                adf_p2, _ = run_adf(log_s)
                kpss_p2, _ = run_kpss(log_s)
                if is_stationary(adf_p2, kpss_p2):
                    transformed = log_s
                    label = "Log Dönüşümü"
                else:
                    transformed = log_s.diff().dropna()
                    label = "Log + Fark"
            elif col in OSCILLATORS:
                # birinci fark
                transformed = series.diff().dropna()
                label = "Birinci Fark"
            else:
                transformed = series.diff().dropna()
                label = "Birinci Fark"
            return transformed, label

        # ----------------------------------------------------------
        # Ana Test Döngüsü
        # ----------------------------------------------------------

        test_cols = [c for c in df.columns if c in
                     PRICE_LIKE | VOLUME_LIKE | VOL_MEASURE | OSCILLATORS]

        results = []
        progress = st.progress(0, text="Testler çalışıyor...")

        for idx, col in enumerate(test_cols):
            series = df[col].dropna()

            # --- Ham seri testleri ---
            adf_p, _ = run_adf(series)
            kpss_p, _ = run_kpss(series)
            stat = is_stationary(adf_p, kpss_p)

            price_like_flag = col in PRICE_LIKE
            dfgls_note = "—"
            pp_note = "—"
            karar_notu = "ADF+KPSS"
            transform_label = "—"
            transform_verdict = "—"

            # Adım 1: Çelişkili → DF-GLS
            if stat is None:
                dfgls_p = dfgls_test(series, price_like=price_like_flag)
                if dfgls_p is not None:
                    dfgls_note = f"p={dfgls_p:.3f}"
                    stat = dfgls_p < 0.05
                    karar_notu = "DF-GLS"

            # Adım 2: Hâlâ belirsiz → PP testi
            if stat is None:
                pp_p = run_pp(series)
                if pp_p is not None:
                    pp_note = f"p={pp_p:.3f}"
                    stat = pp_p < 0.05
                    karar_notu = "PP"

            # Adım 3: Hâlâ belirsiz → muhafazakar: durağan değil say
            if stat is None:
                stat = False
                karar_notu = "Muhafazakar (konsensüs yok)"

            # Ham seri durağan değilse → dönüşüm uygula
            if stat is False:
                transformed, transform_label = get_transform(col, series)
                t_adf_p, _ = run_adf(transformed)
                t_kpss_p, _ = run_kpss(transformed)
                t_stat = is_stationary(t_adf_p, t_kpss_p)

                # Dönüşüm sonrası çelişkili → DF-GLS
                if t_stat is None:
                    t_dfgls_p = dfgls_test(transformed, price_like=False)
                    if t_dfgls_p is not None:
                        t_stat = t_dfgls_p < 0.05

                # Hâlâ belirsiz → PP
                if t_stat is None:
                    t_pp_p = run_pp(transformed)
                    if t_pp_p is not None:
                        t_stat = t_pp_p < 0.05

                # Son çare → muhafazakar
                if t_stat is None:
                    t_stat = False

                if t_stat is True:
                    transform_verdict = "Durağan ✅"
                else:
                    transform_verdict = "Hâlâ Durağan Değil ⚠️"

                ham_sonuc = "Durağan Değil ❌"
                ham_color = "red"
            else:
                ham_sonuc = "Durağan ✅"
                ham_color = "green"

            results.append({
                "Değişken": col,
                "ADF p": f"{adf_p:.3f}" if adf_p is not None else "—",
                "KPSS p": f"{kpss_p:.3f}" if kpss_p is not None else "—",
                "DF-GLS p": dfgls_note,
                "PP p": pp_note,
                "Karar Dayanağı": karar_notu,
                "Ham Sonuç": ham_sonuc,
                "Önerilen Dönüşüm": transform_label,
                "Dönüşüm Sonrası": transform_verdict,
                "_color": ham_color,
            })

            progress.progress((idx + 1) / len(test_cols), text=f"Test ediliyor: {col}")

        progress.empty()

        # ----------------------------------------------------------
        # Sonuç Tabloları
        # ----------------------------------------------------------

        stationary    = [r for r in results if r["_color"] == "green"]
        nonstationary = [r for r in results if r["_color"] == "red"]

        def render_ham_table(rows):
            display = pd.DataFrame([{
                "Değişken":        r["Değişken"],
                "ADF p":           r["ADF p"],
                "KPSS p":          r["KPSS p"],
                "DF-GLS p":        r["DF-GLS p"],
                "PP p":            r["PP p"],
                "Karar Dayanağı":  r["Karar Dayanağı"],
                "Sonuç":           r["Ham Sonuç"],
            } for r in rows])
            st.dataframe(display, use_container_width=True, hide_index=True)

        def render_transform_table(rows):
            display = pd.DataFrame([{
                "Değişken":          r["Değişken"],
                "Karar Dayanağı":    r["Karar Dayanağı"],
                "Önerilen Dönüşüm":  r["Önerilen Dönüşüm"],
                "Dönüşüm Sonrası":   r["Dönüşüm Sonrası"],
            } for r in rows])
            st.dataframe(display, use_container_width=True, hide_index=True)

        if stationary:
            st.markdown("### ✅ Ham Haliyle Durağan")
            render_ham_table(stationary)

        if nonstationary:
            st.markdown("### ❌ Durağan Değil → Dönüşüm Uygulandı")
            render_transform_table(nonstationary)
            still_bad = [r for r in nonstationary if "Hâlâ" in r["Dönüşüm Sonrası"]]
            if still_bad:
                cols_bad = ", ".join(r["Değişken"] for r in still_bad)
                st.warning(
                    f"**{cols_bad}** dönüşüm sonrası hâlâ durağan değil. "
                    "Yapısal kırılma testi (Lee-Strazicich) veya ikinci fark önerilebilir."
                )

        st.markdown("---")
        total   = len(results)
        n_stat  = len(stationary)
        n_non   = len(nonstationary)
        n_fixed = sum(1 for r in nonstationary if r["Dönüşüm Sonrası"] == "Durağan ✅")
        n_cons  = sum(1 for r in results if r["Karar Dayanağı"] == "Muhafazakar (konsensüs yok)")

        st.markdown(f"""
        <div class="info-box">
            <b>Özet:</b> {total} değişken test edildi.<br>
            ✅ Ham durağan: <b>{n_stat}</b> &nbsp;|&nbsp;
            ❌ Durağan değil: <b>{n_non}</b>
            (dönüşümle düzeltilen: <b>{n_fixed}</b>)<br>
            ⚠️ Muhafazakar kararla işlenen: <b>{n_cons}</b>
            {"— tüm testler çelişti, sahte regresyon riskine karşı dönüşüm uygulandı." if n_cons > 0 else ""}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 Test Metodolojisi & Karar Hiyerarşisi"):
            st.markdown("""
**Test Hiyerarşisi (her değişken için sırayla uygulanır):**
1. **ADF + KPSS** → ikisi aynı yönde → karar verilir
2. **DF-GLS** → ADF/KPSS çelişirse devreye girer
3. **PP (Phillips-Perron)** → DF-GLS sonrası hâlâ belirsizse uygulanır
4. **Muhafazakar yaklaşım** → üç test de konsensüs sağlayamazsa "durağan değil" kabul edilir

**Dönüşüm Kuralları:**
| Grup | Değişkenler | Uygulanan Dönüşüm |
|------|-------------|-------------------|
| Fiyat bazlı | Close, Open, High, Low, EMA'lar, BB bantları, Supertrend | Log Return |
| Hacim | Volume | Log Fark |
| Volatilite | ATR, BBW | Log → yetmezse Log+Fark |
| Osilatörler | RSI, MACD, MACD_Signal, MACD_Hist | Birinci Fark |
            """)

        # ============================================================
        # VIF ANALİZİ
        # ============================================================

        st.divider()
        st.subheader("📐 VIF Analizi (Çoklu Doğrusallık)")
        st.caption(
            "Ham seriler ve dönüştürülmüş seriler için ayrı ayrı VIF hesaplanır. "
            "Yüksek VIF değerleri değişkenler arasındaki çoklu doğrusallığa işaret eder."
        )

        vif_threshold = st.slider(
            "VIF Eşiği", min_value=2, max_value=20, value=10, step=1,
            help="Bu eşiğin üzerindeki değişkenler yüksek çoklu doğrusallık içeriyor demektir."
        )

        run_vif = st.button("VIF Analizini Çalıştır")

        if run_vif:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
            except ImportError:
                st.error("statsmodels kütüphanesi gerekli: pip install statsmodels")
                st.stop()

            def calc_vif(data_df):
                """
                NaN içermeyen tam satırlar üzerinde VIF hesaplar.
                Her değişken için VIF döndürür.
                """
                clean = data_df.dropna()
                if clean.shape[0] < clean.shape[1] + 1:
                    return None  # yeterli gözlem yok
                vif_data = []
                for i in range(clean.shape[1]):
                    try:
                        vif_val = variance_inflation_factor(clean.values, i)
                    except Exception:
                        vif_val = np.nan
                    vif_data.append({
                        "Değişken": clean.columns[i],
                        "VIF": round(vif_val, 2)
                    })
                return pd.DataFrame(vif_data)

            def style_vif(val, threshold):
                if isinstance(val, float):
                    if val > threshold:
                        return "color: red; font-weight: bold"
                    elif val > threshold / 2:
                        return "color: orange"
                return ""

            # ----------------------------------------------------------
            # Ham Seriler VIF
            # ----------------------------------------------------------
            st.markdown("### Ham Seriler")

            ham_cols = [c for c in test_cols if c in df.columns]
            ham_df = df[ham_cols].copy()
            vif_ham = calc_vif(ham_df)

            if vif_ham is not None:
                high_ham = vif_ham[vif_ham["VIF"] > vif_threshold]
                st.dataframe(
                    vif_ham.style.applymap(
                        lambda v: style_vif(v, vif_threshold), subset=["VIF"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                if not high_ham.empty:
                    cols_high = ", ".join(high_ham["Değişken"].tolist())
                    st.warning(
                        f"VIF > {vif_threshold}: **{cols_high}** — yüksek çoklu doğrusallık. "
                        "PCA veya değişken çıkarma düşünülebilir."
                    )
                else:
                    st.success(f"Tüm ham değişkenlerde VIF ≤ {vif_threshold}.")
            else:
                st.warning("Ham seriler için yeterli gözlem yok.")

            # ----------------------------------------------------------
            # Dönüştürülmüş Seriler VIF
            # ----------------------------------------------------------
            st.markdown("### Dönüştürülmüş (Durağan) Seriler")

            # Dönüştürülmüş serileri yeniden üret
            transformed_dict = {}
            for col in test_cols:
                series = df[col].dropna()
                # Durağanlık kararını tekrar belirle (results listesinden al)
                r = next((x for x in results if x["Değişken"] == col), None)
                if r is None:
                    continue
                if r["Önerilen Dönüşüm"] == "—":
                    # Ham haliyle durağan → direkt kullan
                    transformed_dict[col] = series
                else:
                    t_series, _ = get_transform(col, series)
                    transformed_dict[col] = t_series

            # Ortak index için hizala
            trans_df = pd.DataFrame(transformed_dict)
            vif_trans = calc_vif(trans_df)

            if vif_trans is not None:
                high_trans = vif_trans[vif_trans["VIF"] > vif_threshold]
                st.dataframe(
                    vif_trans.style.applymap(
                        lambda v: style_vif(v, vif_threshold), subset=["VIF"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                if not high_trans.empty:
                    cols_high = ", ".join(high_trans["Değişken"].tolist())
                    st.warning(
                        f"VIF > {vif_threshold}: **{cols_high}** — dönüşüm sonrası hâlâ yüksek çoklu doğrusallık. "
                        "Bu değişkenler aynı ekonomik fenomeni ölçüyor olabilir. "
                        "PCA ile tek bileşene indirgenmesi önerilebilir."
                    )
                else:
                    st.success(f"Dönüştürülmüş serilerde tüm VIF değerleri ≤ {vif_threshold}.")
            else:
                st.warning("Dönüştürülmüş seriler için yeterli gözlem yok.")

            with st.expander("📖 VIF Yorumlama Rehberi"):
                st.markdown(f"""
**VIF (Variance Inflation Factor) Nedir?**
Bir değişkenin diğer bağımsız değişkenler tarafından ne ölçüde açıklandığını gösterir.

| VIF Değeri | Yorum |
|------------|-------|
| 1 | Çoklu doğrusallık yok |
| 1 – {vif_threshold//2} | Kabul edilebilir |
| {vif_threshold//2} – {vif_threshold} | Orta düzey, dikkat |
| > {vif_threshold} | Yüksek — sorun var |

**Çözüm Seçenekleri:**
- Birbiriyle yüksek korelasyonlu değişkenlerden birini çıkar
- PCA ile boyut indirgeme yap (yorumlanabilirlik kaybı göze alınarak)
- Ridge/Lasso regresyon kullan (ceza terimi çoklu doğrusallığı bastırır)
                """)
