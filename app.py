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
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_bollinger(close, period=20, std_dev=2):
    sma     = close.rolling(window=period).mean()
    std     = close.rolling(window=period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    return bb_upper, bb_lower, (bb_upper - bb_lower) / sma

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    atr         = calc_atr(high, low, close, period)
    hl2         = (high + low) / 2
    upper_band  = hl2 + multiplier * atr
    lower_band  = hl2 - multiplier * atr
    supertrend  = pd.Series(np.nan, index=close.index)
    direction   = pd.Series(1, index=close.index)
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
interval   = INTERVAL_OPTIONS[selected_interval_label]
is_intraday = interval in ("1m", "2m", "5m", "15m", "30m", "1h")
max_days   = INTERVAL_MAX_DAYS.get(interval)

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

    close = df["Close"]; high = df["High"]; low = df["Low"]
    df["EMA_20"]    = calc_ema(close, 20)
    df["EMA_50"]    = calc_ema(close, 50)
    df["EMA_200"]   = calc_ema(close, 200)
    df["RSI"]       = calc_rsi(close)
    df["MACD"]      = calc_macd(close)[0]
    df["ATR"]       = calc_atr(high, low, close)
    df["BB_Upper"], df["BB_Lower"], df["BBW"] = calc_bollinger(close)
    df["Supertrend"] = calc_supertrend(high, low, close)
    df["Return"]     = np.log(close).diff()

    check_cols   = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    zero_or_null = int(((df[check_cols].isnull().any(axis=1)) | (df[check_cols] == 0).any(axis=1)).sum())
    consec_dupes = int((df[check_cols].eq(df[check_cols].shift(1)).all(axis=1)).sum())

    st.markdown(f"""
    <div class="info-box">
        <b>Seçilen aralıktaki {bar_label} sayısı:</b> {len(df):,}<br>
        <b>OHLCV'de boş veya 0 değer taşıyan satır sayısı:</b> {zero_or_null:,}<br>
        <b>Arka arkaya aynı OHLCV satır sayısı:</b> {consec_dupes:,}
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # Veri Seçimi
    # ============================================================

    st.subheader("Veri Seçimi")
    available_cols = list(df.columns)
    st.caption("İndirmek istediğiniz verileri seçin:")
    selected_cols = []
    cols_per_row  = 4
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
            name="Kapanış"
        ))
        fig_px.update_layout(
            title="",
            xaxis=dict(
                title="Tarih",
                rangeslider=dict(visible=True, thickness=0.07),
                rangeselector=dict(
                    buttons=[
                        dict(count=1,  label="1A",  step="month", stepmode="backward"),
                        dict(count=3,  label="3A",  step="month", stepmode="backward"),
                        dict(count=6,  label="6A",  step="month", stepmode="backward"),
                        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                        dict(step="all", label="Tümü"),
                    ],
                    bgcolor="#f0f2f6", activecolor="#0d6efd",
                )
            ),
            yaxis=dict(title="Kapanış", fixedrange=False),
            hovermode="x unified",
            height=420,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_px, use_container_width=True)

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
        alpha = st.slider(
            "Anlamlılık Eşiği (α)",
            min_value=0.01, max_value=0.10, value=0.05, step=0.01,
            help="Bu değerin altındaki p-değerleri istatistiksel olarak anlamlı kabul edilir."
        )

        if st.button("Spearman Korelasyonunu Hesapla"):
            sub = df[numeric_cols].dropna()
            n   = len(sub)
            k   = len(numeric_cols)

            # Spearman rho ve p matrisleri
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

        # Gösterim
        if "spearman_rho" in st.session_state:
            rho_mat = st.session_state["spearman_rho"]
            p_mat   = st.session_state["spearman_p"]
            n       = st.session_state["spearman_n"]
            alpha   = st.session_state["spearman_alpha"]
            numeric_cols = st.session_state["spearman_cols"]

            # --- Isı Haritası ---
            st.markdown("### Korelasyon Isı Haritası")
            fig2, ax2 = plt.subplots(figsize=(max(6, k * 0.9), max(5, k * 0.8)))
            cmap   = plt.cm.RdYlGn
            im     = ax2.imshow(rho_mat.values.astype(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax2, shrink=0.8, label="Spearman ρ")
            ax2.set_xticks(range(k)); ax2.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
            ax2.set_yticks(range(k)); ax2.set_yticklabels(numeric_cols, fontsize=9)
            ax2.set_title(f"{symbol} — Spearman Korelasyon Matrisi (n={n})", fontsize=11, fontweight="bold")

            for i in range(k):
                for j in range(k):
                    val = rho_mat.values[i, j]
                    if not np.isnan(val):
                        sig = (i != j) and (p_mat.values[i, j] < alpha)
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

            # --- Anlamlı Çiftler Tablosu ---
            st.markdown(f"### Anlamlı Korelasyonlar (p < {alpha})")
            pairs = []
            seen  = set()
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
                                "Çok Güçlü"  if abs(rho) >= 0.80 else
                                "Güçlü"      if abs(rho) >= 0.60 else
                                "Orta"       if abs(rho) >= 0.40 else
                                "Zayıf"
                            ),
                        })

            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("ρ", key=abs, ascending=False)

                def color_rho(val):
                    if not isinstance(val, float): return ""
                    c = plt.cm.RdYlGn((val + 1) / 2)
                    hex_c = mcolors.to_hex(c)
                    text  = "black" if abs(val) < 0.5 else "white"
                    return f"background-color: {hex_c}; color: {text}; font-weight: bold"

                st.dataframe(
                    pairs_df.style.applymap(color_rho, subset=["ρ"]),
                    use_container_width=True, hide_index=True
                )

                # Excel indirme
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

            # --- Özet ---
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
