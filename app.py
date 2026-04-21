 import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from itertools import product as iter_product
import requests
import json
import hashlib
import time

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="tahmin.ai", layout="wide")

auto_refresh_on = st.sidebar.toggle("🔄 Canlı Yenileme", value=True)
if auto_refresh_on:
    st_autorefresh(interval=55 * 1000, key="terminal_refresh")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    div[data-testid="stCaption"] { margin-top: -0.5rem; margin-bottom: -0.5rem; }
    h1 { margin-bottom: 0 !important; padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 BORSA TERMİNALİ")
st.caption("YATIRIM TAVSİYESİ İÇERMEZ. ARAŞTIRMA İÇİNDİR.")

# ============================================================
# SESSION STATE VARSAYILANLARI
# ============================================================
_defaults = {
    "sma_short":     20,
    "sma_long":      200,
    "rsi_period":    14,
    "rsi_lower":     30,
    "rsi_upper":     70,
    "bb_period":     20,
    "bb_std":        2.0,
    "macd_fast":     12,
    "macd_slow":     26,
    "macd_signal":   9,
    "z_period":      30,
    "z_thresh":      2.0,
    "adx_period":    14,
    "adx_threshold": 25,
    "st_period":     10,
    "st_multiplier": 3.0,
    "lrc_period":    50,
    "lrc_std_mult":  2.0,
    "wt_n1":         10,
    "wt_n2":         21,
    "obv_short":     10,
    "obv_long":      30,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 🤖 LLM PROVIDER KONFİGÜRASYONU VE AKIŞ FONKSİYONLARI
# ============================================================
LLM_PROVIDERS = {
    "Google": {
        "models":   ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "type":     "gemini",
        "key_url":  "https://aistudio.google.com/apikey",
    },
    "OpenAI": {
        "models":   ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"],
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "type":     "openai",
        "key_url":  "https://platform.openai.com/api-keys",
    },
    "Anthropic": {
        "models":   ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-sonnet-4-6"],
        "endpoint": "https://api.anthropic.com/v1/messages",
        "type":     "anthropic",
        "key_url":  "https://console.anthropic.com/settings/keys",
    },
    "DeepSeek": {
        "models":   ["deepseek-chat", "deepseek-reasoner"],
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "type":     "openai",
        "key_url":  "https://platform.deepseek.com/api_keys",
    },
    "Groq": {
        "models":   ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile",
                     "mixtral-8x7b-32768", "llama-3.1-8b-instant"],
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "type":     "openai",
        "key_url":  "https://console.groq.com/keys",
    },
}

AI_DETAIL_LEVELS = {"Kısa": 1500, "Orta": 4000, "Detaylı": 8000}


def _stream_openai_compat(endpoint, api_key, model, messages, max_tokens, provider_name="OpenAI"):
    """OpenAI-uyumlu streaming (OpenAI, DeepSeek, Groq)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # OpenAI'de gpt-5 / o-serisi modelleri 'max_tokens' yerine 'max_completion_tokens' ister
    token_param = "max_tokens"
    if provider_name == "OpenAI" and any(
        model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
    ):
        token_param = "max_completion_tokens"

    payload = {
        "model": model, "messages": messages, "stream": True,
        token_param: max_tokens, "temperature": 0.4,
    }
    r = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=90)
    try:
        if r.status_code != 200:
            try:
                err_body = r.text
            except Exception:
                err_body = "(yanıt okunamadı)"
            err_msg = err_body[:500]
            try:
                err_json = json.loads(err_body)
                if isinstance(err_json, dict) and "error" in err_json:
                    err_detail = err_json["error"]
                    if isinstance(err_detail, dict):
                        err_msg = err_detail.get("message", err_msg)
                    else:
                        err_msg = str(err_detail)
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            raise RuntimeError(f"HTTP {r.status_code}: {err_msg}")

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        r.close()


def _stream_anthropic(api_key, model, system_prompt, user_prompt, max_tokens):
    """Anthropic Messages API streaming."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model, "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": True, "temperature": 0.4,
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=payload, stream=True, timeout=90)
    try:
        if r.status_code != 200:
            try:
                err_body = r.text
            except Exception:
                err_body = "(yanıt okunamadı)"
            err_msg = err_body[:500]
            try:
                err_json = json.loads(err_body)
                if isinstance(err_json, dict) and "error" in err_json:
                    err_detail = err_json["error"]
                    if isinstance(err_detail, dict):
                        err_msg = err_detail.get("message", err_msg)
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            raise RuntimeError(f"HTTP {r.status_code}: {err_msg}")

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            try:
                chunk = json.loads(data)
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
            except json.JSONDecodeError:
                continue
    finally:
        r.close()


def _stream_gemini(api_key, model, system_prompt, user_prompt, max_tokens):
    """Google Gemini streamGenerateContent (SSE)."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:streamGenerateContent?alt=sse&key={api_key}")

    # --- Thinking davranışı modele göre ayarlanır ---
    # 2.5 Flash: thinking kapatılabilir → kullanıcının max_tokens'ı aynen kalır
    # 2.5 Pro / 3.x: thinking KAPATILAMAZ, 8192'ye kadar token yer →
    #                kullanıcının max_tokens'ına thinking buffer eklenmeli,
    #                aksi halde cevaba yer kalmaz
    gen_config = {"temperature": 0.4}

    if model.startswith("gemini-2.5-flash") or model.startswith("gemini-2.5-flash-lite"):
        # Flash'te thinking'i kapat → tüm bütçe cevaba gider
        gen_config["thinkingConfig"]  = {"thinkingBudget": 0}
        gen_config["maxOutputTokens"] = max_tokens
    elif model.startswith("gemini-2.5-pro") or model.startswith("gemini-3"):
        # Pro ve 3.x'te thinking zorunlu → thinking için 8192 ekstra ayır
        # Böylece kullanıcının istediği `max_tokens` gerçekten cevaba gider
        gen_config["maxOutputTokens"] = max_tokens + 8192
    else:
        # 2.0 ve diğerleri — thinking yok, normal davran
        gen_config["maxOutputTokens"] = max_tokens

    payload = {
        "contents":          [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig":  gen_config,
    }
    r = requests.post(url, json=payload, stream=True, timeout=90)
    try:
        if r.status_code != 200:
            try:
                err_body = r.text
            except Exception:
                err_body = "(yanıt okunamadı)"
            err_msg = err_body[:500]
            try:
                err_json = json.loads(err_body)
                if isinstance(err_json, dict) and "error" in err_json:
                    err_detail = err_json["error"]
                    if isinstance(err_detail, dict):
                        err_msg = err_detail.get("message", err_msg)
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            raise RuntimeError(f"HTTP {r.status_code}: {err_msg}")

        finish_reason = None
        usage_meta    = None
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            try:
                chunk = json.loads(data)
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        # Sadece cevap text'ini ver, "thought: true" parçalarını atla
                        if "text" in part and not part.get("thought", False):
                            yield part["text"]
                    fr = cand.get("finishReason")
                    if fr and fr != "STOP":
                        finish_reason = fr
                # Son chunk'ta usage gelir
                if "usageMetadata" in chunk:
                    usage_meta = chunk["usageMetadata"]
            except json.JSONDecodeError:
                continue

        # --- Debug / Uyarı mesajları ---
        debug_lines = []
        if usage_meta:
            prompt_t   = usage_meta.get("promptTokenCount", 0)
            cand_t     = usage_meta.get("candidatesTokenCount", 0)
            thought_t  = usage_meta.get("thoughtsTokenCount", 0)
            total_t    = usage_meta.get("totalTokenCount", 0)
            debug_lines.append(
                f"📊 Token Kullanımı — Prompt: {prompt_t} · Cevap: {cand_t} · "
                f"Thinking: {thought_t} · Toplam: {total_t}"
            )

        if finish_reason == "MAX_TOKENS":
            pro_hint = ""
            if "pro" in model.lower():
                pro_hint = (
                    " **Not:** `gemini-2.5-pro` modelinde reasoning token kapatılamıyor. "
                    "`gemini-2.5-flash` modeline geçmeyi deneyin."
                )
            yield (
                f"\n\n---\n⚠️ **Yanıt token limitine takıldı** (`MAX_TOKENS`).{pro_hint}"
            )
        elif finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
            yield f"\n\n---\n⚠️ **Yanıt güvenlik filtresi nedeniyle kesildi** (`{finish_reason}`)."
        elif finish_reason:
            yield f"\n\n---\n⚠️ **Yanıt şu sebeple kesildi:** `{finish_reason}`"

        if debug_lines:
            yield "\n\n" + "\n".join(debug_lines)
    finally:
        r.close()


def stream_llm(provider, api_key, model, system_prompt, user_prompt, max_tokens):
    """Birleşik streaming dispatcher."""
    cfg = LLM_PROVIDERS[provider]
    if cfg["type"] == "openai":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        yield from _stream_openai_compat(
            cfg["endpoint"], api_key, model, messages, max_tokens,
            provider_name=provider,
        )
    elif cfg["type"] == "anthropic":
        yield from _stream_anthropic(api_key, model, system_prompt, user_prompt, max_tokens)
    elif cfg["type"] == "gemini":
        yield from _stream_gemini(api_key, model, system_prompt, user_prompt, max_tokens)
    else:
        raise ValueError(f"Bilinmeyen provider tipi: {cfg['type']}")


# ============================================================
# 🔄 NON-STREAMING (toplu yanıt) FONKSİYONLARI
# Streaming'deki yarım-kesilme bug'larından etkilenmez.
# ============================================================
def _parse_http_error(response, default_msg):
    """HTTP hata gövdesinden anlamlı mesaj çıkar."""
    try:
        body = response.text
    except Exception:
        return default_msg
    msg = body[:500]
    try:
        err_json = json.loads(body)
        if isinstance(err_json, dict) and "error" in err_json:
            err_detail = err_json["error"]
            if isinstance(err_detail, dict):
                msg = err_detail.get("message", msg)
            else:
                msg = str(err_detail)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return msg


def _fetch_openai_compat(endpoint, api_key, model, messages, max_tokens, provider_name="OpenAI"):
    """OpenAI uyumlu non-streaming (OpenAI, DeepSeek, Groq)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    token_param = "max_tokens"
    if provider_name == "OpenAI" and any(
        model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
    ):
        token_param = "max_completion_tokens"
    payload = {
        "model": model, "messages": messages, "stream": False,
        token_param: max_tokens, "temperature": 0.4,
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_parse_http_error(r, r.text[:500])}")
    data = r.json()
    try:
        choice   = data["choices"][0]
        text     = choice["message"].get("content", "") or ""
        finish   = choice.get("finish_reason")
    except (KeyError, IndexError):
        raise RuntimeError("Beklenmeyen yanıt formatı")
    usage = data.get("usage", {})
    meta = {
        "finish_reason":    finish,
        "prompt_tokens":    usage.get("prompt_tokens",     0),
        "output_tokens":    usage.get("completion_tokens", 0),
        "thinking_tokens":  0,
        "total_tokens":     usage.get("total_tokens",      0),
    }
    return text, meta


def _fetch_anthropic(api_key, model, system_prompt, user_prompt, max_tokens):
    """Anthropic Messages API non-streaming."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model, "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "stream": False, "temperature": 0.4,
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_parse_http_error(r, r.text[:500])}")
    data = r.json()
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    usage = data.get("usage", {})
    meta = {
        "finish_reason":    data.get("stop_reason"),
        "prompt_tokens":    usage.get("input_tokens",  0),
        "output_tokens":    usage.get("output_tokens", 0),
        "thinking_tokens":  0,
        "total_tokens":     usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }
    return text, meta


def _fetch_gemini(api_key, model, system_prompt, user_prompt, max_tokens):
    """Google Gemini generateContent non-streaming."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")

    gen_config = {"temperature": 0.4}
    if model.startswith("gemini-2.5-flash") or model.startswith("gemini-2.5-flash-lite"):
        gen_config["thinkingConfig"]  = {"thinkingBudget": 0}
        gen_config["maxOutputTokens"] = max_tokens
    elif model.startswith("gemini-2.5-pro") or model.startswith("gemini-3"):
        # Pro ve 3.x'te thinking zorunlu → thinking için ekstra 8192 buffer
        gen_config["maxOutputTokens"] = max_tokens + 8192
    else:
        gen_config["maxOutputTokens"] = max_tokens

    payload = {
        "contents":          [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig":  gen_config,
    }
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {_parse_http_error(r, r.text[:500])}")
    data = r.json()

    text   = ""
    finish = None
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part and not part.get("thought", False):
                text += part["text"]
        if cand.get("finishReason"):
            finish = cand["finishReason"]

    usage = data.get("usageMetadata", {})
    meta = {
        "finish_reason":    finish,
        "prompt_tokens":    usage.get("promptTokenCount",    0),
        "output_tokens":    usage.get("candidatesTokenCount", 0),
        "thinking_tokens":  usage.get("thoughtsTokenCount",   0),
        "total_tokens":     usage.get("totalTokenCount",      0),
    }
    return text, meta


def fetch_llm(provider, api_key, model, system_prompt, user_prompt, max_tokens):
    """Non-streaming birleşik dispatcher. (text, meta_dict) döner."""
    cfg = LLM_PROVIDERS[provider]
    if cfg["type"] == "openai":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return _fetch_openai_compat(
            cfg["endpoint"], api_key, model, messages, max_tokens,
            provider_name=provider,
        )
    elif cfg["type"] == "anthropic":
        return _fetch_anthropic(api_key, model, system_prompt, user_prompt, max_tokens)
    elif cfg["type"] == "gemini":
        return _fetch_gemini(api_key, model, system_prompt, user_prompt, max_tokens)
    else:
        raise ValueError(f"Bilinmeyen provider tipi: {cfg['type']}")


def build_ai_prompt(*, detail, ticker, close, interval, total_score, karar,
                    regime_label, regime_desc, adx, adx_threshold,
                    components, ema200, rsi, stk, macd, macd_sig,
                    atr_high, swing_levels, fib_levels, st_dir,
                    obv_sig, div_rsi, div_macd, rsi_lo, rsi_up):
    """Yapılandırılmış system + user prompt üret."""
    system = (
        "Sen deneyimli bir kurumsal teknik analiz uzmanısın. "
        "SADECE sana verilen sayısal verileri kullan — hiçbir sayıyı uydurma, "
        "tahmin etme veya ek veri varsay. Yatırım tavsiyesi verme; bir "
        "profesyonel trader'ın veriye nasıl yaklaşacağını anlat. "
        "Türkçe yanıt ver, teknik jargon kullanabilirsin ama netlikten "
        "taviz verme. Yanıtını markdown formatında, başlıklar altında "
        "organize et. Somut ve aksiyona dönüştürülebilir ol.\n\n"
        "ÖNEMLİ YORUMLAMA KURALLARI:\n"
        "- **Skor bileşenlerindeki puanlar (örn. +0.7, -1, ×0.7) SKOR AĞIRLIKLARIDIR, "
        "indikatör değeri DEĞİLDİR.** 'MACD Histogram İvmesi: -0.7' bileşeninde -0.7 "
        "histogram değeri değil, bileşenin skora katkısıdır. İndikatörün gerçek sayısal "
        "değeri için sana ayrıca verilen 'Anahtar İndikatör Değerleri' bölümüne bak.\n"
        "- **Genel ADX/RSI/threshold değerleri için kullanıcının kendi ayarlarını kullan** "
        "— jenerik '25', '30', '70' gibi sayılar varsayma; sana prompt'ta verilen "
        "gerçek eşik değerlerini kullan.\n"
        "- **Risk/Ödül (R/R) oranını mutlaka hesapla.** Stop-loss ve hedef seviye öner "
        "dikten sonra: R/R = (hedef - giriş) / (giriş - stop). Eğer R/R < 2:1 ise bunu "
        "'Risk/Ödül oranı düşük, pozisyonu yeniden değerlendirin' şeklinde AÇIKÇA BELİRT.\n\n"
        "ÖNEMLİ FORMATLAMA KURALLARI:\n"
        "- Her cümleyi tam olarak bitir, asla yarıda bırakma.\n"
        "- Son başlığın son cümlesi de tam ve anlamlı olmalı.\n"
        "- Eğer istenen tüm başlıkları cevaplayacak kadar yerin olmayacağını "
        "düşünürsen, her başlığı biraz daha kısa tut ama tamamla — "
        "hiçbir zaman ortada kesme.\n"
        "- Yanıtı bitirirken noktayla bitir, üç nokta veya yarım cümle kullanma."
    )

    # RSI durumu
    if   rsi >= rsi_up: rsi_state = f"aşırı alım (>{rsi_up})"
    elif rsi <= rsi_lo: rsi_state = f"aşırı satım (<{rsi_lo})"
    else:               rsi_state = f"nötr bölge ({rsi_lo}-{rsi_up})"

    # Divergence
    div_parts = []
    if div_rsi  == 1:  div_parts.append("RSI Bullish 🔺")
    if div_rsi  == -1: div_parts.append("RSI Bearish 🔻")
    if div_macd == 1:  div_parts.append("MACD Bullish 🔺")
    if div_macd == -1: div_parts.append("MACD Bearish 🔻")
    div_text = " / ".join(div_parts) if div_parts else "Yok"

    # Destek / Direnç
    sr_lines = []
    if swing_levels:
        below = sorted([s for s in swing_levels if s["price"] < close], key=lambda x: -x["price"])
        above = sorted([s for s in swing_levels if s["price"] > close], key=lambda x: x["price"])
        for i, b in enumerate(below[:2]):
            pct = abs(b["price"] - close) / close * 100
            sr_lines.append(f"Destek-{i+1}: {b['price']:.2f} (%{pct:.2f} altta, {b['touches']}x test)")
        for i, a in enumerate(above[:2]):
            pct = abs(a["price"] - close) / close * 100
            sr_lines.append(f"Direnç-{i+1}: {a['price']:.2f} (%{pct:.2f} üstte, {a['touches']}x test)")
    sr_text = ("\n  - " + "\n  - ".join(sr_lines)) if sr_lines else " Tespit edilmedi"

    # Fibonacci (en yakın 3)
    fib_text = "Hesaplanmadı"
    if fib_levels:
        sorted_fib = sorted(fib_levels.items(), key=lambda x: abs(x[1] - close))[:3]
        fib_text = ", ".join(f"{k} = {v:.2f}" for k, v in sorted_fib)

    # Skor bileşen dökümü
    comp_text = "\n".join(
        f"- **{row['Bileşen']}**: `{row['Puan']}` — {row['Değer']}" for row in components
    )

    # Çıktı şablonu (detay seviyesine göre)
    if detail == "Kısa":
        output_req = (
            "\n## İstenen Çıktı (KISA)\n"
            "Toplamda 3-4 cümle, şu başlıklarda:\n"
            "1. **🎯 Durum**\n"
            "2. **⚠️ Uyarı**\n"
            "3. **📍 Aksiyon** (stop ve hedef varsa R/R oranını parantez içinde belirt)\n"
        )
    elif detail == "Orta":
        output_req = (
            "\n## İstenen Çıktı (ORTA)\n"
            "Şu 5 başlıkta orta uzunlukta yorum yap:\n"
            "1. **🎯 Genel Değerlendirme** — skorun ne anlama geldiği (2-3 cümle)\n"
            "2. **⚠️ Ana Risk** — en kritik uyarı ve nedeni\n"
            "3. **📍 Giriş Senaryosu** — ne beklenmeli, hangi seviyeler aksiyon için uygun\n"
            "4. **🛡️ Risk Yönetimi** — somut stop-loss ve hedef seviyeleri ver, "
            "ardından **Risk/Ödül oranını hesapla**: R/R = (hedef - giriş) / (giriş - stop). "
            "Oran 2:1'den düşükse 'R/R uygun değil' şeklinde açıkça uyar.\n"
            "5. **👁️ Takip Listesi** — dikkat edilmesi gereken 3-4 kritik sinyal\n"
        )
    else:  # Detaylı
        output_req = (
            "\n## İstenen Çıktı (DETAYLI)\n"
            "Şu 7 başlıkta derin analiz yap:\n"
            "1. **🎯 Genel Değerlendirme** — skoru ve rejimi derinlemesine açıkla\n"
            "2. **📊 Bileşen Analizi** — her skor bileşeninin neden o değeri aldığını yorumla "
            "(unutma: puanlar indikatör değeri değil, skor ağırlığıdır)\n"
            "3. **⚠️ Risk Faktörleri** — tüm önemli uyarılar ve neden önemli oldukları\n"
            "4. **📍 Senaryolar** — Boğa / Ayı / Yatay senaryolar için ayrı planlar\n"
            "5. **🛡️ Risk Yönetimi** — pozisyon boyutu, stop-loss, hedef (somut sayılarla) "
            "ve **Risk/Ödül oranı hesabı**. R/R = (hedef - giriş) / (giriş - stop). "
            "Oran uygun değilse (< 2:1) açıkça uyar.\n"
            "6. **📈 İhtimal Değerlendirmesi** — kısa ve orta vadeli muhtemel hareketler\n"
            "7. **👁️ Takip Listesi** — durumu değiştirebilecek kritik sinyaller\n"
        )

    user = f"""## Analiz Edilecek Veri

**Enstrüman:** {ticker.upper()}
**Fiyat:** {close:.4f}
**Zaman Dilimi:** {interval}

## 🎯 KOMBİNE SKOR SONUCU
- **Toplam Skor:** {total_score:+.2f} / ±11
- **Karar:** {karar}
- **Rejim:** {regime_label} (ADX: {adx:.1f}, kullanıcının eşik ayarı: **{adx_threshold}**)
- **Rejim Notu:** {regime_desc}

## 📊 Skor Bileşen Dökümü
⚠️ **Aşağıdaki 'Puan' değerleri SKOR AĞIRLIKLARIDIR, indikatör değerleri DEĞİLDİR.**
Örneğin "Histogram İvmesi -0.7" → histogramın kendisi -0.7 değil, bileşenin puanıdır.
Gerçek indikatör sayılarını aşağıdaki 'Anahtar İndikatör Değerleri' bölümünden oku.

{comp_text}

## 🔑 Anahtar İndikatör Değerleri (GERÇEK SAYILAR)
- **RSI:** {rsi:.1f} — {rsi_state} (kullanıcı eşikleri: alt={rsi_lo}, üst={rsi_up})
- **Stoch RSI %K:** {stk:.1f}
- **MACD:** {macd:+.4f} | **Signal:** {macd_sig:+.4f} | **Histogram:** {macd - macd_sig:+.4f}
- **ADX:** {adx:.1f} (kullanıcının trend eşik ayarı: **{adx_threshold}** — bu sayıyı kullan, 25/30 gibi varsayılanlar kullanma)
- **EMA200:** {ema200:.2f} (fiyat {"üstünde" if close > ema200 else "altında"})
- **SuperTrend:** {"AL ✅" if st_dir == 1 else "SAT ❌"}
- **OBV:** {"Birikim ✅" if obv_sig == 1 else ("Dağıtım ❌" if obv_sig == -1 else "Nötr ⚪")}
- **Divergence:** {div_text}
- **Volatilite (ATR):** {"Yüksek ↑" if atr_high else "Düşük ↓"}

## 📍 Seviyeler
- **Destek / Direnç:**{sr_text}
- **En Yakın Fibonacci:** {fib_text}

---
{output_req}
### Kurallar
- Yukarıda verilmeyen hiçbir sayıyı uydurma veya tahmin etme
- Skor bileşen puanlarını (±0.7, ±1 vb.) indikatör değeri gibi yorumlama
- ADX/RSI eşikleri için yukarıda verilen kullanıcı ayarlarını kullan (25, 30, 70 gibi genel sayılar varsayma)
- Stop-loss ve hedef verdikten sonra Risk/Ödül oranını hesapla ve yorumla
- "Yatırım tavsiyesi" ibaresi kullanma
- Somut ve aksiyona dönüştürülebilir ol
- Markdown formatında yaz (başlıklar, bold, listeler)
"""
    return system, user


def ai_cache_key(ticker, interval, total_score, close, provider, model, detail):
    """Analiz durumu için stabil cache anahtarı."""
    s = f"{ticker}|{interval}|{round(total_score, 2)}|{round(close, 4)}|{provider}|{model}|{detail}"
    return "ai_report_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def clean_half_sentence(text):
    """Yanıt sonunda yarım kalmış cümleyi temizle.
    Gemini Flash streaming bazen cümle ortasında kesiyor — bunu tespit edip
    son tam cümleyi koru. (clean_text, was_cut) döner."""
    if not text or len(text) < 30:
        return text, False

    stripped = text.rstrip()
    if not stripped:
        return text, False

    # Zaten düzgün bir bitirici karakter ile bitiyor mu?
    # (nokta, ünlem, soru, iki nokta, parantez, yıldız, backtick, emoji/özel)
    safe_endings = ".!?:;)]}*`\"'›»"
    if stripped[-1] in safe_endings:
        return text, False

    # Cümle sonlarını bul — SADECE "nokta/ünlem/soru + boşluk veya satır sonu"
    # Bu decimal sayıları (örn. "78.4") yanlışlıkla cümle sonu saymaz.
    import re
    matches = list(re.finditer(r'[.!?](?=\s|$)', stripped))
    if not matches:
        return text, False

    last_end = matches[-1].end()  # son cümle bitişinin konumu

    # Son bitiş çok başlardaysa (metnin %40'ından az), olduğu gibi bırak —
    # muhtemelen kısa bir özetti, müdahale etmeyelim
    if last_end < len(stripped) * 0.4:
        return text, False

    cleaned = stripped[:last_end]
    return cleaned, True


# ============================================================
# 2. YAN PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Veri Ayarları")
    ticker = st.text_input("Ticker Sembolü:", "gc=f")

    period = st.selectbox(
        "Toplam Veri Süresi (Period):",
        options=["1d", "5d", "1mo", "6mo", "1y", "2y", "5y", "max"],
        index=5,
    )

    if period in ["1d", "5d"]:
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 0
    elif period == "1mo":
        interval_options = ["2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 6
    else:
        interval_options = ["1h", "1d", "1wk", "1mo"]
        default_int_idx = 1

    interval = st.selectbox(
        "Mum Aralığı (Interval):", options=interval_options, index=default_int_idx
    )
    st.write("---")
    chart_type = st.radio("📊 Grafik Tipi:", ["Mum", "Çizgi"], horizontal=True)

    # ──────────────────────────────────────────────────────────
    # 🤖 AI RAPOR YORUMCUSU (LLM Provider Seçimi)
    # Sık kullanılan — Sabit Parametreler'in üstünde
    # ──────────────────────────────────────────────────────────
    st.write("---")
    st.subheader("🤖 AI Rapor Yorumcusu")

    ai_provider = st.selectbox(
        "Provider",
        options=list(LLM_PROVIDERS.keys()),
        index=0,
        key="ai_provider_select",
        help="Hangi LLM sağlayıcısını kullanmak istiyorsunuz?",
    )
    _prov_cfg = LLM_PROVIDERS[ai_provider]

    ai_model = st.selectbox(
        "Model",
        options=_prov_cfg["models"],
        index=0,
        key=f"ai_model_{ai_provider}",
    )

    ai_api_key = st.text_input(
        f"{ai_provider} API Key",
        value=("AIzaSyCctdPnlMgy9j3ri5zh6WTtY7kcuAwbyFE" if ai_provider == "Google" else ""),
        type="password",
        key=f"ai_key_{ai_provider}",
        help=f"API key almak için: {_prov_cfg['key_url']}",
    )

    ai_detail = st.select_slider(
        "Detay Seviyesi",
        options=list(AI_DETAIL_LEVELS.keys()),
        value="Orta",
        key="ai_detail_level",
    )
    st.caption(f"Max token: {AI_DETAIL_LEVELS[ai_detail]} · Sıcaklık: 0.4")

    st.write("---")
    st.subheader("Sabit Parametreler")
    ss = st.session_state
    sma_short        = st.slider("SMA Kısa Periyot:",        5,   50,  value=ss["sma_short"])
    sma_long         = st.slider("SMA Uzun Periyot:",        50,  300, value=ss["sma_long"])
    rsi_period       = st.slider("RSI Periyodu:",            7,   21,  value=ss["rsi_period"])
    rsi_lower        = st.slider("RSI Alt Eşik:",            20,  40,  value=ss["rsi_lower"])
    rsi_upper        = st.slider("RSI Üst Eşik:",            60,  80,  value=ss["rsi_upper"])
    rsi_ma_period    = st.slider("RSI MA Periyodu:",         5,   50,  14)
    bb_period        = st.slider("BB Periyodu:",             10,  50,  value=ss["bb_period"])
    bb_std           = st.slider("BB Standart Sapma:",       1.0, 3.0, value=ss["bb_std"],        step=0.5)
    macd_fast        = st.slider("MACD Hızlı EMA:",          5,   20,  value=ss["macd_fast"])
    macd_slow        = st.slider("MACD Yavaş EMA:",          15,  40,  value=ss["macd_slow"])
    macd_signal      = st.slider("MACD Sinyal:",             5,   15,  value=ss["macd_signal"])
    z_period         = st.slider("Z-Score Pencere:",         10,  60,  value=ss["z_period"])
    z_thresh         = st.slider("Z-Score Eşik:",            1.0, 3.0, value=ss["z_thresh"],      step=0.5)
    obv_short        = st.slider("OBV Kısa SMA:",            5,   20,  value=ss["obv_short"])
    obv_long         = st.slider("OBV Uzun SMA:",            15,  50,  value=ss["obv_long"])
    adx_period       = st.slider("ADX Periyodu:",            7,   30,  value=ss["adx_period"])
    adx_threshold    = st.slider("ADX Trend Eşiği:",        15,  35,  value=ss["adx_threshold"])
    atr_period       = st.slider("ATR Periyodu:",            7,   30,  14)
    stoch_rsi_period = st.slider("Stoch RSI Periyodu:",      7,   21,  14)
    stoch_d_period   = st.slider("Stoch RSI %D Smoothing:",  2,   5,   3)
    stoch_lower      = st.slider("Stoch RSI Alt Eşik:",      5,   30,  20)
    stoch_upper      = st.slider("Stoch RSI Üst Eşik:",      70,  95,  80)
    ichi_tenkan      = st.slider("Tenkan-sen:",              5,   20,  9)
    ichi_kijun       = st.slider("Kijun-sen:",               20,  40,  26)
    ichi_senkou_b    = st.slider("Senkou Span B:",           40,  65,  52)
    st_period        = st.slider("SuperTrend ATR Periyodu:", 5,   20,  value=ss["st_period"])
    st_multiplier    = st.slider("SuperTrend Çarpan:",       1.0, 5.0, value=ss["st_multiplier"], step=0.5)
    kama_period      = st.slider("KAMA Etkinlik Periyodu:",  5,   20,  10)
    kama_fast        = st.slider("KAMA Hızlı EMA:",          2,   5,   2)
    kama_slow        = st.slider("KAMA Yavaş EMA:",          20,  40,  30)
    lrc_period       = st.slider("LRC Periyodu:",            20,  100, value=ss["lrc_period"])
    lrc_std_mult     = st.slider("LRC Standart Sapma:",      1.0, 3.0, value=ss["lrc_std_mult"],  step=0.5)
    nw_bandwidth     = st.slider("NW Bant Genişliği (h):",   3,   20,  8)
    nw_window        = st.slider("NW Pencere (son N bar):",  50,  300, 100)
    vwap_band_pct    = st.slider("VWAP Nötr Bant (%):",     0.0, 1.0, 0.1, step=0.05)

    st.write("---")
    st.subheader("📐 Fibonacci Ayarları")
    fib_lookback = st.slider("Fibonacci Lookback (bar):", 20, 300, 100)

    st.write("---")
    st.subheader("〰️ WaveTrend Ayarları")
    wt_n1 = st.slider("WaveTrend Kanal (n1):",    5,  20,  value=ss["wt_n1"])
    wt_n2 = st.slider("WaveTrend Ortalama (n2):", 10, 40,  value=ss["wt_n2"])
    wt_ob = st.slider("WaveTrend Aşırı Alım:",    40, 80,  60)
    wt_os = st.slider("WaveTrend Aşırı Satım:",  -80, -20, -60)

    st.write("---")
    st.subheader("🔀 Divergence Ayarları")
    div_window = st.slider("Divergence Pivot Pencere:", 3, 10, 5)

    # ── Destek/Direnç ve Trend Çizgisi Ayarları ───────────────
    st.write("---")
    st.subheader("📊 Destek / Direnç Ayarları")
    swing_window  = st.slider("S/R Pivot Pencere:",    3,  20, 10,
        help="Tepe/dip tespiti için her yönde bakılacak bar sayısı")
    swing_touches = st.slider("Min. Dokunuş Sayısı:", 1,   5,  1,
        help="1 = tek pivotlu seviyeler de gösterilir (daha fazla çizgi, zayıf güç)")
    swing_atr_k   = st.slider("ATR Tolerans Çarpanı:", 0.2, 2.0, 0.5, step=0.1,
        help="Seviye birleştirme toleransı = bu değer × ATR / fiyat. "
             "Volatil enstrümanlarda yükselt, sakin enstrümanlarda düşür.")
    swing_tol     = 0.003  # fallback (ATR yoksa kullanılır)

    st.write("---")
    st.subheader("📐 Trend Çizgisi Ayarları")
    tl_pivot_window = st.slider("TL Pivot Pencere:",       5,  20,  10,
        help="Trend çizgisi pivot tespiti için pencere genişliği")
    tl_max_lines    = st.slider("Max Çizgi Sayısı:",       1,   5,   3,
        help="Her yönde (destek/direnç) gösterilecek maksimum çizgi")
    tl_tolerance    = st.slider("TL Tolerans (%):",        0.3, 2.0, 1.2, step=0.1,
        help="Pivotun çizgiye dokundu sayılması için fiyat toleransı") / 100
    tl_show_channel = st.checkbox("Kanalları Göster", value=True,
        help="Paralel destek+direnç kanallarını dolgulu göster")
    # ──────────────────────────────────────────────────────────

    st.write("---")
    st.subheader("📊 Backtest Ayarları")
    commission_pct = st.slider("Komisyon (% / işlem):", 0.0, 1.0, 0.1, step=0.01)
    slippage_pct   = st.slider("Slippage (% / işlem):", 0.0, 0.5, 0.05, step=0.01)

    st.write("---")
    st.subheader("🔁 Walk-Forward Optimizasyon")
    n_windows = st.slider("Pencere Sayısı:", 2, 8, 3,
        help="Veri kaç eşit parçaya bölünsün?")
    train_pct = st.slider("Eğitim Oranı (%):", 50, 85, 70, step=5,
        help="Her pencerenin %kaçı eğitim, kalanı test olsun?")
    st.caption(f"{n_windows} pencere · %{train_pct} eğitim / %{100-train_pct} test")

    st.write("---")
    run_opt = st.button("🚀 Algoritmaları Optimize Et", use_container_width=True, type="primary")
    st.info("İpucu: 1 dakikalık analizler için Periyot: 5d, Mum Aralığı: 1m seçiniz.")


# ============================================================
# 3. OPTİMİZASYON PARAMETRE GRİDLERİ
# ============================================================
PARAM_GRIDS = {
    "SMA Crossover":  {"sma_s":         [5, 10, 20, 30],
                       "sma_l":         [50, 100, 150, 200]},
    "RSI":            {"rsi_period":    [10, 14, 21],
                       "rsi_lower":     [25, 30, 35],
                       "rsi_upper":     [65, 70, 75]},
    "Bollinger Bands":{"bb_period":     [15, 20, 30],
                       "bb_std":        [1.5, 2.0, 2.5]},
    "MACD":           {"macd_fast":     [8, 12, 16],
                       "macd_slow":     [20, 26, 30],
                       "macd_signal":   [7, 9, 12]},
    "Mean Reversion": {"z_period":      [20, 30, 50],
                       "z_thresh":      [1.5, 2.0, 2.5]},
    "ADX":            {"adx_period":    [10, 14, 20],
                       "adx_threshold": [20, 25, 30]},
    "SuperTrend":     {"st_period":     [7, 10, 14],
                       "st_multiplier": [2.0, 2.5, 3.0, 3.5]},
    "LR Channel":     {"lrc_period":    [30, 50, 75],
                       "lrc_std_mult":  [1.5, 2.0, 2.5]},
    "WaveTrend":      {"wt_n1":         [8, 10, 14],
                       "wt_n2":         [15, 21, 28]},
    "OBV":            {"obv_short":     [5, 10, 15],
                       "obv_long":      [20, 30, 40]},
}


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================
def safe_scalar(value):
    if isinstance(value, (pd.Series, np.ndarray)):
        return float(value.iloc[0]) if len(value) > 0 else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        unique_tickers = df.columns.get_level_values(1).unique()
        if len(unique_tickers) <= 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    return df


def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm   = pd.Series(plus_dm,  index=high.index, dtype=float)
    minus_dm  = pd.Series(minus_dm, index=high.index, dtype=float)
    alpha     = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sp        = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm        = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di   = 100 * (sp / atr_s.replace(0, np.nan))
    minus_di  = 100 * (sm / atr_s.replace(0, np.nan))
    dx        = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx       = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_kama(close, period=10, fast=2, slow=30):
    ca   = close.values.astype(float)
    kama = np.full(len(ca), np.nan)
    kama[period - 1] = ca[period - 1]
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)
    for i in range(period, len(ca)):
        direction  = abs(ca[i] - ca[i - period])
        volatility = np.sum(np.abs(np.diff(ca[i - period:i + 1])))
        er  = 0.0 if volatility == 0 else direction / volatility
        sc  = (er * (fsc - ssc) + ssc) ** 2
        kama[i] = kama[i - 1] + sc * (ca[i] - kama[i - 1])
    return pd.Series(kama, index=close.index)


def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    hl2 = (high + low) / 2
    ub  = (hl2 + multiplier * atr).values.astype(float)
    lb  = (hl2 - multiplier * atr).values.astype(float)
    ca  = close.values.astype(float)
    ubf = ub.copy()
    lbf = lb.copy()
    direction  = np.ones(len(ca), dtype=float)
    supertrend = np.full(len(ca), np.nan)
    for i in range(1, len(ca)):
        if np.isnan(ubf[i-1]) or np.isnan(lbf[i-1]):
            ubf[i] = ub[i]
            lbf[i] = lb[i]
        else:
            ubf[i] = ub[i] if (ub[i] < ubf[i-1] or ca[i-1] > ubf[i-1]) else ubf[i-1]
            lbf[i] = lb[i] if (lb[i] > lbf[i-1] or ca[i-1] < lbf[i-1]) else lbf[i-1]
        if   ca[i] > ubf[i-1]: direction[i] = 1
        elif ca[i] < lbf[i-1]: direction[i] = -1
        else:                   direction[i] = direction[i-1]
        supertrend[i] = lbf[i] if direction[i] == 1 else ubf[i]
    return (pd.Series(supertrend, index=close.index), pd.Series(direction, index=close.index),
            pd.Series(lbf, index=close.index),        pd.Series(ubf, index=close.index))


def calc_linear_regression_channel(close, period=50, std_mult=2.0):
    n = len(close)
    mid   = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = close.values[i - period + 1:i + 1].astype(float)
        x = np.arange(period)
        sl, ic = np.polyfit(x, y, 1)
        yp  = sl * x + ic
        std = np.std(y - yp)
        mid[i]   = yp[-1]
        upper[i] = yp[-1] + std_mult * std
        lower[i] = yp[-1] - std_mult * std
    return (pd.Series(mid, index=close.index), pd.Series(upper, index=close.index),
            pd.Series(lower, index=close.index))


def calc_nadaraya_watson(close, bandwidth=8, window=100):
    n     = len(close)
    nwl   = np.full(n, np.nan)
    start = max(0, n - window)
    y = close.values[start:].astype(float)
    m = len(y)
    for i in range(m):
        w = np.exp(-((i - np.arange(m)) ** 2) / (2 * bandwidth ** 2))
        nwl[start + i] = np.sum(w * y) / np.sum(w)
    nws = pd.Series(nwl, index=close.index)
    mae = np.nanmean(np.abs(close.values[start:] - nwl[start:]))
    return nws, nws + 2 * mae, nws - 2 * mae


def calc_vwap_daily(high, low, close, volume):
    tp = (high + low + close) / 3
    dk = pd.Series(close.index.date, index=close.index)
    return (tp * volume).groupby(dk).cumsum() / volume.groupby(dk).cumsum().replace(0, np.nan)


# ── YENİ: Swing Destek/Direnç ─────────────────────────────────────────────────
def find_swing_levels(high, low, close, window=10, min_touches=2, tolerance=0.003,
                      atr_series=None, atr_k=0.5):
    """
    Swing High/Low bazlı otomatik destek/direnç tespiti.
    - atr_series verilirse tolerans = atr_k * ATR / fiyat (dinamik, volatiliteye uyumlu)
    - Aksi halde sabit 'tolerance' yüzdesi kullanılır (geriye uyumluluk)
    - Her seviyenin 'broken' alanı vardır: son kapanış seviyeyi kırmışsa True
    """
    n      = len(close)
    levels = []

    for i in range(window, n - window):
        if high.iloc[i] == high.iloc[i - window: i + window + 1].max():
            levels.append(("R", float(high.iloc[i]), i))
        if low.iloc[i] == low.iloc[i - window: i + window + 1].min():
            levels.append(("S", float(low.iloc[i]), i))

    # Dinamik tolerans: her pivot için kendi ATR'sine göre yüzde tolerans
    def _tol_for(price, bar_idx):
        if atr_series is not None and bar_idx < len(atr_series):
            atr_val = float(atr_series.iloc[bar_idx]) if hasattr(atr_series, "iloc") else float(atr_series[bar_idx])
            if not np.isnan(atr_val) and price > 0:
                return max(atr_k * atr_val / price, 0.0005)  # minimum %0.05 taban
        return tolerance

    merged = []
    used   = set()
    for idx, (typ, price, bar) in enumerate(levels):
        if idx in used:
            continue
        tol         = _tol_for(price, bar)
        touches     = [price]
        touch_bars  = [bar]
        for jdx, (typ2, price2, bar2) in enumerate(levels):
            if jdx != idx and jdx not in used:
                if abs(price2 - price) / price < tol:
                    touches.append(price2)
                    touch_bars.append(bar2)
                    used.add(jdx)
        used.add(idx)
        avg_price  = float(np.mean(touches))
        last_touch = max(touch_bars)

        # ── Break detection & role reversal ──
        # Fiyat bir direnci kırıp yukarı geçerse o seviye artık "destek"
        # Fiyat bir desteği kırıp aşağı inerse o seviye artık "direnç"
        last_close = float(close.iloc[-1])
        tol_now = _tol_for(avg_price, n - 1)
        if typ == "R":
            if last_close > avg_price * (1 + tol_now):
                typ = "S"           # direnç kırıldı, destek oldu
                broken = False      # yeni rolüyle aktif
            else:
                broken = False
        else:  # "S"
            if last_close < avg_price * (1 - tol_now):
                typ = "R"           # destek kırıldı, direnç oldu
                broken = False
            else:
                broken = False

        # ── Recency: son dokunuşun yakınlığı (0-1, yeni olan yüksek) ──
        recency = last_touch / max(n - 1, 1)

        # ── Güç skoru: dokunuş sayısı × recency ağırlığı ──
        strength = len(touches) * (0.5 + 0.5 * recency)

        merged.append({
            "type":       typ,
            "price":      avg_price,
            "touches":    len(touches),
            "last_touch": last_touch,
            "broken":     broken,
            "strength":   strength,
        })

    merged = [m for m in merged if m["touches"] >= min_touches]
    merged = sorted(merged, key=lambda x: -x["strength"])[:10]
    return merged
# ──────────────────────────────────────────────────────────────────────────────


# ── YENİ: Diyagonal Trend Çizgileri ───────────────────────────────────────────
def find_trendlines(high, low, close, pivot_window=10, max_lines=3, tolerance=0.012):
    """
    Gelişmiş otomatik trend çizgisi tespiti.
    - Swing high/low pivotları tespit edilir
    - Her ikili kombinasyon için çizgi skoru hesaplanır
       (dokunuş sayısı + yenilik + ihlal cezası)
    - Benzer eğimli çizgiler tekilleştirilir
    - Paralel destek+direnç çiftleri kanal olarak işaretlenir
    Döndürür: (lines, channels)
      lines   : list of dict  {type, x0,y0,x1,y1,slope,touches,last_touch}
      channels: list of dict  {support, resistance}
    """
    n     = len(close)
    dates = close.index

    # Pivot tespiti
    pivot_highs, pivot_lows = [], []
    for i in range(pivot_window, n - pivot_window):
        if high.iloc[i] == high.iloc[i - pivot_window: i + pivot_window + 1].max():
            pivot_highs.append((i, float(high.iloc[i])))
        if low.iloc[i] == low.iloc[i - pivot_window: i + pivot_window + 1].min():
            pivot_lows.append((i, float(low.iloc[i])))

    def _score_line(p1, p2, pivots, violation_series):
        x1, y1 = p1;  x2, y2 = p2
        if x2 == x1: return 0, []
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        touches   = []
        violations = 0
        for xi in range(min(x1, x2), n):
            y_line = slope * xi + intercept
            y_act  = float(violation_series.iloc[xi])
            rel    = (y_act - y_line) / (abs(y_line) + 1e-9)
            # Dokunuş: pivot bu çizgiye yeterince yakın mı?
            for (px, py) in pivots:
                if px == xi and abs(py - y_line) / (abs(y_line) + 1e-9) < tolerance:
                    touches.append((xi, py))
            # İhlal: fiyat destek/direnç çizgisini kırdı mı?
            if slope >= 0 and rel < -tolerance * 3:   violations += 1
            if slope < 0  and rel >  tolerance * 3:   violations += 1
        score = len(touches) - violations * 0.5
        return score, touches

    def _best_lines(pivots, violation_series, line_type):
        if len(pivots) < 2:
            return []
        candidates = []
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                p1, p2 = pivots[i], pivots[j]
                score, touches = _score_line(p1, p2, pivots, violation_series)
                if score < 1.5 or len(touches) < 2:
                    continue
                x1, y1 = p1;  x2, y2 = p2
                slope     = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                y_end     = slope * (n - 1) + intercept
                last_bar  = max(t[0] for t in touches)
                candidates.append({
                    "type":       line_type,
                    "x0":         x1,         "y0": y1,
                    "x1":         n - 1,      "y1": y_end,
                    "slope":      slope,
                    "intercept":  intercept,
                    "touches":    len(touches),
                    "last_touch": last_bar,
                    "score":      score,
                })
        # Sırala: skor desc, yenilik desc
        candidates.sort(key=lambda c: (-c["score"], -c["last_touch"]))
        # Benzer eğimli çizgileri tekilleştir
        unique = []
        for c in candidates:
            dup = any(
                abs(c["slope"] - u["slope"]) / (abs(u["slope"]) + 1e-9) < 0.08
                for u in unique
            )
            if not dup:
                unique.append(c)
            if len(unique) >= max_lines:
                break
        return unique

    support_lines    = _best_lines(pivot_lows,  low,  "support")
    resistance_lines = _best_lines(pivot_highs, high, "resistance")

    # Kanal tespiti: yaklaşık paralel destek + direnç çiftleri
    channels = []
    for sl in support_lines:
        for rl in resistance_lines:
            sdiff = abs(sl["slope"] - rl["slope"]) / (abs(sl["slope"]) + 1e-9)
            if sdiff < 0.12:
                channels.append({"support": sl, "resistance": rl})

    return support_lines + resistance_lines, channels, dates
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================
# FİBONACCİ, WAVETREND, DIVERGENCE
# ============================================================
def calc_fibonacci(high, low, lookback=100):
    recent_high = float(high.rolling(lookback, min_periods=1).max().iloc[-1])
    recent_low  = float(low.rolling(lookback, min_periods=1).min().iloc[-1])
    diff = recent_high - recent_low
    if diff == 0:
        return {}, recent_high, recent_low
    levels = {
        "0.0%":   recent_low,
        "23.6%":  recent_low + 0.236 * diff,
        "38.2%":  recent_low + 0.382 * diff,
        "50.0%":  recent_low + 0.500 * diff,
        "61.8%":  recent_low + 0.618 * diff,
        "78.6%":  recent_low + 0.786 * diff,
        "100.0%": recent_high,
    }
    return levels, recent_high, recent_low


def calc_wavetrend(high, low, close, n1=10, n2=21):
    ap  = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def detect_divergence(price, indicator, window=5):
    n      = len(price)
    result = np.zeros(n)
    pv     = price.values.astype(float)
    iv     = indicator.values.astype(float)
    for i in range(window * 2, n):
        seg_p = pv[max(0, i - window * 4):i + 1]
        seg_i = iv[max(0, i - window * 4):i + 1]
        m     = len(seg_p)
        lows_p = []; lows_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.min(seg_p[j - window:j + window + 1]):
                lows_p.append(seg_p[j])
                lows_i.append(seg_i[j])
        if len(lows_p) >= 2:
            if lows_p[-1] < lows_p[-2] and lows_i[-1] > lows_i[-2]:
                result[i] = 1
        highs_p = []; highs_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.max(seg_p[j - window:j + window + 1]):
                highs_p.append(seg_p[j])
                highs_i.append(seg_i[j])
        if len(highs_p) >= 2:
            if highs_p[-1] > highs_p[-2] and highs_i[-1] < highs_i[-2]:
                result[i] = -1
    return pd.Series(result, index=price.index)


# ============================================================
# 5. SİNYAL FONKSİYONLARI
# ============================================================
def sig_sma(close, atr_high, sma_s=20, sma_l=100):
    sh  = close.rolling(sma_s, min_periods=sma_s).mean()
    sl  = close.rolling(sma_l, min_periods=sma_l).mean()
    sig = np.where(sh > sl, 1, -1)
    sig = np.where(sh.isna() | sl.isna(), 0, sig)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), sh, sl


def sig_rsi_fn(close, rsi_period, rsi_lower=30, rsi_upper=70):
    d   = close.diff()
    g   = d.where(d > 0, 0.0).rolling(rsi_period).mean()
    l   = (-d.where(d < 0, 0.0)).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    sig = np.where(rsi < rsi_lower, 1, np.where(rsi > rsi_upper, -1, 0))
    return pd.Series(sig, index=close.index), rsi


def sig_bb(close, bb_period, bb_std_val=2.0):
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    up  = mid + bb_std_val * std
    lo  = mid - bb_std_val * std
    sig = np.where(close < lo, 1, np.where(close > up, -1, 0))
    return pd.Series(sig, index=close.index), mid, up, lo


def sig_macd(close, atr_high, macd_fast=12, macd_slow=26, macd_sig_p=9):
    ef   = close.ewm(span=macd_fast, adjust=False).mean()
    es   = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ef - es
    ms   = macd.ewm(span=macd_sig_p, adjust=False).mean()
    sig  = np.where(macd > ms, 1, -1)
    sig  = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), macd, ms


def sig_z(close, z_period, z_thresh=2.0):
    zm  = close.rolling(z_period).mean()
    zs  = close.rolling(z_period).std().replace(0, np.nan)
    z   = (close - zm) / zs
    sig = np.where(z < -z_thresh, 1, np.where(z > z_thresh, -1, 0))
    return pd.Series(sig, index=close.index), z


def sig_obv(close, volume, obv_short, obv_long):
    obv = (volume * np.sign(close.diff()).fillna(0)).cumsum()
    s   = obv.rolling(obv_short, min_periods=obv_short).mean()
    l   = obv.rolling(obv_long,  min_periods=obv_long).mean()
    sig = np.where(s > l, 1, -1)
    sig = np.where(s.isna() | l.isna(), 0, sig)
    return pd.Series(sig, index=close.index), obv, s, l


def sig_adx_fn(high, low, close, atr_high, adx_period, adx_threshold=25):
    adx_v, pdi, mdi = calc_adx(high, low, close, period=adx_period)
    sig = np.where(adx_v > adx_threshold, np.where(pdi > mdi, 1, -1), 0)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), adx_v, pdi, mdi


def sig_stochrsi(close, rsi_series, srsi_period, sd_period, sl, su):
    """Stoch RSI — bölge + %K/%D kesişim teyitli sinyal.
    - Aşırı satım (K < sl) VE yukarı dönüş (K > D) → AL (+1)
    - Aşırı alım  (K > su) VE aşağı dönüş (K < D) → SAT (-1)
    - Aksi halde nötr (0)
    Sadece bölgede olmak yetmez — K/D kesişimi dönüş teyidi şarttır.
    """
    rmin = rsi_series.rolling(srsi_period, min_periods=srsi_period).min()
    rmax = rsi_series.rolling(srsi_period, min_periods=srsi_period).max()
    k    = ((rsi_series - rmin) / (rmax - rmin).replace(0, np.nan) * 100).fillna(50).clip(0, 100)
    d    = k.rolling(sd_period).mean()

    # Kesişim teyitli sinyal
    bull = (k < sl) & (k > d)   # Aşırı satımda yukarı dönüş
    bear = (k > su) & (k < d)   # Aşırı alımda aşağı dönüş
    sig  = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(sig, index=close.index), k, d


def sig_ichimoku(high, low, close, atr_high, it, ik, isb):
    tenkan   = (high.rolling(it).max()  + low.rolling(it).min())  / 2
    kijun    = (high.rolling(ik).max()  + low.rolling(ik).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(ik)
    senkou_b = ((high.rolling(isb).max() + low.rolling(isb).min()) / 2).shift(ik)
    ct = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cb = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    sig = np.where((tenkan > kijun) & (close > ct), 1,
                   np.where((tenkan < kijun) & (close < cb), -1, 0))
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), tenkan, kijun, senkou_a, senkou_b


def sig_kama_fn(close, atr_high, kp, kf, ks):
    kama = calc_kama(close, period=kp, fast=kf, slow=ks)
    sig  = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    sig  = np.where(kama.isna(), 0, sig)
    sig  = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), kama


def sig_supertrend_fn(high, low, close, atr_high, stp, stm):
    st, std, lb, ub = calc_supertrend(high, low, close, period=stp, multiplier=stm)
    sig = std.values.copy()
    sig = np.where(st.isna(), 0, sig)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), st, std, lb, ub


def sig_lrc(close, lrc_period, lrc_std_mult=2.0):
    mid, up, lo = calc_linear_regression_channel(close, period=lrc_period, std_mult=lrc_std_mult)
    sig = np.where(close < lo, 1, np.where(close > up, -1, 0))
    sig = np.where(mid.isna(), 0, sig)
    return pd.Series(sig, index=close.index), mid, up, lo


def sig_vwap_fn(high, low, close, volume, vwap_band_pct):
    vwap = calc_vwap_daily(high, low, close, volume)
    band = vwap * (vwap_band_pct / 100)
    sig  = np.where(close > vwap + band, 1, np.where(close < vwap - band, -1, 0))
    sig  = np.where(vwap.isna(), 0, sig)
    return pd.Series(sig, index=close.index), vwap


def sig_wavetrend_fn(high, low, close, n1=10, n2=21, ob=60, os_=-60):
    wt1, wt2   = calc_wavetrend(high, low, close, n1=n1, n2=n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = np.where(cross_up & (wt1 < os_), 1,
                   np.where(cross_down & (wt1 > ob), -1, 0))
    return pd.Series(sig, index=close.index), wt1, wt2


# ============================================================
# 6. BACKTEST YARDIMCISI
# ============================================================
def bars_per_year_from_interval(interval):
    """Interval string'i yıllık bar sayısına çevirir (Sharpe yıllıklandırması için)."""
    m = {
        "1m":  252 * 390,  "2m":  252 * 195,  "5m":  252 * 78,
        "15m": 252 * 26,   "30m": 252 * 13,   "60m": 252 * 6.5,
        "1h":  252 * 6.5,  "1d":  252,        "1wk": 52,         "1mo": 12,
    }
    return m.get(interval, 252)


def _strategy_bar_returns(sig_vals, close_arr):
    """Sinyal + fiyat → bar-bazlı strateji log getirisi (pozisyon 1 bar geciktirilmiş)."""
    sig_vals  = np.asarray(sig_vals)
    close_arr = np.asarray(close_arr, dtype=float)
    if len(sig_vals) < 2 or not (close_arr > 0).all():
        return np.array([])
    position = np.zeros(len(sig_vals))
    in_pos = False
    for i in range(1, len(sig_vals)):
        if not in_pos and sig_vals[i] == 1 and sig_vals[i-1] != 1: in_pos = True
        elif in_pos and sig_vals[i] == -1 and sig_vals[i-1] != -1: in_pos = False
        position[i] = 1.0 if in_pos else 0.0
    pos_lag = np.concatenate(([0.0], position[:-1]))
    log_ret = np.diff(np.log(close_arr), prepend=np.log(close_arr[0]))
    return pos_lag * log_ret


def permutation_pvalue(strat_ret, observed_sharpe, bars_per_year, n_perm=200, seed=42):
    """(Geriye dönük uyumluluk için) Basit permutation test.
    YENİ KODDA stationary_bootstrap_pvalue TERCİH EDİN."""
    strat_ret = np.asarray(strat_ret)
    if len(strat_ret) < 10 or strat_ret.std() == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        shuf = rng.permutation(strat_ret)
        if shuf.std() == 0: continue
        s = float(shuf.mean() / shuf.std() * np.sqrt(bars_per_year))
        if s >= observed_sharpe:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


def stationary_bootstrap_pvalue(strat_ret, observed_sharpe, bars_per_year,
                                 n_boot=200, avg_block_len=10, seed=42):
    """Politis & Romano (1994) Stationary Bootstrap.

    Finansal getirilerin bağımsız olmadığı gerçeğini dikkate alır.
    Blok uzunlukları geometrik dağılımdan seçilir (ortalama = avg_block_len).
    Zaman serisi yapısı (volatility clustering, autocorrelation) korunur.

    Basit permutation'a göre p-değeri genellikle daha yüksek (daha dürüst) çıkar.
    """
    strat_ret = np.asarray(strat_ret)
    n = len(strat_ret)
    if n < 20 or strat_ret.std() == 0:
        return 1.0

    p_geom = 1.0 / max(avg_block_len, 2)  # blok başlangıç olasılığı
    rng = np.random.default_rng(seed)
    count_ge = 0
    valid_iters = 0

    for _ in range(n_boot):
        # Stationary bootstrap örneği oluştur
        boot = np.empty(n, dtype=strat_ret.dtype)
        idx = int(rng.integers(0, n))
        for i in range(n):
            boot[i] = strat_ret[idx]
            # Yeni blok başlatma olasılığı
            if rng.random() < p_geom:
                idx = int(rng.integers(0, n))
            else:
                idx = (idx + 1) % n  # aynı bloğa devam

        if boot.std() == 0:
            continue
        valid_iters += 1
        # Null dağılım: sharpe'ı "getirileri merkezileştirilmiş" örnekle ölç
        # (H0: gerçek Sharpe = 0 varsayımı altında)
        centered = boot - boot.mean()
        if centered.std() == 0:
            continue
        s_boot = float(centered.mean() / centered.std() * np.sqrt(bars_per_year))
        if s_boot >= observed_sharpe:
            count_ge += 1

    if valid_iters == 0:
        return 1.0
    return (count_ge + 1) / (valid_iters + 1)


def _norm_ppf(p):
    """Inverse of the standard normal CDF (scipy bağımsız).
    Peter Acklam's algorithm (1/2003), stdlib-only, ~1e-9 doğruluk.
    Girdi: 0 < p < 1. Çıktı: Φ⁻¹(p).
    """
    from math import sqrt, log
    if p <= 0.0 or p >= 1.0:
        # Aşırı uçlar için yaklaşım (pratik kullanımda olmaz ama güvenli)
        if p <= 0.0: return -float("inf")
        if p >= 1.0: return  float("inf")

    # Katsayılar (Acklam 2003)
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    p_low  = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = sqrt(-2.0 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0):
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Multiple testing ('data snooping') cezasını çıkarır.
    n_trials: kaç parametre kombinasyonu denendiği (örn. grid size)
    n_obs:    örneklem boyutu (bar sayısı)
    skew:     getiri dağılımının çarpıklığı
    kurt:     getiri dağılımının basıklığı (normal = 3)

    DSR > 0 → Gerçekten rastgeleden iyi.
    DSR 0   → Eşik: istatistiksel olarak anlamsız.
    DSR < 0 → Bu Sharpe muhtemelen şans eseri.
    """
    from math import log, sqrt, exp
    if n_trials <= 1 or n_obs <= 1:
        return observed_sharpe  # düzeltme gerekmiyor

    # Euler-Mascheroni sabiti
    emc = 0.5772156649
    # Expected Max Sharpe under null (Bailey & López de Prado 2014, Eq. 6)
    # E[max SR] ≈ sqrt(V[SR]) × ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
    try:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * exp(1)))
        expected_max_sr = (1.0 - emc) * z1 + emc * z2
    except Exception:
        # Fallback: N büyükse Gumbel'den yaklaşık
        expected_max_sr = sqrt(2.0 * log(max(n_trials, 2)))

    # DSR: Probabilistic SR'nin deflate edilmiş hali
    # σ(SR_hat) = sqrt((1 - skew·SR + (kurt-1)/4 · SR²) / (n_obs - 1))
    sr = observed_sharpe
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(n_obs - 1, 1)
    if var_sr <= 0:
        return sr - expected_max_sr
    std_sr = sqrt(var_sr)
    if std_sr == 0:
        return sr - expected_max_sr

    # DSR = (gözlemlenen SR - beklenen max SR) / std(SR)
    # Yüksek pozitif = gerçek, 0 civarı = sınırda, negatif = şans
    dsr = (sr - expected_max_sr) / std_sr
    return dsr


def run_backtest(signal_series, close_arr, cost_pct, bars_per_year=252):
    sig    = signal_series.values if hasattr(signal_series, "values") else signal_series
    sig    = np.asarray(sig)
    close_arr = np.asarray(close_arr, dtype=float)
    trades = []
    in_pos = False
    entry_p = 0.0
    for i in range(1, len(sig)):
        if not in_pos and sig[i] == 1 and sig[i-1] != 1:
            entry_p = float(close_arr[i])
            in_pos  = True
        elif in_pos and sig[i] == -1 and sig[i-1] != -1:
            ep = float(close_arr[i])
            trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)
            in_pos = False
    if in_pos:
        ep = float(close_arr[-1])
        trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)

    # ── Bar-bazlı yıllıklandırılmış Sharpe (akademik standart) ──
    strat_ret = _strategy_bar_returns(sig, close_arr)
    if len(strat_ret) > 1 and strat_ret.std() > 0:
        sharpe_bar = float(strat_ret.mean() / strat_ret.std() * np.sqrt(bars_per_year))
    else:
        sharpe_bar = 0.0

    if not trades:
        return {"total_ret": 0.0, "sharpe": round(sharpe_bar, 4), "sharpe_trade": 0.0, "n": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_dd": 0.0, "pf": 0.0}
    r      = np.array(trades)
    cumul  = 1.0
    peak   = 1.0
    max_dd = 0.0
    for rv in r:
        cumul *= (1 + rv / 100)
        if cumul > peak: peak = cumul
        dd = ((peak - cumul) / peak) * 100
        if dd > max_dd: max_dd = dd
    wins      = r[r > 0]
    losses    = r[r <= 0]
    total_ret = (cumul - 1) * 100
    wr        = len(wins) / len(r) * 100
    sharpe_trade = float(np.mean(r) / np.std(r)) * np.sqrt(len(r)) if len(r) > 1 and np.std(r) > 0 else 0.0
    pf        = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    return {"total_ret": round(total_ret, 4),
            "sharpe":       round(sharpe_bar, 4),     # yıllıklandırılmış bar-bazlı
            "sharpe_trade": round(sharpe_trade, 4),   # eski metrik (referans)
            "n": len(r),
            "win_rate": round(wr, 2),
            "avg_win":  round(float(wins.mean())   if len(wins)   > 0 else 0.0, 4),
            "avg_loss": round(float(losses.mean())  if len(losses) > 0 else 0.0, 4),
            "max_dd":   round(max_dd, 4),
            "pf":       round(pf, 4) if pf != float("inf") else float("inf")}


def _score(stats, metric):
    if metric == "Sharpe":
        return stats["sharpe"]
    elif metric == "Getiri":
        return stats["total_ret"]
    else:
        dd = stats["max_dd"]
        return stats["total_ret"] / dd if dd > 0 else stats["total_ret"]


def optimize_algo(param_grid, signal_fn, close_arr, cost_pct,
                  n_windows=4, train_pct=70, metric="Sharpe", min_trades=5,
                  bars_per_year=252, run_permutation=True, n_perm=200):
    """Gerçek walk-forward optimizasyon:
      1) Her pencerede TRAIN dilimi üzerinde en iyi kombo seçilir
      2) Kazanan kombo TEST diliminde dokunulmamış OOS skoru alır
      3) En çok seçilen (ve en yüksek OOS skora sahip) kombo sistem tarafından döndürülür
      4) Tüm OOS test dilimleri birleştirilip permutation test ile p-değeri hesaplanır
    """
    keys    = list(param_grid.keys())
    combos  = list(iter_product(*param_grid.values()))
    n       = len(close_arr)
    default = {k: v[0] for k, v in param_grid.items()}

    # ── EXPANDING WINDOW ──
    # Rolling pencerelerin aksine, her adımda geçmiş birikir (gerçek trading gibi).
    # Eğitim dilimi [0 → split_i], test dilimi [split_i → end_i].
    # Adımlar: veriyi (n_windows + 1) eşit parçaya böl, her adımda bir ek test dilimi.
    #   Adım 1: train=[0, 2/(n+1)), test=[2/(n+1), 3/(n+1))
    #   Adım 2: train=[0, 3/(n+1)), test=[3/(n+1), 4/(n+1))
    #   ...
    #   Adım n: train=[0, (n+1)/(n+1)=end), — son adım test = son dilim
    # Her test dilimi birbirinden bağımsız OOS, hiçbiri train'de görülmez.
    n_steps = max(n_windows, 2)
    step_size = n // (n_steps + 1)
    if step_size < 15:
        return default, None

    windows = []
    # İlk train minimum ~2 dilim olacak şekilde başla
    min_train_start = 2 * step_size
    for w in range(n_steps):
        train_start = 0
        train_end   = min_train_start + w * step_size
        test_start  = train_end
        test_end    = min(test_start + step_size, n) if w < n_steps - 1 else n
        # Yetersiz veri kontrolü
        if train_end - train_start < 20 or test_end - test_start < 10:
            continue
        windows.append((train_start, train_end, test_end))

    if not windows:
        return default, None

    # Kombolara göre OOS sonuçları
    combo_oos = {combo: [] for combo in combos}  # (test_stats, test_sig_slice, test_price_slice)

    for (ts, te, es) in windows:
        train_arr = close_arr[ts:te]
        test_arr  = close_arr[te:es]

        # Adaptif min_trades: pencere kısaysa alt sınır 3'e iner,
        # uzun pencerelerde kullanıcının ayarladığı tavan geçerli olur
        train_bars = te - ts
        eff_min_trades = max(3, min(min_trades, train_bars // 30))

        # TRAIN: her kombo için skor, en iyiyi bul
        sigs_cache = {}
        best_train_combo = None
        best_train_score = -np.inf
        for combo in combos:
            p = dict(zip(keys, combo))
            sig_full = signal_fn(p)
            if sig_full is None:
                continue
            sig_vals = np.asarray(sig_full.values if hasattr(sig_full, "values") else sig_full)
            sigs_cache[combo] = sig_vals
            train_sig = sig_vals[ts:te]
            train_stats = run_backtest(train_sig, train_arr, cost_pct, bars_per_year)
            if train_stats["n"] < eff_min_trades:
                continue
            sc = _score(train_stats, metric)
            if sc > best_train_score:
                best_train_score = sc
                best_train_combo = combo

        if best_train_combo is None:
            continue

        # TEST: yalnız train-kazananını out-of-sample test et (dokunulmamış)
        test_sig  = sigs_cache[best_train_combo][te:es]
        test_stats = run_backtest(test_sig, test_arr, cost_pct, bars_per_year)
        combo_oos[best_train_combo].append((test_stats, test_sig, test_arr))

    # En iyi kombo: en çok seçilen; eşitlikte en yüksek ortalama OOS skor
    winners = [(c, v) for c, v in combo_oos.items() if v]
    if not winners:
        return default, None

    def _rank_key(item):
        combo, results = item
        sel_count = len(results)
        avg_score = float(np.mean([_score(st, metric) for (st, _, _) in results]))
        return (sel_count, avg_score)

    best_combo, best_results = max(winners, key=_rank_key)
    best_p = dict(zip(keys, best_combo))

    # ── OOS aggregate stats (asla train verisi dahil değil) ──
    pooled_n  = sum(st["n"] for (st, _, _) in best_results)
    cumul = 1.0
    for (st, _, _) in best_results:
        cumul *= (1 + st["total_ret"] / 100)
    pooled_ret = (cumul - 1) * 100
    pooled_max_dd = max((st["max_dd"] for (st, _, _) in best_results), default=0.0)
    valid_stats = [st for (st, _, _) in best_results if st["n"] > 0]
    pooled_wr   = float(np.mean([st["win_rate"] for st in valid_stats])) if valid_stats else 0.0
    pooled_aw   = float(np.mean([st["avg_win"]  for st in valid_stats])) if valid_stats else 0.0
    pooled_al   = float(np.mean([st["avg_loss"] for st in valid_stats])) if valid_stats else 0.0
    finite_pfs  = [st["pf"] for st in valid_stats if st["pf"] != float("inf")]
    pooled_pf   = float(np.mean(finite_pfs)) if finite_pfs else float("inf")

    # ── Bar-bazlı OOS Sharpe: test dilimlerinin strateji getirileri concat ──
    all_strat_ret = []
    for (_, test_sig, test_price) in best_results:
        sr = _strategy_bar_returns(test_sig, test_price)
        if len(sr) > 0:
            # NaN/inf değerleri temizle
            sr = sr[np.isfinite(sr)]
            if len(sr) > 0:
                all_strat_ret.append(sr)
    if all_strat_ret:
        strat_ret_concat = np.concatenate(all_strat_ret)
        if len(strat_ret_concat) > 1 and strat_ret_concat.std() > 0:
            oos_sharpe = float(strat_ret_concat.mean() / strat_ret_concat.std() * np.sqrt(bars_per_year))
            if not np.isfinite(oos_sharpe):
                oos_sharpe = 0.0
        else:
            oos_sharpe = 0.0
    else:
        strat_ret_concat = np.array([])
        oos_sharpe = 0.0

    best_s = {
        "total_ret":     round(pooled_ret, 4),
        "sharpe":        round(oos_sharpe, 4),
        "n":             pooled_n,
        "win_rate":      round(pooled_wr, 2),
        "avg_win":       round(pooled_aw, 4),
        "avg_loss":      round(pooled_al, 4),
        "max_dd":        round(pooled_max_dd, 4),
        "pf":            round(pooled_pf, 4) if pooled_pf != float("inf") else float("inf"),
        "wf_windows":    len(windows),
        "wf_selections": len(best_results),
        "oos_only":      True,
    }

    # ── Stationary Bootstrap p-value (Politis & Romano 1994) ──
    # Basit permutation yerine zaman serisi yapısını koruyan blok bootstrap.
    if run_permutation and len(strat_ret_concat) > 20:
        # Ortalama blok uzunluğu: veri uzunluğunun ~küp köküne yakın (yaygın pratik)
        avg_block = max(5, int(len(strat_ret_concat) ** (1.0 / 3.0)))
        p_value = stationary_bootstrap_pvalue(
            strat_ret_concat, oos_sharpe, bars_per_year,
            n_boot=n_perm, avg_block_len=avg_block
        )
        best_s["p_value"] = round(p_value, 4)

    # ── Deflated Sharpe Ratio (Bailey & López de Prado 2014) ──
    # Multiple testing / data snooping cezası uygula.
    n_trials_grid = len(combos)  # bu algoritma için denenen kombo sayısı
    if n_trials_grid > 1 and len(strat_ret_concat) > 20:
        # Skewness ve kurtosis güvenli hesap (sıfır std, NaN ve inf koruması)
        try:
            std_ret = float(strat_ret_concat.std())
            if std_ret > 1e-12:
                demeaned = strat_ret_concat - strat_ret_concat.mean()
                sk_raw = float((demeaned ** 3).mean() / (std_ret ** 3))
                kt_raw = float((demeaned ** 4).mean() / (std_ret ** 4))
                # Aşırı değerleri kırp (DSR formülü aşırı kurtosis'e karşı hassas)
                sk = sk_raw if np.isfinite(sk_raw) else 0.0
                kt = kt_raw if np.isfinite(kt_raw) and kt_raw > 0 else 3.0
                # Güvenli sınırlar: makul finansal getiri dağılımı için
                sk = max(-5.0, min(5.0, sk))
                kt = max(1.0, min(30.0, kt))
            else:
                sk, kt = 0.0, 3.0

            dsr_val = deflated_sharpe_ratio(
                observed_sharpe=oos_sharpe,
                n_trials=n_trials_grid,
                n_obs=len(strat_ret_concat),
                skew=sk, kurt=kt,
            )
            if np.isfinite(dsr_val):
                best_s["dsr"] = round(float(dsr_val), 4)
                best_s["n_trials"] = n_trials_grid
            else:
                best_s["dsr"] = None
        except Exception as _dsr_err:
            best_s["dsr"] = None
            best_s["dsr_error"] = str(_dsr_err)[:100]

    return best_p, best_s


# ============================================================
# 7. VERİ ÇEKME
# ============================================================
@st.cache_data(ttl=55)
def fetch_live_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i, progress=False)
        return pd.DataFrame() if data is None or data.empty else data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()


PLOTLY_CONFIG = dict(scrollZoom=True, displayModeBar=True,
    modeBarButtonsToAdd=["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    modeBarButtonsToRemove=["lasso2d", "select2d"])


def sub_layout(height=250):
    return dict(template="plotly_dark", height=height, margin=dict(t=30, b=30), dragmode="pan")


# ============================================================
# 8. ANA MANTIK
# ============================================================
if ticker:
    df = fetch_live_data(ticker, period, interval)

    if not df.empty:
        df = flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
        if missing:
            st.error(f"Eksik sütunlar: {missing}.")
            st.stop()

        close     = df["Close"].squeeze()
        high      = df["High"].squeeze()
        low       = df["Low"].squeeze()
        volume    = df["Volume"].squeeze()
        close_arr = close.values
        n_bars    = len(close)

        indicator_min_reqs = {
            "SMA Crossover":    sma_long,
            "Bollinger Bands":  bb_period,
            "RSI":              rsi_period * 2,
            "MACD":             macd_slow + macd_signal,
            "Mean Reversion":   z_period,
            "OBV":              obv_long,
            "ADX":              adx_period * 3,
            "Stoch RSI":        rsi_period + stoch_rsi_period,
            "Ichimoku":         ichi_senkou_b + ichi_kijun,
            "KAMA":             kama_period + kama_slow,
            "SuperTrend":       st_period * 2,
            "LR Channel":       lrc_period,
            "WaveTrend":        wt_n1 + wt_n2,
            "Walk-Forward Opt": 150,
        }

        affected = [
            f"{name} (min {req} mum)"
            for name, req in indicator_min_reqs.items()
            if n_bars < req
        ]

        min_req = max(150, adx_period * 3, ichi_senkou_b)
        if n_bars < min_req:
            if affected:
                st.warning(
                    f"⚠️ Yeterli veri yok: **{n_bars} mum** mevcut, en az **{min_req}** gerekli.\n\n"
                    f"**Etkilenen indikatörler:** {', '.join(affected)}"
                )
            else:
                st.warning(f"Yeterli veri yok: {n_bars} mum, en az {min_req} gerekli.")

        cost_pct    = (commission_pct + slippage_pct) / 100
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]

        # ATR
        tr1        = high - low
        tr2        = (high - close.shift(1)).abs()
        tr3        = (low  - close.shift(1)).abs()
        tr         = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        atr_ma     = atr_series.rolling(atr_period, min_periods=atr_period).mean()
        atr_high   = (atr_series > atr_ma).values

        # ── YENİ: 200 EMA ─────────────────────────────────────────
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
        # ──────────────────────────────────────────────────────────

        # ── Swing Destek/Direnç (yatay) ───────────────────────────
        swing_levels = find_swing_levels(
            high, low, close,
            window=swing_window,
            min_touches=swing_touches,
            tolerance=swing_tol,
            atr_series=atr_series,
            atr_k=swing_atr_k,
        )

        # ── Diyagonal Trend Çizgileri ──────────────────────────────
        trendlines, tl_channels, tl_dates = find_trendlines(
            high, low, close,
            pivot_window=tl_pivot_window,
            max_lines=tl_max_lines,
            tolerance=tl_tolerance,
        )
        # ──────────────────────────────────────────────────────────

        # ============================================================
        # OPTİMİZASYON
        # ============================================================
        OPT_KEY = f"opt_v6_dsr_{ticker}_{period}_{interval}_{n_windows}_{train_pct}"

        if run_opt or OPT_KEY not in st.session_state:
            opt_params = {}
            opt_stats  = {}
            prog       = st.progress(0, text="Optimizasyon başlatılıyor…")
            algo_list  = list(PARAM_GRIDS.keys())

            for idx, algo_name in enumerate(algo_list):
                prog.progress(idx / len(algo_list), text=f"Optimize ediliyor: {algo_name}")
                grid = PARAM_GRIDS[algo_name]

                if algo_name == "SMA Crossover":
                    def make_fn():
                        def fn(p):
                            if p["sma_s"] >= p["sma_l"]: return None
                            s, _, _ = sig_sma(close, atr_high, p["sma_s"], p["sma_l"]); return s
                        return fn
                elif algo_name == "RSI":
                    def make_fn():
                        def fn(p):
                            if p["rsi_lower"] >= p["rsi_upper"]: return None
                            s, _ = sig_rsi_fn(close, p["rsi_period"], p["rsi_lower"], p["rsi_upper"]); return s
                        return fn
                elif algo_name == "Bollinger Bands":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_bb(close, p["bb_period"], p["bb_std"]); return s
                        return fn
                elif algo_name == "MACD":
                    def make_fn():
                        def fn(p):
                            if p["macd_fast"] >= p["macd_slow"]: return None
                            s, _, _ = sig_macd(close, atr_high, p["macd_fast"], p["macd_slow"], p["macd_signal"]); return s
                        return fn
                elif algo_name == "Mean Reversion":
                    def make_fn():
                        def fn(p):
                            s, _ = sig_z(close, p["z_period"], p["z_thresh"]); return s
                        return fn
                elif algo_name == "ADX":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_adx_fn(high, low, close, atr_high, p["adx_period"], p["adx_threshold"]); return s
                        return fn
                elif algo_name == "SuperTrend":
                    def make_fn():
                        def fn(p):
                            s, _, _, _, _ = sig_supertrend_fn(high, low, close, atr_high, p["st_period"], p["st_multiplier"]); return s
                        return fn
                elif algo_name == "LR Channel":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_lrc(close, p["lrc_period"], p["lrc_std_mult"]); return s
                        return fn
                elif algo_name == "WaveTrend":
                    def make_fn():
                        def fn(p):
                            s, _, _ = sig_wavetrend_fn(high, low, close, p["wt_n1"], p["wt_n2"], wt_ob, wt_os); return s
                        return fn
                elif algo_name == "OBV":
                    def make_fn():
                        def fn(p):
                            if p["obv_short"] >= p["obv_long"]: return None
                            s, _, _, _ = sig_obv(close, volume, p["obv_short"], p["obv_long"]); return s
                        return fn

                best_p, best_s = optimize_algo(
                    grid, make_fn(), close_arr, cost_pct,
                    n_windows=n_windows, train_pct=train_pct,
                    metric="Sharpe", min_trades=5,
                    bars_per_year=bars_per_year_from_interval(interval),
                    run_permutation=True, n_perm=200)
                opt_params[algo_name] = best_p
                opt_stats[algo_name]  = best_s if best_s else {"total_ret": 0.0, "sharpe": 0.0, "n": 0, "win_rate": 0.0}

            prog.progress(1.0, text="✅ Optimizasyon tamamlandı!")
            st.session_state[OPT_KEY] = {"params": opt_params, "stats": opt_stats}

            p = opt_params
            st.session_state["sma_short"]     = int(p["SMA Crossover"]["sma_s"])
            st.session_state["sma_long"]      = int(p["SMA Crossover"]["sma_l"])
            st.session_state["rsi_period"]    = int(p["RSI"]["rsi_period"])
            st.session_state["rsi_lower"]     = int(p["RSI"]["rsi_lower"])
            st.session_state["rsi_upper"]     = int(p["RSI"]["rsi_upper"])
            st.session_state["bb_period"]     = int(p["Bollinger Bands"]["bb_period"])
            st.session_state["bb_std"]        = float(p["Bollinger Bands"]["bb_std"])
            st.session_state["macd_fast"]     = int(p["MACD"]["macd_fast"])
            st.session_state["macd_slow"]     = int(p["MACD"]["macd_slow"])
            st.session_state["macd_signal"]   = int(p["MACD"]["macd_signal"])
            st.session_state["z_period"]      = int(p["Mean Reversion"]["z_period"])
            st.session_state["z_thresh"]      = float(p["Mean Reversion"]["z_thresh"])
            st.session_state["adx_period"]    = int(p["ADX"]["adx_period"])
            st.session_state["adx_threshold"] = int(p["ADX"]["adx_threshold"])
            st.session_state["st_period"]     = int(p["SuperTrend"]["st_period"])
            st.session_state["st_multiplier"] = float(p["SuperTrend"]["st_multiplier"])
            st.session_state["lrc_period"]    = int(p["LR Channel"]["lrc_period"])
            st.session_state["lrc_std_mult"]  = float(p["LR Channel"]["lrc_std_mult"])
            st.session_state["wt_n1"]         = int(p["WaveTrend"]["wt_n1"])
            st.session_state["wt_n2"]         = int(p["WaveTrend"]["wt_n2"])
            st.session_state["obv_short"]     = int(p["OBV"]["obv_short"])
            st.session_state["obv_long"]      = int(p["OBV"]["obv_long"])
            st.rerun()

        else:
            opt_params = st.session_state[OPT_KEY]["params"]
            opt_stats  = st.session_state[OPT_KEY]["stats"]

        p_sma  = {"sma_s": sma_short,   "sma_l": sma_long}
        p_rsi  = {"rsi_period": rsi_period, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper}
        p_bb   = {"bb_period": bb_period,   "bb_std": bb_std}
        p_macd = {"macd_fast": macd_fast,   "macd_slow": macd_slow, "macd_signal": macd_signal}
        p_z    = {"z_period": z_period,     "z_thresh": z_thresh}
        p_adx  = {"adx_period": adx_period, "adx_threshold": adx_threshold}
        p_st   = {"st_period": st_period,   "st_multiplier": st_multiplier}
        p_lrc  = {"lrc_period": lrc_period, "lrc_std_mult": lrc_std_mult}
        p_wt   = {"wt_n1": wt_n1,           "wt_n2": wt_n2}

        df["Sig_SMA"], df["SMA_SHORT"], df["SMA_LONG"] = sig_sma(
            close, atr_high, p_sma["sma_s"], p_sma["sma_l"])

        df["Sig_RSI"], df["RSI"] = sig_rsi_fn(
            close, p_rsi["rsi_period"], p_rsi["rsi_lower"], p_rsi["rsi_upper"])
        df["RSI_MA"] = df["RSI"].rolling(rsi_ma_period).mean()

        df["Sig_BB"], df["Mid"], df["Up"], df["Low_BB"] = sig_bb(
            close, p_bb["bb_period"], p_bb["bb_std"])

        df["Sig_MACD"], df["MACD"], df["MACD_S"] = sig_macd(
            close, atr_high, p_macd["macd_fast"], p_macd["macd_slow"], p_macd["macd_signal"])

        df["Sig_Z"], df["Z"] = sig_z(close, p_z["z_period"], p_z["z_thresh"])

        df["Sig_OBV"], df["OBV"], obv_sma_short, obv_sma_long = sig_obv(
            close, volume, obv_short, obv_long)

        df["Sig_ADX"], df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = sig_adx_fn(
            high, low, close, atr_high, p_adx["adx_period"], p_adx["adx_threshold"])

        df["Sig_StochRSI"], df["StochRSI_K"], df["StochRSI_D"] = sig_stochrsi(
            close, df["RSI"], stoch_rsi_period, stoch_d_period, stoch_lower, stoch_upper)

        df["Sig_Ichimoku"], df["Tenkan"], df["Kijun"], df["Senkou_A"], df["Senkou_B"] = sig_ichimoku(
            high, low, close, atr_high, ichi_tenkan, ichi_kijun, ichi_senkou_b)

        df["Sig_KAMA"], df["KAMA"] = sig_kama_fn(
            close, atr_high, kama_period, kama_fast, kama_slow)

        df["Sig_SuperTrend"], df["SuperTrend"], df["ST_Direction"], df["ST_Lower"], df["ST_Upper"] = sig_supertrend_fn(
            high, low, close, atr_high, p_st["st_period"], p_st["st_multiplier"])

        df["Sig_LRC"], df["LRC_Mid"], df["LRC_Upper"], df["LRC_Lower"] = sig_lrc(
            close, p_lrc["lrc_period"], p_lrc["lrc_std_mult"])

        df["NW_Line"], df["NW_Upper"], df["NW_Lower"] = calc_nadaraya_watson(
            close, bandwidth=nw_bandwidth, window=nw_window)

        df["ATR"]      = atr_series
        df["ATR_High"] = atr_high

        if is_intraday:
            df["Sig_VWAP"], df["VWAP"] = sig_vwap_fn(high, low, close, volume, vwap_band_pct)
        else:
            df["Sig_VWAP"] = 0
            df["VWAP"]     = np.nan

        df["Sig_WaveTrend"], df["WT1"], df["WT2"] = sig_wavetrend_fn(
            high, low, close, p_wt["wt_n1"], p_wt["wt_n2"], wt_ob, wt_os)

        fib_levels, fib_high, fib_low = calc_fibonacci(high, low, lookback=fib_lookback)

        df["Div_RSI"]  = detect_divergence(close, df["RSI"],  window=div_window)
        df["Div_MACD"] = detect_divergence(close, df["MACD"], window=div_window)

        # ============================================================
        # ANA GRAFİK + VRP
        # ============================================================
        from plotly.subplots import make_subplots

        bull_st = df["ST_Direction"] == 1
        bear_st = df["ST_Direction"] == -1

        st_dir_shifted = df["ST_Direction"].shift(1).fillna(0)
        st_buy_signal  = (df["ST_Direction"] == 1)  & (st_dir_shifted != 1)
        st_sell_signal = (df["ST_Direction"] == -1) & (st_dir_shifted != -1)

        lp = float(close.iloc[-1])
        pp = float(close.iloc[-2]) if len(close) > 1 else lp

        vrp_bins     = 40
        price_min    = float(low.min())
        price_max    = float(high.max())
        bin_edges    = np.linspace(price_min, price_max, vrp_bins + 1)
        bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_at_price = np.zeros(vrp_bins)
        for i in range(len(df)):
            lo_i  = float(low.iloc[i])
            hi_i  = float(high.iloc[i])
            vol_i = float(volume.iloc[i])
            if hi_i == lo_i:
                idx = np.clip(np.searchsorted(bin_edges, lo_i, side="right") - 1, 0, vrp_bins - 1)
                vol_at_price[idx] += vol_i
            else:
                for b in range(vrp_bins):
                    overlap = min(hi_i, bin_edges[b+1]) - max(lo_i, bin_edges[b])
                    if overlap > 0:
                        vol_at_price[b] += vol_i * overlap / (hi_i - lo_i)

        poc_idx   = int(np.argmax(vol_at_price))
        poc_price = bin_centers[poc_idx]
        max_vol   = vol_at_price.max()
        bar_colors = [
            "rgba(255,165,0,1.0)" if b == poc_idx
            else f"rgba(100,{int(80 + 175*(v/max_vol)) if max_vol > 0 else 200},255,0.85)"
            for b, v in enumerate(vol_at_price)
        ]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.75, 0.25],
            shared_yaxes=True,
            horizontal_spacing=0.0,
        )

        if chart_type == "Mum":
            # ── Sinyal bazlı mum renklendirme ─────────────────────
            _rsi_mid    = (rsi_lower + rsi_upper) / 2
            cyan_raw   = (df["ST_Direction"] == 1) & (df["Sig_OBV"] == 1) & (df["RSI"] < rsi_upper)
            cyan_mask  = cyan_raw & ~cyan_raw.shift(1).fillna(False)
            yellow_mask = (~cyan_mask) & (df["ADX"] < adx_threshold) & (df["RSI"] >= _rsi_mid - 5) & (df["RSI"] <= _rsi_mid + 5)
            red_mask   = (~cyan_mask) & (~yellow_mask) & (df["Close"] < df["Open"]) & (df["MACD"] < df["MACD_S"])
            green_mask = ~cyan_mask & ~yellow_mask & ~red_mask

            _color_defs = [
                ("Cyan AL",  cyan_mask,   "#00ffff"),
                ("Yeşil",    green_mask,  "#00cc66"),
                ("Sarı",     yellow_mask, "#ffcc00"),
                ("Ayı",      red_mask,    "#ff4444"),
            ]
            for _lbl, _mask, _color in _color_defs:
                _rising  = _mask & (df["Close"] >= df["Open"])
                _falling = _mask & (df["Close"] <  df["Open"])
                for _m, _fill, _trace_lbl in [
                    (_rising,  _color,   _lbl + " ↑"),
                    (_falling, "#111111", _lbl + " ↓"),
                ]:
                    if _m.any():
                        fig.add_trace(go.Candlestick(
                            x=df.index[_m],
                            open=df["Open"][_m], high=df["High"][_m],
                            low=df["Low"][_m],   close=df["Close"][_m],
                            name=_trace_lbl,
                            increasing_fillcolor=_fill, increasing_line_color=_color,
                            decreasing_fillcolor=_fill, decreasing_line_color=_color,
                            showlegend=False,
                        ), row=1, col=1)

            # ── Divergence marker katmanı (ana grafik) ────────────
            bull_div = (df["Div_RSI"] == 1) | (df["Div_MACD"] == 1)
            bear_div = (df["Div_RSI"] == -1) | (df["Div_MACD"] == -1)
            if bull_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bull_div], y=df["Low"][bull_div] * 0.998,
                    mode="markers", name="Bullish Div 🔺",
                    marker=dict(symbol="triangle-up", color="lime", size=10),
                ), row=1, col=1)
            if bear_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bear_div], y=df["High"][bear_div] * 1.002,
                    mode="markers", name="Bearish Div 🔻",
                    marker=dict(symbol="triangle-down", color="red", size=16),
                ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="orange", width=1.5)), row=1, col=1)

        # ── Mum renk legend girişleri (dummy scatter) ─────────────
        if chart_type == "Mum":
            for _leg_name, _leg_color in [
                ("🔴 Ayı",         "#ff4444"),
                ("🟡 Kararsız",    "#ffcc00"),
                ("🟢 Boğa",        "#00cc66"),
                ("🔵 Güçlü Boğa",  "#00ffff"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=_leg_name,
                    marker=dict(symbol="square", size=24, color=_leg_color),
                    showlegend=True,
                ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SHORT"],
            name=f"SMA {p_sma['sma_s']}", visible="legendonly",
            line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_LONG"],
            name=f"SMA {p_sma['sma_l']}", visible="legendonly",
            line=dict(color="cyan")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["KAMA"],
            name="KAMA", line=dict(color="violet", width=1.5)), row=1, col=1)

        # ── YENİ: 200 EMA trace ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"],
            name="EMA 200",
            line=dict(color="yellow", width=2, dash="dot"),
            visible="legendonly",
        ), row=1, col=1)
        # ──────────────────────────────────────────────────────────

        fig.add_trace(go.Scatter(
            x=df.index[bull_st], y=df["SuperTrend"][bull_st],
            name="SuperTrend (Boğa çizgi)", mode="lines",
            line=dict(color="rgba(0,255,100,0.5)", width=1.5),
            visible="legendonly", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index[bear_st], y=df["SuperTrend"][bear_st],
            name="SuperTrend (Ayı çizgi)", mode="lines",
            line=dict(color="rgba(255,60,60,0.5)", width=1.5),
            visible="legendonly", showlegend=True), row=1, col=1)

        if st_buy_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_buy_signal],
                y=df["SuperTrend"][st_buy_signal],
                name="SuperTrend AL",
                mode="markers+text",
                marker=dict(symbol="square", color="#00c853", size=18, line=dict(color="#00c853", width=0)),
                text="AL",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
            ), row=1, col=1)

        if st_sell_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_sell_signal],
                y=df["SuperTrend"][st_sell_signal],
                name="SuperTrend SAT",
                mode="markers+text",
                marker=dict(symbol="square", color="#d50000", size=18, line=dict(color="#d50000", width=0)),
                text="SAT",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"],
            name="LRC Orta", visible="legendonly",
            line=dict(color="white", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"],
            name="LRC Üst", visible="legendonly",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"],
            name="LRC Alt", visible="legendonly",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Line"],
            name="NW Orta", visible="legendonly",
            line=dict(color="gold", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Upper"],
            name="NW Üst", visible="legendonly",
            line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Lower"],
            name="NW Alt", visible="legendonly",
            line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,215,0,0.04)"), row=1, col=1)

        if is_intraday:
            fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"],
                name="VWAP", visible="legendonly",
                line=dict(color="yellow", dash="dash", width=1.5)), row=1, col=1)

        FIB_COLORS = {
            "0.0%":   "rgba(128,128,128,0.7)",
            "23.6%":  "rgba(255,165,0,0.8)",
            "38.2%":  "rgba(255,215,0,0.9)",
            "50.0%":  "rgba(255,255,255,0.9)",
            "61.8%":  "rgba(255,215,0,0.9)",
            "78.6%":  "rgba(255,165,0,0.8)",
            "100.0%": "rgba(128,128,128,0.7)",
        }
        for lvl_name, lvl_price in fib_levels.items():
            fig.add_hline(
                y=lvl_price,
                line_dash="dot",
                line_color=FIB_COLORS.get(lvl_name, "gray"),
                line_width=1,
                annotation_text=f"  Fib {lvl_name} {lvl_price:.2f}",
                annotation_font=dict(color=FIB_COLORS.get(lvl_name, "gray"), size=9, family="monospace"),
                annotation_position="top left",
                row=1, col=1,
            )

        # ── Yatay S/R çizgileri (güce göre kalınlık, break'te grileşme) ──
        for lvl in swing_levels:
            is_support = lvl["type"] == "S"
            t          = lvl["touches"]
            broken     = lvl.get("broken", False)

            # Kalınlık: dokunuş sayısına göre (1=ince, 2=orta, 3+=kalın)
            width = 1 if t <= 1 else (2 if t == 2 else 3)
            # Çizgi stili
            dash  = "dash" if t <= 1 else ("dashdot" if t == 2 else "solid")
            # Opaklık: 0.40 tabandan 0.80 tavana — görünür ama bunaltıcı değil
            alpha = min(0.40 + 0.15 * t, 0.80)

            if broken:
                # Kırılmış seviye: gri, yarı saydam
                color = f"rgba(160,160,160,{alpha*0.6:.2f})"
            else:
                color = (f"rgba(0,255,100,{alpha:.2f})" if is_support
                         else f"rgba(255,80,80,{alpha:.2f})")

            fig.add_hline(
                y=lvl["price"],
                line_color=color,
                line_width=width,
                line_dash=dash,
                row=1, col=1,
            )

        # ── Diyagonal Trend Çizgileri (legend toggle destekli) ────
        for tl in trendlines:
            is_sup  = tl["type"] == "support"
            color   = "rgba(0,255,120,0.9)" if is_sup else "rgba(255,80,80,0.9)"
            width   = 1 if tl["touches"] <= 2 else (2 if tl["touches"] <= 4 else 3)
            label   = f"{'↗ Destek' if is_sup else '↘ Direnç'} TL (x{tl['touches']})"
            x0_date = tl_dates[tl["x0"]]
            x1_date = tl_dates[tl["x1"]]
            fig.add_trace(go.Scatter(
                x=[x0_date, x1_date],
                y=[tl["y0"], tl["y1"]],
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash="solid"),
                visible="legendonly",
                legendgroup="trendlines",
                legendgrouptitle_text="Trend Çizgileri" if tl == trendlines[0] else None,
            ), row=1, col=1)

        # ── Kanal dolgusu (legend toggle destekli) ────────────────
        if tl_show_channel:
            for ci, ch in enumerate(tl_channels):
                sl   = ch["support"];  rl = ch["resistance"]
                xi0  = max(sl["x0"], rl["x0"])
                xi1  = sl["x1"]
                xs   = [tl_dates[xi0], tl_dates[xi1],
                        tl_dates[xi1], tl_dates[xi0], tl_dates[xi0]]
                y_s0 = sl["slope"] * xi0 + sl["intercept"]
                y_s1 = sl["slope"] * xi1 + sl["intercept"]
                y_r0 = rl["slope"] * xi0 + rl["intercept"]
                y_r1 = rl["slope"] * xi1 + rl["intercept"]
                ys   = [y_s0, y_s1, y_r1, y_r0, y_s0]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    fill="toself",
                    fillcolor="rgba(100,180,255,0.07)",
                    line=dict(width=0),
                    mode="lines",
                    name=f"Kanal {ci+1}",
                    visible="legendonly",
                    legendgroup="trendlines",
                    showlegend=True,
                ), row=1, col=1)
        # ──────────────────────────────────────────────────────────

        fig.add_trace(go.Bar(
            x=vol_at_price, y=bin_centers,
            orientation="h",
            marker_color=bar_colors,
            name="Hacim Profili",
            showlegend=False,
            hovertemplate="Fiyat: %{y:.2f}<br>Hacim: %{x:,.0f}<extra></extra>",
        ), row=1, col=2)

        fig.add_hline(y=poc_price, line_dash="dash", line_color="orange",
            annotation_text=f"POC {poc_price:.2f}",
            annotation_font=dict(color="orange", size=10, family="monospace"),
            annotation_bgcolor="rgba(255,165,0,0.15)",
            annotation_position="top right", row=1, col=2)
        fig.add_hline(y=lp, line_dash="dot", line_color="lime" if lp >= pp else "red",
            annotation_text=f"  {lp:.2f}",
            annotation_font=dict(color="lime" if lp >= pp else "red", size=12, family="monospace"),
            annotation_bgcolor="rgba(0,255,0,0.12)" if lp >= pp else "rgba(255,0,0,0.12)",
            annotation_position="bottom right", row=1, col=2)

        fig.add_annotation(text=f"<b>{ticker}  {lp:,.4f}</b>",
            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
            font=dict(size=13, color="#007a3d" if lp >= pp else "#cc2200", family="monospace"),
            align="left", bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(200,200,200,0.5)", borderwidth=1, borderpad=4)

        fig.update_layout(
            template="plotly_dark", height=580,
            dragmode="pan",
            xaxis=dict(rangeslider_visible=False),
            xaxis2=dict(showgrid=False, showticklabels=False),
            yaxis2=dict(showticklabels=False),
            legend=dict(
                orientation="v",
                x=-0.02, y=1,
                xanchor="right", yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
                itemwidth=30,
                itemsizing="constant",
                tracegroupgap=4,
            ),
            margin=dict(l=110, r=10, t=30, b=30),
        )

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # ============================================================
        # ANA GRAFİK REHBERİ (Expander)
        # ============================================================
        with st.expander("📖 Ana Grafikte Ne Ne Anlama Geliyor? (Detaylı Rehber)", expanded=False):
            st.markdown("""
### 🕯️ Mum Renkleri

Her mum 4 kategoriden birine atanır. Hiyerarşik sıralama: önce **Cyan** kontrol edilir, olmazsa **Sarı**, olmazsa **Kırmızı**, kalanlar **Yeşil**.

| Renk | Anlamı | Tetikleyici |
|---|---|---|
| 🔵 **Cyan (Güçlü Boğa)** | Taze AL sinyali | SuperTrend yukarı **VE** OBV birikim **VE** RSI aşırı alım değil **VE** önceki barda bu koşul yoktu |
| 🟢 **Yeşil (Boğa)** | Normal yükseliş bağlamı | Diğer üç kategoriye girmeyen mumlar (varsayılan) |
| 🟡 **Sarı (Kararsız)** | Yatay/düşük momentum | ADX zayıf **VE** RSI nötr bölgede (eşiklerin ortası ±5) |
| 🔴 **Kırmızı (Ayı)** | Momentumlu düşüş | Düşüş mumu **VE** MACD negatif |

**Gövde dolgu farkı:** Yükselen mumlar (Close ≥ Open) kategori renginde **dolu**. Düşen mumlar (Close < Open) **içi siyah**, kenarı kategori renginde. Böylece hem renk kategorisi hem yön tek bakışta görünür.

---

### 📈 Hareketli Ortalamalar & Trend

| Çizgi | Renk/Stil | Neyi Gösterir |
|---|---|---|
| **SMA Kısa** | Turuncu | Kısa vadeli trend ortalaması (varsayılan 20 bar). Fiyat altındaysa zayıflık, üstündeyse güç |
| **SMA Uzun** | Cyan | Orta vadeli ortalama (varsayılan 200 bar). Trend yönü anchor'ı |
| **KAMA** | Mor | Kaufman Adaptif MA — volatiliteye göre hız değiştirir. Yatayda düz, trend başlayınca hızlanır |
| **EMA 200** | Sarı, noktalı | Uzun vadeli trend filtresi. Fiyat üstündeyse "boğa piyasası", altındaysa "ayı piyasası" |
| **SuperTrend çizgisi** | Yeşil (boğa) / Kırmızı (ayı) | ATR tabanlı trend takip. Çizginin rengi mevcut rejimi söyler |
| **🔼 SuperTrend AL** | Yeşil kare, beyaz "AL" yazısı | ST rejimi AYI'dan BOĞA'ya geçti — trend değişim sinyali |
| **🔽 SuperTrend SAT** | Kırmızı kare, beyaz "SAT" yazısı | ST rejimi BOĞA'dan AYI'ya geçti |

💡 **İpucu:** SMA kısa > SMA uzun → "altın haç" (golden cross) bağlamı. EMA200 üstünde kalan bir fiyat, SMA ve KAMA'nın da yukarı eğimiyle birleşirse **çok katmanlı trend teyidi** vardır.

---

### 📊 Kanallar & Zarflar

| Element | Renk | Neyi Gösterir |
|---|---|---|
| **LRC Orta** | Beyaz kesikli | Linear Regression Channel — periyoda göre fiyatın istatistiksel orta çizgisi |
| **LRC Üst** | Gri noktalı | Orta + N standart sapma. Fiyat burada = kanalın üst sınırı, olası SAT bölgesi |
| **LRC Alt** | Gri noktalı | Orta - N standart sapma. Fiyat burada = kanalın alt sınırı, olası AL bölgesi |
| **NW Orta** | Altın sarısı | Nadaraya-Watson kernel smoother — fiyatın yumuşatılmış trendi |
| **NW Üst/Alt** | Sarı soluk + dolgu | NW zarfı — fiyat üstündeyse aşırı alım, altındaysa aşırı satım |

**LRC vs NW farkı:** LRC doğrusal regresyon (sabit açı), NW lokal ağırlıklı regresyon (kıvrımlı). Trendli piyasada LRC, dönüş noktalarında NW daha iyi çalışır.

---

### 📐 Fibonacci Seviyeleri

Son `fib_lookback` bar içindeki en yüksek ve en düşük fiyat arasına çizilir. Yedi seviye:

| Seviye | Renk (tipik) | Yorum |
|---|---|---|
| **0.0%** | Kırmızı | Swing dibi (aşağı hareket) / Swing tepesi (yukarı hareket) |
| **23.6%** | Turuncu | Hafif geri çekilme — güçlü trendde burada dönüş beklenir |
| **38.2%** | Sarı | Orta seviye geri çekilme — normal trend correction |
| **50.0%** | Yeşil | Yarı yarıya geri çekilme — psikolojik seviye (Fib değil ama eklenmiştir) |
| **61.8%** | Mavi | Altın oran — en önemli fib, burası kırılırsa trend zayıflar |
| **78.6%** | Mor | Derin geri çekilme — trendin sonu yaklaşıyor |
| **100.0%** | Kırmızı | Swing'in diğer ucu — tam tersine çevirmiş demek |

💡 Fiyatın hangi Fib seviyesinde durduğu Rapor'da "**Büyük Resim**" adımında gösterilir.

---

### 🎯 Yatay Destek / Direnç

Swing pivot tespiti + ATR tabanlı gruplama ile otomatik çiziliyor.

| Görünüm | Anlamı |
|---|---|
| **Yeşil yatay çizgi** | Aktif **destek** (fiyatın altında) |
| **Kırmızı yatay çizgi** | Aktif **direnç** (fiyatın üstünde) |
| **Gri yatay çizgi** | Kırılmış seviye — artık aktif değil, referans için duruyor |

**Kalınlık/stil dokunuş sayısını söyler:**
- **İnce, dash** (— —) → 1 dokunuş (zayıf)
- **Orta, dashdot** (—·—·) → 2 dokunuş (orta)
- **Kalın, solid** (———) → 3+ dokunuş (güçlü)

**🔄 Role-Reversal (Rol Değişimi):**
- Fiyat eski bir direnci kırıp yukarı geçerse → o seviye **destek** rolüne geçer (yeşile döner)
- Fiyat eski bir desteği kırıp aşağı inerse → o seviye **direnç** rolüne geçer (kırmızıya döner)
- Klasik teknik analiz prensibi: "eski direnç yeni destektir"

---

### 📏 Diyagonal Trend Çizgileri (Legend'dan aç/kapa)

Pivot high'ları birleştirince **direnç TL**, pivot low'ları birleştirince **destek TL** oluşur. Legend başlığı "Trend Çizgileri" altında:

| Görünüm | Anlamı |
|---|---|
| **↗ Destek TL (xN)** yeşil | Yükselen trend çizgisi, N dokunuşla doğrulanmış |
| **↘ Direnç TL (xN)** kırmızı | Düşen trend çizgisi, N dokunuşla doğrulanmış |
| **Mavimsi dolgu alan** | Paralel kanal — fiyatın içinde hareket etmesi beklenen koridor |

Dokunuş sayısı (xN) arttıkça çizgi daha kalın çizilir. 5+ dokunuşlu bir trend çizgisinin kırılması çok anlamlıdır.

---

### 🔻 Divergence İşaretleri

| Sembol | Renk | Anlamı |
|---|---|---|
| **🔺 Bullish Div** | Yeşil üçgen (mumun altında) | Fiyat daha düşük dip yaptı **ama** RSI veya MACD daha yüksek dip yaptı → gizli güç, dönüş sinyali |
| **🔻 Bearish Div** | Kırmızı üçgen (mumun üstünde) | Fiyat daha yüksek tepe yaptı **ama** RSI veya MACD daha düşük tepe yaptı → zayıflama, düşüş uyarısı |

💡 Divergence tek başına giriş sinyali değildir — başka teyitlerle birlikte değerlendir.

---

### 📦 Volume Profile (Sağ Panel) & POC

Grafiğin sağında yatay hacim çubukları var. Her çubuk, o fiyat seviyesinde geçmişte **ne kadar hacim** gerçekleştiğini gösterir.

| Element | Renk | Anlamı |
|---|---|---|
| **POC (Point of Control)** | Turuncu kesikli yatay çizgi + etiket | En yüksek hacimli fiyat seviyesi — piyasanın "adil değer"i kabul edilir |
| **Mavi-yeşil tonlu çubuklar** | Yoğunluğa göre renk | Hacim arttıkça daha doygun yeşile kayar |
| **Son fiyat etiketi** | Yeşil (POC üstünde) / Kırmızı (POC altında) | Fiyatın POC'a göre konumu |

**Nasıl yorumlanır?**
- Fiyat POC'un **altında** → piyasa ucuza düşmüş, alıcılar devreye girebilir
- Fiyat POC'un **üstünde** → değerinin üstünde, satış baskısı gelebilir
- **Boş hacim bölgeleri** (az çubuk) = fiyat hızlı geçiyor, güçlü hareket zonu
- **Dolu hacim bölgeleri** = konsolidasyon, güçlü destek/direnç

---

### 💡 Hepsini Birlikte Nasıl Okumalı?

Görsel bir **çoklu-teyit sistemi** olarak tasarlanmış. Tek bir sinyale değil, **birbiriyle örtüşen** sinyallere güvenin:

1. **Büyük resim:** Fiyat EMA200'ün neresinde? Trend mi yatay mı?
2. **Rejim:** SuperTrend ne diyor? Kısa MA uzun MA'nın neresinde?
3. **Seviye:** Fiyat hangi Fib / LRC / S/R seviyesinde?
4. **Momentum:** Mum rengi ne? Cyan/Yeşil mi, Sarı/Kırmızı mı?
5. **Uyarı:** Divergence var mı? Kırılmış seviyeler hangileri?
6. **Hacim:** POC'un neresinde? Volume profile dağılımı nasıl?

Üç veya daha fazla sinyal **aynı yönü gösteriyorsa** konfidans yüksektir. Çelişiyorsa → **bekle**.

> ⚠️ **Not:** Bu rehber sadece grafik elementlerini açıklar. Alt sekmelerdeki göstergelerin (RSI, MACD, ADX vb.) detaylı yorumu her sekmenin kendi "📖 Nasıl Okunur?" bölümündedir.
""")

        # ============================================================
        # ALT GRAFİKLER
        # ============================================================
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "RSI", "MACD", "ADX", "OBV", "Stoch RSI", "Ichimoku", "SuperTrend",
            "KAMA & LRC", "Nadaraya-Watson", "WaveTrend", "Divergence"])

        with tab1:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="rgba(0,200,100,0.9)", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,200,100,0.15)"))
            f.add_trace(go.Scatter(x=df.index, y=df["RSI_MA"],
                name=f"RSI MA({rsi_ma_period})", line=dict(color="yellow", width=1.5, dash="dot")))
            f.add_hline(y=p_rsi["rsi_lower"], line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({p_rsi['rsi_lower']})")
            f.add_hline(y=p_rsi["rsi_upper"], line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({p_rsi['rsi_upper']})")
            f.add_hline(y=50, line_dash="dot", line_color="gray")
            bull_div_rsi = df["Div_RSI"] == 1
            bear_div_rsi = df["Div_RSI"] == -1
            if bull_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_rsi], y=df["RSI"][bull_div_rsi],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_rsi], y=df["RSI"][bear_div_rsi],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 RSI Nasıl Okunur?"):
                st.markdown("""
**RSI (Relative Strength Index)** — 0–100 arasında salınan momentum göstergesidir.

| Bölge | Anlam |
|---|---|
| RSI < Aşırı Satım eşiği | 🟢 Aşırı satılmış → potansiyel AL sinyali |
| RSI > Aşırı Alım eşiği | 🔴 Aşırı alınmış → potansiyel SAT sinyali |
| RSI ~ 50 | ⚪ Nötr bölge |

- **RSI MA (sarı noktalı):** RSI'nın hareketli ortalaması. RSI bu çizgiyi yukarı keserse momentum güçleniyor demektir.
- **Bullish Divergence 🔺:** Fiyat düşük dip yaparken RSI yüksek dip yapıyor → güçlü dönüş sinyali.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe yaparken RSI alçak tepe yapıyor → zayıflama uyarısı.
                """)

        with tab2:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="cyan")))
            f.add_trace(go.Scatter(x=df.index, y=df["MACD_S"], name="Sinyal", line=dict(color="orange")))
            hist = df["MACD"] - df["MACD_S"]
            f.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in hist], opacity=0.5))
            bull_div_macd = df["Div_MACD"] == 1
            bear_div_macd = df["Div_MACD"] == -1
            if bull_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_macd], y=df["MACD"][bull_div_macd],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_macd], y=df["MACD"][bear_div_macd],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 MACD Nasıl Okunur?"):
                st.markdown("""
**MACD (Moving Average Convergence Divergence)** — trend yönü ve momentumu ölçer.

| Unsur | Anlam |
|---|---|
| MACD > Sinyal çizgisi | 🟢 Yukarı momentum → AL eğilimi |
| MACD < Sinyal çizgisi | 🔴 Aşağı momentum → SAT eğilimi |
| Histogram yeşil & büyüyor | 🟢 Momentum güçleniyor |
| Histogram kırmızı & büyüyor | 🔴 Momentum zayıflıyor |

- **Sıfır çizgisi geçişi:** MACD sıfırı yukarı kesiyor = güçlü boğa sinyali; aşağı kesiyor = ayı sinyali.
- **Bullish Divergence 🔺:** Fiyat düşük dip, MACD yüksek dip → trend dönüş öncüsü.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe, MACD alçak tepe → zirve uyarısı.
                """)

        with tab3:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["ADX"],      name="ADX", line=dict(color="yellow", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["PLUS_DI"],  name="+DI", line=dict(color="lime", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["MINUS_DI"], name="-DI", line=dict(color="red",  dash="dot")))
            f.add_hline(y=p_adx["adx_threshold"], line_dash="dash", line_color="white",
                annotation_text=f"Trend Eşiği ({p_adx['adx_threshold']})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 ADX Nasıl Okunur?"):
                st.markdown("""
**ADX (Average Directional Index)** — trendin gücünü ölçer (yön değil, sadece güç).

| ADX Değeri | Trend Gücü |
|---|---|
| < 20 | Zayıf / yatay piyasa |
| 20–25 | Trend oluşuyor |
| > 25 | Güçlü trend |
| > 40 | Çok güçlü trend |

- **+DI (yeşil):** Yukarı yönlü hareketin gücü.
- **-DI (kırmızı):** Aşağı yönlü hareketin gücü.
- **+DI > -DI ve ADX > eşik:** 🟢 Güçlü yükseliş trendi.
- **-DI > +DI ve ADX > eşik:** 🔴 Güçlü düşüş trendi.
- ADX düşükken verilen sinyaller güvenilmezdir.
                """)

        with tab4:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="dodgerblue")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_short,
                name=f"OBV SMA {obv_short}", line=dict(color="orange", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_long,
                name=f"OBV SMA {obv_long}", line=dict(color="cyan", dash="dot")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 OBV Nasıl Okunur?"):
                st.markdown("""
**OBV (On-Balance Volume)** — hacim akışını kümülatif olarak izler; fiyat hareketini önceden haber verebilir.

| Durum | Anlam |
|---|---|
| OBV yükseliyor, fiyat yükseliyor | 🟢 Trend onaylanıyor |
| OBV yükseliyor, fiyat düşüyor | 🟢 Gizli birikim → potansiyel yukarı kırılım |
| OBV düşüyor, fiyat yükseliyor | 🔴 Dağıtım var → zayıflama uyarısı |
| OBV düşüyor, fiyat düşüyor | 🔴 Trend onaylanıyor |

- **Kısa SMA (turuncu) > Uzun SMA (cyan):** OBV momentumu pozitif → AL eğilimi.
- **Kısa SMA < Uzun SMA:** OBV momentumu negatif → SAT eğilimi.
- OBV'nin mutlak değeri değil, eğimi önemlidir.
                """)

        with tab5:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"], name="%K", line=dict(color="magenta")))
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"], name="%D", line=dict(color="orange", dash="dot")))
            f.add_hline(y=stoch_lower, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({stoch_lower})")
            f.add_hline(y=stoch_upper, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({stoch_upper})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Stochastic RSI Nasıl Okunur?"):
                st.markdown("""
**Stochastic RSI** — RSI'ya uygulanan Stochastic göstergesidir. RSI'dan daha hassas ve hızlıdır.

| Bölge | Anlam |
|---|---|
| %K < Aşırı Satım eşiği | 🟢 Aşırı satılmış → AL bölgesi |
| %K > Aşırı Alım eşiği | 🔴 Aşırı alınmış → SAT bölgesi |

- **%K (mor):** Hızlı çizgi — anlık sinyal verir.
- **%D (turuncu noktalı):** %K'nın ortalaması — yavaş, daha güvenilir.
- **%K, %D'yi aşırı satım bölgesinde yukarı kesiyor:** 🟢 Güçlü AL sinyali.
- **%K, %D'yi aşırı alım bölgesinde aşağı kesiyor:** 🔴 Güçlü SAT sinyali.
- RSI aşırı bölgelerde değilken Stoch RSI sinyalleri daha az güvenilirdir.
                """)

        with tab6:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index, y=df["Tenkan"], name="Tenkan-sen", line=dict(color="cyan",  width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Kijun"],  name="Kijun-sen",  line=dict(color="red",   width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_A"], name="Senkou A",
                line=dict(color="lime", width=0.5, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_B"], name="Senkou B",
                line=dict(color="red", width=0.5, dash="dot"),
                fill="tonexty", fillcolor="rgba(100,100,100,0.15)"))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Ichimoku Nasıl Okunur?"):
                st.markdown("""
**Ichimoku Kinko Hyo** — trend yönü, destek/direnç ve momentum'u tek grafikte gösterir.

| Unsur | Renk | Anlam |
|---|---|---|
| Tenkan-sen | Cyan | Kısa vadeli denge çizgisi (9 bar) |
| Kijun-sen | Kırmızı | Orta vadeli denge çizgisi (26 bar) |
| Senkou Span A | Yeşil | Bulutun üst sınırı |
| Senkou Span B | Kırmızı | Bulutun alt sınırı |

**Okuma Kuralları:**
- **Fiyat bulutun üstünde:** 🟢 Yükseliş trendi.
- **Fiyat bulutun altında:** 🔴 Düşüş trendi.
- **Fiyat bulut içinde:** ⚪ Konsolidasyon.
- **Tenkan > Kijun:** 🟢 Kısa vadeli momentum pozitif.
- **Yeşil bulut (Span A > Span B):** Boğa piyasası.
- **Kırmızı bulut (Span B > Span A):** Ayı piyasası.
                """)

        with tab7:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index[bull_st], y=df["SuperTrend"][bull_st],
                name="SuperTrend (Boğa)", mode="lines", line=dict(color="lime", width=2)))
            f.add_trace(go.Scatter(x=df.index[bear_st], y=df["SuperTrend"][bear_st],
                name="SuperTrend (Ayı)", mode="lines", line=dict(color="red", width=2)))
            if st_buy_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_buy_signal], y=df["SuperTrend"][st_buy_signal],
                    name="AL", mode="markers+text",
                    marker=dict(symbol="square", color="#00c853", size=18, line=dict(width=0)),
                    text="AL",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            if st_sell_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_sell_signal], y=df["SuperTrend"][st_sell_signal],
                    name="SAT", mode="markers+text",
                    marker=dict(symbol="square", color="#d50000", size=18, line=dict(width=0)),
                    text="SAT",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 SuperTrend Nasıl Okunur?"):
                st.markdown("""
**SuperTrend** — ATR tabanlı dinamik destek/direnç çizgisidir.

| Durum | Anlam |
|---|---|
| Çizgi yeşil (fiyatın altında) | 🟢 Yükseliş trendi — uzun pozisyon |
| Çizgi kırmızı (fiyatın üstünde) | 🔴 Düşüş trendi — kısa pozisyon |
| 🟩 AL kutusu | ⚡ Ayıdan boğaya geçiş — trend dönüşü |
| 🟥 SAT kutusu | ⚡ Boğadan ayıya geçiş — trend dönüşü |

- **ATR Çarpanı (multiplier):** Yüksek değer → daha az sinyal, daha az gürültü.
- **En güçlü sinyal:** SuperTrend AL/SAT + ADX > eşik değeri kombinasyonu.
- Yatay piyasalarda yanlış sinyal üretebilir; ADX filtresiyle kullanın.
                """)

        with tab8:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(color="white", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["KAMA"], name="KAMA", line=dict(color="violet", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"], name="LRC Orta",
                line=dict(color="white", dash="dash", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"], name="LRC Üst",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"], name="LRC Alt",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot"),
                fill="tonexty", fillcolor="rgba(150,150,150,0.07)"))
            f.update_layout(**sub_layout(height=350))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 KAMA & LR Channel Nasıl Okunur?"):
                st.markdown("""
**KAMA (Kaufman Adaptive Moving Average)** — piyasa koşullarına göre hız adapte eden akıllı bir ortalamadır.

| Durum | Anlam |
|---|---|
| Fiyat > KAMA | 🟢 Yükseliş eğilimi |
| Fiyat < KAMA | 🔴 Düşüş eğilimi |
| KAMA düz seyrediyor | ⚪ Piyasa yatay, bekle |

**LR Channel (Linear Regression Channel)**

| Durum | Anlam |
|---|---|
| Fiyat alt banda değiyor | 🟢 Potansiyel destek / AL bölgesi |
| Fiyat üst banda değiyor | 🔴 Potansiyel direnç / SAT bölgesi |
                """)

        with tab9:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(color="white", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Line"], name="NW Orta", line=dict(color="gold", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Upper"], name="NW Üst",
                line=dict(color="red", width=1, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Lower"], name="NW Alt",
                line=dict(color="lime", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(255,215,0,0.05)"))
            nw_ob = close > df["NW_Upper"]
            nw_os = close < df["NW_Lower"]
            if nw_ob.any():
                f.add_trace(go.Scatter(x=df.index[nw_ob], y=close[nw_ob],
                    name="Aşırı Alım", mode="markers", marker=dict(color="red", size=6)))
            if nw_os.any():
                f.add_trace(go.Scatter(x=df.index[nw_os], y=close[nw_os],
                    name="Aşırı Satım", mode="markers", marker=dict(color="lime", size=6)))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Nadaraya-Watson Nasıl Okunur?"):
                st.markdown("""
**Nadaraya-Watson Envelope** — çekirdek regresyon ile hesaplanan non-parametrik bir zarf göstergesidir.

| Durum | Anlam |
|---|---|
| Fiyat üst zarfın üstünde 🔴 | Aşırı alım — geri çekilme beklenebilir |
| Fiyat alt zarfın altında 🟢 | Aşırı satım — toparlanma beklenebilir |
| Fiyat zarf içinde | Normal seyir |
                """)

        with tab10:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["WT1"], name="WT1",
                line=dict(color="cyan", width=1.5)))
            f.add_trace(go.Scatter(x=df.index, y=df["WT2"], name="WT2",
                line=dict(color="orange", width=1.5, dash="dot")))
            wt_hist = df["WT1"] - df["WT2"]
            f.add_trace(go.Bar(x=df.index, y=wt_hist, name="WT Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in wt_hist], opacity=0.4))
            f.add_hline(y=wt_ob, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({wt_ob})")
            f.add_hline(y=wt_os, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({wt_os})")
            f.add_hline(y=0, line_dash="dot", line_color="gray")
            wt_buy  = df["Sig_WaveTrend"] == 1
            wt_sell = df["Sig_WaveTrend"] == -1
            if wt_buy.any():
                f.add_trace(go.Scatter(x=df.index[wt_buy], y=df["WT1"][wt_buy],
                    name="AL", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if wt_sell.any():
                f.add_trace(go.Scatter(x=df.index[wt_sell], y=df["WT1"][wt_sell],
                    name="SAT", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout(height=300))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 WaveTrend Nasıl Okunur?"):
                st.markdown("""
**WaveTrend (WT_CROSS_LB)** — momentum ve aşırı bölge tespiti için kullanılan osilatördür.

| Unsur | Anlam |
|---|---|
| WT1 (cyan) | Hızlı sinyal çizgisi |
| WT2 (turuncu noktalı) | Yavaş sinyal çizgisi |

- **WT1, WT2'yi aşırı satım bölgesinde yukarı kesiyor 🔺:** Güçlü AL sinyali.
- **WT1, WT2'yi aşırı alım bölgesinde aşağı kesiyor 🔻:** Güçlü SAT sinyali.
                """)

        with tab11:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="red", width=1.5)))
            bull_div_r = df["Div_RSI"]  == 1
            bear_div_r = df["Div_RSI"]  == -1
            bull_div_m = df["Div_MACD"] == 1
            bear_div_m = df["Div_MACD"] == -1
            if bull_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_r], y=close[bull_div_r],
                    name="RSI Bullish Div", mode="markers",
                    marker=dict(color="lime", size=12, symbol="triangle-up")))
            if bear_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_r], y=close[bear_div_r],
                    name="RSI Bearish Div", mode="markers",
                    marker=dict(color="red", size=12, symbol="triangle-down")))
            if bull_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_m], y=close[bull_div_m],
                    name="MACD Bullish Div", mode="markers",
                    marker=dict(color="aquamarine", size=10, symbol="diamond")))
            if bear_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_m], y=close[bear_div_m],
                    name="MACD Bearish Div", mode="markers",
                    marker=dict(color="salmon", size=10, symbol="diamond")))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False,
                title_text="Divergence Noktaları (Fiyat Grafiği Üzerinde)")
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Divergence Nasıl Okunur?"):
                st.markdown("""
**Divergence (Uyumsuzluk)** — fiyat hareketi ile indikatör arasındaki zıtlık; trend dönüşünün erken habercisidir.

| Tür | Fiyat | İndikatör | Anlam |
|---|---|---|---|
| Bullish Div 🔺 | Düşük dip | Yüksek dip | 🟢 Satış baskısı azalıyor → yukarı dönüş olabilir |
| Bearish Div 🔻 | Yüksek tepe | Düşük tepe | 🔴 Alış gücü zayıflıyor → aşağı dönüş olabilir |
                """)

        # ============================================================
        # KARAR TABLOSU
        # ============================================================
        last       = df.iloc[-1]
        last_close = safe_scalar(last["Close"])
        last_ath   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False

        # ── Son bar indikatör değerleri (hem Kombine Skor hem Teknik Rapor kullanır) ──
        r_close    = safe_scalar(last["Close"])
        r_kama     = safe_scalar(last["KAMA"])
        r_adx      = safe_scalar(last["ADX"])
        r_pdi      = safe_scalar(last["PLUS_DI"])
        r_mdi      = safe_scalar(last["MINUS_DI"])
        r_macd     = safe_scalar(last["MACD"])
        r_macds    = safe_scalar(last["MACD_S"])
        r_rsi      = safe_scalar(last["RSI"])
        r_stk      = safe_scalar(last["StochRSI_K"])
        r_std      = safe_scalar(last["ST_Direction"])
        r_lrc_sig  = safe_scalar(last["Sig_LRC"])
        r_lrc_mid  = safe_scalar(last["LRC_Mid"])
        r_lrc_up   = safe_scalar(last["LRC_Upper"])
        r_lrc_lo   = safe_scalar(last["LRC_Lower"])
        r_nw       = safe_scalar(last["NW_Line"])
        r_nw_up    = safe_scalar(last["NW_Upper"])
        r_nw_lo    = safe_scalar(last["NW_Lower"])
        r_vwap     = safe_scalar(last["VWAP"])     if is_intraday else np.nan
        r_vwap_sig = safe_scalar(last["Sig_VWAP"]) if is_intraday else 0
        r_obv_sig  = safe_scalar(last["Sig_OBV"])
        r_div_rsi  = safe_scalar(last["Div_RSI"])
        r_div_mac  = safe_scalar(last["Div_MACD"])
        r_ichi     = safe_scalar(last["Sig_Ichimoku"])
        r_wt1      = safe_scalar(last["WT1"])
        r_atr_hi   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False
        r_ema200   = safe_scalar(last["EMA200"])

        # ── ADAPTİF ADX EŞİĞİ ──────────────────────────────────────
        # Volatiliteye göre ADX eşiğini otomatik ayarla.
        # Yüksek volatilitede (gürültülü) trend için daha yüksek eşik iste;
        # düşük volatilitede ise daha düşük eşik yeterli.
        # Kullanıcının manuel eşiği baz alınır, üzerine volatilite düzeltmesi uygulanır.
        r_atr      = safe_scalar(last["ATR"])
        r_atr_ma   = safe_scalar(atr_ma.iloc[-1]) if len(atr_ma) else np.nan
        if not (np.isnan(r_atr) or np.isnan(r_atr_ma)) and r_atr_ma > 0:
            atr_ratio = r_atr / r_atr_ma
        else:
            atr_ratio = 1.0

        # Volatilite düzeltmesi: ATR oranı ±%20'yi aşarsa ±5 puan oynat
        if atr_ratio > 1.2:
            adx_threshold_adaptive = min(adx_threshold + 5, 40)
            adx_regime_note = f"Yüksek vol. (ATR×{atr_ratio:.2f}) → eşik +5"
        elif atr_ratio < 0.8:
            adx_threshold_adaptive = max(adx_threshold - 5, 15)
            adx_regime_note = f"Düşük vol. (ATR×{atr_ratio:.2f}) → eşik -5"
        else:
            adx_threshold_adaptive = adx_threshold
            adx_regime_note = f"Normal vol. (ATR×{atr_ratio:.2f}) → eşik değişmedi"

        if fib_levels:
            r_fib_closest = min(fib_levels.items(), key=lambda x: abs(x[1] - r_close))
        else:
            r_fib_closest = ("N/A", r_close)

        res        = []

        def trend_dec(raw_dec, atr_ok):
            return raw_dec if atr_ok else "TUT (düşük vol.)"

        lss = safe_scalar(last["SMA_SHORT"])
        lsl = safe_scalar(last["SMA_LONG"])
        if not (np.isnan(lss) or np.isnan(lsl)):
            res.append([trend_dec("AL" if lss > lsl else "SAT", last_ath),
                        f"SMA ({p_sma['sma_s']}/{p_sma['sma_l']})", "Trend yönü."])
        else:
            res.append(["N/A", "SMA Crossover", "Yetersiz veri."])

        lr = safe_scalar(last["RSI"])
        if not np.isnan(lr):
            dec = "AL" if lr < p_rsi["rsi_lower"] else ("SAT" if lr > p_rsi["rsi_upper"] else "TUT")
            res.append([dec, f"RSI ({p_rsi['rsi_period']}) [{p_rsi['rsi_lower']}/{p_rsi['rsi_upper']}]", f"Seviye: {lr:.1f}"])
        else:
            res.append(["N/A", "RSI", "Yetersiz veri."])

        lup = safe_scalar(last["Up"])
        llb = safe_scalar(last["Low_BB"])
        if not any(np.isnan(v) for v in [last_close, llb, lup]):
            dec = "AL" if last_close < llb else ("SAT" if last_close > lup else "TUT")
            res.append([dec, f"Bollinger Bands (σ={p_bb['bb_std']})", "Fiyatın kanaldaki yeri."])
        else:
            res.append(["N/A", "Bollinger Bands", "Yetersiz veri."])

        lm  = safe_scalar(last["MACD"])
        lms = safe_scalar(last["MACD_S"])
        if not (np.isnan(lm) or np.isnan(lms)):
            macd_hist = lm - lms
            hist_color = "🟢 Yeşil" if macd_hist > 0 else ("🔴 Kırmızı" if macd_hist < 0 else "⚪ Sıfır")
            relation   = "MACD > Signal" if lm > lms else ("MACD < Signal" if lm < lms else "MACD = Signal")
            macd_desc  = f"{relation} | Histogram: {macd_hist:+.4f} ({hist_color})"
            res.append([trend_dec("AL" if lm > lms else "SAT", last_ath),
                        f"MACD ({p_macd['macd_fast']},{p_macd['macd_slow']},{macd_signal})", macd_desc])
        else:
            res.append(["N/A", "MACD", "Yetersiz veri."])

        lz = safe_scalar(last["Z"])
        if not np.isnan(lz):
            dec = "AL" if lz < -p_z["z_thresh"] else ("SAT" if lz > p_z["z_thresh"] else "TUT")
            res.append([dec, f"Mean Reversion (z={p_z['z_thresh']})", f"Z: {lz:.2f}"])
        else:
            res.append(["N/A", "Mean Reversion", "Yetersiz veri."])

        lo = safe_scalar(last["Sig_OBV"])
        if lo != 0 and not np.isnan(lo):
            # Son bar OBV SMA değerleri
            obv_s_last = safe_scalar(obv_sma_short.iloc[-1]) if len(obv_sma_short) else np.nan
            obv_l_last = safe_scalar(obv_sma_long.iloc[-1])  if len(obv_sma_long)  else np.nan

            if not (np.isnan(obv_s_last) or np.isnan(obv_l_last)):
                diff     = obv_s_last - obv_l_last
                # Sayıyı okunabilir formata çevir (milyon/milyar)
                def _fmt_vol(v):
                    av = abs(v)
                    if av >= 1e9:  return f"{v/1e9:+.2f}B"
                    if av >= 1e6:  return f"{v/1e6:+.2f}M"
                    if av >= 1e3:  return f"{v/1e3:+.2f}K"
                    return f"{v:+.2f}"
                relation = "Kısa SMA > Uzun SMA" if diff > 0 else "Kısa SMA < Uzun SMA"
                status   = "Birikim ✅" if lo > 0 else "Dağıtım ❌"
                obv_desc = f"{relation} | Fark: {_fmt_vol(diff)} ({status})"
            else:
                obv_desc = "Birikim ✅" if lo > 0 else "Dağıtım ❌"
            res.append(["AL" if lo > 0 else "SAT", f"OBV ({obv_short}/{obv_long})", obv_desc])
        else:
            res.append(["N/A", f"OBV ({obv_short}/{obv_long})", "Yetersiz veri."])

        la   = safe_scalar(last["ADX"])
        lpd  = safe_scalar(last["PLUS_DI"])
        lmd2 = safe_scalar(last["MINUS_DI"])
        if not np.isnan(la):
            # Adaptif eşiği kullan (volatiliteye göre düzeltilmiş)
            adx_eff_thresh = adx_threshold_adaptive
            # DI+/DI- farkı trend yönünün gücünü gösterir
            if not (np.isnan(lpd) or np.isnan(lmd2)):
                di_diff  = lpd - lmd2
                di_info  = f"| +DI: {lpd:.1f} / -DI: {lmd2:.1f} ({'↑' if di_diff > 0 else '↓'} fark: {abs(di_diff):.1f})"
            else:
                di_info = ""
            strength = "Güçlü" if la > adx_eff_thresh else "Zayıf"
            thresh_info = f"eşik: {adx_eff_thresh}"
            if adx_eff_thresh != adx_threshold:
                thresh_info += f" (kullanıcı: {adx_threshold}, adaptif: {adx_eff_thresh})"
            macd_desc = f"ADX: {la:.1f} ({strength}, {thresh_info}) {di_info}"
            if la > adx_eff_thresh:
                res.append([trend_dec("AL" if lpd > lmd2 else "SAT", last_ath), "ADX", macd_desc])
            else:
                res.append(["TUT", "ADX", macd_desc])
        else:
            res.append(["N/A", "ADX", "Yetersiz veri."])

        if is_intraday:
            lv  = safe_scalar(last["VWAP"])
            lvs = safe_scalar(last["Sig_VWAP"])
            if not np.isnan(lv):
                dec = "AL" if lvs == 1 else ("SAT" if lvs == -1 else "TUT")
                res.append([dec, "VWAP", f"VWAP: {lv:.2f} | bant: ±%{vwap_band_pct:.2f}"])
            else:
                res.append(["N/A", "VWAP", "Yetersiz veri."])
        else:
            res.append(["N/A", "VWAP", "Günlük+ periyotta devre dışı."])

        lsk = float(df["StochRSI_K"].iloc[-1])
        lsd = float(df["StochRSI_D"].iloc[-1]) if "StochRSI_D" in df.columns else np.nan
        lss = safe_scalar(last["Sig_StochRSI"])
        if not np.isnan(lsk):
            # Bölge tespiti
            if   lsk < stoch_lower:  bolge = f"Aşırı Satım 🟢 (<{stoch_lower})"
            elif lsk > stoch_upper:  bolge = f"Aşırı Alım 🔴 (>{stoch_upper})"
            else:                    bolge = f"Nötr ⚪ ({stoch_lower}-{stoch_upper})"

            # K/D ilişkisi
            if not np.isnan(lsd):
                if lsk > lsd:   kd_rel = f"K > D ↑ (K:{lsk:.1f} / D:{lsd:.1f})"
                elif lsk < lsd: kd_rel = f"K < D ↓ (K:{lsk:.1f} / D:{lsd:.1f})"
                else:           kd_rel = f"K = D (K:{lsk:.1f} / D:{lsd:.1f})"
            else:
                kd_rel = f"%K: {lsk:.1f}"

            # Teyit durumu: sinyal sadece bölge + K/D uyumluysa oluşur
            if lss == 1:
                teyit = "✅ AL teyidi (aşırı satım + yukarı dönüş)"
                dec = "AL"
            elif lss == -1:
                teyit = "✅ SAT teyidi (aşırı alım + aşağı dönüş)"
                dec = "SAT"
            else:
                # Bölgede ama kesişim teyidi yok
                if lsk < stoch_lower and not np.isnan(lsd) and lsk < lsd:
                    teyit = "⏸ Aşırı satımda ama K < D (dönüş teyidi bekle)"
                elif lsk > stoch_upper and not np.isnan(lsd) and lsk > lsd:
                    teyit = "⏸ Aşırı alımda ama K > D (dönüş teyidi bekle)"
                else:
                    teyit = "Nötr bölgede"
                dec = "TUT"

            stoch_desc = f"{bolge} | {kd_rel} | {teyit}"
            res.append([dec, f"Stoch RSI ({stoch_rsi_period})", stoch_desc])
        else:
            res.append(["N/A", "Stoch RSI", "Yetersiz veri."])

        # ───────── Ichimoku zenginleştirilmiş satır ─────────
        lis = safe_scalar(last["Sig_Ichimoku"])
        l_tenkan = safe_scalar(last["Tenkan"])
        l_kijun  = safe_scalar(last["Kijun"])
        l_seka   = safe_scalar(last["Senkou_A"])
        l_sekb   = safe_scalar(last["Senkou_B"])

        if any(np.isnan([l_tenkan, l_kijun, l_seka, l_sekb])):
            # Senkou'lar 26 bar ileri kaydırıldığı için başlarda NaN olabilir
            res.append(["N/A", "Ichimoku", "Yetersiz veri (Senkou henüz hesaplanmadı)."])
        else:
            # 1) Tenkan-Kijun ilişkisi
            if l_tenkan > l_kijun:
                tk_rel = f"T:{l_tenkan:.1f} > K:{l_kijun:.1f} ↑"
            elif l_tenkan < l_kijun:
                tk_rel = f"T:{l_tenkan:.1f} < K:{l_kijun:.1f} ↓"
            else:
                tk_rel = f"T:{l_tenkan:.1f} = K:{l_kijun:.1f}"

            # 2) Fiyat - Bulut pozisyonu
            cloud_top    = max(l_seka, l_sekb)
            cloud_bottom = min(l_seka, l_sekb)
            if last_close > cloud_top:
                cloud_pos = "Bulut ÜSTÜNDE ✅"
            elif last_close < cloud_bottom:
                cloud_pos = "Bulut ALTINDA ❌"
            else:
                cloud_pos = "Bulut İÇİNDE ⚪"

            # 3) Bulut rengi (Senkou A vs B)
            cloud_color = "Yeşil 🟢" if l_seka > l_sekb else ("Kırmızı 🔴" if l_seka < l_sekb else "Eşit ⚪")

            # 4) Rejim bazlı dinamik uyarı (ADX'e göre)
            #    Adaptif eşik kullanıyoruz — tutarlılık için
            if not np.isnan(la):
                if la > adx_threshold_adaptive:
                    regime_note = f"✅ Trend piyasa — sinyal güvenilir (ADX: {la:.1f})"
                elif la < max(adx_threshold_adaptive - 5, 15):
                    regime_note = f"⚠️ Yatay piyasada aldatıcı (ADX: {la:.1f})"
                else:
                    regime_note = f"⏸ Geçiş rejimi (ADX: {la:.1f})"
            else:
                regime_note = ""

            # Karar + açıklama birleşimi
            desc = f"{tk_rel} | {cloud_pos} | {cloud_color}"
            if regime_note:
                desc += f" | {regime_note}"

            if not last_ath and lis != 0:
                res.append(["TUT (düşük vol.)", "Ichimoku", desc + " | ATR filtresi aktif."])
            elif lis == 1:
                res.append(["AL",  "Ichimoku", desc])
            elif lis == -1:
                res.append(["SAT", "Ichimoku", desc])
            else:
                res.append(["TUT", "Ichimoku", desc])

        lk = safe_scalar(last["KAMA"])
        if not np.isnan(lk):
            res.append([trend_dec("AL" if last_close > lk else "SAT", last_ath),
                        f"KAMA ({kama_period},{kama_fast},{kama_slow})", f"KAMA: {lk:.2f}"])
        else:
            res.append(["N/A", "KAMA", "Yetersiz veri."])

        lst  = safe_scalar(last["SuperTrend"])
        lstd = safe_scalar(last["ST_Direction"])
        if not np.isnan(lst) and not np.isnan(lstd):
            # 1) Yön
            yon = "YUKARI ↑" if lstd == 1 else "AŞAĞI ↓"

            # 2) Çizgi seviyesi ve fiyata uzaklık
            if last_close > 0:
                dist_pct = abs(lst - last_close) / last_close * 100
                if lstd == 1:
                    # Trend yukarı → çizgi altta (destek)
                    uzak_str = f"fiyatın %{dist_pct:.2f} altında (destek)"
                else:
                    # Trend aşağı → çizgi üstte (direnç)
                    uzak_str = f"fiyatın %{dist_pct:.2f} üstünde (direnç)"
                # Flip yakınlığı uyarısı
                if dist_pct < 1.0:
                    uzak_str += " ⚠️ flip yakın"
            else:
                uzak_str = ""

            # 3) Güncel ATR (volatilite bağlamı)
            r_atr_st = safe_scalar(last["ATR"])
            atr_str  = f"ATR: {r_atr_st:.2f}" if not np.isnan(r_atr_st) else ""

            # 4) Flip'ten bu yana bar sayısı (sinyal olgunluğu)
            st_dir_series = df["ST_Direction"].values
            bars_since_flip = 0
            for i in range(len(st_dir_series) - 1, 0, -1):
                if st_dir_series[i] != st_dir_series[i-1]:
                    break
                bars_since_flip += 1
            if bars_since_flip == 0:
                flip_str = "🆕 Yeni flip!"
            elif bars_since_flip < 3:
                flip_str = f"Flip'ten {bars_since_flip} bar (yeni sinyal)"
            else:
                flip_str = f"Flip'ten {bars_since_flip} bar"

            # Birleştir
            parts = [f"Yön: {yon}", f"Çizgi: {lst:.2f} ({uzak_str})"]
            if atr_str:   parts.append(atr_str)
            parts.append(flip_str)
            st_desc = " | ".join(parts)

            res.append([trend_dec("AL" if lstd == 1 else "SAT", last_ath),
                        f"SuperTrend ({p_st['st_period']}, x{p_st['st_multiplier']})", st_desc])
        else:
            res.append(["N/A", "SuperTrend", "Yetersiz veri."])

        llrc = safe_scalar(last["Sig_LRC"])
        llm  = safe_scalar(last["LRC_Mid"])
        if not np.isnan(llm):
            dec = "AL" if llrc == 1 else ("SAT" if llrc == -1 else "TUT")
            res.append([dec, f"LR Channel (σ={p_lrc['lrc_std_mult']})", f"Orta: {llm:.2f}"])
        else:
            res.append(["N/A", "LR Channel", "Yetersiz veri."])

        la2 = safe_scalar(last["ATR"])
        lam = safe_scalar(atr_ma.iloc[-1])
        if not np.isnan(la2):
            res.append(["BİLGİ", "ATR Filtre",
                f"Volatilite: {'Yüksek ↑' if last_ath else 'Düşük ↓'} | ATR: {la2:.2f} | MA: {lam:.2f}"])
        else:
            res.append(["N/A", "ATR Filtre", "Yetersiz veri."])

        lnw = safe_scalar(last["NW_Line"])
        lnu = safe_scalar(last["NW_Upper"])
        lnl = safe_scalar(last["NW_Lower"])
        if not np.isnan(lnw):
            if last_close > lnu:   nw_note = "Üst zarfın üstünde (aşırı alım)"
            elif last_close < lnl: nw_note = "Alt zarfın altında (aşırı satım)"
            else:                  nw_note = f"Zarf içinde. NW: {lnw:.2f}"
            res.append(["BİLGİ", "Nadaraya-Watson", nw_note])
        else:
            res.append(["N/A", "Nadaraya-Watson", "Yetersiz veri."])

        lwt1    = safe_scalar(last["WT1"])
        lwt_sig = safe_scalar(last["Sig_WaveTrend"])
        if not np.isnan(lwt1):
            if lwt1 > wt_ob:   wt_zone = f"Aşırı Alım (WT1={lwt1:.1f})"
            elif lwt1 < wt_os: wt_zone = f"Aşırı Satım (WT1={lwt1:.1f})"
            else:               wt_zone = f"Nötr Bölge (WT1={lwt1:.1f})"
            wt_dec = "AL" if lwt_sig == 1 else ("SAT" if lwt_sig == -1 else "TUT")
            res.append([wt_dec, f"WaveTrend ({p_wt['wt_n1']}/{p_wt['wt_n2']})", wt_zone])
        else:
            res.append(["N/A", "WaveTrend", "Yetersiz veri."])

        # ── YENİ: EMA200 karar satırı ─────────────────────────────
        lema200 = safe_scalar(last["EMA200"])
        if not np.isnan(lema200):
            ema_dec = trend_dec("AL" if last_close > lema200 else "SAT", last_ath)
            res.append([ema_dec, "EMA 200", f"EMA200: {lema200:.2f} | Fiyat {'üstünde ✅' if last_close > lema200 else 'altında ❌'}"])
        else:
            res.append(["N/A", "EMA 200", "Yetersiz veri (min 200 bar gerekli)."])

        # ── YENİ: En yakın S/R seviyesi karar satırı ──────────────
        if swing_levels:
            closest_sr = min(swing_levels, key=lambda x: abs(x["price"] - last_close))
            dist_pct   = abs(closest_sr["price"] - last_close) / last_close * 100
            sr_label   = "Destek" if closest_sr["type"] == "S" else "Direnç"
            res.append(["BİLGİ", "Swing S/R",
                f"En yakın {sr_label}: {closest_sr['price']:.2f} "
                f"(%{dist_pct:.1f} uzakta, {closest_sr['touches']}x dokunuş)"])
        # ──────────────────────────────────────────────────────────

        last_div_rsi  = safe_scalar(last["Div_RSI"])
        last_div_macd = safe_scalar(last["Div_MACD"])
        if last_div_rsi == 1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔺 Bullish Divergence — güçlü dip sinyali olabilir"])
        elif last_div_rsi == -1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔻 Bearish Divergence — zayıflayan momentum"])
        else:
            res.append(["BİLGİ", "Divergence (RSI)", "Aktif divergence yok"])
        if last_div_macd == 1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔺 Bullish Divergence"])
        elif last_div_macd == -1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔻 Bearish Divergence"])
        else:
            res.append(["BİLGİ", "Divergence (MACD)", "Aktif divergence yok"])

        if fib_levels:
            closest_lvl = min(fib_levels.items(), key=lambda x: abs(x[1] - last_close))
            res.append(["BİLGİ", f"Fibonacci ({fib_lookback} bar)",
                        f"En yakın seviye: {closest_lvl[0]} ({closest_lvl[1]:.2f}) | Swing: {fib_low:.2f} — {fib_high:.2f}"])

        c1, c2 = st.columns(2)
        c1.metric("Anlık Fiyat",  f"{last_close:.2f}")
        c2.metric("Zaman Dilimi", f"{interval}")

        # ============================================================
        # 🎯 KOMBİNE SİNYAL SKORU
        # MACD + RSI çekirdekli, ağırlıklı, ADX rejim bilinçli
        # ============================================================
        st.write("---")
        st.subheader("🎯 Kombine Sinyal Skoru")
        st.caption("MACD + RSI çekirdekli, ağırlıklı ve ADX rejim bilinçli skor sistemi.")

        # Önceki bar (histogram ivmesi / RSI yönü için)
        if len(df) >= 2:
            prev = df.iloc[-2]
            r_macd_prev  = safe_scalar(prev["MACD"])
            r_macds_prev = safe_scalar(prev["MACD_S"])
            r_rsi_prev   = safe_scalar(prev["RSI"])
        else:
            r_macd_prev = r_macds_prev = r_rsi_prev = np.nan

        r_sig_stk = safe_scalar(last["Sig_StochRSI"])

        # score_rows: (isim, açıklama, ham_puan, etiket)
        # etiket = "macd"  → ADX çarpanına tabi
        # etiket = "neutral" → çarpan yok
        score_rows = []

        # 1) MACD + RSI ÇEKİRDEK (±3)
        macd_bull    = (not np.isnan(r_macd)) and (not np.isnan(r_macds)) and r_macd > r_macds
        macd_bear    = (not np.isnan(r_macd)) and (not np.isnan(r_macds)) and r_macd < r_macds
        rsi_in_range = (not np.isnan(r_rsi)) and rsi_lower < r_rsi < rsi_upper
        rsi_up       = (not np.isnan(r_rsi_prev)) and r_rsi > r_rsi_prev
        rsi_down     = (not np.isnan(r_rsi_prev)) and r_rsi < r_rsi_prev

        if macd_bull and rsi_in_range:
            if rsi_up:
                core_pts, core_desc = 3,  "Güçlü uyum: MACD↑ + RSI↑ (nötr bölge)"
            elif rsi_down:
                core_pts, core_desc = 1,  "Zayıf uyum: MACD↑ ama RSI aşağı (iç çelişki)"
            else:
                core_pts, core_desc = 2,  "Orta uyum: MACD↑ + RSI nötr bölgede yatay"
        elif macd_bull and not rsi_in_range:
            core_pts, core_desc = 1,  f"Kısmi uyum: MACD↑ ama RSI uçta ({r_rsi:.1f})"
        elif macd_bear and rsi_in_range:
            if rsi_down:
                core_pts, core_desc = -3, "Güçlü uyum: MACD↓ + RSI↓ (nötr bölge)"
            elif rsi_up:
                core_pts, core_desc = -1, "Zayıf uyum: MACD↓ ama RSI yukarı (iç çelişki)"
            else:
                core_pts, core_desc = -2, "Orta uyum: MACD↓ + RSI nötr bölgede yatay"
        elif macd_bear and not rsi_in_range:
            core_pts, core_desc = -1, f"Kısmi uyum: MACD↓ ama RSI uçta ({r_rsi:.1f})"
        else:
            core_pts, core_desc = 0,  "Nötr — MACD ve RSI net sinyal vermiyor"
        score_rows.append(("MACD + RSI Çekirdek", core_desc, core_pts, "macd"))

        # 2) MACD Histogram İvmesi (±1)
        hist_now  = r_macd - r_macds if (not np.isnan(r_macd) and not np.isnan(r_macds)) else np.nan
        hist_prev = r_macd_prev - r_macds_prev if (not np.isnan(r_macd_prev) and not np.isnan(r_macds_prev)) else np.nan
        if not (np.isnan(hist_now) or np.isnan(hist_prev)):
            if hist_now > hist_prev:
                h_pts, h_desc = 1,  f"Artıyor ↑ — momentum güçleniyor (H: {hist_now:+.4f})"
            elif hist_now < hist_prev:
                h_pts, h_desc = -1, f"Azalıyor ↓ — momentum zayıflıyor (H: {hist_now:+.4f})"
            else:
                h_pts, h_desc = 0,  "Sabit"
        else:
            h_pts, h_desc = 0, "Yetersiz veri"
        score_rows.append(("MACD Histogram İvmesi", h_desc, h_pts, "macd"))

        # 3) MACD Zero Line (±1)
        if not np.isnan(r_macd):
            if r_macd > 0:
                z_pts, z_desc = 1,  f"Pozitif bölge (MACD: {r_macd:+.4f})"
            elif r_macd < 0:
                z_pts, z_desc = -1, f"Negatif bölge (MACD: {r_macd:+.4f})"
            else:
                z_pts, z_desc = 0,  "Sıfır çizgisinde"
        else:
            z_pts, z_desc = 0, "N/A"
        score_rows.append(("MACD Zero Line", z_desc, z_pts, "macd"))

        # 4) OBV Teyidi (±1)
        if r_obv_sig == 1:
            o_pts, o_desc = 1,  "Birikim ✅ — hacim fiyatı destekliyor"
        elif r_obv_sig == -1:
            o_pts, o_desc = -1, "Dağıtım ❌ — hacim zayıflıyor"
        else:
            o_pts, o_desc = 0,  "Nötr"
        score_rows.append(("OBV Teyidi", o_desc, o_pts, "neutral"))

        # 5) EMA200 Yön Uyumu (±1)
        if not np.isnan(r_ema200):
            if r_close > r_ema200:
                e_pts, e_desc = 1,  f"Fiyat EMA200 üstünde (uzun vade: +)"
            else:
                e_pts, e_desc = -1, f"Fiyat EMA200 altında (uzun vade: -)"
        else:
            e_pts, e_desc = 0, "N/A"
        score_rows.append(("EMA200 Yön Uyumu", e_desc, e_pts, "neutral"))

        # 6) Divergence (±2 teorik, her biri ±1 katkı)
        d_pts = 0
        d_parts = []
        if r_div_rsi == 1:
            d_pts += 1; d_parts.append("RSI Bullish 🔺")
        if r_div_mac == 1:
            d_pts += 1; d_parts.append("MACD Bullish 🔺")
        if r_div_rsi == -1:
            d_pts -= 1; d_parts.append("RSI Bearish 🔻")
        if r_div_mac == -1:
            d_pts -= 1; d_parts.append("MACD Bearish 🔻")
        d_desc = " / ".join(d_parts) if d_parts else "Divergence yok"
        score_rows.append(("Divergence", d_desc, d_pts, "neutral"))

        # 7) Stoch RSI Teyidi (±1)
        if r_sig_stk == 1:
            s_pts, s_desc = 1,  f"Aşırı satım — giriş bölgesi (K: {r_stk:.1f})"
        elif r_sig_stk == -1:
            s_pts, s_desc = -1, f"Aşırı alım — çıkış bölgesi (K: {r_stk:.1f})"
        else:
            s_pts, s_desc = 0,  f"Nötr (K: {r_stk:.1f})"
        score_rows.append(("Stoch RSI Teyidi", s_desc, s_pts, "neutral"))

        # 8) Seviye Yakınlığı (±1)
        if swing_levels:
            csr_s = min(swing_levels, key=lambda x: abs(x["price"] - r_close))
            dist_pct_s = abs(csr_s["price"] - r_close) / r_close * 100
            if dist_pct_s < 2.0:
                if csr_s["type"] == "S":
                    l_pts, l_desc = 1,  f"Desteğe yakın ✅ ({csr_s['price']:.2f}, %{dist_pct_s:.2f} uzak)"
                else:
                    l_pts, l_desc = -1, f"Dirence yakın ❌ ({csr_s['price']:.2f}, %{dist_pct_s:.2f} uzak)"
            else:
                l_pts, l_desc = 0, f"Seviyelerden uzak (en yakın %{dist_pct_s:.2f})"
        else:
            l_pts, l_desc = 0, "Seviye tespit edilmedi"
        score_rows.append(("Seviye Yakınlığı", l_desc, l_pts, "neutral"))

        # --- ADX REJİM ÇARPANI (rejim bilinçli ağırlıklandırma) ---
        # Volatiliteye göre düzeltilmiş adaptif eşik kullanılır
        if not np.isnan(r_adx):
            if r_adx > adx_threshold_adaptive:
                macd_mult    = 1.3
                regime_label = "Trend Rejim"
                regime_desc  = f"ADX {r_adx:.1f} > {adx_threshold_adaptive} — trend piyasa. MACD bileşenleri güçlendirildi (×1.3). [{adx_regime_note}]"
            elif r_adx < max(adx_threshold_adaptive - 5, 15):
                macd_mult    = 0.7
                regime_label = "Yatay Rejim"
                regime_desc  = f"ADX {r_adx:.1f} düşük — yatay piyasa. MACD bileşenleri zayıflatıldı (×0.7). [{adx_regime_note}]"
            else:
                macd_mult    = 1.0
                regime_label = "Geçiş"
                regime_desc  = f"ADX {r_adx:.1f} — geçiş bölgesi, çarpan uygulanmadı. [{adx_regime_note}]"
        else:
            macd_mult    = 1.0
            regime_label = "N/A"
            regime_desc  = "ADX N/A — çarpan uygulanmadı."

        # Ağırlıklı toplam skor
        final_rows  = []
        total_score = 0.0
        for name, desc, pts, tag in score_rows:
            mult = macd_mult if tag == "macd" else 1.0
            adj  = pts * mult
            total_score += adj
            if mult != 1.0 and pts != 0:
                pts_str = f"{pts:+d} × {mult:.1f} = {adj:+.1f}"
            else:
                pts_str = f"{pts:+d}"
            final_rows.append({"Bileşen": name, "Değer": desc, "Puan": pts_str})

        # Karar etiketi
        if   total_score >= 7:  karar = "🟢 ÇOK GÜÇLÜ AL"
        elif total_score >= 4:  karar = "🟢 GÜÇLÜ AL"
        elif total_score >= 1:  karar = "🟡 ZAYIF AL"
        elif total_score <= -7: karar = "🔴 ÇOK GÜÇLÜ SAT"
        elif total_score <= -4: karar = "🔴 GÜÇLÜ SAT"
        elif total_score <= -1: karar = "🟡 ZAYIF SAT"
        else:                   karar = "⚪ NÖTR"

        # Üst metric paneli
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Toplam Skor", f"{total_score:+.1f}", help="Teorik aralık: -11 ile +11")
        mc2.metric("Karar", karar)
        mc3.metric("Rejim", regime_label)
        mc4.metric("Volatilite", "Yüksek ↑" if r_atr_hi else "Düşük ↓")

        st.caption(regime_desc)

        # Bileşen dökümü tablosu
        score_df = pd.DataFrame(final_rows)

        def score_color(val):
            s = str(val)
            try:
                final_num = float(s.split("=")[-1].strip()) if "=" in s else float(s.strip())
                if   final_num > 0: return "color: #00ff00; font-weight: bold"
                elif final_num < 0: return "color: #ff4b4b; font-weight: bold"
                else:               return "color: #aaaaaa"
            except Exception:
                return "color: #aaaaaa"

        st.dataframe(
            score_df.style.map(score_color, subset=["Puan"]),
            use_container_width=True, hide_index=True
        )

        st.subheader("🔍 Algoritmik Detaylar")
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])

        def color_map(val):
            if val == "AL":    return "color: #00ff00; font-weight: bold"
            if val == "SAT":   return "color: #ff4b4b; font-weight: bold"
            if val == "N/A":   return "color: #ffaa00; font-weight: bold"
            if val == "BİLGİ": return "color: #00bfff; font-weight: bold"
            if "düşük vol." in str(val): return "color: #808495; font-style: italic"
            return "color: #808495; font-weight: bold"

        st.table(res_df.style.map(color_map, subset=["Karar"]))

        # ============================================================
        # BİREYSEL BACKTEST TABLOSU
        # ============================================================
        st.write("---")
        st.header("📊 Bireysel Algoritma Backtesti (Güncel Parametrelerle)")
        st.caption("⚠️ Geçmiş performans gelecekteki sonuçların garantisi değildir.")

        algo_signal_map = {
            f"SMA ({p_sma['sma_s']}/{p_sma['sma_l']})":                                    "Sig_SMA",
            f"RSI (p={p_rsi['rsi_period']} [{p_rsi['rsi_lower']}/{p_rsi['rsi_upper']}])":  "Sig_RSI",
            f"Bollinger Bands (p={p_bb['bb_period']}, σ={p_bb['bb_std']})":                "Sig_BB",
            f"MACD ({p_macd['macd_fast']},{p_macd['macd_slow']},{p_macd['macd_signal']})": "Sig_MACD",
            f"Mean Reversion (p={p_z['z_period']}, z={p_z['z_thresh']})":                  "Sig_Z",
            "OBV":                                                                           "Sig_OBV",
            f"ADX (p={p_adx['adx_period']}, eşik={p_adx['adx_threshold']})":              "Sig_ADX",
            "Stoch RSI":                                                                     "Sig_StochRSI",
            "Ichimoku":                                                                      "Sig_Ichimoku",
            "KAMA":                                                                          "Sig_KAMA",
            f"SuperTrend (p={p_st['st_period']}, x{p_st['st_multiplier']})":               "Sig_SuperTrend",
            f"LR Channel (p={p_lrc['lrc_period']}, σ={p_lrc['lrc_std_mult']})":           "Sig_LRC",
            f"WaveTrend ({p_wt['wt_n1']}/{p_wt['wt_n2']})":                               "Sig_WaveTrend",
        }
        if is_intraday:
            algo_signal_map["VWAP"] = "Sig_VWAP"

        algo_results = []
        for algo_name, sig_col in algo_signal_map.items():
            if sig_col not in df.columns:
                continue
            stats = run_backtest(df[sig_col], close_arr, cost_pct,
                                 bars_per_year=bars_per_year_from_interval(interval))
            algo_results.append({
                "Algoritma":       algo_name,
                "Trade":           stats["n"],
                "Getiri (%)":      round(stats["total_ret"], 2),
                "Win Rate (%)":    round(stats["win_rate"],  1),
                "Ort. Kazanç (%)": round(stats["avg_win"],  2),
                "Ort. Kayıp (%)":  round(stats["avg_loss"], 2),
                "Max DD (%)":      round(stats["max_dd"],   2),
                "Profit Factor":   round(stats["pf"], 2) if stats["pf"] != float("inf") else "∞",
            })

        if algo_results:
            algo_df = pd.DataFrame(algo_results)
            active  = algo_df[algo_df["Trade"] > 0].copy()
            if not active.empty:
                best = active.loc[active["Getiri (%)"].idxmax()]
                st.success(
                    f"🥇 En iyi: **{best['Algoritma']}** — "
                    f"Getiri: %{best['Getiri (%)']}, "
                    f"Win Rate: %{best['Win Rate (%)']}, "
                    f"{int(best['Trade'])} trade")
            def ret_color(val):
                if isinstance(val, (int, float)):
                    if val > 0: return "color: #00ff00"
                    if val < 0: return "color: #ff4b4b"
                return ""
            st.dataframe(algo_df.style.map(ret_color, subset=["Getiri (%)"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Algoritma performansı hesaplanamadı.")

        # ============================================================
        # OPTİMİZASYON ÖZET TABLOSU
        # ============================================================
        st.write("---")
        st.subheader("🧬 Walk-Forward Optimizasyon Sonuçları")
        st.caption(f"{n_windows} pencere · %{train_pct} eğitim / %{100-train_pct} test · kriter: Sharpe (yıllıklandırılmış, **out-of-sample**)")

        def opt_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if not np.isfinite(v):   return "color: #888888"
            if v > 0:  return "color: #00ff00"
            if v < 0:  return "color: #ff4b4b"
            return "color: #888888"  # sıfır için gri — koyu arka planda görünür

        def pval_color(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if not np.isfinite(v): return "color: #888888"
            if v < 0.05:  return "color: #00ff00; font-weight: bold"   # anlamlı
            if v < 0.10:  return "color: #ffcc00"                       # sınırda
            return "color: #aaaaaa"                                     # anlamsız

        def _safe_round(x, nd=2, default=0.0):
            try:
                v = float(x)
                if not np.isfinite(v):
                    return default
                return round(v, nd)
            except (ValueError, TypeError):
                return default

        opt_rows  = []
        for algo_name, grid in PARAM_GRIDS.items():
            p = opt_params.get(algo_name, {})
            s = opt_stats.get(algo_name, {})
            row = {"Algoritma": algo_name}
            param_str            = "  |  ".join(f"{k} = {v}" for k, v in p.items())
            row["Parametreler"]  = param_str
            row["Getiri (%)"]    = _safe_round(s.get("total_ret", 0), 2)
            row["Sharpe (OOS)"]  = _safe_round(s.get("sharpe",    0), 2)
            row["DSR"]           = _safe_round(s.get("dsr", np.nan), 2, default=np.nan)
            row["Trade"]         = int(s.get("n", 0) or 0)
            row["Win Rate (%)"]  = _safe_round(s.get("win_rate",  0), 1)
            sel = s.get("wf_selections", 0); wins = s.get("wf_windows", 0)
            row["Seçim"]         = f"{sel}/{wins}" if wins else "—"
            row["p-değeri"]      = _safe_round(s.get("p_value", np.nan), 4, default=np.nan)
            opt_rows.append(row)

        opt_df     = pd.DataFrame(opt_rows)
        color_cols = [c for c in ["Getiri (%)", "Sharpe (OOS)", "DSR"] if c in opt_df.columns]
        fmt        = {"Getiri (%)": "{:.2f}", "Sharpe (OOS)": "{:.2f}", "DSR": "{:.2f}",
                      "Win Rate (%)": "{:.1f}", "p-değeri": "{:.3f}"}
        fmt        = {k: v for k, v in fmt.items() if k in opt_df.columns}
        styled = opt_df.style.format(fmt, na_rep="—").map(opt_color, subset=color_cols)
        if "p-değeri" in opt_df.columns:
            styled = styled.map(pval_color, subset=["p-değeri"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "💡 **Sharpe (OOS)**: Yalnız out-of-sample test dilimlerinden yıllıklandırılmış risk ayarlı getiri. "
            "**DSR** (Deflated Sharpe Ratio — Bailey & López de Prado 2014): Multiple testing "
            "cezası çıkarılmış. **DSR > 0** → gerçekten rastgeleden iyi; **DSR ≤ 0** → yüksek Sharpe "
            "muhtemelen şans eseri. **p-değeri** (Stationary Bootstrap — Politis & Romano 1994): "
            "< 0.05 → sinyal istatistiksel olarak anlamlı. **Seçim k/n** → kombonun n expanding "
            "adımında k tanesinde train-kazananı olduğu."
        )

        # ============================================================
        # RAPOR BÖLÜMÜ
        # ============================================================
        st.write("---")
        st.header("📋 Teknik Analiz Raporu")
        st.caption("Kural tabanlı otomatik analiz. Yatırım tavsiyesi içermez.")

        steps = []

        # ADIM 1 — Büyük Resim (EMA200 eklendi)
        kama_pos  = "ÜSTÜNDE ✅" if r_close > r_kama else "ALTINDA ❌"
        st_signal = "AL ✅" if r_std == 1 else "SAT ❌"
        poc_dist  = abs(r_close - poc_price) / r_close * 100
        ema200_pos = "üstünde ✅" if r_close > r_ema200 else "altında ❌"
        steps.append({
            "Adım": "1 — Büyük Resim",
            "Gösterge": "Fiyat / KAMA / EMA200 / SuperTrend / POC",
            "Değer": f"Fiyat: {r_close:.2f} | KAMA: {r_kama:.2f} | EMA200: {r_ema200:.2f} | POC: {poc_price:.2f} (%{poc_dist:.1f} uzakta)",
            "Yorum": f"Fiyat KAMA'nın {kama_pos} | EMA200 {ema200_pos} | SuperTrend: {st_signal} | En yakın Fib: {r_fib_closest[0]} ({r_fib_closest[1]:.2f})",
            "Sinyal": "AL" if (r_close > r_kama and r_std == 1 and r_close > r_ema200)
                      else ("SAT" if (r_close < r_kama and r_std == -1 and r_close < r_ema200) else "NÖTR"),
        })

        # ADIM 2 — Trend Gücü (adaptif eşikle)
        if not np.isnan(r_adx):
            adx_str  = f"ADX: {r_adx:.1f}"
            is_strong = r_adx > adx_threshold_adaptive
            adx_sig  = "GÜÇLÜ TREND ✅" if is_strong else "ZAYIF / YATAY ⚠️"
            adx_hint = "Trend sinyallerine güven" if is_strong else "Mean-reversion sinyallerine ağırlık ver"
            eff_str  = f"adaptif eşik: {adx_threshold_adaptive}" if adx_threshold_adaptive != adx_threshold else f"eşik: {adx_threshold}"
            steps.append({
                "Adım": "2 — Trend Gücü",
                "Gösterge": "ADX",
                "Değer": f"{adx_str} ({eff_str})",
                "Yorum": f"{adx_sig} — {adx_hint}",
                "Sinyal": "GÜÇLÜ" if is_strong else "ZAYIF",
            })
        else:
            steps.append({"Adım": "2 — Trend Gücü", "Gösterge": "ADX",
                          "Değer": "N/A", "Yorum": "Yetersiz veri", "Sinyal": "N/A"})

        # ADIM 3 — Trend Yönü
        macd_pos  = r_macd > r_macds
        ichi_bull = r_ichi == 1
        ichi_bear = r_ichi == -1
        macd_str  = "Pozitif ✅" if macd_pos else "Negatif ❌"
        ichi_str  = "Boğa ✅" if ichi_bull else ("Ayı ❌" if ichi_bear else "Nötr ⚠️")
        trend_sig = "AL" if (macd_pos and ichi_bull) else ("SAT" if (not macd_pos and ichi_bear) else "NÖTR")
        steps.append({
            "Adım": "3 — Trend Yönü",
            "Gösterge": "MACD + Ichimoku",
            "Değer": f"MACD: {r_macd:.4f} | Sinyal: {r_macds:.4f}",
            "Yorum": f"MACD: {macd_str} | Ichimoku: {ichi_str}",
            "Sinyal": trend_sig,
        })

        # ADIM 4 — Giriş Noktası
        rsi_os    = r_rsi < rsi_lower
        rsi_ob    = r_rsi > rsi_upper
        stk_os    = r_stk < stoch_lower
        stk_ob    = r_stk > stoch_upper
        rsi_str   = f"Aşırı Satım ✅ ({r_rsi:.1f})" if rsi_os else (f"Aşırı Alım ❌ ({r_rsi:.1f})" if rsi_ob else f"Nötr ({r_rsi:.1f})")
        stk_str   = f"Aşırı Satım ✅ ({r_stk:.1f})" if stk_os else (f"Aşırı Alım ❌ ({r_stk:.1f})" if stk_ob else f"Nötr ({r_stk:.1f})")
        entry_sig = "AL" if (rsi_os or stk_os) else ("SAT" if (rsi_ob or stk_ob) else "BEKLE")
        steps.append({
            "Adım": "4 — Giriş Noktası",
            "Gösterge": "RSI + Stoch RSI",
            "Değer": f"RSI: {r_rsi:.1f} | Stoch %K: {r_stk:.1f}",
            "Yorum": f"RSI: {rsi_str} | Stoch RSI: {stk_str}",
            "Sinyal": entry_sig,
        })

        # ADIM 5 — Seviye Teyidi (S/R eklendi)
        lrc_str   = "Alt banda yakın ✅" if r_lrc_sig == 1 else ("Üst banda yakın ❌" if r_lrc_sig == -1 else "Kanal ortası ⚪")
        nw_str    = ("Üst zarfın üstünde ❌" if r_close > r_nw_up
                     else ("Alt zarfın altında ✅" if r_close < r_nw_lo else "Zarf içinde ⚪"))
        sr_note   = ""
        if swing_levels:
            csr      = min(swing_levels, key=lambda x: abs(x["price"] - r_close))
            sr_note  = f" | En yakın {'Destek ✅' if csr['type']=='S' else 'Direnç ❌'}: {csr['price']:.2f} ({csr['touches']}x)"
        level_sig = "AL" if (r_lrc_sig == 1 or r_close < r_nw_lo) else ("SAT" if (r_lrc_sig == -1 or r_close > r_nw_up) else "NÖTR")
        steps.append({
            "Adım": "5 — Seviye Teyidi",
            "Gösterge": "LR Channel + NW + S/R",
            "Değer": f"LRC Alt: {r_lrc_lo:.2f} | LRC Üst: {r_lrc_up:.2f} | NW: {r_nw:.2f}",
            "Yorum": f"LRC: {lrc_str} | NW: {nw_str}{sr_note}",
            "Sinyal": level_sig,
        })

        # ADIM 5b — VWAP
        if is_intraday and not np.isnan(r_vwap):
            vwap_pos = r_close > r_vwap
            vwap_str = f"Fiyat VWAP {'üstünde ✅' if vwap_pos else 'altında ❌'} | VWAP: {r_vwap:.2f}"
            vwap_dec = "AL" if r_vwap_sig == 1 else ("SAT" if r_vwap_sig == -1 else "NÖTR")
            steps.append({
                "Adım": "5b — VWAP Seviyesi",
                "Gösterge": "VWAP",
                "Değer": f"VWAP: {r_vwap:.2f} | Bant: ±%{vwap_band_pct:.2f}",
                "Yorum": vwap_str,
                "Sinyal": vwap_dec,
            })

        # ADIM 6 — Hacim Onayı
        obv_str = "Birikim ✅ (kısa SMA > uzun SMA)" if r_obv_sig == 1 else ("Dağıtım ❌ (kısa SMA < uzun SMA)" if r_obv_sig == -1 else "Nötr ⚪")
        steps.append({
            "Adım": "6 — Hacim Onayı",
            "Gösterge": "OBV",
            "Değer": f"Sinyal: {int(r_obv_sig)}",
            "Yorum": obv_str,
            "Sinyal": "AL" if r_obv_sig == 1 else ("SAT" if r_obv_sig == -1 else "NÖTR"),
        })

        # ADIM 7 — Uyarı / Divergence
        div_rsi_str  = ("🔺 Bullish Divergence — güçlü dip sinyali" if r_div_rsi == 1
                        else ("🔻 Bearish Divergence — zayıflama uyarısı" if r_div_rsi == -1 else "Yok ✅"))
        div_macd_str = ("🔺 Bullish Divergence" if r_div_mac == 1
                        else ("🔻 Bearish Divergence" if r_div_mac == -1 else "Yok ✅"))
        div_risk  = (r_div_rsi == -1 or r_div_mac == -1)
        div_boost = (r_div_rsi == 1  or r_div_mac == 1)
        steps.append({
            "Adım": "7 — Uyarı Kontrolü",
            "Gösterge": "Divergence (RSI + MACD)",
            "Değer": f"RSI Div: {int(r_div_rsi)} | MACD Div: {int(r_div_mac)}",
            "Yorum": f"RSI: {div_rsi_str} | MACD: {div_macd_str}",
            "Sinyal": "UYARI ❌" if div_risk else ("GÜÇLENDIRICI ✅" if div_boost else "TEMİZ ✅"),
        })

        report_df = pd.DataFrame(steps)

        def report_color(val):
            if "AL" in str(val) and "SAT" not in str(val): return "color: #00ff00; font-weight: bold"
            if "SAT" in str(val):    return "color: #ff4b4b; font-weight: bold"
            if "UYARI" in str(val):  return "color: #ff4b4b; font-weight: bold"
            if "TEMİZ" in str(val) or "GÜÇLEND" in str(val): return "color: #00bfff; font-weight: bold"
            if "NÖTR" in str(val) or "BEKLE" in str(val):    return "color: #aaaaaa"
            if "ZAYIF" in str(val):  return "color: #ffaa00; font-weight: bold"
            if "GÜÇLÜ" in str(val):  return "color: #00ff00; font-weight: bold"
            return ""

        st.dataframe(
            report_df.style.map(report_color, subset=["Sinyal"]),
            use_container_width=True, hide_index=True
        )


        st.write("---")
        st.subheader("📝 Özet")
        ozet_parcalar = []

        if r_adx > adx_threshold_adaptive:
            ozet_parcalar.append(f"ADX {r_adx:.1f} ile **güçlü bir trend** mevcut.")
        else:
            ozet_parcalar.append(f"ADX {r_adx:.1f} — piyasa **yatay seyirde**, mean-reversion sinyalleri daha geçerli.")

        if not np.isnan(r_ema200):
            if r_close > r_ema200:
                ozet_parcalar.append(f"Fiyat EMA200 ({r_ema200:.2f}) üstünde — **uzun vadeli trend pozitif**.")
            else:
                ozet_parcalar.append(f"Fiyat EMA200 ({r_ema200:.2f}) altında — **uzun vadeli trend negatif**.")

        if r_std == 1:
            ozet_parcalar.append("SuperTrend **AL** konumunda, yükseliş trendi destekleniyor.")
        else:
            ozet_parcalar.append("SuperTrend **SAT** konumunda, düşüş baskısı var.")

        if macd_pos:
            ozet_parcalar.append("MACD pozitif — momentum yukarı yönlü.")
        else:
            ozet_parcalar.append("MACD negatif — momentum aşağı yönlü.")

        if rsi_os:
            ozet_parcalar.append(f"RSI {r_rsi:.1f} ile **aşırı satım** bölgesinde — potansiyel giriş noktası.")
        elif rsi_ob:
            ozet_parcalar.append(f"RSI {r_rsi:.1f} ile **aşırı alım** bölgesinde — dikkatli olunmalı.")

        if swing_levels:
            csr = min(swing_levels, key=lambda x: abs(x["price"] - r_close))
            dist = abs(csr["price"] - r_close) / r_close * 100
            ozet_parcalar.append(
                f"En yakın {'destek' if csr['type']=='S' else 'direnç'}: "
                f"**{csr['price']:.2f}** (%{dist:.1f} uzakta, {csr['touches']}x test edilmiş).")

        if is_intraday and not np.isnan(r_vwap):
            if r_close > r_vwap:
                ozet_parcalar.append(f"Fiyat VWAP ({r_vwap:.2f}) üstünde — intraday momentum pozitif.")
            else:
                ozet_parcalar.append(f"Fiyat VWAP ({r_vwap:.2f}) altında — intraday baskı var.")

        if r_obv_sig == 1:
            ozet_parcalar.append("OBV birikim sinyali veriyor — hacim fiyatı destekliyor.")
        elif r_obv_sig == -1:
            ozet_parcalar.append("OBV dağıtım sinyali veriyor — hacim zayıflıyor.")

        if div_risk:
            ozet_parcalar.append("⚠️ **Bearish divergence** tespit edildi — mevcut sinyaller zayıflayabilir.")
        if div_boost:
            ozet_parcalar.append("🔺 **Bullish divergence** mevcut — AL sinyalini güçlendiriyor.")

        st.markdown(" ".join(ozet_parcalar))

        # ============================================================
        # 🤖 AI RAPOR YORUMU (Manuel tetikleme + cache + streaming)
        # ============================================================
        st.write("---")
        st.subheader("🤖 AI Rapor Yorumu")

        if not ai_api_key:
            st.info(
                f"💡 **{ai_provider}** için sidebar'dan API key girerseniz AI yorumcu aktif olur. "
                f"Key almak için: {_prov_cfg['key_url']}"
            )
        else:
            st.caption(
                f"Provider: **{ai_provider}** · Model: **{ai_model}** · "
                f"Detay: **{ai_detail}** (max {AI_DETAIL_LEVELS[ai_detail]} token)"
            )

            _cache_key = ai_cache_key(
                ticker, interval, total_score, r_close,
                ai_provider, ai_model, ai_detail
            )
            _cached = st.session_state.get(_cache_key)

            _bc1, _bc2, _ = st.columns([1.2, 1.4, 3])
            with _bc1:
                _gen_btn = st.button(
                    "📝 Yorum Al", type="primary",
                    use_container_width=True, key="ai_gen_btn"
                )
            with _bc2:
                _regen_btn = st.button(
                    "🔄 Yeniden Üret",
                    use_container_width=True,
                    disabled=(_cached is None),
                    key="ai_regen_btn",
                )

            if _gen_btn or _regen_btn:
                if _regen_btn:
                    st.session_state.pop(_cache_key, None)

                _sys_p, _usr_p = build_ai_prompt(
                    detail=ai_detail, ticker=ticker, close=r_close,
                    interval=interval, total_score=total_score,
                    karar=karar, regime_label=regime_label,
                    regime_desc=regime_desc, adx=r_adx,
                    adx_threshold=adx_threshold_adaptive,
                    components=final_rows, ema200=r_ema200,
                    rsi=r_rsi, stk=r_stk, macd=r_macd, macd_sig=r_macds,
                    atr_high=r_atr_hi, swing_levels=swing_levels,
                    fib_levels=fib_levels, st_dir=r_std,
                    obv_sig=r_obv_sig, div_rsi=r_div_rsi,
                    div_macd=r_div_mac, rsi_lo=rsi_lower, rsi_up=rsi_upper,
                )

                try:
                    _t0 = time.time()

                    with st.spinner(f"🤖 {ai_provider} · {ai_model} yanıt üretiyor..."):
                        _full_text, _meta = fetch_llm(
                            ai_provider, ai_api_key, ai_model,
                            _sys_p, _usr_p, AI_DETAIL_LEVELS[ai_detail]
                        )

                    _dt = time.time() - _t0

                    # Yarım cümle güvenlik ağı (safety net)
                    _cleaned, _was_cut = clean_half_sentence(_full_text)
                    _final = _cleaned

                    # Kesilme uyarıları
                    _finish = (_meta or {}).get("finish_reason", "")
                    _finish_lower = str(_finish).lower() if _finish else ""
                    if _finish_lower in ("max_tokens", "length"):
                        pro_hint = ""
                        if ai_provider == "Google" and "pro" in ai_model.lower():
                            pro_hint = (
                                " **Not:** `gemini-2.5-pro` modelinde reasoning kapatılamıyor. "
                                "`gemini-2.5-flash`'a geçmeyi deneyin."
                            )
                        _final += (
                            f"\n\n---\n⚠️ **Yanıt token limitine takıldı** "
                            f"(`{_finish}`). Detay seviyesini yükseltin.{pro_hint}"
                        )
                    elif _finish_lower in ("safety", "recitation", "blocklist", "content_filter"):
                        _final += f"\n\n---\n⚠️ **Yanıt güvenlik filtresi nedeniyle kesildi** (`{_finish}`)."
                    elif _was_cut:
                        _final += (
                            "\n\n---\n⚠️ *Model yanıtı yarıda bıraktı; son yarım cümle otomatik kaldırıldı. "
                            "Yeniden üretmek için 🔄 tuşuna basabilirsiniz.*"
                        )

                    # Token kullanım satırı
                    if _meta:
                        _prompt_t   = _meta.get("prompt_tokens",   0)
                        _output_t   = _meta.get("output_tokens",   0)
                        _thought_t  = _meta.get("thinking_tokens", 0)
                        _total_t    = _meta.get("total_tokens",    0) or (_prompt_t + _output_t + _thought_t)
                        _final += (
                            f"\n\n📊 Token Kullanımı — Prompt: {_prompt_t} · "
                            f"Cevap: {_output_t} · Thinking: {_thought_t} · Toplam: {_total_t}"
                        )

                    if _cleaned:
                        st.markdown(_final)
                        st.session_state[_cache_key] = _final
                        st.caption(f"✅ Tamamlandı · {_dt:.1f}s · ~{len(_cleaned.split())} kelime")
                    else:
                        st.warning("⚠️ Boş yanıt alındı. Farklı bir model veya detay seviyesi deneyin.")
                except RuntimeError as e:
                    st.error(f"❌ API Hatası: {str(e)}")
                except requests.exceptions.Timeout:
                    st.error("❌ Zaman aşımı — sunucu yanıt vermiyor. Tekrar deneyin.")
                except requests.exceptions.ConnectionError as e:
                    st.error(f"❌ Bağlantı hatası: {str(e)[:300]}")
                except Exception as e:
                    st.error(f"❌ {type(e).__name__}: {str(e)[:400]}")

            elif _cached:
                st.markdown(_cached)
                st.caption(
                    "💾 Cache'den gösteriliyor — fiyat/skor değişince anahtar değişir ve "
                    "yeni yorum gerekir. Manuel yenileme için 🔄 tuşuna basın."
                )

    else:
        st.error("Veri çekilemedi. Ticker veya internet bağlantısını kontrol edin.")
