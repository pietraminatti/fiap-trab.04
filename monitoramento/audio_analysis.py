from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .alerts import Alert

# --------------------------------------------------------------------------------------
# Leitura e features acústicas
# --------------------------------------------------------------------------------------


def load_wav(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(str(path))
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = signal.resample_poly(x, target_sr, sr).astype(np.float32)
        sr = target_sr
    return x, sr


def _frames(x, size, hop):
    n = 1 + max(0, (len(x) - size) // hop)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trechos contíguos True -> lista de (início, fim_exclusivo)."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def extract_features(x: np.ndarray, sr: int) -> dict:
    """Extrai biomarcadores vocais de um sinal mono (float, -1..1)."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    win = np.hanning(frame)
    fr = _frames(x, frame, hop)
    frw = (fr - fr.mean(axis=1, keepdims=True)) * win
    db = 20 * np.log10(np.sqrt(np.mean(frw ** 2, axis=1)) + 1e-9)
    floor = np.percentile(db, 10)
    active = db > max(np.percentile(db, 95) - 35, floor + 6)

    # 1) f0 e periodicidade por autocorrelação com correção da janela (Boersma, 1993)
    lag_min, lag_max = int(sr / 400), int(sr / 75)
    ac_w = np.correlate(win, win, mode="full")[frame - 1:]
    ac_w = ac_w / ac_w[0]
    f0 = np.full(len(fr), np.nan)
    r_peak = np.zeros(len(fr))
    for i in np.where(active)[0]:
        ac = np.correlate(frw[i], frw[i], mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0] / np.maximum(ac_w, 1e-3)
        seg = ac[lag_min:lag_max]
        j = int(np.argmax(seg))
        r_peak[i] = min(seg[j], 0.999)
        if seg[j] > 0.5:
            k = j + lag_min
            a, b, c = ac[k - 1], ac[k], ac[k + 1]
            f0[i] = sr / (k + 0.5 * (a - c) / (a - 2 * b + c + 1e-12))
    voiced = ~np.isnan(f0)

    # 2) sílabas = trechos vozeados (>= 50 ms); frases = sílabas separadas por < 250 ms
    syl = [(a, b) for a, b in _runs(voiced) if (b - a) * hop / sr >= 0.05]
    phrases, pauses = [], []
    for a, b in syl:
        if phrases and (a - phrases[-1][1]) * hop / sr < 0.25:
            phrases[-1] = (phrases[-1][0], b)
        else:
            if phrases:
                pauses.append((phrases[-1][1], a))
            phrases.append((a, b))
    speech_time = sum(b - a for a, b in phrases) * hop / sr or 1e-6
    total_time = ((phrases[-1][1] - phrases[0][0]) * hop / sr) if phrases else 1e-6

    # inspiração audível = ruído não periódico com energia relevante dentro de uma pausa
    breath_events = 0
    for a, b in pauses:
        noisy = (db[a:b] > floor + 15) & (r_peak[a:b] < 0.5)
        if any((e - s) * hop / sr >= 0.1 for s, e in _runs(noisy)):
            breath_events += 1

    # ritmo: intervalos entre inícios de sílabas dentro da mesma frase
    intervals = []
    for pa, pb in phrases:
        onsets = [a for a, b in syl if pa <= a < pb]
        intervals += list(np.diff(onsets) * hop / sr)
    intervals = np.array(intervals)
    rhythm_cv = float(intervals.std() / intervals.mean()) if len(intervals) > 3 else 0.0

    # 3) jitter / shimmer no nível do período glotal (picos do sinal em cada sílaba)
    f0_med = float(np.nanmedian(f0)) if voiced.any() else 0.0
    jit, shim = [], []
    if f0_med:
        T = sr / f0_med
        bl, al = signal.butter(4, min(0.99, 1.5 * f0_med / (sr / 2)))
        lp = signal.filtfilt(bl, al, x)
        for a, b in syl:
            s0, s1 = a * hop + frame // 4, b * hop + frame // 2
            seg = lp[s0:s1]
            if len(seg) < 4 * T:
                continue
            pk, _ = signal.find_peaks(seg, distance=int(0.7 * T), height=0.25 * np.max(seg))
            if len(pk) < 5:
                continue
            per, amp = np.diff(pk).astype(float), seg[pk]
            ok = np.abs(per / np.median(per) - 1) < 0.3
            dp = np.abs(np.diff(per))[ok[1:] & ok[:-1]]
            jit += list(dp / np.mean(per[ok]))
            da = np.abs(np.diff(amp))
            shim += list(da[2:-2] / np.mean(amp)) if len(da) > 6 else []
    rv = np.clip(r_peak[voiced], 1e-3, 0.999)
    hnr = float(np.median(10 * np.log10(rv / (1 - rv)))) if voiced.any() else 0.0

    # 4) intensidade média por sílaba ao longo do tempo (queda = fadiga)
    syl_t = np.array([(a + b) / 2 * hop / sr for a, b in syl])
    syl_db = np.array([db[a:b].mean() for a, b in syl])
    slope = float(np.polyfit(syl_t, syl_db, 1)[0]) if len(syl) > 5 else 0.0

    f0v = f0[voiced]
    pause_d = [(b - a) * hop / sr for a, b in pauses]
    return {
        "duracao_s": round(total_time, 2),
        "f0_medio_hz": round(f0_med, 1),
        "f0_cv": round(float(np.std(f0v) / np.mean(f0v)), 3) if len(f0v) else 0.0,
        "jitter_pct": round(100 * float(np.mean(jit)), 2) if jit else 0.0,
        "shimmer_pct": round(100 * float(np.mean(shim)), 2) if shim else 0.0,
        "hnr_db": round(hnr, 1),
        "taxa_silabas_s": round(len(syl) / speech_time, 2),
        "proporcao_pausas": round(1 - speech_time / total_time, 3) if total_time > 1e-3 else 0.0,
        "pausas_por_min": round(len(pauses) / total_time * 60, 1) if total_time > 1e-3 else 0.0,
        "pausa_media_s": round(float(np.mean(pause_d)), 2) if pause_d else 0.0,
        "irregularidade_ritmo": round(rhythm_cv, 3),
        "inclinacao_intensidade_db_s": round(slope, 2),
        "frase_media_s": round(speech_time / max(1, len(phrases)), 2),
        "inspiracoes_audiveis": breath_events,
    }


def _ramp(x, lo, hi):
    """0 em `lo`, 1 em `hi` (funciona com hi < lo para relações inversas)."""
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def vocal_indicators(f: dict) -> dict:
    """Escores 0–1 de fadiga, disartria e dificuldade respiratória.

    Os pontos de corte partem de referências clínicas (fala normal ~3,5–5 sílabas/s;
    shimmer/jitter elevados na disartria; pausas inspiratórias na dispneia) e foram
    calibrados para este extrator com gravações de referência. Em produção o ideal é
    calibrar por paciente (comparar com consultas anteriores do mesmo paciente).
    """
    fatigue = (0.4 * _ramp(f["taxa_silabas_s"], 3.6, 2.6) + 0.4 * _ramp(f["pausa_media_s"], 0.45, 0.8)
               + 0.2 * _ramp(f["inclinacao_intensidade_db_s"], -0.2, -1.0))
    dysarthria = (0.5 * _ramp(f["shimmer_pct"], 8.5, 12.0) + 0.3 * _ramp(f["jitter_pct"], 1.5, 3.0)
                  + 0.2 * _ramp(f["irregularidade_ritmo"], 0.3, 0.5))
    respiratory = (0.45 * _ramp(f["inspiracoes_audiveis"], 1, 5) + 0.25 * _ramp(f["frase_media_s"], 1.5, 0.6)
                   + 0.15 * _ramp(f["pausas_por_min"], 30, 50) + 0.15 * _ramp(f["hnr_db"], 14, 8))
    return {"fadiga": round(float(fatigue), 2), "disartria": round(float(dysarthria), 2),
            "dificuldade_respiratoria": round(float(respiratory), 2)}


# --------------------------------------------------------------------------------------
# Texto (modo offline). O modo Azure está em azure_services.py
# --------------------------------------------------------------------------------------

CRITICAL_TERMS = {
    "respiratorio": ["falta de ar", "dispneia", "nao consigo respirar", "sufoco", "chiado", "tosse",
                     "catarro", "nao consigo nem terminar uma frase"],
    "cardiovascular": ["dor no peito", "aperto no peito", "palpitacao", "desmaio", "taquicardia"],
    "neurologico": ["fala enrolada", "fala esta enrolada", "formigamento", "fraqueza", "fraca",
                    "tontura", "confusao", "dormencia", "visao turva"],
    "infeccioso": ["febre", "calafrio", "calafrios"],
    "queda": ["cai", "caiu", "queda", "tropecei"],
    "fadiga": ["cansada", "cansado", "cansaco", "exausto", "exausta", "sem forca"],
    "saude_mental": ["quero morrer", "sem esperanca", "nao aguento mais", "ansioso", "preocupado", "preocupada"],
}
_NEGATIONS = ["nao", "nunca", "sem", "nenhum", "nenhuma"]
_POS = ["bem", "melhor", "melhorou", "animada", "animado", "otimo", "tranquilo", "diminuiu", "consegui"]
_NEG = ["piorou", "pior", "dor", "muito", "preocupado", "preocupada", "medo", "fraca", "cansada",
        "falta", "nao consigo", "cai", "tontura"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def offline_text_analysis(text: str) -> dict:
    """Termos críticos (com negação), sentimento léxico e frases-chave simples."""
    norm = normalize(text)
    found = []
    for domain, terms in CRITICAL_TERMS.items():
        for term in terms:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", norm):
                before = norm[max(0, m.start() - 25):m.start()].split()[-3:]
                negated = any(w in _NEGATIONS for w in before)
                found.append({"termo": term, "dominio": domain, "negado": negated})
    # remove termos contidos em outros termos encontrados
    found = [f for f in found if not any(f["termo"] != g["termo"] and f["termo"] in g["termo"] for g in found)]
    uniq = {(f["termo"], f["negado"]): f for f in found}.values()
    pos = sum(len(re.findall(r"\b" + w + r"\b", norm)) for w in _POS)
    neg = sum(len(re.findall(r"\b" + w + r"\b", norm)) for w in _NEG)
    total = pos + neg or 1
    label = "positivo" if pos / total > 0.6 else "negativo" if neg / total > 0.6 else "neutro"
    return {"provedor": "offline (léxico)", "sentimento": label,
            "confianca": {"positivo": round(pos / total, 2), "negativo": round(neg / total, 2)},
            "termos_criticos": list(uniq), "frases_chave": [], "entidades_saude": []}


# --------------------------------------------------------------------------------------
# Consolidação -> alertas
# --------------------------------------------------------------------------------------

_INDICATOR_INFO = {
    "fadiga": ("Alteração vocal compatível com fadiga", ["fadiga", "neurologico"],
               "Avaliar fadiga (anemia, distúrbio hidroeletrolítico, sono) e ajustar intensidade da reabilitação."),
    "disartria": ("Alteração vocal compatível com disartria", ["neurologico"],
                  "Avaliação fonoaudiológica e neurológica; disartria nova/pior pode indicar evento neurológico."),
    "dificuldade_respiratoria": ("Padrão de fala com dificuldade respiratória", ["respiratorio"],
                                 "Aferir FR e SpO₂, auscultar o paciente; considerar oxigenoterapia."),
}


def _clinical_domain(text: str) -> str | None:
    """Domínio clínico de um trecho, segundo o vocabulário de termos críticos (ou None)."""
    norm = normalize(text)
    for domain, terms in CRITICAL_TERMS.items():
        if any(re.search(r"\b" + re.escape(t) + r"\b", norm) for t in terms):
            return domain
    return None


def _critical_findings(text_result: dict) -> list[dict]:
    """Termos críticos não negados, combinando o léxico com as entidades do Azure.

    As entidades de saúde do Azure (SymptomOrSign/Diagnosis) só contam quando caem em um
    domínio clínico do vocabulário crítico: o Azure também marca como sintoma trechos como
    "sentindo bem melhor" ou "dor da cirurgia diminuiu", que não devem gerar alerta. Quando o
    Azure detecta negação ("não sinto dor no peito"), ela prevalece sobre o léxico.
    """
    azure_negated = {normalize(e["texto"]) for e in text_result.get("entidades_saude", []) if e.get("negado")}
    crit = [t for t in text_result.get("termos_criticos", [])
            if not t["negado"] and not any(normalize(t["termo"]) in n for n in azure_negated)]
    for e in text_result.get("entidades_saude", []):
        if e.get("categoria") in ("SymptomOrSign", "Diagnosis") and not e.get("negado"):
            domain = _clinical_domain(e["texto"])
            if domain:
                crit.append({"termo": e["texto"], "dominio": domain})
    return crit


def audio_to_alerts(patient_id: str, timestamp: str, indicators: dict, text_result: dict) -> list[Alert]:
    alerts = []
    for k, score in indicators.items():
        if score >= 0.5:
            title, domains, action = _INDICATOR_INFO[k]
            sev = "ALTO" if score >= 0.75 else "MODERADO"
            alerts.append(Alert(patient_id, timestamp, "audio", sev, title,
                                f"Escore acústico de {k.replace('_', ' ')} = {score:.2f} (0–1).",
                                domains, {"indicador": k, "escore": score}, action))
    crit = _critical_findings(text_result)
    if crit:
        domains = sorted({t["dominio"] for t in crit if t["dominio"]})
        severe_domains = {"cardiovascular", "respiratorio", "neurologico", "queda", "saude_mental"}
        sev = "ALTO" if (set(domains) & severe_domains and text_result.get("sentimento") == "negativo") else "MODERADO"
        # um termo por conceito: sem variações de acento e sem termos contidos em outros
        by_norm = {}
        for t in sorted(crit, key=lambda t: len(t["termo"]), reverse=True):
            by_norm.setdefault(normalize(t["termo"]), t["termo"])
        terms = sorted(v for k, v in by_norm.items() if not any(k != o and k in o for o in by_norm))
        alerts.append(Alert(patient_id, timestamp, "audio", sev, "Termos críticos na consulta",
                            f"Relatos do paciente: {', '.join(terms)}. Sentimento: {text_result.get('sentimento')} "
                            f"({text_result.get('provedor')}).", domains, {"termos": terms},
                            "Médico assistente deve revisar a transcrição e investigar os sintomas relatados."))
    return alerts
