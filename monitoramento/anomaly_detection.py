from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .alerts import Alert
from .config import VITAL_LABELS, VITAL_RANGES

VITALS = list(VITAL_RANGES)

# ======================================================================================
# 1. SINAIS VITAIS
# ======================================================================================


def news2_components(v: dict) -> dict:
    """Pontuação NEWS2 por parâmetro (paciente alerta, em ar ambiente)."""

    def band(x, rules):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        for cond, pts in rules:
            if cond(x):
                return pts
        return 0

    return {
        "resp_rate": band(v.get("resp_rate"), [(lambda x: x <= 8, 3), (lambda x: x <= 11, 1),
                                               (lambda x: x <= 20, 0), (lambda x: x <= 24, 2),
                                               (lambda x: True, 3)]),
        "spo2": band(v.get("spo2"), [(lambda x: x <= 91, 3), (lambda x: x <= 93, 2),
                                     (lambda x: x <= 95, 1), (lambda x: True, 0)]),
        "systolic_bp": band(v.get("systolic_bp"), [(lambda x: x <= 90, 3), (lambda x: x <= 100, 2),
                                                   (lambda x: x <= 110, 1), (lambda x: x <= 219, 0),
                                                   (lambda x: True, 3)]),
        "heart_rate": band(v.get("heart_rate"), [(lambda x: x <= 40, 3), (lambda x: x <= 50, 1),
                                                 (lambda x: x <= 90, 0), (lambda x: x <= 110, 1),
                                                 (lambda x: x <= 130, 2), (lambda x: True, 3)]),
        "temperature": band(v.get("temperature"), [(lambda x: x <= 35.0, 3), (lambda x: x <= 36.0, 1),
                                                   (lambda x: x <= 38.0, 0), (lambda x: x <= 39.0, 1),
                                                   (lambda x: True, 2)]),
    }


# limites "graves" que elevam a severidade de um desvio de faixa
_SEVERE = {"heart_rate": (40, 130), "systolic_bp": (90, 180), "diastolic_bp": (40, 110),
           "spo2": (90, 101), "resp_rate": (8, 25), "temperature": (35.0, 39.0)}
_DOMAIN = {"heart_rate": ["cardiovascular"], "systolic_bp": ["cardiovascular"],
           "diastolic_bp": ["cardiovascular"], "spo2": ["respiratorio"],
           "resp_rate": ["respiratorio"], "temperature": ["infeccioso"]}
# inclinações (por hora, em janela de 3 h) consideradas deterioração; exige >= 3 sinais juntos
_TREND = {"heart_rate": 3.0, "resp_rate": 0.8, "spo2": -0.5, "temperature": 0.15, "systolic_bp": -2.5}
_TREND_WINDOW = 36  # amostras (3 h)
_TREND_MIN_SIGNALS = 3
# MAD mínimo por sinal: evita que variações clinicamente irrelevantes virem z-scores altos
_MAD_FLOOR = {"heart_rate": 3.0, "systolic_bp": 5.0, "diastolic_bp": 4.0, "spo2": 1.0,
              "resp_rate": 1.5, "temperature": 0.25}


class VitalsIsolationForest:
    """Isolation Forest multivariado treinado em períodos de estabilidade clínica."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=contamination,
                                     random_state=random_state)
        self.threshold = None

    def fit(self, df: pd.DataFrame) -> VitalsIsolationForest:
        X = self.scaler.fit_transform(df[VITALS].to_numpy())
        self.model.fit(X)
        return self

    def score_many(self, df: pd.DataFrame) -> np.ndarray:
        """Escore de anomalia (maior = mais anômalo; > 0 = anomalia)."""
        X = self.scaler.transform(df[VITALS].to_numpy(dtype=float))
        return -(self.model.score_samples(X) - self.model.offset_)

    def score(self, sample: dict) -> float:
        return float(self.score_many(pd.DataFrame([sample]))[0])


class StreamingVitalsMonitor:
    """Monitor em tempo real (online) para UM paciente."""

    def __init__(self, patient_id: str, iforest: VitalsIsolationForest | None = None,
                 window: int = 72, persistence: int = 3, z_threshold: float = 4.0,
                 sample_minutes: int = 5):
        self.patient_id = patient_id
        self.iforest = iforest
        self.history: deque = deque(maxlen=window)
        self.persistence = persistence
        self.z_threshold = z_threshold
        self.sample_minutes = sample_minutes
        self._streak: dict[str, int] = {}
        self.artifacts: list[dict] = []
        self.records: list[dict] = []

    # --------------------------------------------------------------
    def _bump(self, key: str, active: bool) -> int:
        if active:
            self._streak[key] = self._streak.get(key, 0) + 1
        else:
            if 0 < self._streak.get(key, 0) < self.persistence and key.startswith("dev:"):
                self.artifacts.append({"patient_id": self.patient_id, "sinal": key[4:],
                                       "amostras": self._streak[key]})
            self._streak[key] = 0
        return self._streak[key]

    def update(self, sample: dict, iforest_score: float | None = None) -> list[Alert]:
        """Processa uma nova amostra e devolve os alertas disparados por ela.

        `iforest_score` permite passar o escore já calculado em lote (equivalente, pois o
        Isolation Forest avalia cada amostra de forma independente)."""
        ts = sample["timestamp"]
        alerts: list[Alert] = []
        hist = np.array(self.history, dtype=float) if self.history else None  # (n, 6)
        rec = {"timestamp": ts, "patient_id": self.patient_id}

        # 1) NEWS2
        comp = news2_components(sample)
        news2 = sum(comp.values())
        rec["news2"] = news2
        n_hi = self._bump("news2_hi", news2 >= 7)
        n_md = self._bump("news2_md", news2 >= 5)
        n_red = self._bump("news2_red", max(comp.values()) == 3)
        if n_hi == 2:
            alerts.append(self._alert(ts, "CRITICO", "NEWS2 ≥ 7 — resposta de emergência",
                                      f"Escore NEWS2 = {news2} sustentado. Componentes: {comp}.",
                                      ["respiratorio", "cardiovascular", "infeccioso"],
                                      {"news2": news2, "componentes": comp},
                                      "Acionar time de resposta rápida; avaliação médica imediata; "
                                      "considerar transferência/escala de cuidado."))
        elif n_md == 2:
            alerts.append(self._alert(ts, "ALTO", "NEWS2 ≥ 5 — resposta urgente",
                                      f"Escore NEWS2 = {news2} em 2 aferições seguidas. Componentes: {comp}.",
                                      ["respiratorio", "cardiovascular", "infeccioso"],
                                      {"news2": news2, "componentes": comp},
                                      "Avaliação médica urgente; aumentar frequência de monitorização "
                                      "para no mínimo de hora em hora."))
        elif n_red == 2 and news2 < 5:
            par = max(comp, key=comp.get)
            alerts.append(self._alert(ts, "ALTO", "Parâmetro isolado em faixa vermelha (NEWS2 = 3)",
                                      f"{VITAL_LABELS[par]} = {sample[par]}.", _DOMAIN.get(par, []),
                                      {"parametro": par, "valor": sample[par]},
                                      "Avaliação médica urgente do parâmetro alterado."))

        # 2) Desvio de faixa clínica / z-score robusto (com persistência)
        for vi, v in enumerate(VITALS):
            x = sample.get(v)
            if x is None or np.isnan(x):
                continue
            lo, hi = VITAL_RANGES[v]
            z = 0.0
            if hist is not None and len(hist) >= 24:
                col = hist[:, vi]
                med = float(np.nanmedian(col))
                mad = max(float(np.nanmedian(np.abs(col - med))) * 1.4826, _MAD_FLOOR[v])
                z = (x - med) / mad
            rec[f"z_{v}"] = round(z, 2)
            out = x < lo or x > hi
            dev = out or abs(z) > self.z_threshold
            streak = self._bump(f"dev:{v}", dev)
            rec[f"dev_{v}"] = dev
            if streak == self.persistence:
                slo, shi = _SEVERE[v]
                severe = x < slo or x > shi
                if out:
                    direction = "acima" if x > hi else "abaixo"
                    title = f"{VITAL_LABELS[v]} {direction} da faixa"
                    sev = "ALTO" if severe else "MODERADO"
                    if v == "systolic_bp" and x >= 180:
                        title, sev = "Crise hipertensiva (PAS ≥ 180 mmHg)", "ALTO"
                    if v == "spo2" and x < 88:
                        sev = "CRITICO"
                else:
                    title = f"{VITAL_LABELS[v]}: desvio abrupto do basal do paciente"
                    sev = "MODERADO"
                alerts.append(self._alert(
                    ts, sev, title,
                    f"{VITAL_LABELS[v]} = {x} por ≥ {self.persistence * self.sample_minutes} min "
                    f"(faixa {lo}–{hi}; z-robusto = {z:.1f}).",
                    _DOMAIN.get(v, []), {"sinal": v, "valor": x, "z_robusto": round(z, 2)},
                    "Confirmar aferição à beira-leito e reavaliar o paciente."))

        # 3) Isolation Forest multivariado
        if self.iforest is not None or iforest_score is not None:
            score = iforest_score if iforest_score is not None else self.iforest.score(sample)
            iso = score > 0
            rec["iforest_score"] = round(float(score), 3)
            rec["iforest_anomaly"] = iso
            if self._bump("iforest", iso) == self.persistence * 2:
                alerts.append(self._alert(
                    ts, "MODERADO", "Padrão multivariado atípico (Isolation Forest)",
                    "Combinação de sinais vitais incomum em relação aos períodos de estabilidade, "
                    "por ≥ 30 min.", ["cardiovascular", "respiratorio"],
                    {"score": rec["iforest_score"]}, "Revisar tendência dos sinais vitais."))

        # 4) Tendência (alerta precoce)
        if hist is not None and len(hist) >= _TREND_WINDOW:
            recent = hist[-(_TREND_WINDOW - 1):]
            y_all = {v: np.append(recent[:, VITALS.index(v)], sample[v]) for v in _TREND}
            hours = np.arange(_TREND_WINDOW) * self.sample_minutes / 60
            worsening = []
            for v, thr in _TREND.items():
                slope = np.polyfit(hours, y_all[v], 1)[0]
                if (thr > 0 and slope > thr) or (thr < 0 and slope < thr):
                    worsening.append((v, round(float(slope), 2)))
            rec["trend_signals"] = len(worsening)
            if self._bump("trend", len(worsening) >= _TREND_MIN_SIGNALS) == 2:
                alerts.append(self._alert(
                    ts, "MODERADO", "Tendência de deterioração clínica",
                    "Piora simultânea em 3 h: " + ", ".join(f"{VITAL_LABELS[v]} ({s:+}/h)" for v, s in worsening),
                    ["infeccioso", "respiratorio", "cardiovascular"], {"tendencias": dict(worsening)},
                    "Alerta precoce: reavaliar paciente, considerar protocolo de sepse se houver foco infeccioso."))

        self.history.append([sample[k] for k in VITALS])
        self.records.append(rec)
        return alerts

    def _alert(self, ts, sev, title, desc, domains, evidence, action) -> Alert:
        return Alert(self.patient_id, ts, "sinais_vitais", sev, title, desc, domains, evidence, action)


def run_vitals_monitoring(vitals: pd.DataFrame, baseline_hours: float = 12.0,
                          on_alert=None) -> tuple[list[Alert], pd.DataFrame, list[dict]]:
    """Treina o Isolation Forest nas primeiras `baseline_hours` (período estável) e
    reproduz o fluxo como streaming, paciente a paciente, em ordem cronológica global."""
    t0 = vitals["timestamp"].min()
    base = vitals[vitals["timestamp"] < t0 + pd.Timedelta(hours=baseline_hours)].dropna(subset=VITALS)
    iforest = VitalsIsolationForest().fit(base) if len(base) > 50 else None
    monitors = {pid: StreamingVitalsMonitor(pid, iforest) for pid in vitals["patient_id"].unique()}
    alerts: list[Alert] = []
    ordered = vitals.sort_values("timestamp", kind="stable")
    scores = iforest.score_many(ordered.fillna(ordered[VITALS].median())) if iforest else [None] * len(ordered)
    for row, sc in zip(ordered.to_dict("records"), scores):
        new = monitors[row["patient_id"]].update(row, sc)
        for a in new:
            if on_alert:
                on_alert(a)
        alerts.extend(new)
    records = pd.DataFrame([r for m in monitors.values() for r in m.records])
    artifacts = [a for m in monitors.values() for a in m.artifacts]
    return alerts, records, artifacts


# ======================================================================================
# 2. PRESCRIÇÕES
# ======================================================================================

DRUG_DB = {
    "dipirona": {"classe": "analgesico", "max_diario_mg": 4000},
    "omeprazol": {"classe": "IBP", "max_diario_mg": 80},
    "enoxaparina": {"classe": "anticoagulante", "max_diario_mg": 200, "alta_vigilancia": True},
    "ceftriaxona": {"classe": "antibiotico", "max_diario_mg": 4000, "curso_min_h": 120},
    "azitromicina": {"classe": "antibiotico_macrolideo", "max_diario_mg": 500},
    "losartana": {"classe": "BRA", "max_diario_mg": 100},
    "valsartana": {"classe": "BRA", "max_diario_mg": 320},
    "atorvastatina": {"classe": "estatina", "max_diario_mg": 80},
    "cetorolaco": {"classe": "AINE", "max_diario_mg": 120},
}
INTERACTIONS = {
    frozenset({"anticoagulante", "AINE"}): ("ALTO", "Risco aumentado de sangramento (anticoagulante + AINE)."),
}


def check_prescriptions(rx: pd.DataFrame) -> list[Alert]:
    """Analisa a evolução das prescrições de cada paciente, versão a versão."""
    alerts: list[Alert] = []
    rx = rx.copy()
    rx["timestamp"] = pd.to_datetime(rx["timestamp"])
    rx["dose_diaria_mg"] = rx["dose_mg"] * rx["vezes_dia"]

    def mk(pid, ts, sev, title, desc, ev, action, domains=("medicacao",)):
        return Alert(pid, ts, "prescricao", sev, title, desc, list(domains), ev, action)

    for pid, prx in rx.groupby("patient_id"):
        versions = sorted(prx["timestamp"].unique())
        prev = None
        for ts in versions:
            cur = prx[prx["timestamp"] == ts].set_index("medicamento")
            ts_s = str(pd.Timestamp(ts))
            for drug, row in cur.iterrows():
                info = DRUG_DB.get(drug, {})
                daily = float(row["dose_diaria_mg"])
                mx = info.get("max_diario_mg")
                if mx and daily > mx:
                    sev = "CRITICO" if daily > 1.5 * mx else "ALTO"
                    alerts.append(mk(pid, ts_s, sev, f"Dose acima do máximo: {drug}",
                                     f"{drug} {row['dose_mg']:.0f} mg × {row['vezes_dia']}/dia = "
                                     f"{daily:.0f} mg/dia (máximo {mx} mg/dia). Prescritor: {row['prescritor']}.",
                                     {"medicamento": drug, "dose_diaria_mg": daily, "max_mg": mx},
                                     "Suspender administração e confirmar com o prescritor/farmácia clínica."))
                if prev is not None and drug in prev.index:
                    old = float(prev.loc[drug, "dose_diaria_mg"])
                    ratio = daily / old if old else np.inf
                    if ratio >= 2.0:
                        sev = "ALTO" if info.get("alta_vigilancia") else "MODERADO"
                        alerts.append(mk(pid, ts_s, sev, f"Aumento abrupto de dose: {drug}",
                                         f"Dose diária passou de {old:.0f} para {daily:.0f} mg (×{ratio:.1f})"
                                         + (" — medicamento de ALTA VIGILÂNCIA." if info.get("alta_vigilancia") else "."),
                                         {"medicamento": drug, "de_mg": old, "para_mg": daily, "fator": round(ratio, 2)},
                                         "Verificar justificativa clínica (ex.: dose terapêutica vs profilática)."))
                    elif ratio <= 0.4:
                        alerts.append(mk(pid, ts_s, "MODERADO", f"Redução abrupta de dose: {drug}",
                                         f"Dose diária passou de {old:.0f} para {daily:.0f} mg.",
                                         {"medicamento": drug, "de_mg": old, "para_mg": daily},
                                         "Confirmar se a redução é intencional."))
            classes = cur.index.map(lambda d: DRUG_DB.get(d, {}).get("classe", d))
            dup = pd.Series(cur.index, index=classes).groupby(level=0).apply(list)
            for cls, drugs in dup.items():
                if len(drugs) > 1:
                    alerts.append(mk(pid, ts_s, "ALTO", f"Duplicidade terapêutica ({cls})",
                                     f"Dois fármacos da mesma classe prescritos juntos: {', '.join(drugs)}.",
                                     {"classe": cls, "medicamentos": drugs},
                                     "Revisar prescrição: manter apenas um agente da classe."))
            present = set(classes)
            for pair, (sev, desc) in INTERACTIONS.items():
                if pair <= present:
                    alerts.append(mk(pid, ts_s, sev, "Interação medicamentosa relevante", desc,
                                     {"classes": sorted(pair), "medicamentos": list(cur.index)},
                                     "Avaliar troca do AINE por analgésico sem efeito antiplaquetário.",
                                     ("medicacao", "hemorragico")))
            if prev is not None:
                for drug in set(prev.index) - set(cur.index):
                    info = DRUG_DB.get(drug, {})
                    if "curso_min_h" in info:
                        started = prx[prx["medicamento"] == drug]["timestamp"].min()
                        dur_h = (pd.Timestamp(ts) - started).total_seconds() / 3600
                        if dur_h < info["curso_min_h"]:
                            alerts.append(mk(pid, ts_s, "ALTO", f"Suspensão abrupta de {info['classe']}: {drug}",
                                             f"{drug} suspenso após {dur_h:.0f} h (curso mínimo esperado "
                                             f"{info['curso_min_h']} h), sem substituto da mesma classe.",
                                             {"medicamento": drug, "duracao_h": dur_h},
                                             "Confirmar com o médico assistente se a suspensão é intencional.",
                                             ("medicacao", "infeccioso")))
            prev = cur
    return alerts


# ======================================================================================
# 3. MOVIMENTAÇÃO DURANTE A INTERNAÇÃO
# ======================================================================================

def _movement_features(df: pd.DataFrame) -> np.ndarray:
    hod = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    return np.column_stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                            np.log1p(df["atividade"].to_numpy()), df["saidas_leito"].to_numpy()])


class MovementMonitor:
    def __init__(self, immobility_hours: int = 4, immobility_threshold: float = 5.0,
                 fall_g: float = 3.0, random_state: int = 42):
        self.immobility_hours = immobility_hours
        self.immobility_threshold = immobility_threshold
        self.fall_g = fall_g
        self.model = IsolationForest(n_estimators=200, contamination=0.02, random_state=random_state)
        self.fitted = False

    def fit(self, reference: pd.DataFrame) -> MovementMonitor:
        self.model.fit(_movement_features(reference))
        self.fitted = True
        return self

    def detect(self, mv: pd.DataFrame) -> tuple[list[Alert], pd.DataFrame]:
        alerts: list[Alert] = []
        out = []
        for pid, df in mv.sort_values("timestamp").groupby("patient_id"):
            df = df.copy()
            df["iforest_anomalia"] = (self.model.predict(_movement_features(df)) == -1) if self.fitted else False
            streak = 0
            for _, r in df.iterrows():
                ts = str(r["timestamp"])
                hod = pd.Timestamp(r["timestamp"]).hour
                rule_hit = False
                if r["impacto_max_g"] >= self.fall_g:
                    rule_hit = True
                    alerts.append(Alert(pid, ts, "movimentacao", "CRITICO", "Possível queda detectada",
                                        f"Pico de aceleração de {r['impacto_max_g']} g com saída do leito.",
                                        ["queda", "neurologico"], {"impacto_g": r["impacto_max_g"]},
                                        "Avaliar o paciente imediatamente; protocolo pós-queda (TC de crânio se anticoagulado)."))
                streak = streak + 1 if r["atividade"] < self.immobility_threshold and r["saidas_leito"] == 0 else 0
                if streak == self.immobility_hours:
                    rule_hit = True
                    alerts.append(Alert(pid, ts, "movimentacao", "ALTO", "Imobilidade prolongada",
                                        f"≥ {self.immobility_hours} h sem mudança de decúbito/atividade.",
                                        ["integridade_pele", "neurologico"], {"horas": streak},
                                        "Reposicionar o paciente (mudança de decúbito) e avaliar pele (escala de Braden)."))
                if 0 <= hod <= 5 and (r["atividade"] > 150 or r["saidas_leito"] >= 2):
                    rule_hit = True
                    alerts.append(Alert(pid, ts, "movimentacao", "MODERADO", "Agitação noturna",
                                        f"Atividade {r['atividade']:.0f} e {r['saidas_leito']} saídas do leito às {hod}h.",
                                        ["neurologico", "queda"], {"atividade": r["atividade"]},
                                        "Avaliar delirium (CAM-ICU), dor e risco de queda; grades elevadas."))
                if r["iforest_anomalia"] and not rule_hit:
                    alerts.append(Alert(pid, ts, "movimentacao", "INFO", "Padrão de movimentação atípico",
                                        "Isolation Forest sinalizou combinação incomum de horário/atividade.",
                                        [], {"atividade": r["atividade"]}, "Registro para acompanhamento."))
            out.append(df)
        return alerts, pd.concat(out, ignore_index=True)
