from __future__ import annotations

import pandas as pd

from .alerts import SEVERITY_ORDER, Alert

MODALITY_WEIGHTS = {"sinais_vitais": 0.30, "prescricao": 0.20, "audio": 0.20,
                    "video": 0.15, "movimentacao": 0.15}
SEVERITY_SCORE = {"INFO": 0.1, "MODERADO": 0.4, "ALTO": 0.7, "CRITICO": 1.0}
DOMAIN_LABEL = {"respiratorio": "respiratório", "cardiovascular": "cardiovascular",
                "neurologico": "neurológico", "queda": "risco de queda", "infeccioso": "infeccioso",
                "medicacao": "medicação", "motor": "motor", "fadiga": "fadiga",
                "hemorragico": "hemorrágico", "integridade_pele": "integridade da pele",
                "seguranca": "segurança", "saude_mental": "saúde mental"}


def _level(score: float) -> str:
    return "CRITICO" if score >= 0.75 else "ALTO" if score >= 0.5 else "MODERADO" if score >= 0.25 else "BAIXO"


def fuse_patient(patient_id: str, alerts: list[Alert]) -> dict:
    mine = [a for a in alerts if a.patient_id == patient_id and a.source in MODALITY_WEIGHTS]
    scores = {}
    for m in MODALITY_WEIGHTS:
        al = [a for a in mine if a.source == m and a.severity != "INFO"]
        if al:
            top = max(SEVERITY_SCORE[a.severity] for a in al)
            scores[m] = round(min(1.0, top + 0.05 * min(len(al) - 1, 4)), 2)
        else:
            scores[m] = 0.0
    base = 0.6 * max(scores.values()) + 0.4 * sum(MODALITY_WEIGHTS[m] * s for m, s in scores.items())

    domain_sources: dict[str, set] = {}
    for a in mine:
        if a.severity == "INFO":
            continue
        for d in a.domains:
            domain_sources.setdefault(d, set()).add(a.source)
    convergent = {d: sorted(s) for d, s in domain_sources.items() if len(s) >= 2}
    bonus = min(0.3, 0.1 * len(convergent))
    risk = round(min(1.0, base + bonus), 3)
    top_alerts = sorted([a for a in mine if a.severity != "INFO"], key=lambda a: -a.level)[:5]
    return {"patient_id": patient_id, "escores_modalidade": scores, "escore_base": round(base, 3),
            "bonus_corroboracao": round(bonus, 2), "risco": risk, "nivel": _level(risk),
            "dominios_convergentes": convergent,
            "principais_alertas": [f"[{a.severity}] {a.source}: {a.title}" for a in top_alerts],
            "n_alertas": len(mine)}


def fuse_all(patient_ids, alerts: list[Alert]) -> list[dict]:
    return [fuse_patient(pid, alerts) for pid in patient_ids]


def fusion_alerts(fused: list[dict], alerts: list[Alert]) -> list[Alert]:
    """Alerta consolidado por paciente — é o que chega ao médico.

    A fusão é recalculada em ordem cronológica (como aconteceria em produção, a cada
    novo alerta). Um alerta de fusão é emitido no **momento** em que o risco do paciente
    atinge ALTO e novamente se escalar para CRÍTICO.
    """
    out = []
    for f in fused:
        pid = f["patient_id"]
        mine = sorted([a for a in alerts if a.patient_id == pid and a.source != "fusao"],
                      key=lambda a: pd.Timestamp(a.timestamp))
        reached = set()
        for i in range(1, len(mine) + 1):
            snap = fuse_patient(pid, mine[:i])
            lvl = snap["nivel"]
            if lvl not in ("ALTO", "CRITICO") or lvl in reached:
                continue
            reached.add(lvl)
            if lvl == "CRITICO":
                reached.add("ALTO")
            window = mine[:i]
            conv = "; ".join(f"{DOMAIN_LABEL.get(d, d)} ({' + '.join(s)})"
                             for d, s in snap["dominios_convergentes"].items())
            actions = []
            for a in sorted(window, key=lambda a: -a.level):
                if a.recommended_action and a.recommended_action not in actions and a.level >= SEVERITY_ORDER["ALTO"]:
                    actions.append(a.recommended_action)
            out.append(Alert(
                pid, window[-1].timestamp, "fusao", lvl,
                f"Risco multimodal {lvl} (escore {snap['risco']:.2f})",
                "Achados convergentes: " + (conv or "nenhum") + ". Principais: "
                + " | ".join(snap["principais_alertas"][:3]),
                list(snap["dominios_convergentes"]), {"escores": snap["escores_modalidade"], "risco": snap["risco"]},
                " ".join(actions[:3])))
    return out
