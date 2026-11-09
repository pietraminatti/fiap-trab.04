from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("fase4.alertas")

SEVERITY_ORDER = {"INFO": 0, "MODERADO": 1, "ALTO": 2, "CRITICO": 3}
ROUTING = {
    "CRITICO": "Médico plantonista + Enfermagem (imediato)",
    "ALTO": "Enfermagem do setor (até 15 min)",
    "MODERADO": "Painel de monitoramento (próxima ronda)",
    "INFO": "Registro / auditoria",
}


@dataclass
class Alert:
    patient_id: str
    timestamp: str  # momento clínico do evento (ISO 8601)
    source: str  # sinais_vitais | prescricao | movimentacao | video | audio | fusao
    severity: str  # INFO | MODERADO | ALTO | CRITICO
    title: str
    description: str
    domains: list[str] = field(default_factory=list)  # respiratorio, cardiovascular, ...
    evidence: dict = field(default_factory=dict)
    recommended_action: str = ""
    routed_to: str = ""
    emitted_at: str = ""

    def __post_init__(self):
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Severidade inválida: {self.severity}")
        # formato único (segundos) para ordenar/comparar alertas de qualquer fonte
        self.timestamp = pd.Timestamp(self.timestamp).isoformat(sep=" ", timespec="seconds")

    @property
    def level(self) -> int:
        return SEVERITY_ORDER[self.severity]


class AlertManager:
    """Recebe alertas de todos os módulos, deduplica e entrega nos canais configurados."""

    def __init__(self, jsonl_path: str | Path | None = None, webhook_url: str | None = None,
                 cooldown_minutes: int = 30, min_webhook_severity: str = "ALTO"):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("", encoding="utf-8")
        self.webhook_url = webhook_url
        self.cooldown = pd.Timedelta(minutes=cooldown_minutes)
        self.min_webhook_level = SEVERITY_ORDER[min_webhook_severity]
        self.alerts: list[Alert] = []
        self.suppressed = 0
        self._last_seen: dict[tuple, pd.Timestamp] = {}

    # ------------------------------------------------------------------
    def emit(self, alert: Alert) -> bool:
        """Registra o alerta. Retorna False se foi suprimido por duplicidade."""
        # severidade faz parte da chave: uma escalada (ALTO -> CRITICO) sempre gera novo alerta
        key = (alert.patient_id, alert.source, alert.title, alert.severity)
        ts = pd.Timestamp(alert.timestamp)
        last = self._last_seen.get(key)
        if last is not None and abs(ts - last) < self.cooldown:
            self.suppressed += 1
            self._last_seen[key] = ts  # janela deslizante: condição persistente não re-alerta
            return False
        self._last_seen[key] = ts
        alert.routed_to = ROUTING[alert.severity]
        alert.emitted_at = datetime.now().isoformat(timespec="seconds")
        self.alerts.append(alert)
        self._deliver(alert)
        return True

    def emit_many(self, alerts) -> int:
        return sum(self.emit(a) for a in alerts)

    # ------------------------------------------------------------------
    def _deliver(self, alert: Alert) -> None:
        msg = (f"[{alert.severity}] {alert.patient_id} | {alert.source} | {alert.title} "
               f"@ {alert.timestamp} -> {alert.routed_to}")
        log_fn = logger.warning if alert.level >= 2 else logger.info
        log_fn(msg)
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(alert), ensure_ascii=False, default=str) + "\n")
        if self.webhook_url and alert.level >= self.min_webhook_level:
            self._send_webhook(alert)

    def _send_webhook(self, alert: Alert) -> None:
        """Envia para um webhook (Microsoft Teams 'Incoming Webhook', Slack, etc.)."""
        try:
            import requests

            text = (f"**{alert.severity} — {alert.title}**\n\nPaciente: {alert.patient_id} | "
                    f"Fonte: {alert.source} | Momento: {alert.timestamp}\n\n{alert.description}\n\n"
                    f"Conduta sugerida: {alert.recommended_action}")
            requests.post(self.webhook_url, json={"text": text}, timeout=5)
        except Exception as exc:  # o alerta já está persistido; falha de rede não derruba o fluxo
            logger.error(f"Falha ao enviar webhook: {exc}")

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame(columns=list(Alert.__dataclass_fields__))
        df = pd.DataFrame([asdict(a) for a in self.alerts])
        df["nivel"] = df["severity"].map(SEVERITY_ORDER)
        return df.sort_values(["nivel", "timestamp"], ascending=[False, True]).drop(columns="nivel")

    def summary(self) -> dict:
        df = self.to_dataframe()
        by_sev = df["severity"].value_counts().to_dict() if len(df) else {}
        by_pat = df["patient_id"].value_counts().to_dict() if len(df) else {}
        return {"total": len(df), "suprimidos_por_duplicidade": self.suppressed,
                "por_severidade": by_sev, "por_paciente": by_pat}
