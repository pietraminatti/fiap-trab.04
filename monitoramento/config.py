from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Pasta projeto4_multimodal/ (tudo da Fase 4 fica dentro dela)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "reports"

try:  # python-dotenv é opcional
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # pragma: no cover
    pass
FIGURES_DIR = OUTPUT_DIR / "figures"


@dataclass
class AzureConfig:
    """Credenciais dos serviços Azure Cognitive Services (AI Services)."""

    speech_key: str | None = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY"))
    speech_region: str | None = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION"))
    language_key: str | None = field(default_factory=lambda: os.getenv("AZURE_LANGUAGE_KEY"))
    language_endpoint: str | None = field(
        default_factory=lambda: os.getenv("AZURE_LANGUAGE_ENDPOINT")
    )
    speech_language: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_LANGUAGE", "pt-BR"))

    @property
    def speech_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_enabled(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


@dataclass
class AlertConfig:
    """Para onde os alertas são enviados."""

    webhook_url: str | None = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL"))
    # não repetir o mesmo alerta (paciente + fonte + título + severidade) enquanto a condição
    # reaparecer em intervalos menores que este (janela deslizante) — reduz fadiga de alarme
    cooldown_minutes: int = field(default_factory=lambda: int(os.getenv("ALERT_COOLDOWN_MIN", "120")))


# Faixas clínicas de referência (adulto em repouso) usadas pelas regras e pelo NEWS2.
VITAL_RANGES = {
    "heart_rate": (50, 110),  # bpm
    "systolic_bp": (100, 160),  # mmHg
    "diastolic_bp": (50, 100),  # mmHg
    "spo2": (94, 100),  # %
    "resp_rate": (10, 22),  # irpm
    "temperature": (35.5, 38.0),  # °C
}

VITAL_LABELS = {
    "heart_rate": "Frequência cardíaca (bpm)",
    "systolic_bp": "PA sistólica (mmHg)",
    "diastolic_bp": "PA diastólica (mmHg)",
    "spo2": "SpO₂ (%)",
    "resp_rate": "Frequência respiratória (irpm)",
    "temperature": "Temperatura (°C)",
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
