from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoramento.azure_services import (  # noqa: E402
    AzureSpeechToText,
    AzureTextAnalytics,
    AzureTextToSpeech,
)
from monitoramento.config import DATA_DIR, AzureConfig  # noqa: E402
from monitoramento.synthetic_data import TRANSCRIPTS  # noqa: E402


def main() -> int:
    cfg = AzureConfig()
    if not (cfg.speech_enabled and cfg.language_enabled):
        print("Credenciais Azure ausentes. Copie .env.example para "
              ".env e preencha as chaves.")
        return 2
    wav = DATA_DIR / "consulta_azure_tts.wav"
    print("1) Azure Text to Speech ...")
    AzureTextToSpeech(cfg.speech_key, cfg.speech_region).synthesize(TRANSCRIPTS["P002"], wav, rate="-10%")
    wav.with_suffix(".txt").write_text(TRANSCRIPTS["P002"], encoding="utf-8")
    print(f"   áudio salvo em {wav}")

    print("2) Azure Speech to Text ...")
    stt = AzureSpeechToText(cfg.speech_key, cfg.speech_region, cfg.speech_language).transcribe(wav)
    print(f"   status={stt['status']} confiança={stt['confianca']}\n   texto: {stt['texto']}")

    print("3) Azure AI Language ...")
    res = AzureTextAnalytics(cfg.language_key, cfg.language_endpoint, cfg.speech_language).analyze(
        stt["texto"] or TRANSCRIPTS["P002"])
    print(json.dumps({k: res[k] for k in ("sentimento", "confianca", "frases_chave", "entidades_saude",
                                          "erros_azure")}, indent=2, ensure_ascii=False))
    return 0 if not res["erros_azure"] and stt["texto"] else 1


if __name__ == "__main__":
    sys.exit(main())
