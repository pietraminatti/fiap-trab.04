from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .audio_analysis import offline_text_analysis
from .config import AzureConfig

logger = logging.getLogger("fase4.azure")
API_VERSION = "2023-04-01"


class AzureServiceError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Speech to Text
# --------------------------------------------------------------------------------------

class AzureSpeechToText:
    provider = "Azure Speech to Text"

    def __init__(self, key: str, region: str, language: str = "pt-BR", timeout: int = 60):
        self.key, self.region, self.language, self.timeout = key, region, language, timeout

    @property
    def url(self) -> str:
        return (f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/"
                f"cognitiveservices/v1?language={self.language}&format=detailed")

    def transcribe(self, wav_path: str | Path) -> dict:
        """Transcreve um WAV PCM 16 bits mono 16 kHz (até 60 s)."""
        headers = {"Ocp-Apim-Subscription-Key": self.key,
                   "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                   "Accept": "application/json"}
        with open(wav_path, "rb") as fh:
            resp = requests.post(self.url, headers=headers, data=fh, timeout=self.timeout)
        if resp.status_code != 200:
            raise AzureServiceError(f"Speech to Text HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        status = body.get("RecognitionStatus")
        best = (body.get("NBest") or [{}])[0]
        return {"provedor": self.provider, "status": status, "texto": body.get("DisplayText", ""),
                "confianca": best.get("Confidence"), "duracao_s": body.get("Duration", 0) / 1e7}


class OfflineTranscriber:
    """Fallback: usa a transcrição de referência `<audio>.txt` (gerada junto com o áudio
    sintético ou fornecida pelo usuário)."""

    provider = "offline (transcrição de referência .txt)"

    def transcribe(self, wav_path: str | Path) -> dict:
        txt = Path(wav_path).with_suffix(".txt")
        text = txt.read_text(encoding="utf-8") if txt.exists() else ""
        return {"provedor": self.provider, "status": "Success" if text else "NoTranscript",
                "texto": text, "confianca": None}


class AzureTextToSpeech:
    """Azure Text to Speech (REST). Usado só para **demonstração**: gera uma consulta
    falada real (WAV 16 kHz) a partir de um texto, para depois transcrevê-la com o
    Speech to Text e mostrar o fluxo completo com voz humana sintetizada."""

    def __init__(self, key: str, region: str, voice: str = "pt-BR-FranciscaNeural", timeout: int = 60):
        self.key, self.region, self.voice, self.timeout = key, region, voice, timeout

    def synthesize(self, text: str, out_path: str | Path, rate: str = "0%") -> Path:
        from xml.sax.saxutils import escape

        ssml = (f"<speak version='1.0' xml:lang='pt-BR'><voice name='{self.voice}'>"
                f"<prosody rate='{rate}'>{escape(text)}</prosody></voice></speak>")
        r = requests.post(f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1",
                          headers={"Ocp-Apim-Subscription-Key": self.key,
                                   "Content-Type": "application/ssml+xml",
                                   "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
                                   "User-Agent": "fase4-monitoramento"},
                          data=ssml.encode("utf-8"), timeout=self.timeout)
        if r.status_code != 200:
            raise AzureServiceError(f"Text to Speech HTTP {r.status_code}: {r.text[:300]}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        return out_path


# --------------------------------------------------------------------------------------
# Language / Text Analytics
# --------------------------------------------------------------------------------------

class AzureTextAnalytics:
    provider = "Azure AI Language (Text Analytics)"

    def __init__(self, key: str, endpoint: str, language: str = "pt-BR", timeout: int = 30,
                 healthcare_language: str | None = None):
        self.key, self.endpoint = key, endpoint.rstrip("/")
        self.language, self.timeout = language, timeout
        # Text Analytics for Health não aceita códigos regionais como "pt-BR" (só de, en, es,
        # fr, it, nl, pt-PT); "pt" é aceito e cobre o português do Brasil.
        self.hc_language = healthcare_language or language.split("-")[0]

    @property
    def _headers(self):
        return {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/json"}

    def _sync(self, kind: str, text: str, parameters: dict | None = None) -> dict:
        body = {"kind": kind, "parameters": parameters or {},
                "analysisInput": {"documents": [{"id": "1", "language": self.language, "text": text}]}}
        r = requests.post(f"{self.endpoint}/language/:analyze-text?api-version={API_VERSION}",
                          headers=self._headers, json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise AzureServiceError(f"{kind} HTTP {r.status_code}: {r.text[:300]}")
        res = r.json()["results"]
        if res.get("errors"):
            raise AzureServiceError(f"{kind}: {res['errors']}")
        return res["documents"][0]

    def sentiment(self, text: str) -> dict:
        doc = self._sync("SentimentAnalysis", text, {"opinionMining": True})
        label = {"positive": "positivo", "negative": "negativo", "neutral": "neutro", "mixed": "misto"}
        return {"sentimento": label.get(doc["sentiment"], doc["sentiment"]),
                "confianca": {"positivo": doc["confidenceScores"]["positive"],
                              "negativo": doc["confidenceScores"]["negative"]},
                "frases": [{"texto": s["text"], "sentimento": label.get(s["sentiment"], s["sentiment"])}
                           for s in doc.get("sentences", [])]}

    def key_phrases(self, text: str) -> list[str]:
        return self._sync("KeyPhraseExtraction", text)["keyPhrases"]

    def healthcare_entities(self, text: str, poll_s: float = 1.0, max_wait_s: float = 60) -> list[dict]:
        body = {"displayName": "fase4-consulta",
                "analysisInput": {"documents": [{"id": "1", "language": self.hc_language, "text": text}]},
                "tasks": [{"kind": "Healthcare", "taskName": "saude"}]}
        r = requests.post(f"{self.endpoint}/language/analyze-text/jobs?api-version={API_VERSION}",
                          headers=self._headers, json=body, timeout=self.timeout)
        if r.status_code != 202:
            raise AzureServiceError(f"Healthcare HTTP {r.status_code}: {r.text[:300]}")
        op = r.headers["operation-location"]
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            job = requests.get(op, headers=self._headers, timeout=self.timeout).json()
            if job["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(poll_s)
        if job.get("status") != "succeeded":
            raise AzureServiceError(f"Healthcare job: {job.get('status')} {job.get('errors')}")
        results = job["tasks"]["items"][0]["results"]
        if results.get("errors") or not results.get("documents"):
            raise AzureServiceError(f"Healthcare: {results.get('errors')}")
        doc = results["documents"][0]
        return [{"texto": e["text"], "categoria": e["category"], "confianca": e["confidenceScore"],
                 "negado": (e.get("assertion") or {}).get("certainty", "").startswith("negative")}
                for e in doc["entities"]]

    def analyze(self, text: str) -> dict:
        """Executa as três análises. Cada uma falha de forma independente (registrando o
        erro) e complementa o resultado offline, para nunca perder os termos críticos."""
        result = offline_text_analysis(text)
        result["provedor"] = self.provider
        result["erros_azure"] = []
        try:
            result.update(self.sentiment(text))
        except Exception as exc:
            result["erros_azure"].append(str(exc))
            logger.error(f"Azure SentimentAnalysis falhou: {exc}")
        try:
            result["frases_chave"] = self.key_phrases(text)
        except Exception as exc:
            result["erros_azure"].append(str(exc))
            logger.error(f"Azure KeyPhraseExtraction falhou: {exc}")
        try:
            result["entidades_saude"] = self.healthcare_entities(text)
        except Exception as exc:
            result["erros_azure"].append(str(exc))
            logger.error(f"Azure Text Analytics for Health falhou: {exc}")
        return result


class OfflineTextAnalytics:
    provider = "offline (léxico)"

    def analyze(self, text: str) -> dict:
        return offline_text_analysis(text)


# --------------------------------------------------------------------------------------
# Fábricas
# --------------------------------------------------------------------------------------

def get_speech_client(cfg: AzureConfig | None = None, force_offline: bool = False):
    cfg = cfg or AzureConfig()
    if cfg.speech_enabled and not force_offline:
        logger.info(f"Speech to Text: Azure (região {cfg.speech_region}, idioma {cfg.speech_language}).")
        return AzureSpeechToText(cfg.speech_key, cfg.speech_region, cfg.speech_language)
    logger.warning("Speech to Text em MODO OFFLINE (defina AZURE_SPEECH_KEY e AZURE_SPEECH_REGION "
                   "para usar o Azure).")
    return OfflineTranscriber()


def get_language_client(cfg: AzureConfig | None = None, force_offline: bool = False):
    cfg = cfg or AzureConfig()
    if cfg.language_enabled and not force_offline:
        logger.info(f"Text Analytics: Azure ({cfg.language_endpoint}).")
        return AzureTextAnalytics(cfg.language_key, cfg.language_endpoint, cfg.speech_language)
    logger.warning("Text Analytics em MODO OFFLINE (defina AZURE_LANGUAGE_KEY e AZURE_LANGUAGE_ENDPOINT "
                   "para usar o Azure).")
    return OfflineTextAnalytics()


def transcribe_with_fallback(client, wav_path: str | Path) -> dict:
    """Usa o Azure e, se a transcrição vier vazia (ex.: áudio sintético não-verbal) ou falhar,
    recorre à transcrição de referência — registrando o motivo."""
    try:
        res = client.transcribe(wav_path)
    except Exception as exc:
        logger.error(f"Falha no Speech to Text ({exc}); usando transcrição de referência.")
        res = {"provedor": "falha Azure", "status": "Error", "texto": ""}
    if not res.get("texto"):
        fb = OfflineTranscriber().transcribe(wav_path)
        if fb["texto"]:
            fb["observacao"] = f"Fallback: {res.get('provedor')} retornou status '{res.get('status')}'."
            return fb
    return res
