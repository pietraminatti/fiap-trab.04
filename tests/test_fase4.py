import json

import numpy as np
import pandas as pd
import pytest

from monitoramento import azure_services
from monitoramento.alerts import Alert, AlertManager
from monitoramento.anomaly_detection import (
    MovementMonitor,
    StreamingVitalsMonitor,
    check_prescriptions,
    news2_components,
    run_vitals_monitoring,
)
from monitoramento.audio_analysis import audio_to_alerts, extract_features, offline_text_analysis, vocal_indicators
from monitoramento.config import AzureConfig
from monitoramento.fusion import fuse_patient, fusion_alerts
from monitoramento.synthetic_data import (
    START,
    generate_movement,
    generate_physio_session,
    generate_prescriptions,
    generate_vitals,
    generate_voice,
)
from monitoramento.video_analysis import PoseAnomalyModel, analyze_session


# ----------------------------------------------------------------------------- sinais vitais
def test_news2_normal_e_grave():
    normal = dict(resp_rate=16, spo2=98, systolic_bp=120, heart_rate=75, temperature=36.8)
    grave = dict(resp_rate=28, spo2=89, systolic_bp=88, heart_rate=135, temperature=39.5)
    assert sum(news2_components(normal).values()) == 0
    assert sum(news2_components(grave).values()) == 3 + 3 + 3 + 3 + 2


def _sample(ts, **kw):
    base = dict(timestamp=ts, heart_rate=75, systolic_bp=120, diastolic_bp=78, spo2=97,
                resp_rate=15, temperature=36.7)
    base.update(kw)
    return base


def test_artefato_isolado_nao_gera_alerta():
    mon = StreamingVitalsMonitor("X")
    alerts = []
    for i in range(60):
        kw = {"spo2": 70} if i == 40 else {}
        alerts += mon.update(_sample(START + pd.Timedelta(minutes=5 * i), **kw))
    assert not alerts
    assert mon.artifacts and mon.artifacts[0]["sinal"] == "spo2"


def test_hipoxemia_persistente_gera_alerta():
    mon = StreamingVitalsMonitor("X")
    alerts = []
    for i in range(60):
        kw = {"spo2": 86} if i >= 40 else {}
        alerts += mon.update(_sample(START + pd.Timedelta(minutes=5 * i), **kw))
    titles = [a.title for a in alerts]
    assert any("SpO" in t for t in titles)
    assert any(a.severity == "CRITICO" for a in alerts)


def test_sepse_detectada_e_controle_quieto():
    vitals, _ = generate_vitals()
    alerts, records, _ = run_vitals_monitoring(vitals)
    p2 = [a for a in alerts if a.patient_id == "P002" and a.level >= 2]
    assert p2, "deterioração do P002 deveria gerar alerta ALTO/CRÍTICO"
    assert min(pd.Timestamp(a.timestamp) for a in p2) >= START + pd.Timedelta(hours=30)
    assert not [a for a in alerts if a.patient_id == "P001" and a.level >= 2]
    assert "news2" in records.columns


# ----------------------------------------------------------------------------- prescrições
def test_regras_de_prescricao():
    titles = {(a.patient_id, a.title) for a in check_prescriptions(generate_prescriptions())}
    assert ("P002", "Dose acima do máximo: dipirona") in titles
    assert ("P002", "Suspensão abrupta de antibiotico: ceftriaxona") in titles
    assert ("P003", "Aumento abrupto de dose: enoxaparina") in titles
    assert ("P003", "Duplicidade terapêutica (BRA)") in titles
    assert ("P003", "Interação medicamentosa relevante") in titles
    assert not [t for t in titles if t[0] == "P001"]


# ----------------------------------------------------------------------------- movimentação
def test_movimentacao_queda_e_imobilidade():
    ref = pd.concat([generate_movement(random_state=s) for s in range(100, 103)])
    alerts, scored = MovementMonitor().fit(ref[ref.patient_id != "P003"]).detect(generate_movement())
    p3 = {a.title for a in alerts if a.patient_id == "P003"}
    assert {"Possível queda detectada", "Imobilidade prolongada", "Agitação noturna"} <= p3
    assert not [a for a in alerts if a.patient_id == "P001" and a.level >= 2]
    assert "iforest_anomalia" in scored


# ----------------------------------------------------------------------------- vídeo
@pytest.fixture(scope="module")
def ref_model():
    ref = analyze_session(generate_physio_session(anomalous=False, random_state=1))
    return PoseAnomalyModel().fit(ref["kinematics"])


def test_video_referencia_conforme(ref_model):
    res = analyze_session(generate_physio_session(anomalous=False, random_state=5), ref_model)
    assert res["summary"]["repeticoes"] == 6
    assert res["summary"]["repeticoes_conformes"] == 6
    assert not [e for e in res["events"] if e["severidade"] != "INFO"]


def test_video_anomalo_detecta_eventos(ref_model):
    res = analyze_session(generate_physio_session(anomalous=True), ref_model)
    tipos = {e["tipo"] for e in res["events"]}
    assert {"queda", "assimetria", "tremor", "sem_supervisao"} <= tipos
    queda = next(e for e in res["events"] if e["tipo"] == "queda")
    assert 25 <= queda["t"] <= 27  # queda injetada aos 25 s
    tremor = next(e for e in res["events"] if e["tipo"] == "tremor")
    assert "6." in tremor["descricao"] or "5." in tremor["descricao"]  # ~6 Hz


# ----------------------------------------------------------------------------- áudio
@pytest.mark.parametrize("profile,expected", [("normal", None), ("dispneia", "dificuldade_respiratoria"),
                                              ("disartria_fadiga", "disartria")])
def test_indicadores_vocais(profile, expected):
    for seed in (21, 22):
        ind = vocal_indicators(extract_features(generate_voice(profile, random_state=seed), 16000))
        if expected is None:
            assert max(ind.values()) < 0.5, ind
        else:
            assert ind[expected] >= 0.5, ind


def test_texto_offline_negacao_e_sentimento():
    res = offline_text_analysis("Estou com falta de ar e febre. Não sinto dor no peito.")
    terms = {(t["termo"], t["negado"]) for t in res["termos_criticos"]}
    assert ("falta de ar", False) in terms and ("dor no peito", True) in terms
    assert offline_text_analysis("Estou bem melhor, consegui dormir")["sentimento"] == "positivo"


# ----------------------------------------------------------------------------- Azure (mock)
class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, json.dumps(body)

    def json(self):
        return self._body


def test_entidades_azure_so_alertam_termos_clinicos_criticos():
    """Entidades do Azure como "sentindo bem melhor" não geram alerta; negação do Azure prevalece."""
    calmo = offline_text_analysis("Estou me sentindo bem melhor. A dor da cirurgia diminuiu.")
    calmo["entidades_saude"] = [
        {"texto": "sentindo bem melhor", "categoria": "SymptomOrSign", "negado": False},
        {"texto": "dor da", "categoria": "SymptomOrSign", "negado": False},
    ]
    assert audio_to_alerts("P001", "2026-11-02 09:00", {}, calmo) == []

    grave = offline_text_analysis("Tenho febre. Não sinto dor no peito.")
    grave["entidades_saude"] = [
        {"texto": "febre", "categoria": "SymptomOrSign", "negado": False},
        {"texto": "dor no peito", "categoria": "SymptomOrSign", "negado": True},
    ]
    (alerta,) = audio_to_alerts("P002", "2026-11-02 09:00", {}, grave)
    assert alerta.evidence["termos"] == ["febre"]
    assert alerta.domains == ["infeccioso"]


def test_azure_speech_to_text_mock(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    calls = {}

    def fake_post(url, headers, data, timeout):
        calls["url"], calls["headers"] = url, headers
        return _Resp(200, {"RecognitionStatus": "Success", "DisplayText": "Estou com falta de ar.",
                           "Duration": 25_000_000, "NBest": [{"Confidence": 0.93}]})

    monkeypatch.setattr(azure_services.requests, "post", fake_post)
    res = azure_services.AzureSpeechToText("k", "brazilsouth").transcribe(wav)
    assert res["texto"] == "Estou com falta de ar." and res["confianca"] == 0.93
    assert "brazilsouth.stt.speech.microsoft.com" in calls["url"] and "language=pt-BR" in calls["url"]
    assert calls["headers"]["Ocp-Apim-Subscription-Key"] == "k"


def test_azure_language_mock(monkeypatch):
    def fake_post(url, headers, json, timeout):
        if json.get("kind") == "SentimentAnalysis":
            return _Resp(200, {"results": {"errors": [], "documents": [{
                "sentiment": "negative", "confidenceScores": {"positive": 0.02, "negative": 0.95},
                "sentences": [{"text": "Falta de ar.", "sentiment": "negative"}]}]}})
        if json.get("kind") == "KeyPhraseExtraction":
            return _Resp(200, {"results": {"errors": [], "documents": [{"keyPhrases": ["falta de ar"]}]}})
        return _Resp(202, {}, {"operation-location": "https://x/jobs/1"})

    def fake_get(url, headers, timeout):
        return _Resp(200, {"status": "succeeded", "tasks": {"items": [{"results": {"documents": [{"entities": [
            {"text": "falta de ar", "category": "SymptomOrSign", "confidenceScore": 0.98},
            {"text": "dor no peito", "category": "SymptomOrSign", "confidenceScore": 0.9,
             "assertion": {"certainty": "negative"}}]}]}}]}})

    monkeypatch.setattr(azure_services.requests, "post", fake_post)
    monkeypatch.setattr(azure_services.requests, "get", fake_get)
    res = azure_services.AzureTextAnalytics("k", "https://x.cognitiveservices.azure.com/").analyze(
        "Tenho falta de ar. Não sinto dor no peito.")
    assert res["sentimento"] == "negativo" and res["frases_chave"] == ["falta de ar"]
    assert {e["texto"]: e["negado"] for e in res["entidades_saude"]} == {"falta de ar": False, "dor no peito": True}
    assert res["erros_azure"] == []


def test_fabricas_sem_credenciais_usam_offline(monkeypatch):
    cfg = AzureConfig(speech_key=None, speech_region=None, language_key=None, language_endpoint=None)
    assert isinstance(azure_services.get_speech_client(cfg), azure_services.OfflineTranscriber)
    assert isinstance(azure_services.get_language_client(cfg), azure_services.OfflineTextAnalytics)
    cfg2 = AzureConfig(speech_key="k", speech_region="eastus", language_key="k", language_endpoint="https://x")
    assert isinstance(azure_services.get_speech_client(cfg2, force_offline=True), azure_services.OfflineTranscriber)


# ----------------------------------------------------------------------------- alertas e fusão
def test_alert_manager_dedup_escalada_e_jsonl(tmp_path):
    m = AlertManager(tmp_path / "a.jsonl", cooldown_minutes=30)
    mk = lambda minute, sev="ALTO": Alert("P1", START + pd.Timedelta(minutes=minute), "sinais_vitais",  # noqa: E731
                                          sev, "SpO2 baixa", "d")
    assert m.emit(mk(0)) and not m.emit(mk(10)) and m.emit(mk(15, "CRITICO")) and m.emit(mk(120))
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[1])["severity"] == "CRITICO"
    assert m.summary()["suprimidos_por_duplicidade"] == 1
    with pytest.raises(ValueError):
        Alert("P1", START, "x", "URGENTE", "t", "d")


def test_fusao_corroboracao():
    ts = str(START)
    alerts = [Alert("P", ts, "sinais_vitais", "MODERADO", "SpO2", "", ["respiratorio"]),
              Alert("P", ts, "audio", "MODERADO", "dispneia", "", ["respiratorio"])]
    only_one = fuse_patient("P", alerts[:1])
    both = fuse_patient("P", alerts)
    assert both["dominios_convergentes"] == {"respiratorio": ["audio", "sinais_vitais"]}
    assert both["risco"] > only_one["risco"] and both["bonus_corroboracao"] == 0.1
    crit = alerts + [Alert("P", ts, "video", "CRITICO", "queda", "", ["queda"])]
    out = fusion_alerts([fuse_patient("P", crit)], crit)
    assert out and out[-1].severity == "CRITICO"


# ----------------------------------------------------------------------------- ponta a ponta
def test_pipeline_completo_offline(tmp_path):
    import run_multimodal_monitoring as pipeline

    res = pipeline.main(["--offline", "--no-gif", "--out", str(tmp_path)])
    failed = [c for c in res["avaliacao"] if not c["ok"]]
    assert not failed, failed
    assert (tmp_path / "alerts.jsonl").exists() and (tmp_path / "painel_alertas.md").exists()
    assert (tmp_path / "relatorio_video_P003.md").exists()
    niveis = {f["patient_id"]: f["nivel"] for f in res["fusao"]}
    assert niveis["P002"] == "CRITICO" and niveis["P003"] == "CRITICO" and niveis["P001"] in ("BAIXO", "MODERADO")
    assert np.isfinite([f["risco"] for f in res["fusao"]]).all()
