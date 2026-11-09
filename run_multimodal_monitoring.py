from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from monitoramento import config
from monitoramento import reporting as rep
from monitoramento.alerts import AlertManager
from monitoramento.anomaly_detection import MovementMonitor, check_prescriptions, run_vitals_monitoring
from monitoramento.audio_analysis import audio_to_alerts, extract_features, load_wav, vocal_indicators
from monitoramento.azure_services import get_language_client, get_speech_client, transcribe_with_fallback
from monitoramento.fusion import fuse_all, fusion_alerts
from monitoramento.synthetic_data import (
    AUDIO_FOR_PATIENT,
    PATIENTS,
    START,
    build_dataset,
    generate_movement,
    generate_physio_session,
    load_vitals_csv,
)
from monitoramento.video_analysis import (
    PROTOCOL,
    PoseAnomalyModel,
    YoloBackend,
    analyze_session,
    events_to_alerts,
    load_openpose_json,
)


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(name)-16s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger("fase4")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "monitoramento.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    for name in ("fase4.azure", "fase4.alertas", "fase4.video"):
        logging.getLogger(name).parent = root
    return logging.getLogger("fase4.pipeline")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="não usar Azure mesmo com credenciais")
    p.add_argument("--video", help="vídeo real (mp4/avi) analisado com YOLOv8-pose + YOLOv8")
    p.add_argument("--openpose", help="pasta com JSONs do OpenPose (BODY_25)")
    p.add_argument("--video-patient", default="P003")
    p.add_argument("--audio", help="WAV real de consulta (mono, 16 kHz recomendado, <= 60 s p/ Azure)")
    p.add_argument("--audio-patient", default="P002")
    p.add_argument("--vitals-csv", help="CSV de sinais vitais (formato do projeto ou PhysioNet BIDMC)")
    p.add_argument("--realtime", type=float, default=0.0,
                   help="segundos de espera por amostra no replay dos sinais vitais (0 = instantâneo)")
    p.add_argument("--no-gif", action="store_true", help="não gerar a animação do esqueleto")
    p.add_argument("--out", default=str(config.OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out)
    azure_cfg = config.AzureConfig()
    alert_cfg = config.AlertConfig()
    manager = AlertManager(out / "alerts.jsonl", alert_cfg.webhook_url, alert_cfg.cooldown_minutes)

    log.info("=" * 78)
    log.info("FASE 4 — Monitoramento multimodal (vídeo + áudio + texto + sinais vitais)")
    log.info("=" * 78)
    modo = {"speech": "Azure" if azure_cfg.speech_enabled and not args.offline else "offline",
            "language": "Azure" if azure_cfg.language_enabled and not args.offline else "offline",
            "webhook": bool(alert_cfg.webhook_url)}
    log.info(f"Modo dos serviços: {modo}")

    # 1. Dados ------------------------------------------------------------------------
    log.info("[1/7] Gerando dados sintéticos em %s", config.DATA_DIR)
    manifest = build_dataset(config.DATA_DIR)
    vitals = pd.read_csv(config.DATA_DIR / "sinais_vitais.csv", parse_dates=["timestamp"])
    if args.vitals_csv:
        real = load_vitals_csv(args.vitals_csv)
        log.info(f"Sinais vitais reais carregados de {args.vitals_csv}: {len(real)} amostras.")
        vitals = pd.concat([vitals, real], ignore_index=True)
    rx = pd.read_csv(config.DATA_DIR / "prescricoes.csv", parse_dates=["timestamp"])
    mv = pd.read_csv(config.DATA_DIR / "movimentacao.csv", parse_dates=["timestamp"])
    results: dict = {"modo_servicos": modo, "pacientes": PATIENTS}

    # 2. Vídeo ------------------------------------------------------------------------
    log.info("[2/7] Análise de vídeo (fisioterapia)")
    reference = generate_physio_session(anomalous=False, random_state=1)
    ref_res = analyze_session(reference)
    ref_model = PoseAnomalyModel().fit(ref_res["kinematics"])
    sessions = {"P001": generate_physio_session(anomalous=False, random_state=5),
                "P003": generate_physio_session(anomalous=True, random_state=3)}
    if args.video:
        log.info(f"Processando vídeo real com YOLOv8: {args.video}")
        sessions[args.video_patient] = YoloBackend().process(args.video, annotated_out=figs / "video_anotado.mp4")
    elif args.openpose:
        sessions[args.video_patient] = load_openpose_json(args.openpose)
    session_start = START + pd.Timedelta(hours=34)  # sessão de fisioterapia às 10:00 do 2º dia
    results["video"] = {}
    for pid, sess in sessions.items():
        res = analyze_session(sess, ref_model if sess.source.startswith("synthetic") else None)
        fig = rep.plot_video_session(res, f"Fisioterapia — {pid} ({sess.source})", figs / f"video_{pid}.png")
        rep.video_report(pid, res, PROTOCOL, f"figures/{fig.name}", out / f"relatorio_video_{pid}.md")
        if not args.no_gif and pid == "P003":
            rep.skeleton_gif(sess, res, figs / f"video_{pid}_esqueleto.gif")
        manager.emit_many(events_to_alerts(pid, res, session_start))
        results["video"][pid] = {"resumo": res["summary"], "eventos": res["events"], "repeticoes": res["reps"]}
        log.info(f"  {pid}: {res['summary']['repeticoes']} reps, {res['summary']['repeticoes_conformes']} conformes, "
                 f"eventos={res['summary']['eventos_por_tipo']}")

    # 3. Áudio ------------------------------------------------------------------------
    log.info("[3/7] Análise de áudio das consultas")
    stt = get_speech_client(azure_cfg, force_offline=args.offline)
    nlp = get_language_client(azure_cfg, force_offline=args.offline)
    audio_files = dict(manifest["audio"])
    if args.audio:
        audio_files[args.audio_patient] = args.audio
    results["audio"] = {}
    audio_plot = {}
    consult_ts = str(START + pd.Timedelta(hours=33))
    for pid, path in audio_files.items():
        x, sr = load_wav(path)
        feats = extract_features(x, sr)
        ind = vocal_indicators(feats)
        tr = transcribe_with_fallback(stt, path)
        txt = nlp.analyze(tr["texto"]) if tr["texto"] else {"provedor": "-", "termos_criticos": [],
                                                            "sentimento": "indisponível"}
        manager.emit_many(audio_to_alerts(pid, consult_ts, ind, txt))
        perfil = AUDIO_FOR_PATIENT.get(pid) if path in manifest["audio"].values() else "real"
        try:
            arquivo = Path(path).resolve().relative_to(config.ROOT_DIR).as_posix()
        except ValueError:
            arquivo = Path(path).name
        results["audio"][pid] = {"arquivo": arquivo, "perfil_simulado": perfil, "features": feats,
                                 "indicadores": ind, "transcricao": tr, "texto": txt}
        audio_plot[pid] = {"signal": x, "sr": sr, "indicadores": ind, "perfil": perfil}
        log.info(f"  {pid}: indicadores={ind} | sentimento={txt.get('sentimento')} | "
                 f"termos={[t['termo'] for t in txt.get('termos_criticos', []) if not t['negado']]}")
    rep.plot_audio(audio_plot, figs / "audio_indicadores.png")

    # 4. Anomalias --------------------------------------------------------------------
    log.info("[4/7] Detecção de anomalias — sinais vitais em streaming (%d amostras)", len(vitals))

    def on_alert(a):
        manager.emit(a)
        if args.realtime:
            time.sleep(args.realtime * 10)

    if args.realtime:
        log.info("Replay em tempo real habilitado (%.3fs por amostra).", args.realtime)
    t0 = time.time()
    vit_alerts, vit_records, artifacts = run_vitals_monitoring(vitals, on_alert=on_alert)
    if args.realtime:
        time.sleep(args.realtime * len(vitals) / 100)
    log.info(f"  {len(vit_alerts)} alertas de sinais vitais em {time.time() - t0:.1f}s; "
             f"{len(artifacts)} excursões transitórias descartadas como artefato.")
    vit_records.to_csv(out / "sinais_vitais_scores.csv", index=False)

    log.info("[4/7] Detecção de anomalias — prescrições")
    manager.emit_many(check_prescriptions(rx))

    log.info("[4/7] Detecção de anomalias — movimentação")
    reference_mv = pd.concat([generate_movement(random_state=s) for s in range(100, 106)])
    reference_mv = reference_mv[reference_mv["patient_id"].isin(["P001", "P002"])]
    mv_alerts, mv_scored = MovementMonitor().fit(reference_mv).detect(mv)
    manager.emit_many(mv_alerts)

    # 5/6. Fusão ----------------------------------------------------------------------
    log.info("[5/7] Fusão multimodal")
    fused = fuse_all(list(PATIENTS), manager.alerts)
    for f in fused:
        log.info(f"  {f['patient_id']}: risco={f['risco']:.2f} ({f['nivel']}) convergentes={f['dominios_convergentes']}")
    manager.emit_many(fusion_alerts(fused, manager.alerts))

    # 7. Relatórios -------------------------------------------------------------------
    log.info("[6/7] Gerando figuras e relatórios")
    adf = manager.to_dataframe()
    for pid in vitals["patient_id"].unique():
        rep.plot_vitals(vitals, adf, pid, figs / f"vitais_{pid}.png")
    rep.plot_movement(mv_scored, adf, figs / "movimentacao.png")
    rep.plot_fusion(fused, figs / "fusao_multimodal.png")
    rep.plot_alert_timeline(adf, figs / "linha_do_tempo_alertas.png")
    summary = manager.summary()
    rep.alerts_panel(adf, summary, out / "painel_alertas.md")
    adf.to_csv(out / "alertas.csv", index=False)

    evaluation = evaluate(manifest, adf, results)
    results.update({"fusao": fused, "alertas_resumo": summary, "avaliacao": evaluation,
                    "artefatos_descartados": artifacts})
    (out / "resultados.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    log.info("[7/7] Avaliação contra o ground truth:")
    for item in evaluation:
        log.info(f"  [{'OK' if item['ok'] else 'FALHOU'}] {item['verificacao']}: {item['resultado']}")
    log.info(f"Alertas: {summary}")
    log.info(f"Saídas em {out.resolve()}")
    return results


def evaluate(manifest: dict, adf: pd.DataFrame, results: dict) -> list[dict]:
    """Confere se as anomalias injetadas foram detectadas (e se o controle ficou quieto)."""
    checks = []
    a = adf.copy()
    a["ts"] = pd.to_datetime(a["timestamp"])
    relevant = a[a["severity"].isin(["ALTO", "CRITICO"])]
    for gt in manifest["ground_truth_vitais"]:
        ini, fim = pd.Timestamp(gt["inicio"]), pd.Timestamp(gt["fim"])
        va = a[(a["patient_id"] == gt["patient_id"]) & (a["source"] == "sinais_vitais")
               & (a["ts"] >= ini) & (a["ts"] <= fim + pd.Timedelta(minutes=30))]
        if gt["tipo"] == "artefato_sensor":
            ok = va[va["title"].str.contains("SpO", na=False)].empty
            checks.append({"verificacao": f"{gt['patient_id']} artefato de oxímetro ignorado", "ok": ok,
                           "resultado": "nenhum alerta" if ok else f"{len(va)} alerta(s)"})
            continue
        ok = not va.empty
        first = va.sort_values("ts").iloc[0] if ok else None
        first_high = va[va["severity"].isin(["ALTO", "CRITICO"])].sort_values("ts")
        res = (f"1º alerta {(first['ts'] - ini).total_seconds() / 60:.0f} min após o início "
               f"[{first['severity']}] '{first['title']}'" if ok else "não detectado")
        if ok and not first_high.empty:
            res += f"; 1º ALTO/CRÍTICO após {(first_high.iloc[0]['ts'] - ini).total_seconds() / 60:.0f} min"
        checks.append({"verificacao": f"{gt['patient_id']} {gt['tipo']} ({ini:%d/%m %H:%M}) detectado",
                       "ok": ok, "resultado": res})
    fp = relevant[(relevant["patient_id"] == "P001") & (relevant["source"] != "fusao")]
    checks.append({"verificacao": "P001 (controle) sem alertas ALTO/CRÍTICO", "ok": fp.empty,
                   "resultado": f"{len(fp)} alerta(s)" + (": " + "; ".join(fp["title"]) if len(fp) else "")})
    vid = results["video"]
    exp = {"queda", "assimetria", "tremor", "sem_supervisao"}
    got = set(vid.get("P003", {}).get("resumo", {}).get("eventos_por_tipo", {}))
    checks.append({"verificacao": "Vídeo P003: queda, assimetria, tremor e falta de supervisão", "ok": exp <= got,
                   "resultado": f"detectados {sorted(got)}"})
    p1 = {k for k in vid.get("P001", {}).get("resumo", {}).get("eventos_por_tipo", {}) if k != "padrao_anomalo"}
    checks.append({"verificacao": "Vídeo P001 (execução correta) sem desvios", "ok": not p1,
                   "resultado": f"eventos {sorted(p1) or 'nenhum'}"})
    expected_audio = {"normal": None, "dispneia": "dificuldade_respiratoria", "disartria_fadiga": "disartria"}
    for pid, r in results["audio"].items():
        prof = r["perfil_simulado"]
        if prof not in expected_audio:
            continue
        ind = r["indicadores"]
        if expected_audio[prof] is None:
            ok = max(ind.values()) < 0.5
            res = f"indicadores {ind}"
        else:
            ok = ind[expected_audio[prof]] >= 0.5
            if prof == "disartria_fadiga":
                ok = ok and ind["fadiga"] >= 0.5
            res = f"indicadores {ind}"
        checks.append({"verificacao": f"Áudio {pid} (perfil {prof})", "ok": ok, "resultado": res})
    return checks


if __name__ == "__main__":
    main()
