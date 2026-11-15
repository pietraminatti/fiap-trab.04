import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoramento import reporting as rep  # noqa: E402
from monitoramento.config import OUTPUT_DIR  # noqa: E402
from monitoramento.video_analysis import PROTOCOL, YoloBackend, analyze_session  # noqa: E402


def _to_h264(path: Path, log: logging.Logger) -> None:
    """O OpenCV grava em MPEG-4 Part 2 (arquivo grande, não toca no navegador). Se o
    imageio-ffmpeg estiver instalado, converte para H.264 com 540 px de largura."""
    try:
        import imageio_ffmpeg
    except ImportError:
        log.info("  (instale imageio-ffmpeg para converter o vídeo anotado para H.264)")
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(path), "-vf", "scale=540:-2",
           "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Analisa um vídeo real de fisioterapia com YOLOv8-pose + YOLOv8.")
    p.add_argument("video", help="arquivo de vídeo (mp4/avi/mov)")
    p.add_argument("--paciente", default="VIDEO-REAL", help="identificador usado no relatório")
    p.add_argument("--stride", type=int, default=1, help="processa 1 a cada N quadros")
    p.add_argument("--out", default=str(OUTPUT_DIR / "video_real"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("fase4.video_real")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Processando {args.video} com YOLOv8-pose (keypoints) + YOLOv8 (detecção de pessoas/objetos)")
    t0 = time.time()
    session = YoloBackend().process(args.video, stride=args.stride, annotated_out=out / "video_anotado.mp4")
    log.info(f"  {len(session.keypoints)} quadros em {time.time() - t0:.0f}s · {session.fps:.1f} fps · "
             f"{session.frame_size[0]}x{session.frame_size[1]}")

    _to_h264(out / "video_anotado.mp4", log)

    res = analyze_session(session)
    fig = rep.plot_video_session(res, f"Fisioterapia — {args.paciente} ({session.source})", out / "cinematica.png")
    rep.video_report(args.paciente, res, PROTOCOL, fig.name, out / "relatorio_video.md")

    s = res["summary"]
    log.info(f"  repetições: {s['repeticoes']} ({s['repeticoes_conformes']} conformes) · eventos: {s['eventos_por_tipo']}")
    for e in res["events"]:
        log.info(f"  [{e['severidade']}] {e['tipo']} @ {e['t']}s — {e['descricao']}")
    resumo = {"resumo": s, "repeticoes": res["reps"], "eventos": res["events"],
              "objetos_detectados": sorted({d["label"] for f in session.detections for d in f})}
    (out / "resultado.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info(f"Saídas em {out}: video_anotado.mp4, cinematica.png, relatorio_video.md, resultado.json")
    return resumo


if __name__ == "__main__":
    main()
