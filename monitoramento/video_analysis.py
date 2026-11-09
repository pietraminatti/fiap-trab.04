from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .alerts import Alert
from .synthetic_data import KP, PoseSession

logger = logging.getLogger("fase4.video")

# Protocolo esperado do exercício (agachamento assistido)
PROTOCOL = {
    "nome": "Agachamento assistido (reabilitação de membros inferiores)",
    "flexao_joelho_min_graus": 110,  # o ângulo do joelho deve chegar a <= 110° em cada repetição
    "assimetria_max_graus": 12,
    "inclinacao_tronco_max_graus": 40,
    "tremor_rms_max_graus": 2.0,  # potência na banda 4–12 Hz
    "pessoas_minimas": 2,  # paciente + fisioterapeuta (supervisão)
    "sem_supervisao_max_s": 3.0,
}

# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


class YoloBackend:
    """YOLOv8-pose + YOLOv8 (ultralytics). Instale com `pip install -r requirements-video.txt`.

    Os pesos (`yolov8n-pose.pt`, `yolov8n.pt`) são baixados automaticamente pelo
    ultralytics na primeira execução.
    """

    def __init__(self, pose_weights: str = "yolov8n-pose.pt", det_weights: str = "yolov8n.pt",
                 conf: float = 0.4):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Backend YOLO requer 'ultralytics' (pip install -r requirements-video.txt).") from exc
        self.pose = YOLO(pose_weights)
        self.det = YOLO(det_weights)
        self.conf = conf

    def process(self, video_path: str | Path, stride: int = 1, annotated_out: str | Path | None = None,
                max_frames: int | None = None) -> PoseSession:  # pragma: no cover - requer modelo
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
        fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / stride
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if annotated_out:
            writer = cv2.VideoWriter(str(annotated_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        kps, dets, idx = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and len(kps) >= max_frames):
                break
            idx += 1
            if (idx - 1) % stride:
                continue
            pr = self.pose(frame, conf=self.conf, verbose=False)[0]
            if pr.keypoints is not None and len(pr.keypoints.data):
                boxes = pr.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                kps.append(pr.keypoints.data.cpu().numpy()[int(np.argmax(areas))])  # maior pessoa = paciente
            else:
                kps.append(np.full((17, 3), np.nan))
            dr = self.det(frame, conf=self.conf, verbose=False)[0]
            dets.append([{"label": dr.names[int(c)], "conf": float(s), "box": b.tolist()}
                         for c, s, b in zip(dr.boxes.cls.cpu().numpy(), dr.boxes.conf.cpu().numpy(),
                                            dr.boxes.xyxy.cpu().numpy())])
            if writer is not None:
                writer.write(pr.plot())
        cap.release()
        if writer is not None:
            writer.release()
        return PoseSession(np.asarray(kps), fps, dets, (W, H), source=f"yolov8:{Path(video_path).name}")


_BODY25_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def load_openpose_json(folder: str | Path, fps: float = 30.0) -> PoseSession:
    """Converte a saída do OpenPose (`--write_json`, modelo BODY_25) em `PoseSession`."""
    files = sorted(Path(folder).glob("*_keypoints.json"))
    kps = []
    for f in files:
        people = json.loads(f.read_text())["people"]
        if not people:
            kps.append(np.full((17, 3), np.nan))
            continue
        arr = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
        kp = arr[_BODY25_TO_COCO].astype(float)
        kp[kp[:, 2] == 0, :2] = np.nan
        kps.append(kp)
    return PoseSession(np.asarray(kps), fps, [[] for _ in kps], source=f"openpose:{Path(folder).name}")


# --------------------------------------------------------------------------------------
# Cinemática
# --------------------------------------------------------------------------------------

def _angle(a, b, c):
    """Ângulo em b (graus) formado pelos segmentos b->a e b->c; arrays (T,2)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compute_kinematics(session: PoseSession) -> pd.DataFrame:
    k = session.keypoints[:, :, :2].astype(float)
    k[session.keypoints[:, :, 2] < 0.3] = np.nan
    # interpolação de keypoints perdidos (oclusões curtas)
    T = len(k)
    flat = pd.DataFrame(k.reshape(T, -1)).interpolate(limit=10, limit_direction="both").to_numpy()
    k = flat.reshape(T, 17, 2)

    g = lambda n: k[:, KP[n]]  # noqa: E731
    knee_l = _angle(g("left_hip"), g("left_knee"), g("left_ankle"))
    knee_r = _angle(g("right_hip"), g("right_knee"), g("right_ankle"))
    hip_mid = (g("left_hip") + g("right_hip")) / 2
    sh_mid = (g("left_shoulder") + g("right_shoulder")) / 2
    ank_mid = (g("left_ankle") + g("right_ankle")) / 2
    trunk = sh_mid - hip_mid
    lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))  # 0 = ereto (y da imagem para baixo)
    stand_h = np.nanmedian((ank_mid[:, 1] - sh_mid[:, 1])[: max(5, int(session.fps))])
    hip_h = (ank_mid[:, 1] - hip_mid[:, 1]) / (stand_h + 1e-9)
    df = pd.DataFrame({
        "t": np.arange(T) / session.fps,
        "joelho_esq": knee_l, "joelho_dir": knee_r,
        "assimetria": np.abs(knee_l - knee_r),
        "inclinacao_tronco": lean, "altura_quadril": hip_h,
    })
    df["vel_quadril"] = np.gradient(df["altura_quadril"].to_numpy()) * session.fps
    df["pessoas"] = [sum(d["label"] == "person" for d in f) for f in session.detections] \
        if session.detections else np.nan
    return df


# --------------------------------------------------------------------------------------
# Detecção de desvios
# --------------------------------------------------------------------------------------

_FEATS = ["joelho_esq", "joelho_dir", "assimetria", "inclinacao_tronco", "altura_quadril", "vel_quadril"]


class PoseAnomalyModel:
    """Isolation Forest treinado com uma sessão de referência (execução correta)."""

    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=0.01, random_state=random_state)

    def fit(self, kin: pd.DataFrame) -> PoseAnomalyModel:
        self.model.fit(self.scaler.fit_transform(kin[_FEATS].fillna(0)))
        return self

    def predict(self, kin: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(kin[_FEATS].fillna(0))) == -1


def _segments(mask: np.ndarray, t: np.ndarray, min_len_s: float) -> list[tuple[float, float]]:
    segs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if t[i - 1] - t[start] >= min_len_s:
                segs.append((float(t[start]), float(t[i - 1])))
            start = None
    return segs


def analyze_session(session: PoseSession, reference_model: PoseAnomalyModel | None = None,
                    protocol: dict = PROTOCOL) -> dict:
    """Retorna cinemática, repetições, eventos detectados e resumo da sessão."""
    kin = compute_kinematics(session)
    fps, t = session.fps, kin["t"].to_numpy()
    events: list[dict] = []
    mean_knee = kin[["joelho_esq", "joelho_dir"]].mean(axis=1).to_numpy()

    # Queda / perda de equilíbrio: quadril desce perto do chão e tronco tomba
    fall_mask = (kin["altura_quadril"] < 0.3) & (kin["inclinacao_tronco"] > 55)
    fall_segs = _segments(fall_mask.to_numpy(), t, 0.3)
    fall_t = fall_segs[0][0] - 1.5 if fall_segs else np.inf  # inclui a fase de desequilíbrio

    # Repetições = vales do ângulo médio do joelho (as posteriores à queda não são avaliadas)
    valleys, _ = signal.find_peaks(-signal.medfilt(mean_knee, 5), distance=int(2 * fps), prominence=20)
    valleys = [v for v in valleys if t[v] < fall_t]
    reps = []
    for n, v in enumerate(valleys, 1):
        w = slice(max(0, v - int(0.3 * fps)), v + int(0.3 * fps))
        rep = {"repeticao": n, "t": round(float(t[v]), 2),
               "joelho_esq_min": round(float(np.nanmin(kin["joelho_esq"].iloc[w])), 1),
               "joelho_dir_min": round(float(np.nanmin(kin["joelho_dir"].iloc[w])), 1),
               "assimetria": round(float(kin["assimetria"].iloc[w].median()), 1),
               "tronco_max": round(float(kin["inclinacao_tronco"].iloc[w].max()), 1)}
        rep["conforme"] = True
        if max(rep["joelho_esq_min"], rep["joelho_dir_min"]) > protocol["flexao_joelho_min_graus"]:
            rep["conforme"] = False
            lado = "direito" if rep["joelho_dir_min"] > rep["joelho_esq_min"] else "esquerdo"
            events.append({"t": rep["t"], "tipo": "deficit_amplitude", "severidade": "MODERADO",
                           "descricao": f"Rep. {n}: joelho {lado} não atingiu {protocol['flexao_joelho_min_graus']}° "
                                        f"(mín. {max(rep['joelho_esq_min'], rep['joelho_dir_min'])}°)."})
        if rep["assimetria"] > protocol["assimetria_max_graus"]:
            rep["conforme"] = False
            events.append({"t": rep["t"], "tipo": "assimetria", "severidade": "MODERADO",
                           "descricao": f"Rep. {n}: assimetria de {rep['assimetria']}° entre joelhos "
                                        f"(limite {protocol['assimetria_max_graus']}°) — possível hemiparesia/compensação."})
        if rep["tronco_max"] > protocol["inclinacao_tronco_max_graus"]:
            rep["conforme"] = False
            events.append({"t": rep["t"], "tipo": "compensacao_tronco", "severidade": "MODERADO",
                           "descricao": f"Rep. {n}: inclinação do tronco {rep['tronco_max']}°."})
        reps.append(rep)

    # Tremor: RMS na banda 4–12 Hz em janelas de 2 s
    tremor_rms = np.zeros(len(t))
    if fps >= 15:  # Nyquist: precisa de fps > 8 Hz; exigimos 15 fps para margem
        hi = min(12.0, fps / 2 - 0.5)
        b, a = signal.butter(3, [4.0 / (fps / 2), hi / (fps / 2)], "band")
        band = signal.filtfilt(b, a, np.nan_to_num(mean_knee, nan=np.nanmean(mean_knee)))
        win = int(2 * fps)
        tremor_rms = np.sqrt(pd.Series(band ** 2).rolling(win, center=True, min_periods=1).mean()).to_numpy()
        kin["tremor_rms"] = tremor_rms
        for s, e in _segments(tremor_rms > protocol["tremor_rms_max_graus"], t, 1.0):
            f, pxx = signal.welch(band[int(s * fps):int(e * fps) + 1], fs=fps, nperseg=min(64, int((e - s) * fps)))
            events.append({"t": round(s, 2), "fim": round(e, 2), "tipo": "tremor", "severidade": "MODERADO",
                           "descricao": f"Tremor de ~{f[np.argmax(pxx)]:.1f} Hz entre {s:.1f}s e {e:.1f}s "
                                        f"(RMS máx. {tremor_rms[int(s * fps):int(e * fps) + 1].max():.1f}°)."})

    for s, e in fall_segs:
        vmin = kin["vel_quadril"][(t >= s - 1.5) & (t <= e)].min()
        events.append({"t": round(s, 2), "tipo": "queda", "severidade": "CRITICO",
                       "descricao": f"Perda de equilíbrio/queda aos {s:.1f}s (quadril a "
                                    f"{kin['altura_quadril'][t >= s].iloc[0]:.2f} da altura, velocidade {vmin:.2f}/s)."})

    # Supervisão (YOLO: contagem de pessoas)
    if kin["pessoas"].notna().all():
        alone = (kin["pessoas"] < protocol["pessoas_minimas"]).to_numpy()
        for s, e in _segments(alone, t, protocol["sem_supervisao_max_s"]):
            events.append({"t": round(s, 2), "fim": round(e, 2), "tipo": "sem_supervisao", "severidade": "ALTO",
                           "descricao": f"Paciente sem fisioterapeuta no campo de visão de {s:.1f}s a {e:.1f}s."})

    # Isolation Forest contra a sessão de referência
    if reference_model is not None:
        kin["iforest_anomalia"] = reference_model.predict(kin)
        smooth = pd.Series(kin["iforest_anomalia"].astype(float)).rolling(int(fps / 2), center=True,
                                                                        min_periods=1).mean() > 0.5
        for s, e in _segments(smooth.to_numpy(), t, 0.5):
            events.append({"t": round(s, 2), "fim": round(e, 2), "tipo": "padrao_anomalo", "severidade": "INFO",
                           "descricao": f"Movimento fora do padrão da sessão de referência ({s:.1f}s–{e:.1f}s)."})

    events.sort(key=lambda ev: ev["t"])
    n_ok = sum(r["conforme"] for r in reps)
    summary = {
        "fonte": session.source, "duracao_s": round(float(t[-1]), 1), "fps": fps,
        "repeticoes": len(reps), "repeticoes_conformes": n_ok,
        "interrompida_por_queda": bool(fall_segs),
        "eventos": len(events),
        "eventos_por_tipo": pd.Series([e["tipo"] for e in events]).value_counts().to_dict() if events else {},
        "amplitude_media_esq": round(float(np.mean([r["joelho_esq_min"] for r in reps])), 1) if reps else None,
        "amplitude_media_dir": round(float(np.mean([r["joelho_dir_min"] for r in reps])), 1) if reps else None,
    }
    return {"kinematics": kin, "reps": reps, "events": events, "summary": summary}


_EVENT_DOMAINS = {"queda": ["queda", "neurologico"], "tremor": ["neurologico"],
                  "assimetria": ["neurologico", "motor"], "deficit_amplitude": ["motor"],
                  "compensacao_tronco": ["motor"], "sem_supervisao": ["queda", "seguranca"],
                  "padrao_anomalo": ["motor"]}
_ACTIONS = {"queda": "Interromper a sessão, avaliar o paciente e registrar evento adverso.",
            "sem_supervisao": "Garantir supervisão contínua durante exercícios com risco de queda.",
            "tremor": "Comunicar neurologia; avaliar fadiga muscular e medicações.",
            "assimetria": "Ajustar plano de reabilitação (fortalecimento do lado deficitário).",
            "deficit_amplitude": "Reavaliar amplitude de movimento e dor.",
            "compensacao_tronco": "Corrigir postura; reduzir carga/profundidade."}


def events_to_alerts(patient_id: str, result: dict, session_start: pd.Timestamp) -> list[Alert]:
    """Converte eventos da sessão em alertas (eventos repetidos do mesmo tipo viram um só)."""
    alerts = []
    by_type: dict[str, list[dict]] = {}
    for e in result["events"]:
        by_type.setdefault(e["tipo"], []).append(e)
    for tipo, evs in by_type.items():
        sev = max((e["severidade"] for e in evs), key=lambda s: ["INFO", "MODERADO", "ALTO", "CRITICO"].index(s))
        desc = " ".join(e["descricao"] for e in evs[:3]) + (f" (+{len(evs) - 3} ocorrências)" if len(evs) > 3 else "")
        alerts.append(Alert(patient_id, str(session_start + pd.Timedelta(seconds=evs[0]["t"])), "video", sev,
                            f"Fisioterapia: {tipo.replace('_', ' ')}", desc, _EVENT_DOMAINS.get(tipo, []),
                            {"ocorrencias": len(evs), "tempos_s": [e["t"] for e in evs]},
                            _ACTIONS.get(tipo, "Revisar a gravação da sessão.")))
    return alerts
