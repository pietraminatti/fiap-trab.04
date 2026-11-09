from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

PATIENTS = {
    "P001": {"idade": 58, "leito": "UTI-01", "diagnostico": "Pós-operatório de colecistectomia"},
    "P002": {"idade": 71, "leito": "UTI-02", "diagnostico": "Pneumonia comunitária"},
    "P003": {"idade": 66, "leito": "UTI-03", "diagnostico": "AVC isquêmico em reabilitação"},
}

START = pd.Timestamp("2026-11-01 00:00:00")
SAMPLE_MINUTES = 5
HOURS = 48

# --------------------------------------------------------------------------------------
# 1. Sinais vitais (séries temporais)
# --------------------------------------------------------------------------------------

_BASELINE = {
    "P001": dict(heart_rate=76, systolic_bp=122, diastolic_bp=78, spo2=97.5, resp_rate=15, temperature=36.7),
    "P002": dict(heart_rate=84, systolic_bp=128, diastolic_bp=80, spo2=95.8, resp_rate=17, temperature=37.2),
    "P003": dict(heart_rate=72, systolic_bp=142, diastolic_bp=86, spo2=96.8, resp_rate=15, temperature=36.6),
}
_NOISE = dict(heart_rate=2.5, systolic_bp=4.0, diastolic_bp=3.0, spo2=0.5, resp_rate=0.8, temperature=0.08)


def generate_vitals(random_state: int = 42) -> tuple[pd.DataFrame, list[dict]]:
    """Gera 48 h de sinais vitais (1 amostra / 5 min) para os 3 pacientes.

    Returns:
        (df, ground_truth) — df com colunas timestamp, patient_id e os 6 sinais vitais;
        ground_truth lista os eventos injetados (para avaliação).
    """
    rng = np.random.default_rng(random_state)
    n = HOURS * 60 // SAMPLE_MINUTES
    t_h = np.arange(n) * SAMPLE_MINUTES / 60.0
    circadian = np.sin(2 * np.pi * (t_h - 9) / 24)  # pico ~15h, vale ~3h
    frames, truth = [], []

    for pid, base in _BASELINE.items():
        data = {}
        for vital, mu in base.items():
            amp = {"heart_rate": 4, "systolic_bp": 5, "diastolic_bp": 3, "spo2": 0.3,
                   "resp_rate": 0.8, "temperature": 0.25}[vital]
            # ruído AR(1) para manter correlação temporal realista
            e = rng.normal(0, _NOISE[vital], n)
            ar = signal.lfilter([1.0], [1.0, -0.6], e) * np.sqrt(1 - 0.36)
            data[vital] = mu + amp * circadian + ar

        if pid == "P002":  # deterioração séptica progressiva a partir de 30 h
            ramp = np.clip((t_h - 30) / 10, 0, 1)
            data["heart_rate"] += 38 * ramp
            data["resp_rate"] += 10 * ramp
            data["temperature"] += 1.9 * ramp
            data["spo2"] -= 6.5 * ramp
            data["systolic_bp"] -= 30 * ramp
            data["diastolic_bp"] -= 16 * ramp
            truth.append({"patient_id": "P002", "tipo": "deterioracao_sepse",
                          "inicio": str(START + pd.Timedelta(hours=30)),
                          "fim": str(START + pd.Timedelta(hours=48))})
        if pid == "P003":  # dois picos hipertensivos de 40 min
            for h0 in (14.0, 38.5):
                m = (t_h >= h0) & (t_h < h0 + 0.67)
                data["systolic_bp"][m] += 48
                data["diastolic_bp"][m] += 20
                data["heart_rate"][m] += 12
                truth.append({"patient_id": "P003", "tipo": "crise_hipertensiva",
                              "inicio": str(START + pd.Timedelta(hours=h0)),
                              "fim": str(START + pd.Timedelta(hours=h0 + 0.67))})
        if pid == "P001":  # artefato de sensor isolado (dedo fora do oxímetro): NÃO deve alertar
            k = int(20 * 60 / SAMPLE_MINUTES)
            data["spo2"][k] = 71.0
            truth.append({"patient_id": "P001", "tipo": "artefato_sensor",
                          "inicio": str(START + pd.Timedelta(hours=20)),
                          "fim": str(START + pd.Timedelta(hours=20, minutes=5))})

        df = pd.DataFrame(data)
        df["spo2"] = df["spo2"].clip(upper=100)
        df.insert(0, "timestamp", START + pd.to_timedelta(np.arange(n) * SAMPLE_MINUTES, unit="min"))
        df.insert(1, "patient_id", pid)
        frames.append(df.round(1))

    return pd.concat(frames, ignore_index=True), truth


def load_vitals_csv(path: str | Path, patient_id: str | None = None) -> pd.DataFrame:
    """Carrega sinais vitais reais em CSV.

    Aceita o formato deste projeto (timestamp, patient_id, heart_rate, ...) **ou** o CSV
    de numéricos do dataset PhysioNet *BIDMC PPG and Respiration*
    (`bidmc_XX_Numerics.csv`, colunas `Time [s]`, `HR`, `PULSE`, `RESP`, `SpO2`).
    Sinais ausentes ficam NaN e são ignorados pelas regras.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Time [s]" in df.columns:  # PhysioNet BIDMC
        df = df.rename(columns={"HR": "heart_rate", "RESP": "resp_rate", "SpO2": "spo2"})
        df["timestamp"] = START + pd.to_timedelta(df["Time [s]"], unit="s")
        df["patient_id"] = patient_id or Path(path).stem.split("_Numerics")[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ("heart_rate", "systolic_bp", "diastolic_bp", "spo2", "resp_rate", "temperature"):
        if col not in df.columns:
            df[col] = np.nan
    return df[["timestamp", "patient_id", "heart_rate", "systolic_bp", "diastolic_bp",
               "spo2", "resp_rate", "temperature"]]


# --------------------------------------------------------------------------------------
# 2. Prescrições (evolução do tratamento)
# --------------------------------------------------------------------------------------

def generate_prescriptions() -> pd.DataFrame:
    """Histórico de prescrições (cada linha = uma versão da prescrição de um fármaco)."""
    rows = [
        # P001 – estável, ajustes esperados
        ("P001", 0, "dipirona", 1000, 4, "EV", "Dr. A"),
        ("P001", 0, "omeprazol", 40, 1, "EV", "Dr. A"),
        ("P001", 0, "enoxaparina", 40, 1, "SC", "Dr. A"),
        ("P001", 24, "dipirona", 1000, 3, "VO", "Dr. A"),  # descalonamento normal
        ("P001", 24, "omeprazol", 20, 1, "VO", "Dr. A"),
        ("P001", 24, "enoxaparina", 40, 1, "SC", "Dr. A"),
        # P002 – pneumonia
        ("P002", 0, "ceftriaxona", 1000, 2, "EV", "Dr. B"),
        ("P002", 0, "azitromicina", 500, 1, "EV", "Dr. B"),
        ("P002", 0, "dipirona", 1000, 4, "EV", "Dr. B"),
        ("P002", 0, "enoxaparina", 40, 1, "SC", "Dr. B"),
        ("P002", 24, "azitromicina", 500, 1, "EV", "Dr. B"),
        ("P002", 24, "dipirona", 2000, 4, "EV", "Dr. C"),  # 8 g/dia > máximo (erro)
        ("P002", 24, "enoxaparina", 40, 1, "SC", "Dr. C"),
        # ceftriaxona some às 24 h -> suspensão abrupta de antibiótico
        ("P002", 36, "azitromicina", 500, 1, "EV", "Dr. C"),
        ("P002", 36, "dipirona", 1000, 4, "EV", "Dr. C"),
        ("P002", 36, "enoxaparina", 40, 1, "SC", "Dr. C"),
        # P003 – AVC
        ("P003", 0, "losartana", 50, 1, "VO", "Dr. D"),
        ("P003", 0, "atorvastatina", 40, 1, "VO", "Dr. D"),
        ("P003", 0, "enoxaparina", 40, 1, "SC", "Dr. D"),
        ("P003", 12, "losartana", 50, 1, "VO", "Dr. D"),
        ("P003", 12, "atorvastatina", 40, 1, "VO", "Dr. D"),
        ("P003", 12, "enoxaparina", 120, 2, "SC", "Dr. E"),  # dose x6 sem justificativa
        ("P003", 12, "valsartana", 160, 1, "VO", "Dr. E"),  # duplicidade de classe (BRA)
        ("P003", 30, "losartana", 50, 1, "VO", "Dr. E"),
        ("P003", 30, "atorvastatina", 40, 1, "VO", "Dr. E"),
        ("P003", 30, "enoxaparina", 120, 2, "SC", "Dr. E"),
        ("P003", 30, "valsartana", 160, 1, "VO", "Dr. E"),
        ("P003", 30, "cetorolaco", 30, 3, "EV", "Dr. E"),  # AINE + anticoagulante
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "hora", "medicamento", "dose_mg",
                                     "vezes_dia", "via", "prescritor"])
    df.insert(1, "timestamp", START + pd.to_timedelta(df.pop("hora"), unit="h"))
    return df


# --------------------------------------------------------------------------------------
# 3. Movimentação do paciente durante a internação (acelerômetro de leito/pulseira)
# --------------------------------------------------------------------------------------

def generate_movement(random_state: int = 7) -> pd.DataFrame:
    """Contagens de atividade por hora (acelerômetro), saídas do leito e impacto (g)."""
    rng = np.random.default_rng(random_state)
    rows = []
    for pid in PATIENTS:
        for h in range(HOURS):
            hod = h % 24
            awake = 7 <= hod <= 22
            activity = rng.gamma(6, 30) if awake else rng.gamma(2, 8)
            bed_exits = int(rng.poisson(0.35 if awake else 0.03))
            impact_g = round(float(rng.uniform(0.9, 1.6)), 2)
            if pid == "P003":
                if 8 <= h <= 13:  # 6 h sem mudança de decúbito (risco de lesão por pressão)
                    activity, bed_exits = rng.uniform(0, 4), 0
                if hod in (2, 3) and h >= 24:  # agitação noturna na 2ª noite
                    activity, bed_exits = rng.uniform(260, 320), 2
                if h == 27:  # queda
                    impact_g, bed_exits = 4.3, 1
            rows.append({"timestamp": START + pd.Timedelta(hours=h), "patient_id": pid,
                         "atividade": round(float(activity), 1), "saidas_leito": bed_exits,
                         "impacto_max_g": impact_g})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Sessão de fisioterapia: keypoints COCO-17 (mesmo formato do YOLOv8-pose / OpenPose)
# --------------------------------------------------------------------------------------

COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


@dataclass
class PoseSession:
    """Saída comum de qualquer backend de vídeo."""

    keypoints: np.ndarray  # (T, 17, 3) -> x, y (pixels, origem no topo), confiança
    fps: float
    detections: list[list[dict]]  # por frame: [{"label", "conf", "box": [x1,y1,x2,y2]}]
    frame_size: tuple[int, int] = (640, 480)
    source: str = "synthetic"


def _leg(ankle, flex_deg, l_shank, l_thigh):
    f = np.deg2rad(flex_deg) / 2
    knee = ankle + l_shank * np.array([np.sin(f), np.cos(f)])
    hip = knee + l_thigh * np.array([-np.sin(f), np.cos(f)])
    return knee, hip


def generate_physio_session(anomalous: bool, duration_s: float = 30.0, fps: float = 15.0,
                            random_state: int = 3) -> PoseSession:
    """Simula agachamentos assistidos (vista lateral), 1 repetição a cada 5 s.

    Protocolo esperado: flexão de joelho até ~95°, simetria entre lados (< 12°) e tronco
    com inclinação < 40°. Na versão anômala: déficit de amplitude e assimetria do lado
    direito (hemiparesia), tremor de 6 Hz a partir de 15 s, saída do fisioterapeuta do
    campo de visão aos 18 s e perda de equilíbrio/queda aos 25 s.
    """
    rng = np.random.default_rng(random_state)
    W, H = 640, 480
    T = int(duration_s * fps)
    kps = np.zeros((T, 17, 3))
    dets: list[list[dict]] = []
    scale = 120.0  # pixels por "unidade" de segmento
    for i in range(T):
        t = i / fps
        depth = (1 - np.cos(2 * np.pi * t / 5.0)) / 2  # 0 (em pé) -> 1 (agachado)
        target_l = 95.0
        target_r = 135.0 if anomalous else 97.0
        knee_l = 175 - depth * (175 - target_l)
        knee_r = 175 - depth * (175 - target_r)
        if anomalous and t >= 15:
            tremor = 7.0 * np.sin(2 * np.pi * 6.0 * t)
            knee_l += tremor
            knee_r += tremor
        lean = np.deg2rad(8 + 22 * depth)
        fall = anomalous and t >= 25.0
        ground = np.array([3.0, 0.0])
        a_l, a_r = ground + [0.05, 0], ground - [0.05, 0]
        k_l, h_l = _leg(a_l, 180 - knee_l, 1.0, 1.0)
        k_r, h_r = _leg(a_r, 180 - knee_r, 1.0, 1.0)
        hip = (h_l + h_r) / 2
        if fall:  # corpo tomba para trás e quadril desce até o chão em ~1 s
            p = min((t - 25.0) / 1.0, 1.0)
            lean = np.deg2rad(8 + 22 * depth) * (1 - p) - np.deg2rad(75) * p
            drop = p * (hip[1] - 0.25)
            hip = hip - [0.6 * p, drop]
            h_l, h_r = h_l - [0.6 * p, drop], h_r - [0.6 * p, drop]
            k_l, k_r = k_l - [0.3 * p, drop * 0.6], k_r - [0.3 * p, drop * 0.6]
        sh = hip + 1.25 * np.array([np.sin(lean), np.cos(lean)])
        head = sh + 0.35 * np.array([np.sin(lean), np.cos(lean)])
        arm = np.deg2rad(80 * depth)  # braços elevam à frente para equilíbrio
        elbow = sh + 0.55 * np.array([np.sin(arm), -np.cos(arm)])
        wrist = elbow + 0.5 * np.array([np.sin(arm), -np.cos(arm)])

        pts = {
            "nose": head + [0.08, 0], "left_eye": head + [0.07, 0.04], "right_eye": head + [0.06, 0.04],
            "left_ear": head + [-0.03, 0.02], "right_ear": head + [-0.04, 0.02],
            "left_shoulder": sh + [0.03, 0], "right_shoulder": sh - [0.03, 0],
            "left_elbow": elbow + [0.03, 0], "right_elbow": elbow - [0.03, 0],
            "left_wrist": wrist + [0.03, 0], "right_wrist": wrist - [0.03, 0],
            "left_hip": h_l, "right_hip": h_r, "left_knee": k_l, "right_knee": k_r,
            "left_ankle": a_l, "right_ankle": a_r,
        }
        for name, p in pts.items():
            x = p[0] * scale + rng.normal(0, 1.5)
            y = H - 40 - p[1] * scale + rng.normal(0, 1.5)
            kps[i, KP[name]] = (x, y, 0.92)

        xs, ys = kps[i, :, 0], kps[i, :, 1]
        frame_dets = [{"label": "person", "conf": 0.93,
                       "box": [float(xs.min() - 15), float(ys.min() - 25), float(xs.max() + 15), float(ys.max() + 5)]}]
        therapist_present = not (anomalous and t >= 18.0)
        if therapist_present:
            frame_dets.append({"label": "person", "conf": 0.90, "box": [470.0, 120.0, 560.0, 440.0]})
        frame_dets.append({"label": "chair", "conf": 0.81, "box": [80.0, 300.0, 170.0, 440.0]})
        dets.append(frame_dets)

    return PoseSession(keypoints=kps, fps=fps, detections=dets, frame_size=(W, H),
                       source="synthetic-anomalo" if anomalous else "synthetic-referencia")


# --------------------------------------------------------------------------------------
# 5. Gravações de voz (consultas) + transcrições de referência
# --------------------------------------------------------------------------------------

AUDIO_PROFILES = {
    # f0 (Hz), jitter (fração), shimmer (fração), sílabas/s, prob. pausa entre sílabas,
    # duração média de pausa (s), ruído aspirado, decaimento de intensidade, irregularidade ritmo
    "normal": dict(f0=205, jitter=0.004, shimmer=0.025, rate=4.6, pause_p=0.07, pause_s=0.35,
                   breath=0.012, decay=0.0, irregular=0.08, breaths=False),
    "dispneia": dict(f0=215, jitter=0.009, shimmer=0.05, rate=4.0, pause_p=0.33, pause_s=0.55,
                     breath=0.06, decay=0.15, irregular=0.15, breaths=True),
    "disartria_fadiga": dict(f0=165, jitter=0.024, shimmer=0.11, rate=2.6, pause_p=0.25, pause_s=0.7,
                             breath=0.03, decay=0.55, irregular=0.45, breaths=False),
}

TRANSCRIPTS = {
    "P001": ("Bom dia, doutor. Estou me sentindo bem melhor hoje. A dor da cirurgia diminuiu "
             "bastante, consegui dormir a noite toda e já estou caminhando no corredor. "
             "Estou animada para ir para casa."),
    "P002": ("Doutor, eu estou com muita falta de ar, piorou desde ontem à noite. Tenho febre e "
             "calafrios, a tosse está com catarro amarelado e sinto uma dor no peito quando "
             "respiro fundo. Estou muito preocupado, não consigo nem terminar uma frase."),
    "P003": ("Eu... estou muito cansada. A perna direita continua fraca e eu tenho tontura "
             "quando levanto. Ontem à noite eu caí perto da cama. A fala está enrolada e às "
             "vezes sinto formigamento no braço. Não sinto dor no peito."),
}
AUDIO_FOR_PATIENT = {"P001": "normal", "P002": "dispneia", "P003": "disartria_fadiga"}


def _formant_filter(x, sr, formants):
    y = np.zeros_like(x)
    for fc, bw, gain in formants:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r * r]
        y += gain * signal.lfilter(b, a, x)
    return y


def generate_voice(profile: str, duration_s: float = 12.0, sr: int = 16000,
                   random_state: int = 11) -> np.ndarray:
    """Sintetiza uma 'fala' com fonte glotal (trem de pulsos com jitter/shimmer),
    filtro de formantes e envelope silábico. Não é inteligível — serve para extrair
    biomarcadores acústicos com ground truth conhecido."""
    p = AUDIO_PROFILES[profile]
    rng = np.random.default_rng(random_state)
    n = int(duration_s * sr)
    out = np.zeros(n)
    t = 0.3
    vowels = [[(700, 90, 1.0), (1220, 110, 0.5), (2600, 160, 0.25)],
              [(300, 70, 1.0), (2300, 120, 0.4), (3000, 170, 0.2)],
              [(500, 80, 1.0), (900, 100, 0.5), (2500, 160, 0.2)]]
    while t < duration_s - 0.5:
        syl = max(0.08, rng.normal(1.0 / p["rate"] * 0.75, 0.02 + p["irregular"] * 0.08))
        i0, i1 = int(t * sr), min(n, int((t + syl) * sr))
        L = i1 - i0
        progress = t / duration_s
        f0 = p["f0"] * (1 - 0.12 * p["decay"] * progress) * rng.normal(1, 0.04)
        src = np.zeros(L)
        pos = 0.0
        amp = 1.0
        while pos < L:
            period = sr / f0 * (1 + rng.normal(0, p["jitter"]))
            amp = max(0.2, 1 + rng.normal(0, p["shimmer"]))
            k = int(pos)
            if k < L:
                src[k] = amp
            pos += period
        src = signal.lfilter([1], [1, -0.97], src)
        voiced = _formant_filter(src, sr, vowels[rng.integers(0, 3)])
        env = np.sin(np.pi * np.linspace(0, 1, L)) ** 0.8
        loud = (1 - p["decay"] * progress)
        seg = voiced * env * loud + rng.normal(0, p["breath"], L) * env
        out[i0:i1] += seg
        t += syl + max(0.02, rng.normal(0.06, 0.02 * (1 + 3 * p["irregular"])))
        if rng.random() < p["pause_p"]:
            pause = abs(rng.normal(p["pause_s"], 0.1))
            if p["breaths"]:  # inspiração audível (ruído filtrado) no meio da pausa
                b0, b1 = int((t + 0.1) * sr), min(n, int((t + 0.1 + 0.35) * sr))
                if b1 > b0:
                    noise = signal.lfilter(*signal.butter(2, [500 / (sr / 2), 3000 / (sr / 2)], "band"),
                                           rng.normal(0, 1, b1 - b0))
                    out[b0:b1] += 0.25 * noise * np.hanning(b1 - b0)
            t += pause
    out += rng.normal(0, 0.002, n)  # ruído de fundo do consultório
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.8
    return out.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return path


def build_dataset(data_dir: str | Path) -> dict:
    """Gera todos os dados sintéticos em `data_dir` e devolve um manifesto."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    vitals, truth = generate_vitals()
    vitals.to_csv(data_dir / "sinais_vitais.csv", index=False)
    generate_prescriptions().to_csv(data_dir / "prescricoes.csv", index=False)
    generate_movement().to_csv(data_dir / "movimentacao.csv", index=False)
    audio = {}
    for pid, profile in AUDIO_FOR_PATIENT.items():
        wav = save_wav(data_dir / f"consulta_{pid}.wav", generate_voice(profile, random_state=10 + int(pid[1:])))
        (data_dir / f"consulta_{pid}.txt").write_text(TRANSCRIPTS[pid], encoding="utf-8")
        audio[pid] = str(wav)
    manifest = {"pacientes": PATIENTS, "ground_truth_vitais": truth, "audio": audio,
                "perfis_audio": AUDIO_FOR_PATIENT}
    (data_dir / "manifesto.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return manifest
