from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import VITAL_LABELS, VITAL_RANGES  # noqa: E402
from .synthetic_data import KP  # noqa: E402

SEV_COLOR = {"INFO": "#9aa0a6", "MODERADO": "#f9ab00", "ALTO": "#e8710a", "CRITICO": "#d93025"}
_BONES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
          ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
          ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
          ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
          ("nose", "left_shoulder"), ("nose", "right_shoulder")]


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(nenhum)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------

def plot_vitals(vitals: pd.DataFrame, alerts_df: pd.DataFrame, patient_id: str, out: Path) -> Path:
    df = vitals[vitals["patient_id"] == patient_id]
    al = alerts_df[(alerts_df["patient_id"] == patient_id) & (alerts_df["source"] == "sinais_vitais")] \
        if len(alerts_df) else alerts_df
    fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    for ax, v in zip(axes, VITAL_RANGES):
        ax.plot(df["timestamp"], df[v], lw=0.9, color="#1a73e8")
        lo, hi = VITAL_RANGES[v]
        ax.axhspan(lo, hi, color="#34a853", alpha=0.08)
        ax.set_ylabel(VITAL_LABELS[v], fontsize=8)
        for _, a in al.iterrows():
            ax.axvline(pd.Timestamp(a["timestamp"]), color=SEV_COLOR[a["severity"]], lw=1, alpha=0.7)
    axes[0].set_title(f"Sinais vitais — {patient_id} (faixa verde = referência; linhas = alertas)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_video_session(result: dict, title: str, out: Path) -> Path:
    kin = result["kinematics"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(kin["t"], kin["joelho_esq"], label="joelho esquerdo")
    axes[0].plot(kin["t"], kin["joelho_dir"], label="joelho direito")
    axes[0].axhline(110, ls="--", color="gray", lw=0.8, label="alvo de flexão (110°)")
    axes[0].set_ylabel("ângulo (°)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].plot(kin["t"], kin["inclinacao_tronco"], color="#9334e6", label="inclinação do tronco (°)")
    axes[1].plot(kin["t"], kin["altura_quadril"] * 100, color="#188038", label="altura do quadril (% em pé)")
    axes[1].legend(fontsize=8, loc="upper left")
    if "tremor_rms" in kin:
        axes[2].plot(kin["t"], kin["tremor_rms"], color="#d93025", label="tremor RMS 4–12 Hz (°)")
    if kin["pessoas"].notna().all():
        axes[2].step(kin["t"], kin["pessoas"], color="#5f6368", label="pessoas detectadas (YOLO)")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].set_xlabel("tempo (s)")
    for e in result["events"]:
        if e["severidade"] == "INFO":
            continue
        for ax in axes:
            ax.axvspan(e["t"], e.get("fim", e["t"] + 0.3), color=SEV_COLOR[e["severidade"]], alpha=0.15)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def skeleton_gif(session, result: dict, out: Path, step: int = 3) -> Path:
    """Animação do esqueleto com o evento ativo destacado (para o vídeo de demonstração)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    k = session.keypoints
    W, H = session.frame_size
    fig, ax = plt.subplots(figsize=(5, 4))
    frames = list(range(0, len(k), step))

    def draw(i):
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        t = i / session.fps
        active = [e for e in result["events"] if e["severidade"] != "INFO"
                  and e["t"] - 0.4 <= t <= e.get("fim", e["t"] + 1.5)]
        color = SEV_COLOR[max(active, key=lambda e: ["MODERADO", "ALTO", "CRITICO"].index(e["severidade"]))
                          ["severidade"]] if active else "#1a73e8"
        for a, b in _BONES:
            pa, pb = k[i, KP[a]], k[i, KP[b]]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=3)
        ax.scatter(k[i, :, 0], k[i, :, 1], s=12, color="black", zorder=3)
        for d in session.detections[i] if session.detections else []:
            x1, y1, x2, y2 = d["box"]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#5f6368", lw=0.8, ls="--"))
            ax.text(x1, y1 - 3, d["label"], fontsize=6, color="#5f6368")
        ax.set_title(f"t = {t:4.1f}s  " + (active[-1]["tipo"].upper() if active else "OK"), fontsize=9,
                     color=color)

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(out, writer=PillowWriter(fps=session.fps / step))
    plt.close(fig)
    return out


def plot_audio(audio_results: dict, out: Path) -> Path:
    n = len(audio_results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), gridspec_kw={"width_ratios": [3, 1]})
    axes = np.atleast_2d(axes)
    for row, (pid, r) in zip(axes, audio_results.items()):
        x, sr = r["signal"], r["sr"]
        row[0].plot(np.arange(len(x)) / sr, x, lw=0.4, color="#1a73e8")
        row[0].set_title(f"{pid} — perfil: {r.get('perfil', 'real')}", fontsize=9)
        row[0].set_xlabel("s", fontsize=8)
        ind = r["indicadores"]
        row[1].barh(list(ind), list(ind.values()),
                    color=["#d93025" if v >= 0.75 else "#f9ab00" if v >= 0.5 else "#34a853" for v in ind.values()])
        row[1].set_xlim(0, 1)
        row[1].axvline(0.5, ls="--", color="gray", lw=0.8)
        row[1].tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_movement(mv: pd.DataFrame, alerts_df: pd.DataFrame, out: Path) -> Path:
    pids = list(mv["patient_id"].unique())
    fig, axes = plt.subplots(len(pids), 1, figsize=(12, 2.4 * len(pids)), sharex=True)
    for ax, pid in zip(np.atleast_1d(axes), pids):
        d = mv[mv["patient_id"] == pid]
        ax.bar(d["timestamp"], d["atividade"], width=0.035, color="#1a73e8")
        ax.set_ylabel(pid)
        if len(alerts_df):
            for _, a in alerts_df[(alerts_df["patient_id"] == pid) & (alerts_df["source"] == "movimentacao")
                                  & (alerts_df["severity"] != "INFO")].iterrows():
                ax.axvline(pd.Timestamp(a["timestamp"]), color=SEV_COLOR[a["severity"]], lw=2)
    np.atleast_1d(axes)[0].set_title("Atividade por hora (acelerômetro) e alertas de movimentação")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_fusion(fused: list[dict], out: Path) -> Path:
    mods = list(fused[0]["escores_modalidade"])
    M = np.array([[f["escores_modalidade"][m] for m in mods] + [f["risco"]] for f in fused])
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * len(fused)))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mods) + 1), mods + ["RISCO FUNDIDO"], rotation=20, fontsize=8)
    ax.set_yticks(range(len(fused)), [f"{f['patient_id']} ({f['nivel']})" for f in fused])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Fusão multimodal: escore por modalidade e risco final")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_alert_timeline(alerts_df: pd.DataFrame, out: Path) -> Path:
    pids = sorted(alerts_df["patient_id"].unique())
    df = alerts_df[alerts_df["severity"] != "INFO"].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    sources = ["sinais_vitais", "prescricao", "movimentacao", "video", "audio", "fusao"]
    fig, ax = plt.subplots(figsize=(12, 1 + 1.1 * len(pids)))
    for i, pid in enumerate(pids):
        d = df[df["patient_id"] == pid]
        for _, a in d.iterrows():
            y = i + (sources.index(a["source"]) - 2.5) * 0.12
            ax.scatter(a["ts"], y, color=SEV_COLOR[a["severity"]], s=40, marker="os^DvP"[sources.index(a["source"])],
                       edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(pids)), pids)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="os^DvP"[i], ls="", color="gray", label=s) for i, s in enumerate(sources)]
    handles += [Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in SEV_COLOR.items() if s != "INFO"]
    ax.legend(handles=handles, fontsize=7, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.25))
    ax.set_title("Linha do tempo dos alertas (forma = fonte, cor = severidade)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Relatórios Markdown
# --------------------------------------------------------------------------------------

def video_report(patient_id: str, result: dict, protocol: dict, figure: str, out: Path) -> Path:
    s = result["summary"]
    reps = pd.DataFrame(result["reps"])
    ev = pd.DataFrame(result["events"])
    if len(ev):
        ev = ev[[c for c in ["t", "fim", "tipo", "severidade", "descricao"] if c in ev.columns]].fillna("")
    falhas = [e for e in result["events"] if e["severidade"] in ("ALTO", "CRITICO")]
    conclusao = ("**Sessão com desvios relevantes** — " + "; ".join(sorted({e["tipo"] for e in falhas}))
                 if falhas else ("Sessão com desvios leves." if result["events"] else "Sessão conforme o protocolo."))
    interrompida = " — **sessão interrompida por queda**" if s.get("interrompida_por_queda") else ""
    md = f"""# Relatório automático — Análise de vídeo ({patient_id})

- **Fonte:** `{s['fonte']}` · duração {s['duracao_s']} s · {s['fps']} fps
- **Protocolo:** {protocol['nome']}
- **Repetições avaliadas:** {s['repeticoes']} ({s['repeticoes_conformes']} conformes){interrompida}
- **Amplitude média (ângulo mínimo do joelho):** esq. {s['amplitude_media_esq']}° · dir. {s['amplitude_media_dir']}°
- **Conclusão:** {conclusao}

## Critérios do protocolo
| Critério | Limite |
|---|---|
| Flexão do joelho em cada repetição | ≤ {protocol['flexao_joelho_min_graus']}° |
| Assimetria entre joelhos | ≤ {protocol['assimetria_max_graus']}° |
| Inclinação do tronco | ≤ {protocol['inclinacao_tronco_max_graus']}° |
| Tremor (RMS 4–12 Hz) | ≤ {protocol['tremor_rms_max_graus']}° |
| Supervisão | ≥ {protocol['pessoas_minimas']} pessoas em cena; ausência ≤ {protocol['sem_supervisao_max_s']} s |

## Repetições
{_md_table(reps)}
## Desvios / eventos detectados
{_md_table(ev)}
![cinemática]({figure})
"""
    out.write_text(md, encoding="utf-8")
    return out


def alerts_panel(alerts_df: pd.DataFrame, summary: dict, out: Path) -> Path:
    cols = ["severity", "patient_id", "timestamp", "source", "title", "routed_to", "recommended_action"]
    show = alerts_df[alerts_df["severity"] != "INFO"][cols] if len(alerts_df) else alerts_df
    md = f"""# Painel de alertas — equipe médica

Total: **{summary['total']}** alertas · suprimidos por duplicidade: {summary['suprimidos_por_duplicidade']}
· por severidade: {summary['por_severidade']}

{_md_table(show.rename(columns={'severity': 'Sev.', 'patient_id': 'Paciente', 'timestamp': 'Momento',
                                 'source': 'Fonte', 'title': 'Alerta', 'routed_to': 'Destino',
                                 'recommended_action': 'Conduta sugerida'}))}
"""
    out.write_text(md, encoding="utf-8")
    return out
