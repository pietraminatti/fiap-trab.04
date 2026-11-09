from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "reports"
FIG = OUT / "figures"
SEV_ICON = {"CRITICO": "🔴", "ALTO": "🟠", "MODERADO": "🟡", "INFO": "⚪", "BAIXO": "🟢"}

st.set_page_config(page_title="Monitoramento Multimodal", page_icon="🩺", layout="wide")

if not (OUT / "resultados.json").exists():
    st.error("Execute primeiro: `python run_multimodal_monitoring.py`")
    st.stop()

res = json.loads((OUT / "resultados.json").read_text(encoding="utf-8"))
alerts = pd.read_json(OUT / "alerts.jsonl", lines=True)
alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])

st.title("🩺 Monitoramento multimodal de pacientes")
modo = res["modo_servicos"]
st.caption(f"Speech to Text: **{modo['speech']}** · Text Analytics: **{modo['language']}** · "
           f"Webhook: **{'ativo' if modo['webhook'] else 'não configurado'}**")

# ---- Cartões de risco -----------------------------------------------------------------
cols = st.columns(len(res["fusao"]))
for col, f in zip(cols, res["fusao"]):
    info = res["pacientes"][f["patient_id"]]
    col.metric(f"{SEV_ICON[f['nivel']]} {f['patient_id']} · {info['leito']}", f"{f['risco']:.2f} · {f['nivel']}")
    col.caption(f"{info['diagnostico']} · {f['n_alertas']} alertas")

# ---- Fila de alertas ------------------------------------------------------------------
st.subheader("Fila de alertas")
c1, c2, c3 = st.columns(3)
pacs = c1.multiselect("Paciente", sorted(alerts["patient_id"].unique()), default=sorted(alerts["patient_id"].unique()))
sevs = c2.multiselect("Severidade", ["CRITICO", "ALTO", "MODERADO", "INFO"], default=["CRITICO", "ALTO", "MODERADO"])
fontes = c3.multiselect("Fonte", sorted(alerts["source"].unique()), default=sorted(alerts["source"].unique()))
view = alerts[alerts["patient_id"].isin(pacs) & alerts["severity"].isin(sevs) & alerts["source"].isin(fontes)]
order = {"CRITICO": 0, "ALTO": 1, "MODERADO": 2, "INFO": 3}
view = view.assign(_o=view["severity"].map(order)).sort_values(["_o", "timestamp"], ascending=[True, False])

if "ack" not in st.session_state:
    st.session_state.ack = set()
for i, a in view.iterrows():
    key = f"{a['patient_id']}|{a['timestamp']}|{a['title']}"
    done = key in st.session_state.ack
    with st.expander(f"{SEV_ICON[a['severity']]} {a['severity']} · {a['patient_id']} · {a['title']} · "
                     f"{a['timestamp']:%d/%m %H:%M}" + ("  ✅ ciente" if done else ""),
                     expanded=a["severity"] == "CRITICO" and not done):
        st.write(a["description"])
        st.markdown(f"**Fonte:** {a['source']} · **Destino:** {a['routed_to']}")
        if a["recommended_action"]:
            st.info(f"Conduta sugerida: {a['recommended_action']}")
        st.json(a["evidence"], expanded=False)
        if not done and st.button("Registrar ciência", key=f"ack{i}"):
            st.session_state.ack.add(key)
            st.rerun()

# ---- Evidências por paciente ----------------------------------------------------------
st.subheader("Evidências por paciente")
pid = st.selectbox("Paciente", list(res["pacientes"]))
tabs = st.tabs(["Sinais vitais", "Vídeo", "Áudio", "Movimentação", "Fusão"])
with tabs[0]:
    st.image(str(FIG / f"vitais_{pid}.png"))
with tabs[1]:
    if pid in res["video"]:
        gif = FIG / f"video_{pid}_esqueleto.gif"
        if gif.exists():
            st.image(str(gif))
        st.image(str(FIG / f"video_{pid}.png"))
        st.markdown((OUT / f"relatorio_video_{pid}.md").read_text(encoding="utf-8").split("![")[0])
    else:
        st.write("Sem sessão de vídeo para este paciente.")
with tabs[2]:
    a = res["audio"].get(pid)
    if a:
        audio_path = Path(__file__).parent / a["arquivo"]
        if audio_path.exists():
            st.audio(str(audio_path))
        st.write(f"**Transcrição** ({a['transcricao']['provedor']}): {a['transcricao']['texto']}")
        st.write(f"**Sentimento:** {a['texto'].get('sentimento')} · **Indicadores vocais:** {a['indicadores']}")
        st.dataframe(pd.DataFrame([a["features"]]))
    st.image(str(FIG / "audio_indicadores.png"))
with tabs[3]:
    st.image(str(FIG / "movimentacao.png"))
with tabs[4]:
    st.image(str(FIG / "fusao_multimodal.png"))
    st.json(next(f for f in res["fusao"] if f["patient_id"] == pid))
    st.image(str(FIG / "linha_do_tempo_alertas.png"))
