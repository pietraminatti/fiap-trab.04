#!/bin/sh
set -e
case "$1" in
  pipeline) shift; exec python run_multimodal_monitoring.py "$@" ;;
  dashboard) python run_multimodal_monitoring.py --no-gif && exec streamlit run app_dashboard.py --server.address=0.0.0.0 --server.port=8501 ;;
  *) exec "$@" ;;
esac
