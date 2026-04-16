"""Streamlit Cloud entry point — Realtime Anomaly Detection SRE Dashboard.

Streamlit Cloud ищет этот файл в корне репозитория/проекта.
Запускает дашборд из пакета anomaly.dashboard.

Locally:
    cd 05-realtime-anomaly
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем директорию проекта в путь, чтобы пакет anomaly был импортируемым.
# На Streamlit Cloud CWD — корень репозитория, поэтому явно указываем путь.
_PROJECT_DIR = Path(__file__).parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from anomaly.dashboard.app import main  # noqa: E402

main()
