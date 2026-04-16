"""Streamlit Cloud entry point — Real Estate Pricing Demo.

Streamlit Cloud ищет этот файл в корне репозитория/проекта.
Запускает дашборд из пакета pricing.dashboard.

Locally:
    cd 07-realestate-pricing
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем директорию проекта в путь, чтобы пакет pricing был импортируемым.
# На Streamlit Cloud CWD — корень репозитория, поэтому явно указываем путь.
_PROJECT_DIR = Path(__file__).parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from pricing.dashboard.app import main  # noqa: E402

main()
