"""
streamlit_app.py
----------------
Root entrypoint for Streamlit Cloud deployment.
Delegates directly to app.dashboard.main()
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dashboard import main

if __name__ == "__main__":
    main()
