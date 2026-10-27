"""Entry point for Streamlit Community Cloud: runs the WildGuard app in SYSTEM CODE/app/app.py.

    streamlit run streamlit_app.py        (from the repository root)
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "SYSTEM CODE" / "app" / "app.py"),
               run_name="__main__")
