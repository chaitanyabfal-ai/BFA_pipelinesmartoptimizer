@echo off
cd /d "%~dp0"
if not exist venv ( python -m venv venv & call venv\Scripts\activate & pip install -r requirements-desktop.txt ) else ( call venv\Scripts\activate )
python desktop_app.py
