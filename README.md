# Pipeline Smart Optimizer — Bharat Flow Analytics

Flow: **splash page → login → dashboard** (Steps 1-4).

## Start
Windows: `run_desktop.bat` · macOS/Linux: `./run_desktop.sh` (native window; first run creates `venv`).
Browser: `pip install -r requirements.txt` then `streamlit run streamUIdashboard.py`.

## Login
Default: **admin / bfa@123**. Change with environment variables `BFA_USERNAME` / `BFA_PASSWORD` before starting.

## Data
Place `Upstream_data.csv`, `Downstream_data.csv`, `elevation_profile.csv` and `diameter.csv` next to
`streamUIdashboard.py` (or upload via the left pane). Without them the app uses embedded example data.

## Logo
`assets/BFAPL_LOGO.png` is the supplied logo, unmodified; the app only scales its display width.

## Desktop downloads
`desktop_app.py` enables webview downloads and sets `BFA_DESKTOP=1`; Step 4 then also shows
**Save to Downloads** buttons that write the PDF/CSV straight into your Downloads folder.
