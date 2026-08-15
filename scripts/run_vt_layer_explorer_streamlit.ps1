# Launch the Vt frequency / Kpod layer explorer.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try { & python -m streamlit run "streamlit_apps/vt_layer_explorer_app.py" } finally { Pop-Location }
