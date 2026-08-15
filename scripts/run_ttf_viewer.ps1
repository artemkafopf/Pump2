# Launch the TTF viewer - binned Свод fact panel by any operating characteristic.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try { & python -m streamlit run "streamlit_apps/ttf_viewer.py" } finally { Pop-Location }
