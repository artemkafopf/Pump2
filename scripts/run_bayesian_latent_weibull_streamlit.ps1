$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonCommand = $null

if (Test-Path $venvPython) {
    $pythonCommand = $venvPython
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCommand = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCommand = "python"
}
else {
    throw "No Python launcher was found. Install Python, then install 'streamlit_apps/requirements.txt'."
}

$env:PYTHONPATH = Join-Path $repoRoot "backend"

Push-Location $repoRoot
try {
    if ($pythonCommand -eq "py") {
        & py -m streamlit run "streamlit_apps/bayesian_latent_weibull.py"
    }
    elseif ($pythonCommand -eq "python") {
        & python -m streamlit run "streamlit_apps/bayesian_latent_weibull.py"
    }
    else {
        & $pythonCommand -m streamlit run "streamlit_apps/bayesian_latent_weibull.py"
    }
}
finally {
    Pop-Location
}
