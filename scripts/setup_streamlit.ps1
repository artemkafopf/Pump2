$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyVersions = & py -0p 2>$null
        if ($pyVersions -match "-V:3\.13") {
            & py -3.13 -m venv --clear .venv
        }
        else {
            & py -m venv --clear .venv
        }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv --clear .venv
    }
    else {
        throw "No Python launcher was found. Install Python first."
    }

    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment creation did not produce .venv\\Scripts\\python.exe."
    }

    & $venvPython -m pip install -r "streamlit_apps/requirements.txt"
}
finally {
    Pop-Location
}
