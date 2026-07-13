param(
    [string]$Name = "Pump2ProductionRisk",
    [switch]$OneFile,
    [switch]$SkipLauncher
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# Prefer the repo venv so the build works without an activated shell.
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

# Probe for PyInstaller without letting pip's stderr WARNING become a terminating
# error under ErrorActionPreference=Stop (Windows PowerShell 5.1 NativeCommandError).
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
cmd /c "`"$Python`" -m pip show pyinstaller >nul 2>nul"
$needInstall = ($LASTEXITCODE -ne 0)
if ($needInstall) {
    & $Python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }
}
$ErrorActionPreference = $prevEap

$pyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--name", $Name,
    "--paths", "backend",
    "--add-data", "results\esp_survival_vba_models;results\esp_survival_vba_models",
    "--hidden-import", "analysis.data.equipment_big",
    "--hidden-import", "analysis.data.pump_type_parser",
    "--hidden-import", "scripts.data_utils",
    "--hidden-import", "openpyxl.cell._writer",
    # heavy packages the workflow never imports (analysis/__init__ is lazy):
    "--exclude-module", "torch",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pytest",
    "--exclude-module", "hypothesis",
    "--exclude-module", "sympy",
    "--exclude-module", "scipy",
    "--exclude-module", "pyarrow",      # projection.parquet falls back to csv.gz
    "--exclude-module", "PIL",
    "--exclude-module", "cryptography",
    "--exclude-module", "lxml",
    "--exclude-module", "IPython",
    "--exclude-module", "jedi",
    "--exclude-module", "tkinter",
    "--exclude-module", "numba",
    "--exclude-module", "llvmlite",
    "--exclude-module", "shap",
    "scripts\run\production_risk.py"
)

if ($OneFile) {
    $pyInstallerArgs = @("--onefile") + $pyInstallerArgs
}

& $Python -m PyInstaller @pyInstallerArgs

if (-not $SkipLauncher) {
    if ($OneFile) {
        $launcherDir = "dist"
        $exePath = "dist\$Name.exe"
    } else {
        $launcherDir = "dist\$Name"
        $exePath = "dist\$Name\$Name.exe"
    }
    & $Python scripts\deploy\build_production_risk_launcher.py `
        --output "$launcherDir\ProductionRiskLauncher.xlsm" `
        --exe $exePath
}

Write-Host ""
Write-Host "Built executable under:" -ForegroundColor Green
if ($OneFile) {
    Write-Host "  dist\$Name.exe"
} else {
    Write-Host "  dist\$Name\$Name.exe"
}
if (-not $SkipLauncher) {
    Write-Host "  $launcherDir\ProductionRiskLauncher.xlsm"
}
