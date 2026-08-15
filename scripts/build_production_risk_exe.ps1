param(
    [string]$Name = "Pump2ProductionRisk",
    [switch]$OneFile,
    [switch]$SkipLauncher,
    # By default the build REUSES PyInstaller's analysis/build cache, which makes
    # iterative rebuilds (param/code tweaks) 2-3x faster.  Pass -Clean to force a
    # full from-scratch rebuild (use if you changed dependencies or hit a stale-cache
    # issue).  NOTE: model parameters in the bundle esp_*.csv files (Ql/Kpod/uncertainty
    # betas + enable flags) are read at runtime, so tuning them needs NO rebuild at all —
    # just edit dist\<Name>\_internal\results\esp_survival_vba_models\<bundle>\esp_*.csv.
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# Prefer the repo venv so the build works without an activated shell.
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$predictionNameCodes = @(
    0x041E, 0x0442, 0x043A, 0x0430, 0x0437, 0x044B, 0x0020,
    0x0441, 0x0432, 0x043E, 0x0434, 0x0020,
    0x0441, 0x0020,
    0x0430, 0x043D, 0x0430, 0x043B, 0x0438, 0x0437, 0x043E, 0x043C,
    0x002E, 0x0078, 0x006C, 0x0073, 0x0078
)
$predictionFileName = -join ($predictionNameCodes | ForEach-Object { [char]$_ })
$PredictionSource = Join-Path $RepoRoot (Join-Path "data\inputs" $predictionFileName)
if (-not (Test-Path -LiteralPath $PredictionSource)) {
    throw "prediction source workbook not found: $PredictionSource"
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

# Preflight the shipped model hooks before freezing.  This does not run the
# forecast or build outputs; it prevents packaging a source tree that points away
# from the accepted bundle or accidentally restores retired manual factors.
$preflight = @'
import sys
from pathlib import Path

repo = Path.cwd()
for item in (repo, repo / "backend"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.workflows.production_risk import failure_rate as fr

required = (
    "_SURVIVAL_WEIGHT_FIELDS",
    "_CALIBRATION_FACTORS",
    "_REPORTING_FIELD_CALIBRATION_FACTORS",
)
missing = [name for name in required if not hasattr(fr, name)]
if missing:
    raise SystemExit("missing failure-rate calibration hooks: " + ", ".join(missing))
from analysis.workflows.production_risk import config as C
if C.BUNDLE_DATE != "2026-07-15-mc2023plus":
    raise SystemExit(f"unexpected production-risk bundle date: {C.BUNDLE_DATE}")
if fr._CALIBRATION_FACTORS or fr._REPORTING_FIELD_CALIBRATION_FACTORS:
    raise SystemExit("manual failure-rate calibration factors must be retired for the accepted bundle")
if not C.observed_failures_path(C.BUNDLE_DATE).exists():
    raise SystemExit("accepted bundle is missing esp_observed_failures.csv")
if not getattr(C, "KPOD_HAZARD_ENABLED", False):
    raise SystemExit("Kpod hazard dial is disabled (the active load/IOR stress term)")
print(
    "Production-risk model preflight: "
    f"survival={sorted(fr._SURVIVAL_WEIGHT_FIELDS)}; "
    f"bundle={C.BUNDLE_DATE}; "
    "manual_factors=retired; "
    f"ql_hazard_enabled={C.QL_HAZARD_ENABLED}; "
    f"uncertainty_enabled={C.UNCERTAINTY_HAZARD_ENABLED}; "
    f"kpod_dial={C.KPOD_HAZARD_SOURCE}"
)
'@
$preflightPath = Join-Path $env:TEMP "pump2_preflight_$PID.py"
Set-Content -Path $preflightPath -Value $preflight -Encoding UTF8
try {
    & $Python $preflightPath
    if ($LASTEXITCODE -ne 0) { throw "failure-rate calibration preflight failed" }
}
finally {
    Remove-Item -LiteralPath $preflightPath -ErrorAction SilentlyContinue
}

$pyInstallerArgs = @(
    "--noconfirm",
    "--name", $Name,
    "--paths", "backend",
    "--add-data", "results\esp_survival_vba_models;results\esp_survival_vba_models",
    "--hidden-import", "analysis.data.equipment_big",
    "--hidden-import", "analysis.data.pump_type_parser",
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
    # ML-comparison / refit-only deps — NOT used by the shipped forecast (base_p50 +
    # Kpod stress).  catboost alone was ~346 MB of the packaged EXE; these are reachable
    # only via the opt-in failure_rate_catboost (--catboost-compare, CLI-only) and
    # hazard_refit_c (CLI refit tool), never from run.py, so excluding them is safe.
    "--exclude-module", "catboost",
    "--exclude-module", "plotly",
    "--exclude-module", "sklearn",
    "--exclude-module", "statsmodels",
    "--exclude-module", "graphviz",
    "scripts\run\production_risk.py"
)

if ($Clean) {
    $pyInstallerArgs = @("--clean") + $pyInstallerArgs
}
if ($OneFile) {
    $pyInstallerArgs = @("--onefile") + $pyInstallerArgs
}

& $Python -m PyInstaller @pyInstallerArgs

if ($OneFile) {
    $distRoot = "dist"
} else {
    $distRoot = "dist\$Name"
}
$predictionDestDir = Join-Path $distRoot "data\inputs"
$predictionDest = Join-Path $predictionDestDir (Split-Path -Leaf $PredictionSource)
New-Item -ItemType Directory -Force -Path $predictionDestDir | Out-Null
Copy-Item -LiteralPath $PredictionSource -Destination $predictionDest -Force

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
        --exe $exePath `
        --prediction-workbook $predictionDest
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
Write-Host "  $predictionDest"
