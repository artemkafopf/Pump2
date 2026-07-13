# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts\\run\\production_risk.py'],
    pathex=['backend'],
    binaries=[],
    datas=[('results\\esp_survival_vba_models', 'results\\esp_survival_vba_models')],
    hiddenimports=['analysis.data.equipment_big', 'analysis.data.pump_type_parser', 'scripts.data_utils', 'openpyxl.cell._writer'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'matplotlib', 'pytest', 'hypothesis', 'sympy', 'scipy', 'pyarrow', 'PIL', 'cryptography', 'lxml', 'IPython', 'jedi', 'tkinter', 'numba', 'llvmlite', 'shap'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Pump2ProductionRisk',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Pump2ProductionRisk',
)
