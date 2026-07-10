Attribute VB_Name = "ThisWorkbook"
Option Explicit

' ============================================================
' ThisWorkbook — Auto-load model registry on open.
' Paste this into the ThisWorkbook code module (not a .bas module).
' ============================================================

Private Sub Workbook_Open()
    ' Load both registries automatically when the workbook opens.
    ' This avoids the first-call delay on large batches.
    Call LoadModelRegistry
    Call LoadModeMix        ' Phase B mode-mix planning layer (ESP_ModeMix)
    Call LoadHazardLayer    ' operational + completion overlay (ESP_CoxCoeffs/ESP_RunCov)

    ' Force a full recalc so cached UDF errors do not remain visible on open.
    On Error Resume Next
    Application.CalculateFull

    ' If the simplified forecast sheets exist, refresh their history-derived fields.
    CalculateSimplePrediction
    CalculateMultiPrediction
    On Error GoTo 0
End Sub
