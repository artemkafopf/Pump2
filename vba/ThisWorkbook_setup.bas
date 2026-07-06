Attribute VB_Name = "ThisWorkbook"
Option Explicit

' ============================================================
' ThisWorkbook — Auto-load model registry on open.
' Paste this into the ThisWorkbook code module (not a .bas module).
' ============================================================

Private Sub Workbook_Open()
    ' Load model registry automatically when the workbook opens.
    ' This avoids the first-call delay on large batches.
    Call LoadModelRegistry
End Sub
