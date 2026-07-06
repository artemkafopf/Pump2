Attribute VB_Name = "mdlBatchProcess"
Option Explicit

' ============================================================
' mdlBatchProcess -- Batch macros and utilities.
'
' RunPredictions()      -- fills output columns in the data sheet.
' ImportModelCSV()      -- loads updated parameters from Python export.
' RefreshModels()       -- clears registry cache.
' ExportSurvivalCurve() -- writes S(t) table to a new sheet.
' ============================================================

' Input column letters (1-based column numbers)
Private Const COL_FIELD      As Long = 1   ' A  Field
Private Const COL_CONTRACTOR As Long = 4   ' D  Contractor
Private Const COL_AGE        As Long = 9   ' I  Age in service (days)
Private Const COL_H2S        As Long = 75  ' BW H2S class (sour/nonsour)
Private Const COL_FLAG       As Long = 76  ' BX Failure Flag

' Returns the data sheet name using ChrW() to avoid Cyrillic literals in source.
' ChrW() accepts 0-65535; Chr() only accepts 0-255 and would throw Error 5.
' ChrW(1057)&ChrW(1074)&ChrW(1086)&ChrW(1076) = Svod (Russian sheet name)
Private Function DataSheetName() As String
    Static s As String
    If s = "" Then s = ChrW(1057) & ChrW(1074) & ChrW(1086) & ChrW(1076)
    DataSheetName = s
End Function

' Source columns for Cox covariate computation (verified 2026-07-02).
' delta_bep     = Q_actual / Q_nominal - 1   (computed from cols 21 and 60)
' p_bot         = Рзаб, bottomhole pressure  (direct read, col 31)
' n_stages_ratio = Кол.ступеней / Q_nominal  (computed from cols 64 and 60)
Private Const COL_FLOW_ACT As Long = 21  ' U   Дебит жидк., м³/сут (actual liquid rate)
Private Const COL_FLOW_NOM As Long = 60  ' BH  Ном. Произв. м³/сут (nominal pump rate)
Private Const COL_P_BOT    As Long = 31  ' AE  Рзаб, atm            (bottomhole pressure)
Private Const COL_STAGES   As Long = 64  ' BL  Кол.ступеней          (stage count)

' Reference-value fallbacks for missing data: substituting these makes theta = 1
' (Cox layer neutral — equals baseline prediction).  Matches mdlCoxHR REF_* constants.
Private Const COX_REF_BEP  As Double = -0.342  ' population-mean delta_bep
Private Const COX_REF_PBOT As Double = 110.3   ' population-mean p_bot, atm
Private Const COX_REF_NSR  As Double = 2.249   ' population-mean n_stages_ratio

' Output columns (written after the last existing column = 77)
Private Const OUT_START      As Long = 80  ' update if Cox input columns added inline

Private Const OUT_STRATUM    As Long = 80  ' BY
Private Const OUT_B10        As Long = 81  ' BZ
Private Const OUT_B50        As Long = 82  ' CA
Private Const OUT_B90        As Long = 83  ' CB
Private Const OUT_RUL        As Long = 84  ' CC
Private Const OUT_TTF_PRED   As Long = 85  ' CD
Private Const OUT_SF_NOW     As Long = 86  ' CE
Private Const OUT_COMPONENT  As Long = 87  ' CF
Private Const OUT_HAZARD     As Long = 88  ' CG
Private Const OUT_DEGEN      As Long = 89  ' CH
' Cox-adjusted output columns
Private Const OUT_THETA      As Long = 90  ' CI  ESP_Theta (hazard multiplier)
Private Const OUT_B50_COX    As Long = 91  ' CJ  ESP_B50_Cox
Private Const OUT_RUL_COX    As Long = 92  ' CK  ESP_RUL_Cox

' ------------------------------------------------------------------
' RunPredictions -- Write all prediction columns to the data sheet.
' Skips rows with Failure Flag = -1.
' Shows a progress bar in the status bar.
' ------------------------------------------------------------------
Public Sub RunPredictions()
    Dim ws As Worksheet
    Dim lastRow As Long, i As Long
    Dim field As String, h2sRaw As String, ctrRaw As String
    Dim ageDays As Double, flag As Integer
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Dim flowAct As Double, flowNom As Double, stagesN As Double
    Dim deltaBEP As Double, pBot As Double, nStagesR As Double

    On Error GoTo ErrHandler

    Set ws = ThisWorkbook.Worksheets(DataSheetName())
    lastRow = ws.Cells(ws.Rows.Count, COL_FIELD).End(xlUp).Row

    ' Write output headers
    ws.Cells(1, OUT_STRATUM).Value   = "ESP_Stratum"
    ws.Cells(1, OUT_B10).Value       = "ESP_B10"
    ws.Cells(1, OUT_B50).Value       = "ESP_B50"
    ws.Cells(1, OUT_B90).Value       = "ESP_B90"
    ws.Cells(1, OUT_RUL).Value       = "ESP_RUL"
    ws.Cells(1, OUT_TTF_PRED).Value  = "ESP_TTF_Pred"
    ws.Cells(1, OUT_SF_NOW).Value    = "ESP_SF_Now"
    ws.Cells(1, OUT_COMPONENT).Value = "ESP_Component"
    ws.Cells(1, OUT_HAZARD).Value    = "ESP_Hazard"
    ws.Cells(1, OUT_DEGEN).Value     = "ESP_Degenerate"
    ws.Cells(1, OUT_THETA).Value     = "ESP_Theta"
    ws.Cells(1, OUT_B50_COX).Value   = "ESP_B50_Cox"
    ws.Cells(1, OUT_RUL_COX).Value   = "ESP_RUL_Cox"
    ws.Rows(1).Font.Bold = True

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    ' Ensure model registry is loaded
    EnsureRegistryLoaded

    Dim t0 As Long
    t0 = Timer

    For i = 2 To lastRow
        ' Progress every 500 rows
        If (i Mod 500) = 0 Then
            Application.StatusBar = "ESP predictions: row " & i & " of " & lastRow & _
                                    "  (" & Format((i - 1) / (lastRow - 1) * 100, "0") & "%)"
            DoEvents
        End If

        field   = Trim(CStr(ws.Cells(i, COL_FIELD).Value))
        ctrRaw  = Trim(CStr(ws.Cells(i, COL_CONTRACTOR).Value))
        h2sRaw  = Trim(CStr(ws.Cells(i, COL_H2S).Value))
        flag    = CInt(Val(CStr(ws.Cells(i, COL_FLAG).Value)))

        ' Skip suspended runs
        If flag = -1 Then
            Dim col As Long
            For col = OUT_START To OUT_DEGEN
                ws.Cells(i, col).Value = ""
            Next col
            GoTo NextRow
        End If

        ageDays = Val(CStr(ws.Cells(i, COL_AGE).Value))
        If ageDays < 0 Then ageDays = 0

        ' Resolve model
        Dim h2sCls As String, ctrGrp As String
        h2sCls = H2SClass(h2sRaw)
        ctrGrp = ContractorGroup(ctrRaw)
        If ctrGrp = "unk" Then ctrGrp = "Pooled"

        Dim found As Boolean
        found = LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen)

        ws.Cells(i, OUT_STRATUM).Value = IIf(found, stratumKey, "#NoModel")
        ws.Cells(i, OUT_DEGEN).Value   = IIf(found, isDegen, "")

        If Not found Then
            ws.Cells(i, OUT_B10).Value      = ""
            ws.Cells(i, OUT_B50).Value      = ""
            ws.Cells(i, OUT_B90).Value      = ""
            ws.Cells(i, OUT_RUL).Value      = ""
            ws.Cells(i, OUT_TTF_PRED).Value = ""
            ws.Cells(i, OUT_SF_NOW).Value   = ""
            ws.Cells(i, OUT_COMPONENT).Value = ""
            ws.Cells(i, OUT_HAZARD).Value   = ""
            GoTo NextRow
        End If

        ' Quantiles (same regardless of running/failed)
        ws.Cells(i, OUT_B10).Value = LatentQuantile(0.1, w1, b1, e1, b2, e2)
        ws.Cells(i, OUT_B50).Value = LatentQuantile(0.5, w1, b1, e1, b2, e2)
        ws.Cells(i, OUT_B90).Value = LatentQuantile(0.9, w1, b1, e1, b2, e2)

        ' Survival probability at current age
        ws.Cells(i, OUT_SF_NOW).Value = LatentSF(ageDays, w1, b1, e1, b2, e2)

        ' Component assignment at current age
        If LatentC1Posterior(ageDays, w1, b1, e1, b2, e2) >= 0.5 Then
            ws.Cells(i, OUT_COMPONENT).Value = "C1 (early failure)"
        Else
            ws.Cells(i, OUT_COMPONENT).Value = "C2 (wear-out)"
        End If

        ' Hazard at current age
        ws.Cells(i, OUT_HAZARD).Value = LatentHazard(ageDays, w1, b1, e1, b2, e2)

        ' RUL vs TTF split on failure flag
        If flag = 0 Then
            ' Running pump → RUL
            Dim rul As Double
            rul = LatentRUL(ageDays, w1, b1, e1, b2, e2)
            ws.Cells(i, OUT_RUL).Value      = IIf(rul > 0, rul, "")
            ws.Cells(i, OUT_TTF_PRED).Value = ""
        Else
            ' Failed pump → model B50 as predicted TTF
            ws.Cells(i, OUT_RUL).Value      = ""
            ws.Cells(i, OUT_TTF_PRED).Value = LatentQuantile(0.5, w1, b1, e1, b2, e2)
        End If

        ' ── Cox-adjusted columns ─────────────────────────────────────────────
        ' Compute covariates from source columns; fall back to reference values
        ' when data is missing so that CoxTheta returns 1 (neutral adjustment).
        flowAct = Val(CStr(ws.Cells(i, COL_FLOW_ACT).Value))
        flowNom = Val(CStr(ws.Cells(i, COL_FLOW_NOM).Value))
        pBot    = Val(CStr(ws.Cells(i, COL_P_BOT).Value))
        stagesN = Val(CStr(ws.Cells(i, COL_STAGES).Value))

        If flowNom > 0 Then
            deltaBEP = flowAct / flowNom - 1#
            nStagesR = stagesN / flowNom
        Else
            deltaBEP = COX_REF_BEP
            nStagesR = COX_REF_NSR
        End If
        If pBot <= 0 Then pBot = COX_REF_PBOT

        Dim theta As Double
        theta = CoxTheta(deltaBEP, pBot, nStagesR)
        ws.Cells(i, OUT_THETA).Value = theta

        Dim e1x As Double, e2x As Double
        e1x = ApplyCoxEta(e1, b1, theta)
        e2x = ApplyCoxEta(e2, b2, theta)

        ws.Cells(i, OUT_B50_COX).Value = LatentQuantile(0.5, w1, b1, e1x, b2, e2x)

        If flag = 0 Then
            Dim rulCox As Double
            rulCox = LatentRUL(ageDays, w1, b1, e1x, b2, e2x)
            ws.Cells(i, OUT_RUL_COX).Value = IIf(rulCox > 0, rulCox, "")
        Else
            ws.Cells(i, OUT_RUL_COX).Value = ""
        End If

NextRow:
    Next i

    ' Format output columns
    With ws.Range(ws.Cells(2, OUT_B10), ws.Cells(lastRow, OUT_HAZARD))
        .NumberFormat = "0.0"
    End With
    ws.Columns(OUT_DEGEN).NumberFormat = "General"
    With ws.Range(ws.Cells(2, OUT_THETA), ws.Cells(lastRow, OUT_THETA))
        .NumberFormat = "0.000"
    End With
    With ws.Range(ws.Cells(2, OUT_B50_COX), ws.Cells(lastRow, OUT_RUL_COX))
        .NumberFormat = "0.0"
    End With

    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.StatusBar = False

    MsgBox "Predictions complete: " & (lastRow - 1) & " rows in " & _
           Format(Timer - t0, "0.0") & "s.", vbInformation
    Exit Sub

ErrHandler:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.StatusBar = False
    MsgBox "Error at row " & i & ": " & Err.Description, vbCritical
End Sub

' ------------------------------------------------------------------
' RefreshModels — Force reload of the model registry.
' Call after editing the ESP_Models sheet.
' ------------------------------------------------------------------
Public Sub RefreshModels()
    RefreshRegistry
    MsgBox "Model registry reloaded.", vbInformation
End Sub

' ------------------------------------------------------------------
' ImportModelCSV — Load updated parameters from a CSV file.
' The CSV must have the same column order as ESP_Models sheet.
' Path: e.g. results\esp_survival_vba_models\YYYY-MM-DD\esp_models.csv
' ------------------------------------------------------------------
Public Sub ImportModelCSV(Optional ByVal csvPath As String = "")
    If csvPath = "" Then
        ' Show file picker
        Dim fd As FileDialog
        Set fd = Application.FileDialog(msoFileDialogFilePicker)
        fd.Title = "Select ESP model CSV"
        fd.Filters.Add "CSV files", "*.csv"
        If fd.Show = False Then Exit Sub
        csvPath = fd.SelectedItems(1)
    End If

    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets("ESP_Models")
    On Error GoTo 0
    If ws Is Nothing Then
        Call CreateModelSheet
        Set ws = ThisWorkbook.Worksheets("ESP_Models")
    End If

    ' Clear existing data (keep header)
    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    If lastRow > 1 Then ws.Rows("2:" & lastRow).Delete

    ' Read CSV line by line
    Dim fileNum As Integer
    fileNum = FreeFile
    Open csvPath For Input As #fileNum

    Dim lineText As String, fields() As String
    Dim rowNum As Long
    rowNum = 2
    Dim isHeader As Boolean
    isHeader = True

    Do While Not EOF(fileNum)
        Line Input #fileNum, lineText
        If isHeader Then
            isHeader = False  ' skip header row
        Else
            fields = Split(lineText, ",")
            Dim j As Integer
            For j = 0 To Application.WorksheetFunction.Min(UBound(fields), 11)
                ws.Cells(rowNum, j + 1).Value = Trim(Replace(fields(j), """", ""))
            Next j
            rowNum = rowNum + 1
        End If
    Loop
    Close #fileNum

    RefreshRegistry
    MsgBox "Imported " & (rowNum - 2) & " model rows from CSV.", vbInformation
End Sub

' ------------------------------------------------------------------
' ExportSurvivalCurve — Write S(t) table to a new sheet.
' Useful for plotting the model curve for a given stratum.
' ------------------------------------------------------------------
Public Sub ExportSurvivalCurve(ByVal field As String, _
                                ByVal h2s As String, _
                                ByVal contractor As String, _
                                Optional ByVal tMax As Double = 2000, _
                                Optional ByVal nPoints As Integer = 200)
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Dim h2sCls As String, ctrGrp As String

    h2sCls = H2SClass(h2s)
    ctrGrp = ContractorGroup(contractor)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"

    If Not LookupModel(Trim(field), h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        MsgBox "No model found for: " & field & " / " & h2s & " / " & contractor, vbExclamation
        Exit Sub
    End If

    ' Create or clear output sheet
    Dim wsOut As Worksheet
    Dim shName As String
    shName = Left("Curve_" & stratumKey, 31)

    On Error Resume Next
    Set wsOut = ThisWorkbook.Worksheets(shName)
    On Error GoTo 0
    If wsOut Is Nothing Then
        Set wsOut = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        wsOut.Name = shName
    Else
        wsOut.Cells.Clear
    End If

    ' Headers
    wsOut.Cells(1, 1).Value = "t_days"
    wsOut.Cells(1, 2).Value = "S(t)"
    wsOut.Cells(1, 3).Value = "C1_contrib"
    wsOut.Cells(1, 4).Value = "C2_contrib"
    wsOut.Cells(1, 5).Value = "Hazard"
    wsOut.Rows(1).Font.Bold = True

    Dim step As Double
    step = tMax / nPoints
    Dim i As Integer, t As Double
    For i = 0 To nPoints
        t = i * step
        wsOut.Cells(i + 2, 1).Value = t
        wsOut.Cells(i + 2, 2).Value = LatentSF(t, w1, b1, e1, b2, e2)
        wsOut.Cells(i + 2, 3).Value = w1 * WeibullSF(t, b1, e1)
        wsOut.Cells(i + 2, 4).Value = (1# - w1) * WeibullSF(t, b2, e2)
        wsOut.Cells(i + 2, 5).Value = LatentHazard(t, w1, b1, e1, b2, e2)
    Next i

    wsOut.Columns("A:E").AutoFit

    ' Add sparkline-style chart
    Dim chartObj As ChartObject
    Set chartObj = wsOut.ChartObjects.Add(Left:=wsOut.Columns("G").Left, _
                                          Top:=wsOut.Rows(1).Top, _
                                          Width:=400, Height:=250)
    With chartObj.Chart
        .ChartType = xlLine
        .SetSourceData Source:=wsOut.Range(wsOut.Cells(1, 1), wsOut.Cells(nPoints + 2, 2))
        .HasTitle = True
        .ChartTitle.Text = "Survival curve: " & stratumKey
        .Axes(xlCategory).HasTitle = True
        .Axes(xlCategory).AxisTitle.Text = "Days in service"
        .Axes(xlValue).HasTitle = True
        .Axes(xlValue).AxisTitle.Text = "S(t)"
        .Axes(xlValue).MinimumScale = 0
        .Axes(xlValue).MaximumScale = 1
    End With

    MsgBox "Survival curve for '" & stratumKey & "' exported to sheet '" & shName & "'.", _
           vbInformation
End Sub
