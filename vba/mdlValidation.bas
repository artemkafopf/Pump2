Attribute VB_Name = "mdlValidation"
Option Explicit

' ============================================================
' mdlValidation -- ValidateRegistry harness (agents/analyses/vba_model_v2.md T6).
'
' Writes a pass/fail table to the ESP_Validation scratch sheet (cell output --
' never MsgBox).  Run from the Immediate window:  ValidateRegistry
'
' Checks:
'   1. Registry row count / stratum coverage vs the ESP_Version manifest.
'   2. Every stratum: VBA-bisection B50 vs the imported b50 column (tol 1%).
'   3. theta==1 equivalence: ESP_B50_Cox == ESP_B50 (deprecate-keep, Cox shelved).
'   4. Clock spot-checks against imported values (not hardcoded numbers) +
'      Global_Pooled fallback resolution.
'   5. Deliberate failure probe: ImportModelCSV must REFUSE a clock=run_days file.
' ============================================================

Private Const VAL_SHEET As String = "ESP_Validation"
Private Const TOL As Double = 0.01   ' 1% relative tolerance

Private Function VersionValue(ByVal key As String) As String
    On Error GoTo NoVal
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("ESP_Version")
    Dim lastRow As Long, i As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    For i = 1 To lastRow
        If LCase(Trim(CStr(ws.Cells(i, 1).Value))) = LCase(key) Then
            VersionValue = Trim(CStr(ws.Cells(i, 2).Value))
            Exit Function
        End If
    Next i
NoVal:
    VersionValue = ""
End Function

Private Sub AddResult(ByVal ws As Worksheet, ByRef r As Long, _
                      ByVal name As String, ByVal detail As String, _
                      ByVal passed As Boolean, ByRef nFail As Long)
    ws.Cells(r, 1).Value = name
    ws.Cells(r, 2).Value = detail
    ws.Cells(r, 3).Value = IIf(passed, "PASS", "FAIL")
    If Not passed Then
        nFail = nFail + 1
        ws.Cells(r, 3).Interior.Color = RGB(255, 199, 206)
    Else
        ws.Cells(r, 3).Interior.Color = RGB(198, 239, 206)
    End If
    r = r + 1
End Sub

Private Function RelClose(ByVal a As Double, ByVal b As Double) As Boolean
    If b = 0 Then RelClose = (Abs(a) < 1E-06) : Exit Function
    RelClose = (Abs(a - b) / Abs(b) <= TOL)
End Function

Public Sub ValidateRegistry()
    Dim ws As Worksheet
    Dim r As Long, nFail As Long
    Dim nRows As Long
    Dim manRows As String
    Dim i As Long, nChecked As Long, nB50Fail As Long
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double, b50 As Double
    Dim recomputed As Double
    Dim eqOK As Boolean
    Dim fallbackKey As String
    Dim beforeN As Long, afterN As Long, probePath As String

    If Not PrepWorkbook() Then Exit Sub
    Set ws = FreshSheet(VAL_SHEET)   ' robust to a protected existing sheet
    If ws Is Nothing Then Exit Sub

    ws.Cells(1, 1).Value = "check"
    ws.Cells(1, 2).Value = "detail"
    ws.Cells(1, 3).Value = "result"
    ws.Rows(1).Font.Bold = True

    On Error GoTo FailHandler

    RefreshRegistry
    RefreshModeMix

    r = 2 : nFail = 0
    nRows = RegistryRowCount()

    ' ── Check 1: row count / coverage vs manifest ────────────────────────────
    manRows = VersionValue("esp_models rows")
    If manRows = "" Then
        AddResult ws, r, "1. row count vs manifest", _
                  "ESP_Version not imported; loaded rows=" & nRows & " (manifest check skipped)", _
                  (nRows > 0), nFail
    Else
        AddResult ws, r, "1. row count vs manifest", _
                  "loaded=" & nRows & " manifest=" & manRows, _
                  (CStr(nRows) = Trim(manRows)), nFail
    End If

    ' Clock stamp is ttf_mix
    AddResult ws, r, "1b. registry clock", "clock=" & RegistryClock(), _
              (LCase(RegistryClock()) = "ttf_mix"), nFail

    ' ── Check 2: B50 bisection vs imported b50 (every stratum) ────────────────
    nChecked = 0 : nB50Fail = 0
    For i = 1 To nRows
        If RegistryParamsAt(i, w1, b1, e1, b2, e2, b50) Then
            If b50 > 0 Then
                recomputed = LatentQuantile(0.5, w1, b1, e1, b2, e2)
                nChecked = nChecked + 1
                If Not RelClose(recomputed, b50) Then
                    nB50Fail = nB50Fail + 1
                    AddResult ws, r, "2. B50 drift", _
                              RegistryStratumAt(i) & ": VBA=" & Format(recomputed, "0.0") & _
                              " imported=" & Format(b50, "0.0"), False, nFail
                End If
            End If
        End If
    Next i
    AddResult ws, r, "2. B50 bisection vs imported (all strata, tol 1%)", _
              nChecked & " checked, " & nB50Fail & " outside tol", (nB50Fail = 0), nFail

    ' ── Check 2b: B20 <= B50 <= B80 ordering (every stratum) ──────────────────
    Dim nOrdFail As Long
    nOrdFail = 0
    For i = 1 To nRows
        If RegistryParamsAt(i, w1, b1, e1, b2, e2, b50) Then
            If b50 > 0 Then
                Dim q20 As Double, q50 As Double, q80 As Double
                q20 = LatentQuantile(0.2, w1, b1, e1, b2, e2)
                q50 = LatentQuantile(0.5, w1, b1, e1, b2, e2)
                q80 = LatentQuantile(0.8, w1, b1, e1, b2, e2)
                If Not (q20 <= q50 + 0.001 And q50 <= q80 + 0.001) Then nOrdFail = nOrdFail + 1
            End If
        End If
    Next i
    AddResult ws, r, "2b. quantile ordering B20<=B50<=B80", _
              nChecked & " checked, " & nOrdFail & " out of order", (nOrdFail = 0), nFail

    ' ── Check 3: theta==1 equivalence (legacy 3-covariate Cox RETIRED) ────────
    eqOK = ThetaEquiv("Ya", "nonsour", "") And _
           ThetaEquiv("Vt", "sour", "slb") And _
           ThetaEquiv("Az", "nonsour", "")
    AddResult ws, r, "3. ESP_B50_Cox == ESP_B50 (theta==1)", _
              "Cox enabled=" & CBool(CoxEnabled()), eqOK, nFail

    ' ── Check 4: clock spot-checks vs imported values ────────────────────────
    SpotCheck ws, r, nFail, "Ya", "nonsour", "brt"
    SpotCheck ws, r, nFail, "Vt", "sour", "slb"
    SpotCheck ws, r, nFail, "Ya", "nonsour", "oth"

    ' Global_Pooled fallback for an unmodelled field. Use the core resolver here
    ' instead of the worksheet wrapper to avoid UDF-in-macro break-state issues.
    fallbackKey = ResolveStratumKey("Bt", "nonsour", "Pooled")
    AddResult ws, r, "4c. Global_Pooled fallback", _
              "ESP_Stratum('Bt','nonsour','')=" & fallbackKey, _
              (fallbackKey = "Global_Pooled"), nFail

    ' ── Check 5: doctored-CSV refusal probe (clock=run_days) ─────────────────
    beforeN = RegistryRowCount()
    probePath = Environ("TEMP") & "\esp_models_doctored_probe.csv"
    WriteDoctoredCsv probePath
    Call ImportModelCSV(probePath)   ' must refuse -> ESP_Models untouched
    RefreshRegistry
    afterN = RegistryRowCount()
    On Error Resume Next
    Kill probePath
    On Error GoTo 0
    AddResult ws, r, "5. import refusal (clock=run_days)", _
              "rows before=" & beforeN & " after=" & afterN & " (unchanged => refused)", _
              (beforeN = afterN And beforeN > 0), nFail

    ' ── Check 6: hazard layer status (informational; layer is optional) ───────
    RefreshHazardLayer
    Dim hzN As Long
    hzN = HazardCovariateCount()
    If hzN = 0 Then
        AddResult ws, r, "6. hazard layer", _
                  "ESP_CoxCoeffs not imported (operational/completion overlay optional)", _
                  True, nFail
    Else
        ' When disabled, the multiplier must be neutral (theta = 1); when enabled it
        ' must at least be a finite positive number. Probe a synthetic key (falls back
        ' to theta = 1 for an unknown run either way -> always neutral here).
        AddResult ws, r, "6. hazard layer", _
                  "loaded " & hzN & " covariates; enabled=" & HazardEnabled() & _
                  " (SENSITIVITY overlay -> ESP_RUL_sens / ESP_dRUL_*; not OOS-validated). " & _
                  "Unknown-run theta=" & Format(HazardThetaForKey("__none__", 1), "0.000"), _
                  (HazardThetaForKey("__none__", 1) > 0), nFail
    End If

    ' ── Summary ──────────────────────────────────────────────────────────────
    r = r + 1
    ws.Cells(r, 1).Value = "SUMMARY"
    ws.Cells(r, 2).Value = IIf(nFail = 0, "ALL GREEN", nFail & " FAILURE(S)")
    ws.Cells(r, 3).Value = IIf(nFail = 0, "PASS", "FAIL")
    ws.Cells(r, 1).Font.Bold = True
    ws.Cells(r, 3).Font.Bold = True
    ws.Cells(r, 3).Interior.Color = IIf(nFail = 0, RGB(198, 239, 206), RGB(255, 199, 206))

    ws.Columns("A:C").AutoFit
    ws.Activate
    Exit Sub

FailHandler:
    ws.Cells(r, 1).Value = "X. internal validation error"
    ws.Cells(r, 2).Value = "Err " & Err.Number & ": " & Err.Description
    ws.Cells(r, 3).Value = "FAIL"
    ws.Cells(r, 3).Interior.Color = RGB(255, 199, 206)
    ws.Columns("A:C").AutoFit
    ws.Activate
End Sub

' ESP_B50_Cox (all covariates at 0) must equal ESP_B50 exactly while Cox is shelved.
Private Function ThetaEquiv(ByVal f As String, ByVal h As String, ByVal c As String) As Boolean
    Dim a As Variant, b As Variant
    a = ESP_B50(f, h, c)
    b = ESP_B50_Cox(f, h, c, 0#, 0#, 0#)
    If IsError(a) Or IsError(b) Then ThetaEquiv = False : Exit Function
    ThetaEquiv = (Abs(CDbl(a) - CDbl(b)) < 1E-06)
End Function

' ESP_B50 (UDF path, Latin-code resolution) vs the imported b50 column.
Private Sub SpotCheck(ByVal ws As Worksheet, ByRef r As Long, ByRef nFail As Long, _
                      ByVal f As String, ByVal h As String, ByVal c As String)
    Dim modelB50 As Double, b50Imp As Double
    Dim h2sCls As String, ctrGrp As String
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Dim ok As Boolean, detail As String

    On Error GoTo SpotFail

    h2sCls = H2SClass(h)
    ctrGrp = ContractorGroup(c)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not GetParams(f, h, c, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        AddResult ws, r, "4. spot-check " & f & "/" & h & "/" & c, _
                  f & "_" & h2sCls & "_" & ctrGrp & ": model params unavailable", _
                  False, nFail
        Exit Sub
    End If
    modelB50 = LatentQuantile(0.5, w1, b1, e1, b2, e2)
    b50Imp = ImportedB50(f, h2sCls, ctrGrp)
    If b50Imp <= 0 Then
        ok = False
        detail = f & "_" & h2sCls & "_" & ctrGrp & ": imported B50 unavailable"
    Else
        ok = RelClose(modelB50, b50Imp)
        detail = f & "_" & h2sCls & "_" & ctrGrp & ": ESP_B50=" & Format(modelB50, "0.0") & _
                 " imported=" & Format(b50Imp, "0.0")
    End If
    AddResult ws, r, "4. spot-check " & f & "/" & h & "/" & c, detail, ok, nFail
    Exit Sub

SpotFail:
    AddResult ws, r, "4. spot-check " & f & "/" & h & "/" & c, _
              "ERR " & Err.Number & ": " & Err.Description, False, nFail
End Sub

' Write a schema-valid ESP models CSV whose rows carry clock=run_days (must be refused).
Private Sub WriteDoctoredCsv(ByVal path As String)
    Dim fn As Integer
    fn = FreeFile
    Open path For Output As #fn
    Print #fn, "stratum,field,h2s_class,contractor_group,model_kind,w1,beta1,eta1,beta2,eta2,b20,b50,b80,b50_lo,b50_hi,uptime_factor,n_runs,n_failures,pct_mixed_clock,clock,fit_date"
    Print #fn, "Ya_nonsour_Pooled,Ya,nonsour,Pooled,k2,0.5,0.9,140,1.5,800,15,250,1200,200,300,0.8,1281,731,0.19,run_days,2026-07-07"
    Close #fn
End Sub
