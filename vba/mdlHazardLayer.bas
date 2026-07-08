Attribute VB_Name = "mdlHazardLayer"
Option Explicit

' ============================================================
' mdlHazardLayer -- operational + completion hazard overlay (Prompt A/B).
'
' Sheet-driven, N-covariate version of the shelved theta layer. Reads:
'   ESP_CoxCoeffs  (covariate, beta, reference_value, mode, enabled, clock)
'   ESP_RunCov     (well_key_norm, run_seq, <covariate columns>, covariate_available)
' Both imported by ImportBundle from esp_cox_coeffs.csv / esp_run_covariates.csv
' (backend/analysis/workflows/esp_survival/hazard_layer.py).
'
' STATUS: the operational+completion subset FAILED the out-of-sample gate (it
' forecasts worse than the stratum baseline OOS -- see the hazard manifest and
' model report S3.8). So `enabled` ships FALSE and the layer is DISPLAY-ONLY:
'   * ESP_HazardFlags surfaces the modes a run's operating covariates would elevate
'     (advisory: spares/crew/ops-policy hint), and NEVER changes B50/RUL.
' The multiplier machinery is kept intact for a future theta: flip the ESP_CoxCoeffs
' `enabled` column to TRUE (only if a re-fit passes the OOS gate) and HazardTheta
' eta-rescales the baseline via mdlCoxHR.ApplyCoxEta.
'
' Flag rule: a covariate flags its mode when its would-be hazard multiplier
' exp(beta*(x-ref)) >= FLAG_HR (1.15, i.e. +15%). Display only.
'
' VBA rules honored: ChrW only; no MsgBox in UDF paths; no reserved-word variable
' names; Option Explicit; Private module state via public accessors.
' ============================================================

Private Const COEFF_SHEET As String = "ESP_CoxCoeffs"
Private Const RUNCOV_SHEET As String = "ESP_RunCov"
Private Const FLAG_HR As Double = 1.15
Private Const THETA_MIN As Double = 0.05
Private Const THETA_MAX As Double = 20#

' Coefficient cache
Private hCov()  As String
Private hBeta() As Double
Private hRef()  As Double
Private hMode() As String
Private hGroup() As String
Private hNum    As Long
Private hEnabled As Boolean

' Run-covariate cache: key = well_key_norm & "|" & run_seq -> Dictionary(covName -> Double)
Private mRunCov  As Object
Private mHzLoaded As Boolean

' ---- header helpers (BOM-safe, matching the other modules) ----
Private Function CleanHdr(ByVal s As String) As String
    s = Trim(s)
    If Len(s) >= 3 Then
        If AscW(Mid(s, 1, 1)) = 239 And AscW(Mid(s, 2, 1)) = 187 And AscW(Mid(s, 3, 1)) = 191 Then s = Mid(s, 4)
    End If
    If Len(s) > 0 Then
        If AscW(Left(s, 1)) = 65279 Then s = Mid(s, 2)
    End If
    CleanHdr = LCase(Trim(s))
End Function

Private Function HeaderCol(ByVal ws As Worksheet, ByVal name As String) As Long
    Dim lastCol As Long, c As Long
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column
    For c = 1 To lastCol
        If CleanHdr(CStr(ws.Cells(1, c).Value)) = LCase(name) Then HeaderCol = c : Exit Function
    Next c
    HeaderCol = 0
End Function

Private Function SafeDbl(ByVal v As Variant) As Double
    On Error Resume Next
    Dim s As String
    s = Trim(CStr(v))
    If s = "" Then SafeDbl = 0# Else SafeDbl = CDbl(Val(s))
    On Error GoTo 0
End Function

' ------------------------------------------------------------------
' Load ESP_CoxCoeffs + ESP_RunCov into caches. Silent if sheets absent.
' ------------------------------------------------------------------
Public Sub LoadHazardLayer()
    hNum = 0 : hEnabled = False
    Set mRunCov = CreateObject("Scripting.Dictionary")
    mHzLoaded = False

    Dim wc As Worksheet
    On Error Resume Next
    Set wc = ThisWorkbook.Worksheets(COEFF_SHEET)
    On Error GoTo 0
    If wc Is Nothing Then Exit Sub

    Dim cCov As Long, cBeta As Long, cRef As Long, cMode As Long, cEn As Long, cGrp As Long
    cCov = HeaderCol(wc, "covariate")
    cBeta = HeaderCol(wc, "beta")
    cRef = HeaderCol(wc, "reference_value")
    cMode = HeaderCol(wc, "mode")
    cGrp = HeaderCol(wc, "hazard_group")
    cEn = HeaderCol(wc, "enabled")
    If cCov = 0 Or cBeta = 0 Or cRef = 0 Then Exit Sub

    Dim lastRow As Long, i As Long
    lastRow = wc.Cells(wc.Rows.Count, cCov).End(xlUp).Row
    If lastRow < 2 Then Exit Sub

    ReDim hCov(1 To lastRow - 1)
    ReDim hBeta(1 To lastRow - 1)
    ReDim hRef(1 To lastRow - 1)
    ReDim hMode(1 To lastRow - 1)
    ReDim hGroup(1 To lastRow - 1)
    For i = 2 To lastRow
        If Trim(CStr(wc.Cells(i, cCov).Value)) = "" Then GoTo NextC
        hNum = hNum + 1
        hCov(hNum) = LCase(Trim(CStr(wc.Cells(i, cCov).Value)))
        hBeta(hNum) = SafeDbl(wc.Cells(i, cBeta).Value)
        hRef(hNum) = SafeDbl(wc.Cells(i, cRef).Value)
        hMode(hNum) = IIf(cMode > 0, LCase(Trim(CStr(wc.Cells(i, cMode).Value))), "")
        hGroup(hNum) = IIf(cGrp > 0, LCase(Trim(CStr(wc.Cells(i, cGrp).Value))), "")
        If cEn > 0 Then
            Dim ev As String
            ev = UCase(Trim(CStr(wc.Cells(i, cEn).Value)))
            If ev = "TRUE" Or ev = "1" Or ev = "-1" Or ev = "YES" Then hEnabled = True
        End If
NextC:
    Next i

    ' ── run covariates ───────────────────────────────────────────────────────
    Dim wr As Worksheet
    On Error Resume Next
    Set wr = ThisWorkbook.Worksheets(RUNCOV_SHEET)
    On Error GoTo 0
    If Not wr Is Nothing And hNum > 0 Then
        Dim cWell As Long, cSeq As Long
        cWell = HeaderCol(wr, "well_key_norm")
        cSeq = HeaderCol(wr, "run_seq")
        If cWell > 0 And cSeq > 0 Then
            ' column index for each coefficient covariate in ESP_RunCov
            Dim covCol() As Long
            ReDim covCol(1 To hNum)
            Dim j As Long
            For j = 1 To hNum
                covCol(j) = HeaderCol(wr, hCov(j))
            Next j
            Dim lastR As Long, r As Long
            lastR = wr.Cells(wr.Rows.Count, cWell).End(xlUp).Row
            For r = 2 To lastR
                Dim wk As String, key As String
                wk = LCase(Trim(CStr(wr.Cells(r, cWell).Value)))
                If wk = "" Then GoTo NextR
                key = wk & "|" & Trim(CStr(wr.Cells(r, cSeq).Value))
                Dim d As Object
                Set d = CreateObject("Scripting.Dictionary")
                For j = 1 To hNum
                    If covCol(j) > 0 Then
                        Dim cell As String
                        cell = Trim(CStr(wr.Cells(r, covCol(j)).Value))
                        If cell <> "" And IsNumeric(cell) Then d(hCov(j)) = CDbl(cell)
                    End If
                Next j
                If Not mRunCov.Exists(key) Then mRunCov.Add key, d
            Next r
NextR:
        End If
    End If

    mHzLoaded = (hNum > 0)
End Sub

Public Sub EnsureHazardLoaded()
    If Not mHzLoaded Then LoadHazardLayer
End Sub

Public Sub RefreshHazardLayer()
    mHzLoaded = False
    LoadHazardLayer
End Sub

Public Function HazardEnabled() As Boolean
    EnsureHazardLoaded
    HazardEnabled = hEnabled
End Function

Public Function HazardCovariateCount() As Long
    EnsureHazardLoaded
    HazardCovariateCount = hNum
End Function

' ------------------------------------------------------------------
' HazardThetaForKey -- gated hazard multiplier for a run.
' Returns 1 unless the layer is enabled (deprecate-keep). Missing covariates
' contribute 0 (fall back to reference). Clamped.
' ------------------------------------------------------------------
' Core: theta over the covariates in `group` (empty = all groups). Missing covariate
' contributes 0 (fall back to reference). Gated by `enabled`; clamped.
Private Function ThetaFromKey(ByVal wellNorm As String, ByVal runSeq As Variant, _
                             ByVal group As String) As Double
    EnsureHazardLoaded
    If Not hEnabled Or hNum = 0 Then ThetaFromKey = 1# : Exit Function
    Dim key As String, lp As Double, i As Long
    key = LCase(Trim(wellNorm)) & "|" & Trim(CStr(runSeq))
    If Not mRunCov.Exists(key) Then ThetaFromKey = 1# : Exit Function
    Dim d As Object, g As String
    Set d = mRunCov(key)
    g = LCase(Trim(group))
    lp = 0#
    For i = 1 To hNum
        If (g = "" Or hGroup(i) = g) Then
            If d.Exists(hCov(i)) Then lp = lp + hBeta(i) * (CDbl(d(hCov(i))) - hRef(i))
        End If
    Next i
    Dim th As Double
    th = Exp(lp)
    If th < THETA_MIN Then th = THETA_MIN
    If th > THETA_MAX Then th = THETA_MAX
    ThetaFromKey = th
End Function

Public Function HazardThetaForKey(ByVal wellNorm As String, ByVal runSeq As Variant) As Double
    HazardThetaForKey = ThetaFromKey(wellNorm, runSeq, "")
End Function

Public Function HazardThetaGroupForKey(ByVal wellNorm As String, ByVal runSeq As Variant, _
                                       ByVal group As String) As Double
    HazardThetaGroupForKey = ThetaFromKey(wellNorm, runSeq, group)
End Function

' ------------------------------------------------------------------
' Hazard-adjusted RUL / B50 for a run: rescale the stratum baseline's etas by the
' group theta (group="" = all) via mdlCoxHR.ApplyCoxEta. Returns -1 if no model.
' The baseline itself is never modified (these feed the *_sens columns only).
' ------------------------------------------------------------------
Public Function HazardAdjustedRUL(ByVal field As String, ByVal h2s As String, _
                                  ByVal ctr As String, ByVal age As Double, _
                                  ByVal wellNorm As String, ByVal runSeq As Variant, _
                                  ByVal group As String) As Double
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean, h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s) : ctrGrp = ContractorGroup(ctr)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        HazardAdjustedRUL = -1# : Exit Function
    End If
    If age < 0 Then age = 0
    Dim th As Double
    th = ThetaFromKey(wellNorm, runSeq, group)
    HazardAdjustedRUL = LatentRUL(age, w1, b1, ApplyCoxEta(e1, b1, th), b2, ApplyCoxEta(e2, b2, th))
End Function

Public Function HazardAdjustedB50(ByVal field As String, ByVal h2s As String, _
                                  ByVal ctr As String, ByVal wellNorm As String, _
                                  ByVal runSeq As Variant, ByVal group As String) As Double
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean, h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s) : ctrGrp = ContractorGroup(ctr)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        HazardAdjustedB50 = -1# : Exit Function
    End If
    Dim th As Double
    th = ThetaFromKey(wellNorm, runSeq, group)
    HazardAdjustedB50 = LatentQuantile(0.5, w1, b1, ApplyCoxEta(e1, b1, th), b2, ApplyCoxEta(e2, b2, th))
End Function

' ------------------------------------------------------------------
' HazardFlagsForKey -- advisory: distinct modes a run's covariates elevate
' (would-be per-covariate HR >= FLAG_HR). Computed regardless of `enabled`; it
' NEVER changes survival numbers. Returns "" when nothing is elevated.
' ------------------------------------------------------------------
Public Function HazardFlagsForKey(ByVal wellNorm As String, ByVal runSeq As Variant) As String
    EnsureHazardLoaded
    If hNum = 0 Then HazardFlagsForKey = "" : Exit Function
    Dim key As String
    key = LCase(Trim(wellNorm)) & "|" & Trim(CStr(runSeq))
    If Not mRunCov.Exists(key) Then HazardFlagsForKey = "" : Exit Function
    Dim d As Object
    Set d = mRunCov(key)

    Dim seen As Object
    Set seen = CreateObject("Scripting.Dictionary")
    Dim i As Long
    For i = 1 To hNum
        If d.Exists(hCov(i)) Then
            Dim hr As Double
            hr = Exp(hBeta(i) * (CDbl(d(hCov(i))) - hRef(i)))
            If hr >= FLAG_HR And hMode(i) <> "" And hMode(i) <> "all-cause" Then
                If Not seen.Exists(hMode(i)) Then seen.Add hMode(i), True
            End If
        End If
    Next i

    Dim out As String, k As Variant
    out = ""
    For Each k In seen.Keys
        out = out & IIf(out = "", "", "; ") & CStr(k)
    Next k
    HazardFlagsForKey = out
End Function

' ================================================================
' Worksheet UDFs
' ================================================================

' =ESP_HazardFlags(<well cell>, <run_seq>) -- advisory elevated modes (display only).
Public Function ESP_HazardFlags(ByVal wellCell As Range, ByVal runSeq As Variant) As Variant
    On Error GoTo ErrHandler
    ESP_HazardFlags = HazardFlagsForKey(LCase(Trim(CStr(wellCell.Cells(1, 1).Value))), runSeq)
    Exit Function
ErrHandler:
    ESP_HazardFlags = CVErr(xlErrValue)
End Function

' =ESP_HazardTheta(<well cell>, <run_seq>) -- gated multiplier (1 unless enabled).
Public Function ESP_HazardTheta(ByVal wellCell As Range, ByVal runSeq As Variant) As Variant
    On Error GoTo ErrHandler
    ESP_HazardTheta = HazardThetaForKey(LCase(Trim(CStr(wellCell.Cells(1, 1).Value))), runSeq)
    Exit Function
ErrHandler:
    ESP_HazardTheta = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' delta_RUL / delta_B50 for one hazard group (one-group-at-a-time marginal):
' adjusted minus baseline, in OPERATING days. + = longer life than baseline,
' - = shorter. Zero at the reference profile / when the layer is disabled.
' SENSITIVITY (USER-OVERRIDE, not OOS-validated).
'   =ESP_dRUL(A2, BW2, D2, I2, <well cell>, <run_seq>, "kpod")
'   =ESP_dB50(A2, BW2, D2, <well cell>, <run_seq>, "vintage")
' ------------------------------------------------------------------
Public Function ESP_dRUL(ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                         ByVal age As Double, ByVal wellCell As Range, _
                         ByVal runSeq As Variant, ByVal group As String) As Variant
    On Error GoTo ErrHandler
    Dim wn As String
    wn = LCase(Trim(CStr(wellCell.Cells(1, 1).Value)))
    Dim adj As Double, base As Double
    adj = HazardAdjustedRUL(field, h2s, ctr, age, wn, runSeq, group)
    base = HazardAdjustedRUL(field, h2s, ctr, age, wn, runSeq, "__baseline__")   ' theta=1
    If adj < 0 Or base < 0 Then ESP_dRUL = CVErr(xlErrNA) : Exit Function
    ESP_dRUL = adj - base
    Exit Function
ErrHandler:
    ESP_dRUL = CVErr(xlErrValue)
End Function

Public Function ESP_dB50(ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                         ByVal wellCell As Range, ByVal runSeq As Variant, _
                         ByVal group As String) As Variant
    On Error GoTo ErrHandler
    Dim wn As String
    wn = LCase(Trim(CStr(wellCell.Cells(1, 1).Value)))
    Dim adj As Double, base As Double
    adj = HazardAdjustedB50(field, h2s, ctr, wn, runSeq, group)
    base = HazardAdjustedB50(field, h2s, ctr, wn, runSeq, "__baseline__")
    If adj < 0 Or base < 0 Then ESP_dB50 = CVErr(xlErrNA) : Exit Function
    ESP_dB50 = adj - base
    Exit Function
ErrHandler:
    ESP_dB50 = CVErr(xlErrValue)
End Function

' Public group list (order = RunPredictions delta columns).
Public Function HazardGroupList() As Variant
    HazardGroupList = Array("GLF", "load", "kpod", "frequency", "curvature", _
                            "well_history", "vintage")
End Function
