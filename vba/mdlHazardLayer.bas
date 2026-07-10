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

Private Function TryGetNumeric(ByVal v As Variant, ByRef outVal As Double) As Boolean
    On Error GoTo FailNum
    If IsError(v) Then Exit Function
    If IsEmpty(v) Then Exit Function
    Dim s As String
    s = Trim(CStr(v))
    If s = "" Then Exit Function
    If Not IsNumeric(s) Then Exit Function
    outVal = CDbl(Val(s))
    TryGetNumeric = True
    Exit Function
FailNum:
    TryGetNumeric = False
End Function

Private Function Log1pSafe(ByVal v As Variant) As Double
    Dim x As Double
    x = SafeDbl(v)
    If x <= -1# Then
        Log1pSafe = 0#
    Else
        Log1pSafe = Log(1# + x)
    End If
End Function

Private Function Clamp01(ByVal v As Variant) As Double
    Dim x As Double
    x = SafeDbl(v)
    If x < 0# Then x = 0#
    If x > 1# Then x = 1#
    Clamp01 = x
End Function

Private Function YearFromVariant(ByVal v As Variant) As Long
    On Error GoTo FailYear
    If IsDate(v) Then
        YearFromVariant = Year(CDate(v))
        Exit Function
    End If
    Dim s As String, y As Long
    s = Trim(CStr(v))
    If Len(s) >= 4 Then
        y = CLng(Val(Left$(s, 4)))
        If y >= 1900 And y <= 2100 Then
            YearFromVariant = y
            Exit Function
        End If
    End If
FailYear:
    YearFromVariant = 0
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

Private Function BuildDirectCovDict(ByVal glf As Variant, _
                                    ByVal loadStd As Variant, _
                                    ByVal loadMean As Variant, _
                                    ByVal fracKpodUnder As Variant, _
                                    ByVal kpodMean As Variant, _
                                    ByVal freqAbove55Pct As Variant, _
                                    ByVal freqSteps As Variant, _
                                    ByVal curvatureDeg10m As Variant, _
                                    ByVal runSeq As Variant, _
                                    ByVal daysSincePrevFailure As Variant, _
                                    ByVal installDate As Variant) As Object
    Dim d As Object
    Dim x As Double
    Set d = CreateObject("Scripting.Dictionary")
    ' Missing / non-parseable values are omitted so that this covariate
    ' contributes a neutral "no hazard adjustment" instead of acting like 0.
    If TryGetNumeric(glf, x) Then
        If x > -1# Then d("log_glf_mean_opdays") = Log(1# + x)
    End If
    If TryGetNumeric(loadStd, x) Then d("load_std_early") = x
    If TryGetNumeric(loadMean, x) Then d("load_mean") = x
    If TryGetNumeric(fracKpodUnder, x) Then d("frac_kpod_below_0p7") = Clamp01(x)
    If TryGetNumeric(kpodMean, x) Then d("kpod_freq_mean") = x
    If TryGetNumeric(freqAbove55Pct, x) Then d("freq_above_55hz_pct_early") = x
    If TryGetNumeric(freqSteps, x) Then d("n_freq_steps_per_100d") = x
    If TryGetNumeric(curvatureDeg10m, x) Then d("curvature_deg10m") = x
    If TryGetNumeric(runSeq, x) Then
        If x > -1# Then d("log_run_seq") = Log(1# + x)
    End If
    If TryGetNumeric(daysSincePrevFailure, x) Then
        If x > -1# Then d("log_days_since_prev_failure") = Log(1# + x)
    End If

    Dim y As Long
    y = YearFromVariant(installDate)
    If y > 0 Then
        d("install_pre2020") = IIf(y <= 2019, 1#, 0#)
        d("install_2023plus") = IIf(y >= 2023, 1#, 0#)
    End If
    Set BuildDirectCovDict = d
End Function

Private Function ThetaFromDict(ByVal d As Object, ByVal group As String, _
                               Optional ByVal gateEnabled As Boolean = True) As Double
    EnsureHazardLoaded
    If hNum = 0 Then ThetaFromDict = 1# : Exit Function
    If gateEnabled And Not hEnabled Then ThetaFromDict = 1# : Exit Function

    Dim lp As Double, i As Long, g As String
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
    ThetaFromDict = th
End Function

Private Function HazardFlagsFromDict(ByVal d As Object) As String
    EnsureHazardLoaded
    If hNum = 0 Then HazardFlagsFromDict = "" : Exit Function

    Dim seen As Object
    Set seen = CreateObject("Scripting.Dictionary")
    Dim i As Long, hr As Double
    For i = 1 To hNum
        If d.Exists(hCov(i)) Then
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
    HazardFlagsFromDict = out
End Function

Private Function HazardModeRu(ByVal modeName As String) As String
    Select Case LCase(Trim(modeName))
        Case "hydraulic"
            HazardModeRu = ChrW(1043) & ChrW(1080) & ChrW(1076) & ChrW(1088) & ChrW(1072) & ChrW(1074) & ChrW(1083) & ChrW(1080) & ChrW(1095) & ChrW(1077) & ChrW(1089) & ChrW(1082) & ChrW(1080) & ChrW(1081)
        Case "electro-thermal"
            HazardModeRu = ChrW(1069) & ChrW(1083) & ChrW(1077) & ChrW(1082) & ChrW(1090) & ChrW(1088) & ChrW(1086) & ChrW(1090) & ChrW(1077) & ChrW(1087) & ChrW(1083) & ChrW(1086) & ChrW(1074) & ChrW(1086) & ChrW(1081)
        Case "protector"
            HazardModeRu = ChrW(1043) & ChrW(1080) & ChrW(1076) & ChrW(1088) & ChrW(1086) & ChrW(1079) & ChrW(1072) & ChrW(1097) & ChrW(1080) & ChrW(1090) & ChrW(1072)
        Case "other"
            HazardModeRu = ChrW(1055) & ChrW(1088) & ChrW(1086) & ChrW(1095) & ChrW(1077) & ChrW(1077)
        Case Else
            HazardModeRu = modeName
    End Select
End Function

Private Function HazardFlagsRuText(ByVal flagsText As String) As String
    Dim parts() As String, i As Long, out As String
    flagsText = Trim(flagsText)
    If flagsText = "" Then HazardFlagsRuText = "" : Exit Function
    parts = Split(flagsText, ";")
    out = ""
    For i = LBound(parts) To UBound(parts)
        out = out & IIf(out = "", "", "; ") & HazardModeRu(Trim(parts(i)))
    Next i
    HazardFlagsRuText = out
End Function

Public Function HazardThetaForKey(ByVal wellNorm As String, ByVal runSeq As Variant) As Double
    HazardThetaForKey = ThetaFromKey(wellNorm, runSeq, "")
End Function

Public Function HazardThetaGroupForKey(ByVal wellNorm As String, ByVal runSeq As Variant, _
                                       ByVal group As String) As Double
    HazardThetaGroupForKey = ThetaFromKey(wellNorm, runSeq, group)
End Function

Public Function HazardThetaDirect(ByVal glf As Variant, _
                                  ByVal loadStd As Variant, _
                                  ByVal loadMean As Variant, _
                                  ByVal fracKpodUnder As Variant, _
                                  ByVal kpodMean As Variant, _
                                  ByVal freqAbove55Pct As Variant, _
                                  ByVal freqSteps As Variant, _
                                  ByVal curvatureDeg10m As Variant, _
                                  ByVal runSeq As Variant, _
                                  ByVal daysSincePrevFailure As Variant, _
                                  ByVal installDate As Variant) As Double
    Dim d As Object
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    HazardThetaDirect = ThetaFromDict(d, "", True)
End Function

Public Function HazardThetaGroupDirect(ByVal group As String, _
                                       ByVal glf As Variant, _
                                       ByVal loadStd As Variant, _
                                       ByVal loadMean As Variant, _
                                       ByVal fracKpodUnder As Variant, _
                                       ByVal kpodMean As Variant, _
                                       ByVal freqAbove55Pct As Variant, _
                                       ByVal freqSteps As Variant, _
                                       ByVal curvatureDeg10m As Variant, _
                                       ByVal runSeq As Variant, _
                                       ByVal daysSincePrevFailure As Variant, _
                                       ByVal installDate As Variant) As Double
    Dim d As Object
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    HazardThetaGroupDirect = ThetaFromDict(d, group, True)
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

Public Function HazardAdjustedRULDirect(ByVal age As Double, _
                                        ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                        ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                        ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                        ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                        ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                        ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant, _
                                        Optional ByVal group As String = "") As Double
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean, h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s) : ctrGrp = ContractorGroup(ctr)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        HazardAdjustedRULDirect = -1# : Exit Function
    End If
    If age < 0 Then age = 0
    Dim d As Object, th As Double
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    th = ThetaFromDict(d, group, True)
    HazardAdjustedRULDirect = LatentRUL(age, w1, b1, ApplyCoxEta(e1, b1, th), b2, ApplyCoxEta(e2, b2, th))
End Function

Public Function HazardAdjustedB50Direct(ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                        ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                        ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                        ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                        ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                        ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant, _
                                        Optional ByVal group As String = "") As Double
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean, h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s) : ctrGrp = ContractorGroup(ctr)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        HazardAdjustedB50Direct = -1# : Exit Function
    End If
    Dim d As Object, th As Double
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    th = ThetaFromDict(d, group, True)
    HazardAdjustedB50Direct = LatentQuantile(0.5, w1, b1, ApplyCoxEta(e1, b1, th), b2, ApplyCoxEta(e2, b2, th))
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

Public Function HazardFlagsDirect(ByVal glf As Variant, _
                                  ByVal loadStd As Variant, _
                                  ByVal loadMean As Variant, _
                                  ByVal fracKpodUnder As Variant, _
                                  ByVal kpodMean As Variant, _
                                  ByVal freqAbove55Pct As Variant, _
                                  ByVal freqSteps As Variant, _
                                  ByVal curvatureDeg10m As Variant, _
                                  ByVal runSeq As Variant, _
                                  ByVal daysSincePrevFailure As Variant, _
                                  ByVal installDate As Variant) As String
    Dim d As Object
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    HazardFlagsDirect = HazardFlagsFromDict(d)
End Function

Public Function HazardAdjustedFailProbWindowDirect(ByVal age As Double, ByVal horizonDays As Double, _
                                                   ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                                   ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                                   ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                                   ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                                   ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                                   ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant, _
                                                   Optional ByVal group As String = "") As Double
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean, h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s) : ctrGrp = ContractorGroup(ctr)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    If Not LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        HazardAdjustedFailProbWindowDirect = -1# : Exit Function
    End If
    If age < 0 Then age = 0
    If horizonDays <= 0 Then HazardAdjustedFailProbWindowDirect = 0# : Exit Function
    Dim d As Object, th As Double, e1x As Double, e2x As Double
    Dim s0 As Double, s1 As Double
    Set d = BuildDirectCovDict(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                               daysSincePrevFailure, installDate)
    th = ThetaFromDict(d, group, True)
    e1x = ApplyCoxEta(e1, b1, th)
    e2x = ApplyCoxEta(e2, b2, th)
    s0 = LatentSF(age, w1, b1, e1x, b2, e2x)
    If s0 <= 1E-12 Then
        HazardAdjustedFailProbWindowDirect = 1#
        Exit Function
    End If
    s1 = LatentSF(age + horizonDays, w1, b1, e1x, b2, e2x)
    HazardAdjustedFailProbWindowDirect = 1# - (s1 / s0)
    If HazardAdjustedFailProbWindowDirect < 0# Then HazardAdjustedFailProbWindowDirect = 0#
    If HazardAdjustedFailProbWindowDirect > 1# Then HazardAdjustedFailProbWindowDirect = 1#
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

' =ESP_HazardTheta_Direct(glf, load_std, load_mean, frac_kpod_under, kpod, pct55,
'                         freq_steps, curvature, run_seq, days_since_prev_failure, install_date)
Public Function ESP_HazardTheta_Direct(ByVal glf As Variant, _
                                       ByVal loadStd As Variant, _
                                       ByVal loadMean As Variant, _
                                       ByVal fracKpodUnder As Variant, _
                                       ByVal kpodMean As Variant, _
                                       ByVal freqAbove55Pct As Variant, _
                                       ByVal freqSteps As Variant, _
                                       ByVal curvatureDeg10m As Variant, _
                                       ByVal runSeq As Variant, _
                                       ByVal daysSincePrevFailure As Variant, _
                                       ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    ESP_HazardTheta_Direct = HazardThetaDirect(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                                               daysSincePrevFailure, installDate)
    Exit Function
ErrHandler:
    ESP_HazardTheta_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_HazardFlags_Direct(ByVal glf As Variant, _
                                       ByVal loadStd As Variant, _
                                       ByVal loadMean As Variant, _
                                       ByVal fracKpodUnder As Variant, _
                                       ByVal kpodMean As Variant, _
                                       ByVal freqAbove55Pct As Variant, _
                                       ByVal freqSteps As Variant, _
                                       ByVal curvatureDeg10m As Variant, _
                                       ByVal runSeq As Variant, _
                                       ByVal daysSincePrevFailure As Variant, _
                                       ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    ESP_HazardFlags_Direct = HazardFlagsDirect(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                                               freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                                               daysSincePrevFailure, installDate)
    Exit Function
ErrHandler:
    ESP_HazardFlags_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_HazardFlags_RU_Direct(ByVal glf As Variant, _
                                          ByVal loadStd As Variant, _
                                          ByVal loadMean As Variant, _
                                          ByVal fracKpodUnder As Variant, _
                                          ByVal kpodMean As Variant, _
                                          ByVal freqAbove55Pct As Variant, _
                                          ByVal freqSteps As Variant, _
                                          ByVal curvatureDeg10m As Variant, _
                                          ByVal runSeq As Variant, _
                                          ByVal daysSincePrevFailure As Variant, _
                                          ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    ESP_HazardFlags_RU_Direct = HazardFlagsRuText(HazardFlagsDirect(glf, loadStd, loadMean, fracKpodUnder, kpodMean, _
                                                                    freqAbove55Pct, freqSteps, curvatureDeg10m, runSeq, _
                                                                    daysSincePrevFailure, installDate))
    Exit Function
ErrHandler:
    ESP_HazardFlags_RU_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_B50_Sens_Direct(ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                    ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                    ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                    ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                    ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                    ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    Dim v As Double
    v = HazardAdjustedB50Direct(field, h2s, ctr, glf, loadStd, loadMean, fracKpodUnder, _
                                kpodMean, freqAbove55Pct, freqSteps, curvatureDeg10m, _
                                runSeq, daysSincePrevFailure, installDate, "")
    If v < 0 Then
        ESP_B50_Sens_Direct = CVErr(xlErrNA)
    Else
        ESP_B50_Sens_Direct = v
    End If
    Exit Function
ErrHandler:
    ESP_B50_Sens_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_RUL_Sens_Direct(ByVal age As Double, _
                                    ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                    ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                    ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                    ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                    ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                    ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    Dim v As Double
    v = HazardAdjustedRULDirect(age, field, h2s, ctr, glf, loadStd, loadMean, fracKpodUnder, _
                                kpodMean, freqAbove55Pct, freqSteps, curvatureDeg10m, _
                                runSeq, daysSincePrevFailure, installDate, "")
    If v < 0 Then
        ESP_RUL_Sens_Direct = CVErr(xlErrNA)
    Else
        ESP_RUL_Sens_Direct = v
    End If
    Exit Function
ErrHandler:
    ESP_RUL_Sens_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_FailProb30_Sens_Direct(ByVal age As Double, _
                                           ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                           ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                           ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                           ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                           ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                           ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    Dim v As Double
    v = HazardAdjustedFailProbWindowDirect(age, 30#, field, h2s, ctr, glf, loadStd, loadMean, fracKpodUnder, _
                                           kpodMean, freqAbove55Pct, freqSteps, curvatureDeg10m, _
                                           runSeq, daysSincePrevFailure, installDate, "")
    If v < 0 Then
        ESP_FailProb30_Sens_Direct = CVErr(xlErrNA)
    Else
        ESP_FailProb30_Sens_Direct = v
    End If
    Exit Function
ErrHandler:
    ESP_FailProb30_Sens_Direct = CVErr(xlErrValue)
End Function

Public Function ESP_FailProbX_Sens_Direct(ByVal age As Double, ByVal horizonDays As Double, _
                                          ByVal field As String, ByVal h2s As String, ByVal ctr As String, _
                                          ByVal glf As Variant, ByVal loadStd As Variant, ByVal loadMean As Variant, _
                                          ByVal fracKpodUnder As Variant, ByVal kpodMean As Variant, _
                                          ByVal freqAbove55Pct As Variant, ByVal freqSteps As Variant, _
                                          ByVal curvatureDeg10m As Variant, ByVal runSeq As Variant, _
                                          ByVal daysSincePrevFailure As Variant, ByVal installDate As Variant) As Variant
    On Error GoTo ErrHandler
    Dim v As Double
    v = HazardAdjustedFailProbWindowDirect(age, horizonDays, field, h2s, ctr, glf, loadStd, loadMean, fracKpodUnder, _
                                           kpodMean, freqAbove55Pct, freqSteps, curvatureDeg10m, _
                                           runSeq, daysSincePrevFailure, installDate, "")
    If v < 0 Then
        ESP_FailProbX_Sens_Direct = CVErr(xlErrNA)
    Else
        ESP_FailProbX_Sens_Direct = v
    End If
    Exit Function
ErrHandler:
    ESP_FailProbX_Sens_Direct = CVErr(xlErrValue)
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
