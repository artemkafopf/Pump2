Attribute VB_Name = "mdlPublicFunctions"
Option Explicit

' ============================================================
' mdlPublicFunctions -- Worksheet-callable UDFs.
'
' CLOCK NOTE (agents/analyses/vba_model_v2.md §0.1 = document-only):
'   The baselines are fitted on the OPERATING-time clock (ttf_mix).  Every time
'   argument (age t / t0) and every time output (RUL, B20/B50/B80, TTF) is in
'   OPERATING days, not calendar days.  No conversion is applied here -- feeding
'   calendar age overstates a pump's age.  ESP_Uptime returns the per-stratum
'   uptime factor u_s = mean(ttf_mix/run_days) so the user can convert manually
'   (age_op ~= age_cal * u_s ; RUL_cal ~= RUL_op / u_s).
'
' All functions accept raw strings directly from the spreadsheet:
'   field      -> column A  (Field name, e.g. "Az", "Ya", "Vt")
'   h2s        -> column BW (raw Russian cell value, mapped to sour/nonsour)
'   contractor -> column D  (raw Russian cell value, mapped to brt/slb/oth)
'
' Usage examples (row 2):
'   =ESP_RUL(I2, A2, BW2, D2)        <- RUL in OPERATING days
'   =ESP_B50(A2, BW2, D2)            <- median TTF in OPERATING days
'   =ESP_ModelKind(A2, BW2, D2)      <- "k2" / "k1_aic" / "k1_degenerate"
'   =ESP_Uptime(A2, BW2, D2)         <- per-stratum uptime factor (audit)
'   =ESP_RunSeq(<well cell>)         <- run sequence number on that well
'   =ESP_Predict(I2, BX2, A2, BW2, D2)
'   =ESP_Debug(A2, BW2, D2)          <- diagnostic: returns stratum or error text
' ============================================================

' ------------------------------------------------------------------
' Shared helper: resolve raw cell inputs to model parameters.
' Public so mdlCoxHR UDFs (ESP_B50_Cox etc.) can use the same pipeline.
' Returns True if a model was found (False -> caller returns #N/A).
' ------------------------------------------------------------------
Public Function GetParams(ByVal fieldRaw As String, _
                            ByVal h2sRaw As String, _
                            ByVal contractorRaw As String, _
                            ByRef w1 As Double, _
                            ByRef b1 As Double, ByRef e1 As Double, _
                            ByRef b2 As Double, ByRef e2 As Double, _
                            ByRef stratumKey As String, _
                            ByRef isDegen As Boolean) As Boolean
    Dim h2sCls As String, ctrGrp As String

    h2sCls = H2SClass(h2sRaw)
    ctrGrp = ContractorGroup(contractorRaw)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"

    GetParams = LookupModel(Trim(fieldRaw), h2sCls, ctrGrp, _
                            w1, b1, e1, b2, e2, stratumKey, isDegen)
End Function

' ------------------------------------------------------------------
' ESP_Debug -- diagnostic UDF, returns text instead of a number.
' Use to verify the model pipeline is working before testing numerics.
'   =ESP_Debug(A2, BW2, D2)
' Returns: "OK: <stratum>" or "NO_MODEL: <key tried>" or "ERR: <msg>"
' ------------------------------------------------------------------
Public Function ESP_Debug(ByVal field As String, _
                           ByVal h2s As String, _
                           ByVal contractor As String) As String
    On Error GoTo ErrHandler

    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Dim h2sCls As String, ctrGrp As String

    h2sCls = H2SClass(h2s)
    ctrGrp = ContractorGroup(contractor)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"

    Dim key As String
    key = Trim(field) & "_" & h2sCls & "_" & ctrGrp

    If GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_Debug = "OK: " & stratumKey & _
                    " w1=" & Format(w1, "0.000") & _
                    " e1=" & Format(e1, "0.0") & _
                    " e2=" & Format(e2, "0.0")
    Else
        ESP_Debug = "NO_MODEL: tried " & key & " -> " & _
                    Trim(field) & "_" & h2sCls & "_Pooled -> Global_Pooled"
    End If
    Exit Function
ErrHandler:
    ESP_Debug = "ERR: " & Err.Number & " " & Err.Description
End Function

' ------------------------------------------------------------------
' ESP_SF -- Survival probability S(t) at age t.
' =ESP_SF(I2, A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_SF(ByVal t As Double, _
                        ByVal field As String, _
                        ByVal h2s As String, _
                        ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_SF = CVErr(xlErrNA) : Exit Function
    End If
    If t < 0 Then t = 0
    ESP_SF = LatentSF(t, w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_SF = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_TTF_Mean -- Model mean TTF (exact closed form).
' =ESP_TTF_Mean(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_TTF_Mean(ByVal field As String, _
                              ByVal h2s As String, _
                              ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_TTF_Mean = CVErr(xlErrNA) : Exit Function
    End If
    ESP_TTF_Mean = LatentMean(w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_TTF_Mean = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B50 -- Median predicted TTF (50th percentile).
' =ESP_B50(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_B50(ByVal field As String, _
                         ByVal h2s As String, _
                         ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B50 = CVErr(xlErrNA) : Exit Function
    End If
    ESP_B50 = LatentQuantile(0.5, w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_B50 = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B20 -- 20th percentile TTF (early-risk bound; replaces B10).
' =ESP_B20(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_B20(ByVal field As String, _
                         ByVal h2s As String, _
                         ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B20 = CVErr(xlErrNA) : Exit Function
    End If
    ESP_B20 = LatentQuantile(0.2, w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_B20 = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B80 -- 80th percentile TTF (maintenance horizon; replaces B90).
' =ESP_B80(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_B80(ByVal field As String, _
                         ByVal h2s As String, _
                         ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B80 = CVErr(xlErrNA) : Exit Function
    End If
    ESP_B80 = LatentQuantile(0.8, w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_B80 = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_RUL -- Remaining Useful Life conditioned on surviving to t0.
' =ESP_RUL(I2, A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_RUL(ByVal t0 As Double, _
                         ByVal field As String, _
                         ByVal h2s As String, _
                         ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_RUL = CVErr(xlErrNA) : Exit Function
    End If
    If t0 < 0 Then t0 = 0

    Dim rul As Double
    rul = LatentRUL(t0, w1, b1, e1, b2, e2)
    ESP_RUL = IIf(rul > 0, rul, "")
    Exit Function
ErrHandler:
    ESP_RUL = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_Hazard -- Instantaneous hazard rate at age t (per day).
' =ESP_Hazard(I2, A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_Hazard(ByVal t As Double, _
                            ByVal field As String, _
                            ByVal h2s As String, _
                            ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_Hazard = CVErr(xlErrNA) : Exit Function
    End If
    If t < 0 Then t = 0
    ESP_Hazard = LatentHazard(t, w1, b1, e1, b2, e2)
    Exit Function
ErrHandler:
    ESP_Hazard = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_Component -- "C1 (early failure)" or "C2 (wear-out)".
' =ESP_Component(I2, A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_Component(ByVal t As Double, _
                               ByVal field As String, _
                               ByVal h2s As String, _
                               ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_Component = CVErr(xlErrNA) : Exit Function
    End If
    If t < 0 Then t = 0

    If LatentC1Posterior(t, w1, b1, e1, b2, e2) >= 0.5 Then
        ESP_Component = "C1 (early failure)"
    Else
        ESP_Component = "C2 (wear-out)"
    End If
    Exit Function
ErrHandler:
    ESP_Component = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_Stratum -- Stratum key resolved (audit column).
' =ESP_Stratum(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_Stratum(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String) As String
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Call GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen)
    ESP_Stratum = stratumKey
    Exit Function
ErrHandler:
    ESP_Stratum = "ERR:" & Err.Number
End Function

' ------------------------------------------------------------------
' ESP_IsDegenerate -- TRUE if the resolved model is flagged degenerate.
' =ESP_IsDegenerate(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_IsDegenerate(ByVal field As String, _
                                  ByVal h2s As String, _
                                  ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_IsDegenerate = CVErr(xlErrNA) : Exit Function
    End If
    ESP_IsDegenerate = isDegen
    Exit Function
ErrHandler:
    ESP_IsDegenerate = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_ModelKind -- registry model kind, reported distinctly (T5):
'   "k2"            K=2 mixture (AIC-preferred, non-degenerate)
'   "k1_aic"        single Weibull preferred by AIC (a valid fit, NOT degenerate)
'   "k1_degenerate" K=2 degenerated (w1>0.75) -> collapsed to single Weibull
' =ESP_ModelKind(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_ModelKind(ByVal field As String, _
                              ByVal h2s As String, _
                              ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s)
    ctrGrp = ContractorGroup(contractor)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    Dim k As String
    k = ResolveModelKind(field, h2sCls, ctrGrp)
    ESP_ModelKind = IIf(k = "", CVErr(xlErrNA), k)
    Exit Function
ErrHandler:
    ESP_ModelKind = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_Uptime -- per-stratum uptime factor u_s = mean(ttf_mix / run_days).
' Audit / manual clock conversion (§0.1 document-only): age_op ~= age_cal * u_s.
' =ESP_Uptime(A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_Uptime(ByVal field As String, _
                           ByVal h2s As String, _
                           ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Dim h2sCls As String, ctrGrp As String
    h2sCls = H2SClass(h2s)
    ctrGrp = ContractorGroup(contractor)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"
    Dim u As Double
    u = UptimeFactor(field, h2sCls, ctrGrp)
    ESP_Uptime = IIf(u > 0, u, CVErr(xlErrNA))
    Exit Function
ErrHandler:
    ESP_Uptime = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_RunSeq -- run sequence number of the passed well cell within its column
' (1 = first run on that well; counts this row and every earlier row with the
' same well). Well name normalised trim + lower-case, matching the Python
' normalize_well_key. DISPLAY-ONLY risk flag: log_run_seq was the largest theta
' coefficient, but theta failed the OOS gate -- so it ships as a flag, never a
' multiplier. Pass the well cell, e.g. =ESP_RunSeq(C2).
' ------------------------------------------------------------------
Public Function ESP_RunSeq(ByVal wellCell As Range) As Variant
    On Error GoTo ErrHandler
    Dim target As String
    target = LCase(Trim(CStr(wellCell.Cells(1, 1).Value)))
    If target = "" Then ESP_RunSeq = "" : Exit Function

    Dim ws As Worksheet, col As Long, thisRow As Long
    Set ws = wellCell.Worksheet
    col = wellCell.Column
    thisRow = wellCell.Row
    If thisRow <= 2 Then ESP_RunSeq = 1 : Exit Function

    Dim arr As Variant
    arr = ws.Range(ws.Cells(2, col), ws.Cells(thisRow, col)).Value  ' >=2 rows -> 2D
    Dim i As Long, cnt As Long
    cnt = 0
    For i = 1 To UBound(arr, 1)
        If LCase(Trim(CStr(arr(i, 1)))) = target Then cnt = cnt + 1
    Next i
    ESP_RunSeq = cnt
    Exit Function
ErrHandler:
    ESP_RunSeq = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_Predict -- Combined prediction for batch use:
'   Running pump (flag=0)  -> RUL
'   Failed pump  (flag=1)  -> B50 (model median TTF)
'   Suspended    (flag=-1) -> ""
' =ESP_Predict(I2, BX2, A2, BW2, D2)
' ------------------------------------------------------------------
Public Function ESP_Predict(ByVal ageDays As Double, _
                             ByVal failureFlag As Integer, _
                             ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String) As Variant
    On Error GoTo ErrHandler
    Select Case failureFlag
        Case 0
            ESP_Predict = ESP_RUL(ageDays, field, h2s, contractor)
        Case 1
            ESP_Predict = ESP_B50(field, h2s, contractor)
        Case Else
            ESP_Predict = ""
    End Select
    Exit Function
ErrHandler:
    ESP_Predict = CVErr(xlErrValue)
End Function
