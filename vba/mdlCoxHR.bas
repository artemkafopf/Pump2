Attribute VB_Name = "mdlCoxHR"
Option Explicit

' ============================================================
' mdlCoxHR -- Cox proportional hazards covariate adjustment  [SHELVED / DEPRECATE-KEEP].
'
' STATUS (agents/analyses/vba_model_v2.md T3, §0.2 = deprecate-keep):
'   The covariate theta layer is DECOMMISSIONED for forecasting.  The merged
'   18-term theta failed the out-of-sample gate (temporal holdout C-index ~= chance
'   at both cutoffs), as did Phase D's dynamic layer.  Program verdict: NO covariate
'   theta ships -- strata + K=2 baselines are the forecasting engine.
'   See results/model_report/2026-07-07/README.md §3.8 for the full reasoning.
'
'   These UDFs remain CALLABLE for backward compatibility, but return theta == 1
'   (i.e. exactly the baseline result) UNLESS an ESP_CoxCoeffs sheet exists AND
'   carries an explicit cell labelled "enabled" set to TRUE.  The stale phase5e
'   coefficient sheet has been removed; RunPredictions no longer writes Cox columns.
'
'   RE-ENABLE PATH (Phase E, only if a future theta passes the OOS gate): populate
'   ESP_CoxCoeffs from theta_final_coeffs.csv (the 18-term fit,
'   results/phase_c_joint_theta/.../theta_final_coeffs.csv) with enabled=TRUE, and
'   wire CoxTheta to read the sheet.  The eta-rescaling math below (ApplyCoxEta) is
'   the exact, unchanged, tested mechanism it would use -- kept alive for that day.
'
' Legacy 3-covariate reference (pre-Phase-C, superseded -- kept only so a re-enable
' has a worked example): delta_bep, p_bot, n_stages_ratio; refs -0.342 / 110.3 / 2.249.
'
' Cox -> Weibull eta rescaling (exact for Weibull baseline):
'   eta_eff = eta / theta ^ (1 / beta)
'   theta > 1 -> shorter life ; theta < 1 -> longer life ; theta = 1 -> baseline.
'
' VBA lessons applied: no MsgBox inside UDFs; ChrW() for Unicode; no cross-module
' Private references (accessors via mdlModelRegistry).
' ============================================================

' ── Legacy fitted beta coefficients (pre-Phase-C; only used if re-enabled) ──
Private Const BETA_BEP      As Double = 0.360455
Private Const BETA_PBOT     As Double = 0.000466
Private Const BETA_NSTAGES  As Double = -0.052032

' ── Reference profile (population mean) ─────────────────────
Private Const REF_BEP       As Double = -0.342
Private Const REF_PBOT      As Double = 110.3
Private Const REF_NSTAGES   As Double = 2.249

' ── Clamp bounds for theta (guards against extreme covariate values) ─
Private Const THETA_MIN     As Double = 0.05
Private Const THETA_MAX     As Double = 20#

' ------------------------------------------------------------------
' CoxEnabled -- the deprecate-keep switch.  TRUE only if an ESP_CoxCoeffs sheet
' exists AND a cell labelled "enabled" (col A) holds TRUE (col B).  Never MsgBox.
' ------------------------------------------------------------------
Public Function CoxEnabled() As Boolean
    On Error GoTo NotEnabled
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("ESP_CoxCoeffs")
    ' v2 schema: an "enabled" HEADER column (row 1), one value per covariate row.
    Dim lastCol As Long, c As Long, enCol As Long
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column
    enCol = 0
    For c = 1 To lastCol
        If LCase(Trim(CStr(ws.Cells(1, c).Value))) = "enabled" Then enCol = c : Exit For
    Next c
    If enCol > 0 Then
        Dim v As String
        v = UCase(Trim(CStr(ws.Cells(2, enCol).Value)))
        CoxEnabled = (v = "TRUE" Or v = "1" Or v = "-1" Or v = "YES")
        Exit Function
    End If
    ' Legacy fallback: a cell labelled "enabled" in column A.
    Dim lastRow As Long, i As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    For i = 1 To lastRow
        If LCase(Trim(CStr(ws.Cells(i, 1).Value))) = "enabled" Then
            CoxEnabled = (UCase(Trim(CStr(ws.Cells(i, 2).Value))) = "TRUE")
            Exit Function
        End If
    Next i
NotEnabled:
    CoxEnabled = False
End Function

' ------------------------------------------------------------------
' CoxTheta -- hazard multiplier.  DEPRECATE-KEEP: returns 1 (baseline) unless the
' Cox layer is explicitly re-enabled via ESP_CoxCoeffs!enabled=TRUE.
' ------------------------------------------------------------------
Public Function CoxTheta(ByVal deltaBEP As Double, _
                          ByVal pBot As Double, _
                          ByVal nStagesRatio As Double) As Double
    ' RETIRED: the legacy 3-covariate theta (delta_bep, p_bot, n_stages_ratio) is a
    ' pre-Phase-C fit, superseded by the operational+completion+cohort hazard layer in
    ' mdlHazardLayer (ESP_HazardTheta / ESP_dRUL). It always returns 1 so the legacy
    ' ESP_*_Cox UDFs equal the baseline. The `enabled` flag on ESP_CoxCoeffs now drives
    ' the NEW layer, not this. To use covariate adjustment, use mdlHazardLayer.
    CoxTheta = 1#
End Function

' ------------------------------------------------------------------
' ApplyCoxEta -- rescale one Weibull eta by theta.
'   eta_eff = eta / theta ^ (1 / beta)
' ------------------------------------------------------------------
Public Function ApplyCoxEta(ByVal eta As Double, _
                             ByVal beta As Double, _
                             ByVal theta As Double) As Double
    If beta <= 0 Or theta <= 0 Or eta <= 0 Then
        ApplyCoxEta = eta
        Exit Function
    End If
    ApplyCoxEta = eta / (theta ^ (1# / beta))
End Function

' ================================================================
' PUBLIC UDFs
' ================================================================

' ------------------------------------------------------------------
' ESP_Theta -- hazard multiplier for a pump with given covariates.
'   theta > 1  -> worse than population average
'   theta < 1  -> better than population average
'   theta = 1  -> average pump (all covariates at reference values)
'
'   =ESP_Theta(delta_bep, p_bot, n_stages_ratio)
'   =ESP_Theta(BZ2, CA2, CB2)
' ------------------------------------------------------------------
Public Function ESP_Theta(ByVal deltaBEP As Double, _
                           ByVal pBot As Double, _
                           ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    ESP_Theta = CoxTheta(deltaBEP, pBot, nStagesRatio)
    Exit Function
ErrHandler:
    ESP_Theta = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_SF_Cox -- Cox-adjusted survival probability S(t|x).
'   =ESP_SF_Cox(t, field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_SF_Cox(ByVal t As Double, _
                            ByVal field As String, _
                            ByVal h2s As String, _
                            ByVal contractor As String, _
                            ByVal deltaBEP As Double, _
                            ByVal pBot As Double, _
                            ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_SF_Cox = CVErr(xlErrNA) : Exit Function
    End If
    If t < 0 Then t = 0

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    Dim e1x As Double, e2x As Double
    e1x = ApplyCoxEta(e1, b1, theta)
    e2x = ApplyCoxEta(e2, b2, theta)

    ESP_SF_Cox = LatentSF(t, w1, b1, e1x, b2, e2x)
    Exit Function
ErrHandler:
    ESP_SF_Cox = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B50_Cox -- Cox-adjusted median TTF.
'   =ESP_B50_Cox(field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_B50_Cox(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B50_Cox = CVErr(xlErrNA) : Exit Function
    End If

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    ESP_B50_Cox = LatentQuantile(0.5, w1, b1, _
                                 ApplyCoxEta(e1, b1, theta), _
                                 b2, ApplyCoxEta(e2, b2, theta))
    Exit Function
ErrHandler:
    ESP_B50_Cox = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B20_Cox -- (legacy, retired) 20th percentile TTF; CoxTheta=1 -> baseline B20.
'   =ESP_B20_Cox(field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_B20_Cox(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B20_Cox = CVErr(xlErrNA) : Exit Function
    End If

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    ESP_B20_Cox = LatentQuantile(0.2, w1, b1, _
                                 ApplyCoxEta(e1, b1, theta), _
                                 b2, ApplyCoxEta(e2, b2, theta))
    Exit Function
ErrHandler:
    ESP_B20_Cox = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B80_Cox -- (legacy, retired) 80th percentile TTF; CoxTheta=1 -> baseline B80.
'   =ESP_B80_Cox(field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_B80_Cox(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B80_Cox = CVErr(xlErrNA) : Exit Function
    End If

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    ESP_B80_Cox = LatentQuantile(0.8, w1, b1, _
                                 ApplyCoxEta(e1, b1, theta), _
                                 b2, ApplyCoxEta(e2, b2, theta))
    Exit Function
ErrHandler:
    ESP_B80_Cox = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_RUL_Cox -- Cox-adjusted RUL conditioned on surviving to t0.
'   =ESP_RUL_Cox(t0, field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_RUL_Cox(ByVal t0 As Double, _
                             ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_RUL_Cox = CVErr(xlErrNA) : Exit Function
    End If
    If t0 < 0 Then t0 = 0

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    Dim e1x As Double, e2x As Double
    e1x = ApplyCoxEta(e1, b1, theta)
    e2x = ApplyCoxEta(e2, b2, theta)

    Dim rul As Double
    rul = LatentRUL(t0, w1, b1, e1x, b2, e2x)
    ESP_RUL_Cox = IIf(rul > 0, rul, "")
    Exit Function
ErrHandler:
    ESP_RUL_Cox = CVErr(xlErrValue)
End Function
