Attribute VB_Name = "mdlCoxHR"
Option Explicit

' ============================================================
' mdlCoxHR -- Cox proportional hazards covariate adjustment.
'
' Adds three covariates on top of the K=2 latent Weibull baseline:
'
'   delta_bep      = Q_actual / Q_nominal - 1
'                    (negative = under-loaded, positive = over-loaded)
'   p_bot          = bottomhole pressure, atm
'   n_stages_ratio = n_stages / q_nominal  (stages per m3/day nominal)
'
' Fitted by stratified Cox (cluster-robust SE on well_key),
' 9 strata = Field x H2S class.  Source: phase5e, 2026-07-02.
'
' Reference profile (theta = 1, i.e. population-mean pump):
'   delta_bep_ref      = -0.342
'   p_bot_ref          = 110.3 atm
'   n_stages_ratio_ref =   2.249
'
' theta(x) = exp( b_bep * (delta_bep - ref_bep)
'               + b_p   * (p_bot     - ref_p)
'               + b_n   * (n_stages  - ref_n) )
'
' Cox -> Weibull eta rescaling (exact for Weibull baseline):
'   eta_eff = eta / theta ^ (1 / beta)
'
' theta > 1  higher risk than average pump  -> eta_eff < eta (shorter life)
' theta < 1  lower  risk than average pump  -> eta_eff > eta (longer life)
' theta = 1  average covariate profile      -> baseline unchanged
'
' Public UDFs added:
'   ESP_Theta(delta_bep, p_bot, n_stages_ratio)
'   ESP_SF_Cox(t, field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
'   ESP_B50_Cox(field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
'   ESP_B10_Cox(field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
'   ESP_B90_Cox(field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
'   ESP_RUL_Cox(t0, field, h2s, ctr, delta_bep, p_bot, n_stages_ratio)
'
' VBA lessons applied:
'   - No MsgBox inside UDFs
'   - ChrW() for any Unicode output
'   - No cross-module Private references (EnsureRegistryLoaded via mdlModelRegistry)
' ============================================================

' ── Fitted beta coefficients ─────────────────────────────────
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
' CoxTheta -- core computation: theta = exp(beta . (x - x_ref))
' Exported as ESP_Theta UDF.
' ------------------------------------------------------------------
Public Function CoxTheta(ByVal deltaBEP As Double, _
                          ByVal pBot As Double, _
                          ByVal nStagesRatio As Double) As Double
    Dim lp As Double
    lp = BETA_BEP    * (deltaBEP     - REF_BEP)    _
       + BETA_PBOT   * (pBot         - REF_PBOT)   _
       + BETA_NSTAGES * (nStagesRatio - REF_NSTAGES)

    Dim th As Double
    th = Exp(lp)
    If th < THETA_MIN Then th = THETA_MIN
    If th > THETA_MAX Then th = THETA_MAX
    CoxTheta = th
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
' ESP_B10_Cox -- Cox-adjusted 10th percentile TTF.
'   =ESP_B10_Cox(field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_B10_Cox(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B10_Cox = CVErr(xlErrNA) : Exit Function
    End If

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    ESP_B10_Cox = LatentQuantile(0.1, w1, b1, _
                                 ApplyCoxEta(e1, b1, theta), _
                                 b2, ApplyCoxEta(e2, b2, theta))
    Exit Function
ErrHandler:
    ESP_B10_Cox = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_B90_Cox -- Cox-adjusted 90th percentile TTF.
'   =ESP_B90_Cox(field, h2s, contractor, delta_bep, p_bot, n_stages_ratio)
' ------------------------------------------------------------------
Public Function ESP_B90_Cox(ByVal field As String, _
                             ByVal h2s As String, _
                             ByVal contractor As String, _
                             ByVal deltaBEP As Double, _
                             ByVal pBot As Double, _
                             ByVal nStagesRatio As Double) As Variant
    On Error GoTo ErrHandler
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    If Not GetParams(field, h2s, contractor, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        ESP_B90_Cox = CVErr(xlErrNA) : Exit Function
    End If

    Dim theta As Double
    theta = CoxTheta(deltaBEP, pBot, nStagesRatio)

    ESP_B90_Cox = LatentQuantile(0.9, w1, b1, _
                                 ApplyCoxEta(e1, b1, theta), _
                                 b2, ApplyCoxEta(e2, b2, theta))
    Exit Function
ErrHandler:
    ESP_B90_Cox = CVErr(xlErrValue)
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
