Attribute VB_Name = "mdlLatentWeibull"
Option Explicit

' ============================================================
' mdlLatentWeibull — K=2 latent Weibull mixture model.
'
' Parameterisation:
'   S(t) = w1*exp(-(t/e1)^b1) + (1-w1)*exp(-(t/e2)^b2)
'   C1 = early-failure component (small eta1)
'   C2 = wear-out component     (large eta2)
'
' Depends on: mdlMath (WeibullSF, WeibullMean, GammaApprox,
'             SimpsonLatentSF, BisectQuantile)
' ============================================================

' Integration cap: stop at T_MAX_FACTOR * max(e1,e2) or T_MAX_ABS days
Private Const T_MAX_FACTOR As Double = 15#
Private Const T_MAX_ABS    As Double = 6000#
Private Const RUL_STEPS    As Long   = 600   ' Simpson steps for RUL integral

' ------------------------------------------------------------------
' Mixture survival function
' ------------------------------------------------------------------
Public Function LatentSF(ByVal t As Double, _
                          ByVal w1 As Double, _
                          ByVal b1 As Double, ByVal e1 As Double, _
                          ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim w2 As Double
    w2 = 1# - w1
    LatentSF = w1 * WeibullSF(t, b1, e1) + w2 * WeibullSF(t, b2, e2)
End Function

' ------------------------------------------------------------------
' Mixture hazard rate h(t) = f(t)/S(t)
' f(t) = w1*(b1/e1)*(t/e1)^(b1-1)*exp(-(t/e1)^b1)
'       + w2*(b2/e2)*(t/e2)^(b2-1)*exp(-(t/e2)^b2)
' ------------------------------------------------------------------
Public Function LatentHazard(ByVal t As Double, _
                              ByVal w1 As Double, _
                              ByVal b1 As Double, ByVal e1 As Double, _
                              ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim w2 As Double, s As Double, f As Double
    Dim f1 As Double, f2 As Double

    If t <= 0 Then LatentHazard = 0# : Exit Function

    w2 = 1# - w1
    f1 = w1 * (b1 / e1) * ((t / e1) ^ (b1 - 1#)) * Exp(-(t / e1) ^ b1)
    f2 = w2 * (b2 / e2) * ((t / e2) ^ (b2 - 1#)) * Exp(-(t / e2) ^ b2)
    f = f1 + f2
    s = LatentSF(t, w1, b1, e1, b2, e2)

    If s < 1E-12 Then
        LatentHazard = 0#
    Else
        LatentHazard = f / s
    End If
End Function

' ------------------------------------------------------------------
' Mixture mean TTF (exact closed form — no integration):
'   E[T] = w1*eta1*Gamma(1+1/b1) + w2*eta2*Gamma(1+1/b2)
' ------------------------------------------------------------------
Public Function LatentMean(ByVal w1 As Double, _
                            ByVal b1 As Double, ByVal e1 As Double, _
                            ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim w2 As Double
    w2 = 1# - w1
    LatentMean = w1 * WeibullMean(b1, e1) + w2 * WeibullMean(b2, e2)
End Function

' ------------------------------------------------------------------
' p-th quantile: smallest t such that S(t) <= 1-p.
'   B50 = LatentQuantile(0.50, ...)   (median)
'   B10 = LatentQuantile(0.10, ...)   (early risk bound)
'   B90 = LatentQuantile(0.90, ...)   (conservative bound)
' ------------------------------------------------------------------
Public Function LatentQuantile(ByVal p As Double, _
                                ByVal w1 As Double, _
                                ByVal b1 As Double, ByVal e1 As Double, _
                                ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim tHi As Double
    tHi = Application.WorksheetFunction.Min(T_MAX_FACTOR * Application.WorksheetFunction.Max(e1, e2), T_MAX_ABS)
    ' Survival target = 1 - p
    LatentQuantile = BisectQuantile(w1, b1, e1, b2, e2, 1# - p, 0#, tHi)
End Function

' ------------------------------------------------------------------
' Remaining Useful Life conditioned on survival to t0:
'   RUL(t0) = E[T - t0 | T > t0]
'           = (1/S(t0)) * integral_{t0}^{Tmax} S(t) dt
' Returns 0 if the pump is effectively dead (S(t0) < 1e-9).
' ------------------------------------------------------------------
Public Function LatentRUL(ByVal t0 As Double, _
                           ByVal w1 As Double, _
                           ByVal b1 As Double, ByVal e1 As Double, _
                           ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim s0 As Double, tMax As Double, integral As Double

    If t0 < 0 Then t0 = 0#

    s0 = LatentSF(t0, w1, b1, e1, b2, e2)
    If s0 < 1E-09 Then
        LatentRUL = 0#
        Exit Function
    End If

    tMax = Application.WorksheetFunction.Min( _
               T_MAX_FACTOR * Application.WorksheetFunction.Max(e1, e2), _
               T_MAX_ABS)

    integral = SimpsonLatentSF(w1, b1, e1, b2, e2, t0, tMax, RUL_STEPS)
    LatentRUL = integral / s0
End Function

' ------------------------------------------------------------------
' Posterior probability that a pump at age t belongs to C1 (early mode).
' P(Z=1 | T=t) = w1*f1(t) / [w1*f1(t) + w2*f2(t)]
' Useful for component assignment: > 0.5 → "C1 (early failure likely)"
' ------------------------------------------------------------------
Public Function LatentC1Posterior(ByVal t As Double, _
                                   ByVal w1 As Double, _
                                   ByVal b1 As Double, ByVal e1 As Double, _
                                   ByVal b2 As Double, ByVal e2 As Double) As Double
    Dim w2 As Double, p1 As Double, p2 As Double

    If t <= 0 Then LatentC1Posterior = w1 : Exit Function

    w2 = 1# - w1
    p1 = w1 * (b1 / e1) * ((t / e1) ^ (b1 - 1#)) * Exp(-(t / e1) ^ b1)
    p2 = w2 * (b2 / e2) * ((t / e2) ^ (b2 - 1#)) * Exp(-(t / e2) ^ b2)

    If (p1 + p2) < 1E-15 Then
        LatentC1Posterior = w1
    Else
        LatentC1Posterior = p1 / (p1 + p2)
    End If
End Function
