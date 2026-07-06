Attribute VB_Name = "mdlMath"
Option Explicit

' ============================================================
' mdlMath — Pure numerical utilities
' No dependencies on other modules.
' ============================================================

' ------------------------------------------------------------------
' Weibull survival function: exp(-(t/eta)^beta)
' ------------------------------------------------------------------
Public Function WeibullSF(ByVal t As Double, _
                           ByVal beta As Double, _
                           ByVal eta As Double) As Double
    If t <= 0 Then
        WeibullSF = 1#
    ElseIf eta <= 0 Or beta <= 0 Then
        WeibullSF = 0#
    Else
        WeibullSF = Exp(-(t / eta) ^ beta)
    End If
End Function

' ------------------------------------------------------------------
' Gamma function via Lanczos approximation (g=7, n=9 coefficients).
' Accurate to ~15 significant digits for Re(z) > 0.5.
' For z < 0.5 uses reflection: Gamma(z)*Gamma(1-z) = pi/sin(pi*z)
' ------------------------------------------------------------------
Public Function GammaApprox(ByVal z As Double) As Double
    Const PI As Double = 3.14159265358979324#
    Dim g As Double, n As Integer
    Dim c(0 To 8) As Double
    Dim x As Double, t As Double, ser As Double
    Dim i As Integer

    c(0) = 0.99999999999980993
    c(1) = 676.5203681218851
    c(2) = -1259.1392167224028
    c(3) = 771.32342877765313
    c(4) = -176.61502916214059
    c(5) = 12.507343278686905
    c(6) = -0.13857109526572012
    c(7) = 9.9843695780195716E-06
    c(8) = 1.5056327351493116E-07

    If z < 0.5 Then
        GammaApprox = PI / (Sin(PI * z) * GammaApprox(1# - z))
        Exit Function
    End If

    z = z - 1#
    x = c(0)
    For i = 1 To 8
        x = x + c(i) / (z + i)
    Next i
    t = z + 8# - 0.5
    GammaApprox = Sqr(2# * PI) * t ^ (z + 0.5) * Exp(-t) * x
End Function

' ------------------------------------------------------------------
' Weibull mean: eta * Gamma(1 + 1/beta)
' ------------------------------------------------------------------
Public Function WeibullMean(ByVal beta As Double, _
                             ByVal eta As Double) As Double
    If beta <= 0 Or eta <= 0 Then
        WeibullMean = 0#
    Else
        WeibullMean = eta * GammaApprox(1# + 1# / beta)
    End If
End Function

' ------------------------------------------------------------------
' Composite Simpson's rule:
' Approximates integral of LatentSF from tLo to tHi using nSteps steps.
' nSteps must be even; if odd it is incremented by 1.
' ------------------------------------------------------------------
Public Function SimpsonLatentSF(ByVal w1 As Double, _
                                 ByVal b1 As Double, ByVal e1 As Double, _
                                 ByVal b2 As Double, ByVal e2 As Double, _
                                 ByVal tLo As Double, ByVal tHi As Double, _
                                 ByVal nSteps As Long) As Double
    Dim h As Double, s As Double, t As Double
    Dim i As Long

    If nSteps < 2 Then nSteps = 2
    If nSteps Mod 2 <> 0 Then nSteps = nSteps + 1
    If tHi <= tLo Then
        SimpsonLatentSF = 0#
        Exit Function
    End If

    h = (tHi - tLo) / nSteps

    ' Endpoints
    s = LatentSF(tLo, w1, b1, e1, b2, e2) + LatentSF(tHi, w1, b1, e1, b2, e2)

    ' Inner points
    For i = 1 To nSteps - 1
        t = tLo + i * h
        If i Mod 2 = 0 Then
            s = s + 2# * LatentSF(t, w1, b1, e1, b2, e2)
        Else
            s = s + 4# * LatentSF(t, w1, b1, e1, b2, e2)
        End If
    Next i

    SimpsonLatentSF = s * h / 3#
End Function

' ------------------------------------------------------------------
' Bisection root-finder.
' Finds t in [lo, hi] such that LatentSF(t) = target.
' LatentSF is monotone decreasing, so target must be in
' [LatentSF(hi), LatentSF(lo)].
' tol: convergence tolerance in days (default 0.05).
' ------------------------------------------------------------------
Public Function BisectQuantile(ByVal w1 As Double, _
                                ByVal b1 As Double, ByVal e1 As Double, _
                                ByVal b2 As Double, ByVal e2 As Double, _
                                ByVal target As Double, _
                                ByVal lo As Double, ByVal hi As Double, _
                                Optional ByVal tol As Double = 0.05) As Double
    Dim mid As Double, fMid As Double
    Dim iter As Integer

    ' Safety clamp
    If target >= 1# Then BisectQuantile = 0# : Exit Function
    If target <= 0# Then BisectQuantile = hi  : Exit Function

    For iter = 1 To 100
        mid = (lo + hi) / 2#
        fMid = LatentSF(mid, w1, b1, e1, b2, e2)
        If Abs(fMid - target) < 1E-08 Or (hi - lo) < tol Then
            Exit For
        End If
        ' SF is decreasing: if fMid > target, we need larger t
        If fMid > target Then
            lo = mid
        Else
            hi = mid
        End If
    Next iter

    BisectQuantile = mid
End Function
