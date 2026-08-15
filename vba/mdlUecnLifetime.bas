Attribute VB_Name = "mdlUecnLifetime"
Option Explicit

' ============================================================
' mdlUecnLifetime — ready-to-use VBA implementation for
' UECN lifetime metrics from pre-fitted Weibull parameters.
'
' Based on the specification in the request:
'   - select pattern: all-three vs only-debit
'   - compute phi via log deviations from references
'   - derive eta/beta
'   - compute median, mean, RMST(tau), residual resource at age t
' ============================================================

Private Const DEFAULT_TAU As Double = 730#
Private Const DEBIT_MIN As Double = 47#
Private Const DEBIT_MAX As Double = 823#
Private Const DEBIT_REF As Double = 250#
Private Const FREQ_REF As Double = 50#
Private Const KPOD_REF As Double = 0.70#

Public Function UECN_Median(ByVal debit As Variant, _
                            ByVal frequency As Variant, _
                            ByVal kpod As Variant, _
                            ByVal h2sClass As Variant, _
                            ByVal contractor As Variant, _
                            Optional ByVal tau As Double = DEFAULT_TAU) As Double
    Dim metrics As Variant
    metrics = UECN_Metrics(debit, frequency, kpod, h2sClass, contractor, 0#, tau)
    UECN_Median = CDbl(metrics(0))
End Function

Public Function UECN_Mean(ByVal debit As Variant, _
                          ByVal frequency As Variant, _
                          ByVal kpod As Variant, _
                          ByVal h2sClass As Variant, _
                          ByVal contractor As Variant, _
                          Optional ByVal tau As Double = DEFAULT_TAU) As Double
    Dim metrics As Variant
    metrics = UECN_Metrics(debit, frequency, kpod, h2sClass, contractor, 0#, tau)
    UECN_Mean = CDbl(metrics(1))
End Function

Public Function UECN_RMST(ByVal debit As Variant, _
                          ByVal frequency As Variant, _
                          ByVal kpod As Variant, _
                          ByVal h2sClass As Variant, _
                          ByVal contractor As Variant, _
                          Optional ByVal tau As Double = DEFAULT_TAU) As Double
    Dim metrics As Variant
    metrics = UECN_Metrics(debit, frequency, kpod, h2sClass, contractor, 0#, tau)
    UECN_RMST = CDbl(metrics(2))
End Function

Public Function UECN_ResidualResource(ByVal debit As Variant, _
                                      ByVal frequency As Variant, _
                                      ByVal kpod As Variant, _
                                      ByVal h2sClass As Variant, _
                                      ByVal contractor As Variant, _
                                      ByVal age As Double, _
                                      Optional ByVal tau As Double = DEFAULT_TAU) As Double
    Dim metrics As Variant
    metrics = UECN_Metrics(debit, frequency, kpod, h2sClass, contractor, age, tau)
    UECN_ResidualResource = CDbl(metrics(3))
End Function

' Returns a 4-element array ordered as:
'   0 = median
'   1 = mean (MRL at age 0)
'   2 = RMST over [0, tau]
'   3 = residual resource at age t
Public Function UECN_Metrics(ByVal debit As Variant, _
                              ByVal frequency As Variant, _
                              ByVal kpod As Variant, _
                              ByVal h2sClass As Variant, _
                              ByVal contractor As Variant, _
                              Optional ByVal age As Double = 0#, _
                              Optional ByVal tau As Double = DEFAULT_TAU) As Variant
    Dim beta As Double, etaRef As Double, phi As Double, eta As Double
    Dim pattern As String
    Dim q As Double, s As Double
    Dim median As Double, mean As Double, rmst As Double, residual As Double
    Dim debitValue As Double, freqValue As Double, kpodValue As Double
    Dim debitKnown As Boolean, freqKnown As Boolean, kpodKnown As Boolean
    Dim x As Double, p As Double, sf As Double
    Dim results(0 To 3) As Double

    If tau <= 0# Then tau = DEFAULT_TAU

    pattern = ResolvePattern(h2sClass, debit, frequency, kpod)
    beta = BetaForPattern(pattern, h2sClass)
    etaRef = EtaRefForPattern(pattern, h2sClass)

    debitValue = SafeNumeric(debit)
    freqValue = SafeNumeric(frequency)
    kpodValue = SafeNumeric(kpod)

    debitKnown = IsNumeric(debit) And Not IsMissingValue(debit)
    freqKnown = IsNumeric(frequency) And Not IsMissingValue(frequency)
    kpodKnown = IsNumeric(kpod) And Not IsMissingValue(kpod)

    If debitKnown Then
        q = Clamp(debitValue, DEBIT_MIN, DEBIT_MAX)
    Else
        q = DEBIT_REF
    End If

    s = 0#
    If IsAcidicH2S(h2sClass) Then
        ' Acidic case uses only-debit form and ignores frequency / kpod.
        s = 0#
    ElseIf pattern = "all-three" Then
        s = s + GDebAllThree * Log(q / DEBIT_REF)
        If freqKnown Then s = s + GFreqAllThree * (freqValue - FREQ_REF)
        If kpodKnown Then s = s + GKpodAllThree * (kpodValue - KPOD_REF)
    Else
        s = s + GDebOnlyDebit * Log(q / DEBIT_REF)
    End If

    phi = ContractorMultiplierForPattern(pattern, h2sClass, contractor) * Exp(s)
    eta = etaRef * phi

    If beta <= 0# Or eta <= 0# Then
        UECN_Metrics = results
        Exit Function
    End If

    median = eta * (Log(2#) ^ (1# / beta))
    mean = eta * GammaApprox(1# + 1# / beta)

    x = (tau / eta) ^ beta
    p = GammaLowerRegularized(1# / beta, x)
    rmst = mean * p

    If age <= 0# Then
        residual = mean
    Else
        sf = Exp(-(age / eta) ^ beta)
        x = (age / eta) ^ beta
        p = GammaLowerRegularized(1# / beta, x)
        residual = mean * (1# - p) / sf
    End If

    results(0) = median
    results(1) = mean
    results(2) = rmst
    results(3) = residual
    UECN_Metrics = results
End Function

Private Function ResolvePattern(ByVal h2sClass As Variant, _
                                ByVal debit As Variant, _
                                ByVal frequency As Variant, _
                                ByVal kpod As Variant) As String
    If IsAcidicH2S(h2sClass) Then
        ResolvePattern = "only-debit"
        Exit Function
    End If

    If Not IsNumeric(debit) Or IsMissingValue(debit) Then
        ResolvePattern = "only-debit"
        Exit Function
    End If

    If Not IsNumeric(frequency) Or IsMissingValue(frequency) Then
        ResolvePattern = "only-debit"
        Exit Function
    End If

    If Not IsNumeric(kpod) Or IsMissingValue(kpod) Then
        ResolvePattern = "only-debit"
        Exit Function
    End If

    ResolvePattern = "all-three"
End Function

Private Function BetaForPattern(ByVal pattern As String, _
                                ByVal h2sClass As Variant) As Double
    If IsAcidicH2S(h2sClass) Then
        BetaForPattern = 1.37
        Exit Function
    End If

    If pattern = "all-three" Then
        BetaForPattern = 1.28
    Else
        BetaForPattern = 1.15
    End If
End Function

Private Function EtaRefForPattern(ByVal pattern As String, _
                                   ByVal h2sClass As Variant) As Double
    If IsAcidicH2S(h2sClass) Then
        EtaRefForPattern = 149#
        Exit Function
    End If

    If pattern = "all-three" Then
        EtaRefForPattern = 570#
    Else
        EtaRefForPattern = 571#
    End If
End Function

Private Function ContractorMultiplierForPattern(ByVal pattern As String, _
                                                 ByVal h2sClass As Variant, _
                                                 ByVal contractor As Variant) As Double
    Dim contractorName As String

    contractorName = NormalizeContractor(contractor)

    If IsAcidicH2S(h2sClass) Then
        Select Case contractorName
            Case "brt": ContractorMultiplierForPattern = 1#
            Case "slb": ContractorMultiplierForPattern = 1.05
            Case "oth": ContractorMultiplierForPattern = 0.645
            Case Else: ContractorMultiplierForPattern = 1#
        End Select
        Exit Function
    End If

    If pattern = "all-three" Then
        Select Case contractorName
            Case "brt": ContractorMultiplierForPattern = 1#
            Case "slb": ContractorMultiplierForPattern = 0.728
            Case "oth": ContractorMultiplierForPattern = 0.635
            Case Else: ContractorMultiplierForPattern = 1#
        End Select
    Else
        Select Case contractorName
            Case "brt": ContractorMultiplierForPattern = 1#
            Case "slb": ContractorMultiplierForPattern = 0.79
            Case "oth": ContractorMultiplierForPattern = 0.54
            Case Else: ContractorMultiplierForPattern = 1#
        End Select
    End If
End Function

Private Function NormalizeContractor(ByVal contractor As Variant) As String
    Dim val As String

    If IsMissingValue(contractor) Then
        NormalizeContractor = "brt"
        Exit Function
    End If

    val = Trim$(LCase$(CStr(contractor)))
    Select Case val
        Case "brt", "брт", "1"
            NormalizeContractor = "brt"
        Case "slb", "сlb", "2"
            NormalizeContractor = "slb"
        Case "oth", "отх", "3"
            NormalizeContractor = "oth"
        Case Else
            NormalizeContractor = "brt"
    End Select
End Function

Private Function IsAcidicH2S(ByVal h2sClass As Variant) As Boolean
    Dim val As String

    If IsMissingValue(h2sClass) Then
        IsAcidicH2S = False
        Exit Function
    End If

    val = Trim$(LCase$(CStr(h2sClass)))
    Select Case val
        Case "кислые", "кислый", "acidic", "acid", "sour", "к"
            IsAcidicH2S = True
        Case Else
            IsAcidicH2S = False
    End Select
End Function

Private Function IsMissingValue(ByVal value As Variant) As Boolean
    If IsEmpty(value) Or IsNull(value) Then
        IsMissingValue = True
        Exit Function
    End If

    If VarType(value) = vbString Then
        If Trim$(CStr(value)) = vbNullString Then
            IsMissingValue = True
            Exit Function
        End If
    End If

    IsMissingValue = False
End Function

Private Function SafeNumeric(ByVal value As Variant) As Double
    If IsNumeric(value) Then
        SafeNumeric = CDbl(value)
    Else
        SafeNumeric = 0#
    End If
End Function

Private Function Clamp(ByVal x As Double, ByVal lo As Double, ByVal hi As Double) As Double
    If x < lo Then
        Clamp = lo
    ElseIf x > hi Then
        Clamp = hi
    Else
        Clamp = x
    End If
End Function

Private Function GammaLowerRegularized(ByVal s As Double, ByVal x As Double) As Double
    Dim gln As Double
    Dim sum As Double, term As Double
    Dim k As Long

    If s <= 0# Or x <= 0# Then
        GammaLowerRegularized = 0#
        Exit Function
    End If

    On Error GoTo FallbackSeries
    GammaLowerRegularized = Application.WorksheetFunction.Gamma_Dist(x, s, 1#, True)
    Exit Function

FallbackSeries:
    gln = GammaApprox(s)
    If gln <= 0# Then
        GammaLowerRegularized = 0#
        Exit Function
    End If

    ' Series form for lower regularized gamma:
    ' P(s, x) = e^-x * x^s / Gamma(s) * sum_{k=0}^∞ x^k / (s(s+1)...(s+k))
    sum = 1# / s
    term = 1# / s
    For k = 1 To 200
        term = term * x / (s + k)
        sum = sum + term
        If Abs(term) < 1E-14 * Abs(sum) Then Exit For
    Next k

    GammaLowerRegularized = Exp(-x) * (x ^ s) * sum / gln
End Function

Private Function GammaApprox(ByVal z As Double) As Double
    Const PI As Double = 3.14159265358979324#
    Dim g As Double, n As Integer
    Dim c(0 To 8) As Double
    Dim x As Double, t As Double
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

' Optional constants to make the main formula explicit.
Private Const GDebAllThree As Double = -0.130
Private Const GFreqAllThree As Double = -0.0142
Private Const GKpodAllThree As Double = -0.729
Private Const GDebOnlyDebit As Double = -0.305

Public Sub DemoUECNChecks()
    Debug.Print "Case A all-three slb:";
    Debug.Print UECN_Metrics(400#, 52#, 0.9, "некислые", "slb", 0#, 730#)(0);
    Debug.Print UECN_Metrics(400#, 52#, 0.9, "некислые", "slb", 0#, 730#)(1);
    Debug.Print UECN_Metrics(400#, 52#, 0.9, "некислые", "slb", 0#, 730#)(2);

    Debug.Print "Case B only-debit slb:";
    Debug.Print UECN_Metrics(400#, 0#, 0#, "некислые", "slb", 0#, 730#)(0);
    Debug.Print UECN_Metrics(400#, 0#, 0#, "некислые", "slb", 0#, 730#)(1);
    Debug.Print UECN_Metrics(400#, 0#, 0#, "некислые", "slb", 0#, 730#)(2);

    Debug.Print "Case C acidic slb:";
    Debug.Print UECN_Metrics(0#, 0#, 0#, "кислые", "slb", 0#, 730#)(0);
    Debug.Print UECN_Metrics(0#, 0#, 0#, "кислые", "slb", 0#, 730#)(1);
    Debug.Print UECN_Metrics(0#, 0#, 0#, "кислые", "slb", 0#, 730#)(2);
End Sub
