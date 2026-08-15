Attribute VB_Name = "ModuleFailureV5"
Option Explicit

' ============================================================================
'  ESP operating-life model, deploy form of the v5.2 fit.  Do NOT refit here.
'  Source of truth: results/production_risk_field_v52/<date>/tables/
'                     final_sets_rational.csv        Kpod and water-cut forms
'                     freq_final_coefficients.csv    frequency form
'                     deploy_tune_block.csv / deploy_qnom_block.csv
'
'    theta_total = level_contractor * ThetaQnom * ThetaFreq5 * ThetaKpod5 * ThetaWcut5
'    eta_eff     = eta0 * theta_total ^ (-1/beta0)
'    life (days) = RMST(0,730)
'
'  --- What changed from V4 --------------------------------------------------
'  1. THE SHAPES ARE NOW FITTED, and they live in this module as rational
'     coefficients.  V4 built its frequency curve out of the operator's own
'     anchors (a cubic in log theta through 40/50/60/70 Hz) and its Kpod curve
'     out of a plateau plus two arm exponents - shapes the operator DEFINED.
'     v5.2 fits both layers on Ya's counting-process frame (34 985 monthly
'     intervals, 1 226 runs, covariates lagged 3 months) and the shipped object
'     is the fitted curve.  The sheet therefore keeps LEVELS, not shape knobs.
'
'  2. TWO SHIPPED SETS PER LAYER.  "Bazovyi" and "Stress" are separate fits,
'     not one curve scaled: matching them at 35 Hz still leaves a 0.27 gap at
'     41 Hz.  Both coefficient sets are here; the scenario selector picks one.
'
'  3. OPERATOR CONTROL IS AN EXPONENT, applied per arm on top of the fit:
'
'         theta_deploy(x) = theta_fit(x) ^ gamma
'         gamma = Ln(theta_target) / Ln(theta_fit(x_anchor))
'
'     Three properties make this the right knob: theta(ref) stays exactly 1 for
'     any gamma (Ln theta_fit(ref) = 0); gamma = 1 reproduces the fit, so "no
'     opinion" is the default rather than a special case; and the arms are
'     independent, so 40 Hz and 60 Hz can be set without touching each other.
'     WHAT IT CANNOT DO: move where the curve turns.  The exponent scales log
'     theta, so extrema keep their position and only their depth changes.  A
'     target below 1 flips the arm's sign, and the clamp then flattens it to 1 -
'     which is the honest way to say "no penalty on this arm".
'
'  4. WATER CUT IS A NEW LAYER and it ships ON.  The switch is kept because the
'     layer is the weakest of the three: in DAYS its curve is much flatter than
'     the fact (571 days at 80 % against 497 at 30-60 %, where the model moves
'     life by 30), so it can be dropped without touching the other two.
'
'     The sheet has no water-cut column: it derives cut from cumulative recovery
'     through the HV curve, inside the oil-rate formula.  The renewal loop below
'     reproduces that same lookup from the oil-rate column plus NIZ and TIZ, so
'     no helper columns were added to either workbook.
'
'  5. Kpod REFERENCE MOVED to exactly 0.8 (V4 had a plateau 0.7..0.95).  The
'     fitted layer is pinned at 0.8 and has no plateau - it rises on BOTH sides,
'     shallowly.  Reported range 1.00..1.12 base, 1.00..1.47 stress.
'
'  6. FREQUENCY IS ON THE DEVIATION AXIS.  The fit is in Hz away from the pump's
'     nominal, clamped to -15..+20 (35..70 Hz at a 50 Hz nominal) and floored at
'     theta = 1, because the fitted base curve dips ~0.1 % below 1 at nominal and
'     a sub-1 multiplier at the reference point would be read as a bonus.
'
'  --- Unchanged from V4 ------------------------------------------------------
'  Per-field baselines and contractor levels (TuneBlock), the nameplate polyline
'  (QnomBlock), stratum key resolution with the Fleet fallback, the Qnom ladder
'  fallback to the LARGEST nominal, and the day-grid renewal bookkeeping.
'
'  --- Reference point --------------------------------------------------------
'  theta(Kpod 0.8) = 1 and theta(25 % water cut) = 1 exactly - both layers are
'  pinned there by construction.  Frequency is the exception: its numerator is not
'  multiplied by u, so the fitted value at nominal is whatever the fit produced -
'  0.9992 for the base curve (raised to 1 by the floor) and 1.0055 for stress.
'
'  Consequence, measured: under STRESS the reference life is 523.13 days against a
'  bare baseline of 524.02, i.e. 0.17 % shorter.  CM4_RmstRef and CM4_LifeAtRef
'  therefore agree under «Áàçîâûé» but not under «Ñòðåññ».  Left as is by explicit
'  decision - the deviation is far below the precision anything else is quoted at.
' ============================================================================

' --- v5.2 fitted layer coefficients ----------------------------------------
' Frequency, on the deviation axis:  u = Clamp(dev, -15, 20) / 10
'     Ln theta = (a0 + a1 u + ... ) / (1 + u * (b0 + b1 u + b2 u^2))
' Note the numerator is NOT multiplied by u here (it is for the other two
' layers), so theta(0) is a0 rather than exactly 1 - hence the floor at 1.
' Max |dTheta| against the fitted table: 0.0175 base, 0.0989 stress on a dense
' grid; 0.0007 / 0.0018 at 40 / 45 / 55 / 60 Hz, which is where it is read.
Private Const FA0_B As Double = -0.0008212625912
Private Const FA1_B As Double = 0.003900963071
Private Const FA2_B As Double = 0.2193743376
Private Const FA3_B As Double = -0.09858808433
Private Const FA4_B As Double = 0.3041728279
Private Const FB0_B As Double = -0.4746388993
Private Const FB1_B As Double = 1.848669429
Private Const FB2_B As Double = 0.1245599728

Private Const FA0_S As Double = 0.005526769523
Private Const FA1_S As Double = -0.1571024758
Private Const FA2_S As Double = 0.4723855629
Private Const FA3_S As Double = 0.005712809603
Private Const FA4_S As Double = 0#
Private Const FB0_S As Double = 0.08027070802
Private Const FB1_S As Double = 0.140555742
Private Const FB2_S As Double = -0.06156767127

' Kpod:  u = (Clamp(k, 0.4, 1.8) - 0.8) / 0.6
'     Ln theta = u * (a0 + a1 u) / (1 + u * (b0 + b1 u))
' Flat below 0.4 by decision: the fit has nothing under it there (the lowest
' equal-count bin starts at 0.45) and an unclamped rational keeps rising.
Private Const KA0_B As Double = -0.044287585
Private Const KA1_B As Double = 0.20526101
Private Const KB0_B As Double = -0.033041251
Private Const KB1_B As Double = 0.97026351

Private Const KA0_S As Double = -0.024292629
Private Const KA1_S As Double = 0.29992627
Private Const KB0_S As Double = 0.23177276
Private Const KB1_S As Double = 0#

' Water cut, per cent:  u = (Clamp(w, 0, 100) - 25) / 30
'     Ln theta = u * (a0 + a1 u) / (1 + u * (b0 + b1 u))
' One shipped shape only - there is no stress variant, because the layer has
' not been validated against the Svod and inventing a bound for it would be
' claiming more than we know.
Private Const WA0 As Double = 0.14802941
Private Const WA1 As Double = -0.10012055
Private Const WB0 As Double = 0.24529177
Private Const WB1 As Double = -0.054688025

Private Const FREQ_REF  As Double = 50#         ' pump nominal assumed by the sheets
Private Const FREQ_DLO  As Double = -15#        ' clamp on the deviation axis
Private Const FREQ_DHI  As Double = 20#
Private Const FREQ_ANC_LO As Double = -10#      ' 40 Hz - the lower anchor
Private Const FREQ_ANC_HI As Double = 10#       ' 60 Hz - the upper anchor

Private Const KPOD_REF As Double = 0.8
' BOTH curves continue to zero, and BOTH use the STRESS rational on the LOWER arm.  That is a
' consequence of the decision "theta(0) must reach 1.5", not a simplification:
'   * the base shape is too flat (1.166 at zero) - forcing 1.5 out of it needs gamma = 2.65,
'     which also lifts the middle of the arm (0.4: 1.087 -> 1.246, 0.6: 1.034 -> 1.093) and
'     makes the BASE scenario harsher than STRESS exactly where data exists.  Acceptance
'     caught it: base life 247 days against 270 for stress;
'   * the stress shape is steeper near zero and nearly identical near the reference, so the
'     same decision is reached by WEAKENING it (gamma = 0.495): 0.4 comes out at 1.092 against
'     the fit's 1.087, i.e. the data region is untouched and all the added severity sits below
'     Kpod 0.3.  Scenario ordering then holds BY CONSTRUCTION - one shape, gamma = 1 for stress
'     and gamma < 1 for base, so stress cannot end up softer.
' Upper arms stay their own: base 1.121 at Kpod 1.6, stress 1.466.
Private Const KPOD_LO  As Double = 0#
Private Const KPOD_HI  As Double = 1.8
Private Const KPOD_ANC_LO As Double = 0.2
Private Const KPOD_ANC_HI As Double = 1.6

Private Const WCUT_REF As Double = 25#
Private Const WCUT_LO  As Double = 0#
Private Const WCUT_HI  As Double = 100#
Private Const WCUT_ANC As Double = 95#

Private Const QN_GUARD As Double = 60#          ' first knot; flat below
Private Const QN_CAP   As Double = 1600#        ' last knot; flat above

' --- baseline fallback, used ONLY when no TuneBlock is passed ----------------
' V4 fell back to Vt's own strata here, which quietly priced an unwired sheet as
' if it were Vt.  v5.2 falls back to Fleet - the pooled fit over the whole stock -
' for the nonsour side, which is the honest default for an unidentified field.
' The sour side keeps Vt_sour because sour IS a Vt-only split.
' eta is the Weibull scale that reproduces the shipped RMST_ref at that beta:
'   Fleet   beta 1.0489102, RMST_ref 472.336276 -> eta  749.22430
'   Vt_sour beta 1.3399178, RMST_ref 177.696445 -> eta  193.78229
Private Const NB_BETA As Double = 1.0489102
Private Const NB_ETA  As Double = 749.2243
Private Const SB_BETA As Double = 1.3399178
Private Const SB_ETA  As Double = 193.78229
Private Const NB_SLB  As Double = 1.14944
Private Const NB_OTH  As Double = 2.53074
Private Const SB_SLB  As Double = 1.19628
Private Const SB_OTH  As Double = 1.74982

' --- CtrlBlock defaults (the fit's own values => gamma = 1 everywhere) -------
Private Const D_SET   As Double = 1#            ' 1 = base shape, 2 = stress shape
Private Const D_ONF   As Double = 1#            ' frequency layer on
Private Const D_T40   As Double = 1.2129        ' = ThetaFreqBase5(-10, base)
Private Const D_T60   As Double = 1.1869        ' = ThetaFreqBase5(+10, base)
Private Const D_ONK   As Double = 1#            ' Kpod layer on
' DECISION, not the fit: a fully unloaded pump must cost at least 1.5 on hazard.  On the
' stress-shaped lower arm that is gamma = 0.495, giving theta 0 -> 1.500, 0.2 -> 1.233,
' 0.4 -> 1.092, 0.6 -> 1.022 - the data region is left within 0.5 % of the fit and all the
' added severity sits below Kpod 0.3.
Private Const D_K02   As Double = 1.2326
Private Const D_K16   As Double = 1.1209        ' = ThetaKpodBase5(1.6, base)
Private Const D_ONW   As Double = 1#            ' water cut layer ON (switch kept, see note 4)
Private Const D_W95   As Double = 0.855         ' = ThetaWcutBase5(95)

Private Const REF_QNOM As Double = 250#
Private Const REF_FREQ As Double = 50#
Private Const REF_KPOD As Double = 0.8
Private Const REF_WCUT As Double = 25#
' ---------------------------------------------------------------------------
'  Public UDFs
' ---------------------------------------------------------------------------

' Operating life in days, RMST(0,730).
Public Function CM4_LifeDays(ByVal Sour As Double, ByVal contractor As String, _
    ByVal qnom As Double, ByVal freq As Double, ByVal kpod As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal CtrlBlock As Range = Nothing, _
    Optional ByVal QnomBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "", _
    Optional ByVal Wcut As Double = -1#) As Double
    Dim isSour As Boolean, b0 As Double, e0 As Double, key As String
    Dim brtC As Double, slbC As Double, othC As Double
    Dim iSet As Double, onF As Double, t40 As Double, t60 As Double
    Dim onK As Double, k02 As Double, k16 As Double, onW As Double, w95 As Double
    Dim theta As Double, etaEff As Double
    On Error GoTo EH
    isSour = (Sour >= 0.5)
    key = CM4_ResolveKey(FieldKey, Sour, TuneBlock, QnomBlock)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, key)
    Call ReadCtrl5(CtrlBlock, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    theta = ContractorMult4(contractor, brtC, slbC, othC) _
          * ThetaQnom(isSour, qnom, QnomBlock, key) _
          * ThetaOper5(freq, kpod, Wcut, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    If theta < 0.000001 Then theta = 0.000001
    etaEff = e0 * theta ^ (-1# / b0)
    CM4_LifeDays = RMST730v4(etaEff, b0)
    Exit Function
EH:
    CM4_LifeDays = CVErr(xlErrValue)
End Function

' Bare baseline life (all theta = 1).  NOT the life at 50 Hz while theta(50)
' differs from 1 - use CM4_LifeAtRef for that.
Public Function CM4_RmstRef(ByVal Sour As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "") As Double
    Dim isSour As Boolean, b0 As Double, e0 As Double
    Dim brtC As Double, slbC As Double, othC As Double
    isSour = (Sour >= 0.5)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, _
                   CM4_ResolveKey(FieldKey, Sour, TuneBlock))
    CM4_RmstRef = RMST730v4(e0, b0)
End Function

' Life at the reference operating point: brt, Qnom 250, 50 Hz, Kpod in plateau.
' This is what a sheet should divide by when it wants a relative multiplier.
Public Function CM4_LifeAtRef(ByVal Sour As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal CtrlBlock As Range = Nothing, _
    Optional ByVal QnomBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "") As Double
    ' The fitted layers are all pinned at the reference point, so it is a
    ' constant now instead of a plateau midpoint read out of the CtrlBlock.
    CM4_LifeAtRef = CM4_LifeDays(Sour, "brt", REF_QNOM, REF_FREQ, REF_KPOD, _
                                 TuneBlock, CtrlBlock, QnomBlock, FieldKey, REF_WCUT)
End Function

' Individual layers, so the sheet can chart them.
Public Function CM4_ThetaQnom(ByVal Sour As Double, ByVal qnom As Double, _
    Optional ByVal QnomBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "", _
    Optional ByVal TuneBlock As Range = Nothing) As Double
    CM4_ThetaQnom = ThetaQnom((Sour >= 0.5), qnom, QnomBlock, _
                              CM4_ResolveKey(FieldKey, Sour, TuneBlock, QnomBlock))
End Function

Public Function CM4_ThetaFreq(ByVal freq As Double, _
    Optional ByVal CtrlBlock As Range = Nothing) As Double
    Dim iSet As Double, onF As Double, t40 As Double, t60 As Double
    Dim onK As Double, k02 As Double, k16 As Double, onW As Double, w95 As Double
    Call ReadCtrl5(CtrlBlock, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    If onF < 0.5 Then CM4_ThetaFreq = 1#: Exit Function
    CM4_ThetaFreq = ThetaFreq5(freq, iSet, t40, t60)
End Function

Public Function CM4_ThetaKpod(ByVal kpod As Double, _
    Optional ByVal CtrlBlock As Range = Nothing) As Double
    Dim iSet As Double, onF As Double, t40 As Double, t60 As Double
    Dim onK As Double, k02 As Double, k16 As Double, onW As Double, w95 As Double
    Call ReadCtrl5(CtrlBlock, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    If onK < 0.5 Then CM4_ThetaKpod = 1#: Exit Function
    CM4_ThetaKpod = ThetaKpod5(kpod, iSet, k02, k16)
End Function

' Water-cut multiplier.  Returns 1 while the layer is switched off in 3.3, so a
' chart drawn against it shows a flat line rather than a curve the model is not
' actually applying.
Public Function CM4_ThetaWcut(ByVal Wcut As Double, _
    Optional ByVal CtrlBlock As Range = Nothing) As Double
    Dim iSet As Double, onF As Double, t40 As Double, t60 As Double
    Dim onK As Double, k02 As Double, k16 As Double, onW As Double, w95 As Double
    Call ReadCtrl5(CtrlBlock, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    If onW < 0.5 Then CM4_ThetaWcut = 1#: Exit Function
    CM4_ThetaWcut = ThetaWcut5(Wcut, w95)
End Function

Public Function CM4_Contractor(ByVal name As String, ByVal Sour As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "") As Double
    Dim isSour As Boolean, b0 As Double, e0 As Double
    Dim brtC As Double, slbC As Double, othC As Double
    isSour = (Sour >= 0.5)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, _
                   CM4_ResolveKey(FieldKey, Sour, TuneBlock))
    CM4_Contractor = ContractorMult4(name, brtC, slbC, othC)
End Function

' ---------------------------------------------------------------------------
'  Layers
' ---------------------------------------------------------------------------

' Nameplate layer: piecewise-linear in LOG theta between the 8 fitted knots,
' flat below 60 and above 1600.  Deliberately NOT a closed form: the fitted
' object IS a polyline, and interpolating it is exact where a smooth surrogate
' would invent curvature that was never estimated.
Private Function ThetaQnom(ByVal isSour As Boolean, ByVal qnom As Double, _
    ByVal QnomBlock As Range, Optional ByVal key As String = "") As Double
    Dim kn(0 To 7) As Double, th(0 To 7) As Double
    Dim q As Double, i As Long, w As Double
    If qnom <= 0# Then ThetaQnom = 1#: Exit Function
    kn(0) = 60#:  kn(1) = 100#:  kn(2) = 160#:  kn(3) = 250#
    kn(4) = 400#: kn(5) = 640#:  kn(6) = 1000#: kn(7) = 1600#
    If isSour Then
        ' th(7) is CLAMPED to th(6) (fitted 1.203401).  Free-fitted the sour arm turns DOWN
        ' above 1000 on 20 runs / 14 events, which would model a 1600 pump as longer-lived
        ' than a 1000 one.  Deployment guard, not a refit - see unified_v4.deploy_theta_qnom.
        th(0) = 0.555767: th(1) = 0.790572: th(2) = 1.051675: th(3) = 1#
        th(4) = 1.11554:  th(5) = 1.334801: th(6) = 1.366776: th(7) = 1.366776
    Else
        th(0) = 0.817254: th(1) = 0.754386: th(2) = 0.91978:  th(3) = 1#
        th(4) = 1.30472:  th(5) = 1.635608: th(6) = 1.63824:  th(7) = 1.826061
    End If
    Call ReadQnomBlock(QnomBlock, isSour, kn, th, key)

    q = Clamp4(qnom, QN_GUARD, QN_CAP)
    If q <= kn(0) Then ThetaQnom = th(0): Exit Function
    For i = 0 To 6
        If q <= kn(i + 1) Then
            w = (Log(q) - Log(kn(i))) / (Log(kn(i + 1)) - Log(kn(i)))
            ThetaQnom = Exp(Log(th(i)) * (1# - w) + Log(th(i + 1)) * w)
            Exit Function
        End If
    Next i
    ThetaQnom = th(7)
End Function

' Frequency layer, fitted rational on the deviation axis with a per-arm operator
' exponent.  See header notes 2, 3 and 6.
Private Function ThetaFreq5(ByVal f As Double, ByVal iSet As Double, _
    ByVal t40 As Double, ByVal t60 As Double) As Double
    Dim dev As Double, lt As Double, g As Double, anc As Double
    If f <= 0# Then ThetaFreq5 = 1#: Exit Function
    dev = Clamp4(f - FREQ_REF, FREQ_DLO, FREQ_DHI)
    lt = LnThetaFreq5(dev, iSet)
    If dev < 0# Then
        anc = LnThetaFreq5(FREQ_ANC_LO, iSet)
        g = ArmGamma5(anc, t40)
    ElseIf dev > 0# Then
        anc = LnThetaFreq5(FREQ_ANC_HI, iSet)
        g = ArmGamma5(anc, t60)
    Else
        g = 1#
    End If
    ThetaFreq5 = Exp(g * lt)
    If ThetaFreq5 < 1# Then ThetaFreq5 = 1#     ' floor: see header note 6
End Function

' WARNING: Horner is written as SEQUENTIAL assignments, not one nested expression.  The obvious
'   num = FA0 + u * (FA1 + u * (FA2 + u * (FA3 + u * FA4)))
' compiles but raises run-time error 16, "Expression too complex", every single time: VBA's
' expression-depth limit counts named constants, and four levels of them is past it.  It fails
' at every input, so it looks like a broken layer rather than a parser limit.  The Kpod and
' water-cut layers are only two deep and survive the nested form; they are unrolled here too
' so the three read identically and nobody re-nests one of them later.
Private Function LnThetaFreq5(ByVal dev As Double, ByVal iSet As Double) As Double
    Dim u As Double, num As Double, den As Double
    u = Clamp4(dev, FREQ_DLO, FREQ_DHI) / 10#
    If iSet >= 1.5 Then
        num = FA4_S
        num = FA3_S + u * num
        num = FA2_S + u * num
        num = FA1_S + u * num
        num = FA0_S + u * num
        den = FB2_S
        den = FB1_S + u * den
        den = FB0_S + u * den
        den = 1# + u * den
    Else
        num = FA4_B
        num = FA3_B + u * num
        num = FA2_B + u * num
        num = FA1_B + u * num
        num = FA0_B + u * num
        den = FB2_B
        den = FB1_B + u * den
        den = FB0_B + u * den
        den = 1# + u * den
    End If
    If den < 0.2 Then den = 0.2                 ' pole guard; the fits stay well clear
    LnThetaFreq5 = num / den
End Function

' Kpod layer.  Pinned at 0.8 by construction (u = 0 => Ln theta = 0), flat below
' 0.4, per-arm operator exponent.
Private Function ThetaKpod5(ByVal k As Double, ByVal iSet As Double, _
    ByVal k02 As Double, ByVal k16 As Double) As Double
    Dim kk As Double, lt As Double, g As Double, anc As Double
    If k < 0# Then ThetaKpod5 = 1#: Exit Function
    kk = Clamp4(k, KPOD_LO, KPOD_HI)
    lt = LnThetaKpod5(kk, iSet)
    If kk < KPOD_REF Then
        anc = LnThetaKpod5(KPOD_ANC_LO, iSet)
        g = ArmGamma5(anc, k02)
    ElseIf kk > KPOD_REF Then
        anc = LnThetaKpod5(KPOD_ANC_HI, iSet)
        g = ArmGamma5(anc, k16)
    Else
        g = 1#
    End If
    ThetaKpod5 = Exp(g * lt)
    If ThetaKpod5 < 0.000001 Then ThetaKpod5 = 0.000001
End Function

' Coefficients are picked PER ARM, not per scenario: the lower arm is the stress rational for
' both scenarios (see the KPOD_LO note), the upper arm is each scenario's own.
Private Function LnThetaKpod5(ByVal k As Double, ByVal iSet As Double) As Double
    Dim u As Double, num As Double, den As Double, kk As Double
    kk = Clamp4(k, KPOD_LO, KPOD_HI)
    u = (kk - KPOD_REF) / 0.6
    If (kk < KPOD_REF) Or (iSet >= 1.5) Then
        num = KA1_S
        num = KA0_S + u * num
        den = KB1_S
        den = KB0_S + u * den
    Else
        num = KA1_B
        num = KA0_B + u * num
        den = KB1_B
        den = KB0_B + u * den
    End If
    num = u * num
    den = 1# + u * den
    If den < 0.2 Then den = 0.2
    LnThetaKpod5 = num / den
End Function

' Water-cut layer, one shipped shape, single operator exponent anchored at 95 %.
' Accepts either a fraction or per cent: <= 1 is read as a fraction, which makes
' exactly 1 ambiguous (1 % or 100 %) and resolves it as 100 % - the same rule the
' fitting frame uses, so the two cannot disagree.
Private Function ThetaWcut5(ByVal w As Double, ByVal w95 As Double) As Double
    Dim ww As Double, lt As Double, g As Double
    If w < 0# Then ThetaWcut5 = 1#: Exit Function
    ww = w: If ww <= 1# Then ww = ww * 100#
    ww = Clamp4(ww, WCUT_LO, WCUT_HI)
    lt = LnThetaWcut5(ww)
    g = ArmGamma5(LnThetaWcut5(WCUT_ANC), w95)
    ThetaWcut5 = Exp(g * lt)
    If ThetaWcut5 < 0.000001 Then ThetaWcut5 = 0.000001
End Function

Private Function LnThetaWcut5(ByVal w As Double) As Double
    Dim u As Double, den As Double
    Dim num As Double
    u = (Clamp4(w, WCUT_LO, WCUT_HI) - WCUT_REF) / 30#
    num = WA1
    num = WA0 + u * num
    num = u * num
    den = WB1
    den = WB0 + u * den
    den = 1# + u * den
    If den < 0.2 Then den = 0.2
    LnThetaWcut5 = num / den
End Function

' gamma that maps the fitted anchor onto the operator's target, in log theta.
' No leverage when the fit itself is flat at the anchor (Ln theta ~ 0): the
' exponent multiplies zero, so any target is unreachable and 1 is the honest
' answer rather than a division blow-up.
Private Function ArmGamma5(ByVal lnFit As Double, ByVal target As Double) As Double
    If Abs(lnFit) < 0.000001 Then ArmGamma5 = 1#: Exit Function
    If target <= 0# Then ArmGamma5 = 1#: Exit Function
    ArmGamma5 = Log(target) / lnFit
End Function

' Water cut at a point on the recovery curve, reproducing the sheet's own
' XLOOKUP(cum_share, HV!P, HV!Q, , 1) - match mode 1 is smallest-greater-or-equal,
' NOT nearest.  Returns -1 when the table is unusable, which the callers read as
' "no water-cut information" and pass through as theta = 1.
Private Function WcutFromHv5(ByVal cumOil As Double, ByVal niz As Double, _
    ByVal tiz As Double, ByRef hv As Variant) As Double
    Dim share As Double, r As Long, bestX As Double, bestY As Double
    Dim x As Double, y As Double, found As Boolean, lastY As Double
    If niz <= 0# Then WcutFromHv5 = -1#: Exit Function
    share = (cumOil + (niz - tiz)) / niz
    found = False: lastY = -1#
    For r = LBound(hv, 1) To UBound(hv, 1)
        If IsNumeric(hv(r, 1)) And IsNumeric(hv(r, 2)) Then
            x = CDbl(hv(r, 1)): y = CDbl(hv(r, 2))
            lastY = y
            If x >= share Then
                If (Not found) Or (x < bestX) Then bestX = x: bestY = y: found = True
            End If
        End If
    Next r
    If found Then
        WcutFromHv5 = bestY
    ElseIf lastY >= 0# Then
        WcutFromHv5 = lastY                     ' past the end of the curve: hold the last cut
    Else
        WcutFromHv5 = -1#
    End If
End Function

' ---------------------------------------------------------------------------
'  Day-grid renewal schedule (v4 replacement for FailureScheduleDyn)
' ---------------------------------------------------------------------------
' Returns a column of 1 (running) / 0 (failure day) over a daily grid, consuming
' 1/life per day and firing when the accumulated fraction reaches 1, then
' carrying the remainder into the next run.  Identical bookkeeping to the v1
' FailureScheduleDyn - only the hazard layers change.
'
' The nameplate needs no new input column: the v1 function ALREADY derives it,
' looking the standard nominal up from Ql at each restart edge in order to form
' Kpod = Ql/nominal.  v4 simply feeds that same nominal to ThetaQnom instead of
' feeding Ql to ThetaQl.  So the sheet's existing arguments carry over unchanged
' and the two versions can be compared column against column.
'
'   I15 = CM4_FailureScheduleDyn($C$15:$C$1108,$F$15:$F$1108,$G$15:$G$1108,$K$4,
'                                KRS!$B$41:$B$57,$M$4,$N$4,1,TuneBlock,CtrlBlock,QnomBlock,
'                                $N$4)          '  <- UN / Uchastok nedr
Public Function CM4_FailureScheduleDyn(ByVal QlRange As Range, _
    ByVal ContractorRange As Range, ByVal FreqRange As Range, _
    ByVal InitNominal As Double, ByVal NominalTable As Range, _
    ByVal DownDays As Double, ByVal Sour As Double, _
    Optional ByVal StepDays As Double = 1#, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal CtrlBlock As Range = Nothing, _
    Optional ByVal QnomBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "", _
    Optional ByVal OilRange As Range = Nothing, _
    Optional ByVal HvTable As Range = Nothing, _
    Optional ByVal Niz As Double = 0#, _
    Optional ByVal Tiz As Double = 0#) As Variant
    Dim vq As Variant, vc As Variant, vf As Variant, tbl As Variant
    Dim vo As Variant, hv As Variant, useW As Boolean
    Dim i As Long, n As Long, dd As Long, key As String
    Dim ql As Double, freq As Double, kpod As Double, contractor As String
    Dim wcut As Double, cumOil As Double
    Dim isSour As Boolean, theta As Double, b0 As Double, e0 As Double
    Dim brtC As Double, slbC As Double, othC As Double
    Dim iSet As Double, onF As Double, t40 As Double, t60 As Double
    Dim onK As Double, k02 As Double, k16 As Double, onW As Double, w95 As Double
    Dim etaEff As Double, life As Double, consumed As Double, nominal As Double
    Dim res() As Variant, iv() As Long
    On Error GoTo CalcErr
    If StepDays <= 0 Then StepDays = 1#
    dd = CLng(DownDays): If dd < 1 Then dd = 1
    isSour = (Sour >= 0.5)
    key = CM4_ResolveKey(FieldKey, Sour, TuneBlock, QnomBlock)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, key)
    Call ReadCtrl5(CtrlBlock, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
    vq = QlRange.Value2: vc = ContractorRange.Value2: vf = FreqRange.Value2
    tbl = NominalTable.Value2
    useW = (onW >= 0.5) And (Not OilRange Is Nothing) And (Not HvTable Is Nothing) And (Niz > 0#)
    If useW Then vo = OilRange.Value2: hv = HvTable.Value2
    cumOil = 0#
    n = UBound(vq, 1)
    ReDim res(1 To n, 1 To 1)
    ReDim iv(1 To n)
    consumed = 0#
    nominal = InitNominal
    For i = 1 To n
        ' dynamic nominal reset at the restart edge (J[i-1]=1 AND J[i-2]=0)
        If i >= 2 Then
            If Jflag4(iv, i - 1, dd) = 1 And Jflag4(iv, i - 2, dd) = 0 Then
                nominal = LookupNominal4(SafeNum4(vq(i, 1)), tbl)
            End If
        End If
        ql = SafeNum4(vq(i, 1))
        contractor = CStr(vc(i, 1))
        freq = SafeNum4(vf(i, 1))
        If nominal <= 0# Then nominal = InitNominal
        kpod = ql / nominal
        ' Water cut is read off the recovery curve at the oil produced BEFORE this
        ' day, which is exactly the argument the sheet's own oil-rate formula uses.
        wcut = -1#
        If useW Then
            wcut = WcutFromHv5(cumOil, Niz, Tiz, hv)
            cumOil = cumOil + SafeNum4(vo(i, 1)) * StepDays
        End If
        theta = ContractorMult4(contractor, brtC, slbC, othC) _
              * ThetaQnom(isSour, nominal, QnomBlock, key) _
              * ThetaOper5(freq, kpod, wcut, iSet, onF, t40, t60, onK, k02, k16, onW, w95)
        If theta < 0.000001 Then theta = 0.000001
        etaEff = e0 * theta ^ (-1# / b0)
        life = RMST730v4(etaEff, b0) / StepDays
        If life < 0.000001 Then life = 0.000001
        consumed = consumed + 1# / life
        If consumed >= 1# Then
            iv(i) = 0: res(i, 1) = 0: consumed = consumed - 1#
        Else
            iv(i) = 1: res(i, 1) = 1
        End If
    Next i
    CM4_FailureScheduleDyn = res
    Exit Function
CalcErr:
    CM4_FailureScheduleDyn = CVErr(xlErrValue)
End Function

' Uptime J at a given day: 0 if any failure (iv=0) within the last dd days.
' Days before 1 count as running.  Only reads already-computed iv entries.
Private Function Jflag4(ByRef iv() As Long, ByVal day As Long, ByVal dd As Long) As Long
    Dim lo As Long, j As Long
    If day < 1 Then Jflag4 = 1: Exit Function
    lo = day - dd + 1: If lo < 1 Then lo = 1
    For j = lo To day
        If iv(j) = 0 Then Jflag4 = 0: Exit Function
    Next j
    Jflag4 = 1
End Function

' Smallest standard nominal >= ql (exact-or-next-larger).  Above the top of the
' ladder there is no such pump, so fall back to the LARGEST catalogue nominal.
'
' The old fallback was 999999, which made Kpod = ql/999999 collapse to the 0.15
' clamp (theta_Kpod ~ 2.3) while theta_Qnom pinned at the 1600 cap - a scenario
' with Ql above the ladder lost ~40 % of its life for a purely cosmetic reason.
' The largest ladder entry is the honest reading: the pump is undersized, so
' Kpod > 1, which both layers can express.  No observed run in any field exceeds
' Ql 1500, so this only ever fires on a user-entered scenario.
Private Function LookupNominal4(ByVal ql As Double, ByRef tbl As Variant) As Double
    Dim r As Long, v As Double, best As Double, found As Boolean
    Dim topNom As Double
    found = False: best = 0#: topNom = 0#
    For r = LBound(tbl, 1) To UBound(tbl, 1)
        If IsNumeric(tbl(r, 1)) Then
            v = CDbl(tbl(r, 1))
            If v > topNom Then topNom = v
            If v >= ql Then
                If (Not found) Or (v < best) Then best = v: found = True
            End If
        End If
    Next r
    If found Then
        LookupNominal4 = best
    ElseIf topNom > 0# Then
        LookupNominal4 = topNom
    Else
        LookupNominal4 = 999999#          ' empty or invalid nominal table
    End If
End Function

Private Function ContractorMult4(ByVal name As String, ByVal brtC As Double, _
    ByVal slbC As Double, ByVal othC As Double) As Double
    Dim nm As String
    nm = LCase$(Trim$(name))
    If Len(nm) = 0 Then ContractorMult4 = brtC: Exit Function
    If IsBrt4(nm) Then
        ContractorMult4 = brtC
    ElseIf IsSlb4(nm) Then
        ContractorMult4 = slbC
    Else
        ContractorMult4 = othC
    End If
End Function

' ---------------------------------------------------------------------------
'  Parameter blocks
' ---------------------------------------------------------------------------

' Stratum key for a (field, sour) pair.  The workbooks' field cell holds a
' calculator code - Au, Az, Ic, Mc, Mr, Vt, Ya - and the parameter blocks are keyed by
' stratum, which is the same string except where a field is split on H2S (only Vt is).
' Resolution order, so a half-filled block degrades instead of erroring:
'   1. "<field>_sour" / "<field>_nonsour"
'   2. "<field>"
'   3. "Fleet" - the POOLED fit over every run in the mart.  Several codes the workbooks
'      price (Ki, Ma, Bt, and thin Da) have no field model at all; falling back to Vt as
'      this used to would price them on a sour-capable field with an unusually steep
'      nameplate slope.  Fleet is the honest default for an unmodelled field.
'   4. the Vt row, only if the block carries no Fleet row either.
Public Function CM4_ResolveKey(ByVal FieldKey As String, ByVal Sour As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal QnomBlock As Range = Nothing) As String
    Dim isSour As Boolean, suffixed As String, bare As String
    isSour = (Sour >= 0.5)
    bare = Trim$(FieldKey)
    suffixed = bare & IIf(isSour, "_sour", "_nonsour")
    If Len(bare) > 0 Then
        If KeyRow4(TuneBlock, suffixed) > 0 Or KeyRow4(QnomBlock, suffixed) > 0 Then
            CM4_ResolveKey = suffixed
            Exit Function
        ElseIf KeyRow4(TuneBlock, bare) > 0 Or KeyRow4(QnomBlock, bare) > 0 Then
            CM4_ResolveKey = bare
            Exit Function
        End If
    End If
    If KeyRow4(TuneBlock, "Fleet") > 0 Or KeyRow4(QnomBlock, "Fleet") > 0 Then
        CM4_ResolveKey = "Fleet"
    Else
        CM4_ResolveKey = IIf(isSour, "Vt_sour", "Vt_nonsour")
    End If
End Function

' Row index of a key inside a keyed block (column 1 holds the key), else 0.
' A block whose first column is numeric is the LEGACY layout and never matches.
Private Function KeyRow4(ByVal blk As Range, ByVal key As String) As Long
    Dim v As Variant, r As Long
    KeyRow4 = 0
    If blk Is Nothing Then Exit Function
    On Error GoTo EH
    v = blk.Value2
    For r = LBound(v, 1) To UBound(v, 1)
        If VarType(v(r, 1)) = vbString Then
            If LCase$(Trim$(CStr(v(r, 1)))) = LCase$(Trim$(key)) Then
                KeyRow4 = r
                Exit Function
            End If
        End If
    Next r
    Exit Function
EH:
    KeyRow4 = 0
End Function

' TuneBlock, two accepted layouts:
'   LEGACY  2 rows x 5 cols  [beta, RMST_ref, brt, slb, oth]; row1 nonsour, row2 sour.
'   KEYED   N rows x 6 cols  [key, beta, RMST_ref, brt, slb, oth]  - one row per stratum,
'           which is what makes the workbook's UN cell switch the model rather than just
'           the downtime and the repair cost.
' Nothing, or a key that is not in the block -> the unified v4 Vt defaults.
Private Sub ReadTune4(ByVal TuneBlock As Range, ByVal isSour As Boolean, _
    ByRef b0 As Double, ByRef e0 As Double, _
    ByRef brtC As Double, ByRef slbC As Double, ByRef othC As Double, _
    Optional ByVal key As String = "")
    Dim tb As Variant, ri As Long, c0 As Long, rmstRef As Double
    If isSour Then
        b0 = SB_BETA: e0 = SB_ETA: brtC = 1#: slbC = SB_SLB: othC = SB_OTH
    Else
        b0 = NB_BETA: e0 = NB_ETA: brtC = 1#: slbC = NB_SLB: othC = NB_OTH
    End If
    If TuneBlock Is Nothing Then Exit Sub
    On Error GoTo EH
    tb = TuneBlock.Value2
    If Len(key) > 0 And TuneBlock.Columns.Count >= 6 Then
        ri = KeyRow4(TuneBlock, key)
        If ri = 0 Then Exit Sub
        c0 = 1                                  ' keyed: values start in column 2
    Else
        If TuneBlock.Rows.Count < 2 Or TuneBlock.Columns.Count < 5 Then Exit Sub
        If isSour Then ri = 2 Else ri = 1
        c0 = 0
    End If
    If SafeNum4(tb(ri, c0 + 1)) > 0# Then b0 = SafeNum4(tb(ri, c0 + 1))
    rmstRef = SafeNum4(tb(ri, c0 + 2))
    If rmstRef > 0# Then e0 = EtaFromRmst4(rmstRef, b0)
    If SafeNum4(tb(ri, c0 + 3)) > 0# Then brtC = SafeNum4(tb(ri, c0 + 3))
    If SafeNum4(tb(ri, c0 + 4)) > 0# Then slbC = SafeNum4(tb(ri, c0 + 4))
    If SafeNum4(tb(ri, c0 + 5)) > 0# Then othC = SafeNum4(tb(ri, c0 + 5))
    Exit Sub
EH:
End Sub

' CtrlBlock = 9 cells (a column or a row), in order:
'   1  curve set                   1 = base shape, 2 = stress shape
'   2  frequency layer on/off      0 = off (theta = 1), otherwise on
'   3  theta_freq @ 40 Hz          lower-arm anchor
'   4  theta_freq @ 60 Hz          upper-arm anchor
'   5  Kpod layer on/off
'   6  theta_Kpod @ 0.2            lower-arm anchor
'   7  theta_Kpod @ 1.6            upper-arm anchor
'   8  water-cut layer on/off      SHIPPED OFF
'   9  theta_wcut @ 95 %           single anchor for the whole curve
'
' The anchors are TARGETS, not coefficients: each one is turned into an exponent
' on the fitted curve for its own arm, so the reference point stays exactly 1 and
' leaving an anchor at its shipped value reproduces the fit.
'
' A block shorter than 9 cells is ignored entirely rather than half-applied, so a
' sheet still carrying the V4 10-cell layout falls back to the defaults above
' instead of reading the old plateau numbers as if they were anchors.
Private Sub ReadCtrl5(ByVal CtrlBlock As Range, _
    ByRef iSet As Double, ByRef onF As Double, ByRef t40 As Double, ByRef t60 As Double, _
    ByRef onK As Double, ByRef k02 As Double, ByRef k16 As Double, _
    ByRef onW As Double, ByRef w95 As Double)
    Dim flat(1 To 9) As Double, n As Long, c As Range
    iSet = D_SET: onF = D_ONF: t40 = D_T40: t60 = D_T60
    onK = D_ONK: k02 = D_K02: k16 = D_K16
    onW = D_ONW: w95 = D_W95
    If CtrlBlock Is Nothing Then Exit Sub
    On Error GoTo EH
    If CtrlBlock.Cells.Count < 9 Then Exit Sub
    n = 0
    For Each c In CtrlBlock.Cells
        n = n + 1
        If n > 9 Then Exit For
        flat(n) = SafeNum4(c.Value2)
    Next c
    If flat(1) > 0# Then iSet = flat(1)
    onF = flat(2)                               ' 0 is a legitimate value: layer off
    If flat(3) > 0# Then t40 = flat(3)
    If flat(4) > 0# Then t60 = flat(4)
    onK = flat(5)
    If flat(6) > 0# Then k02 = flat(6)
    If flat(7) > 0# Then k16 = flat(7)
    onW = flat(8)
    If flat(9) > 0# Then w95 = flat(9)
    Exit Sub
EH:
End Sub

' The three layers as one factor, honouring the on/off switches.  Kept in one
' place so the point calculation and the renewal loop cannot drift apart.
Private Function ThetaOper5(ByVal freq As Double, ByVal kpod As Double, ByVal wcut As Double, _
    ByVal iSet As Double, ByVal onF As Double, ByVal t40 As Double, ByVal t60 As Double, _
    ByVal onK As Double, ByVal k02 As Double, ByVal k16 As Double, _
    ByVal onW As Double, ByVal w95 As Double) As Double
    Dim th As Double
    th = 1#
    If onF >= 0.5 Then th = th * ThetaFreq5(freq, iSet, t40, t60)
    If onK >= 0.5 Then th = th * ThetaKpod5(kpod, iSet, k02, k16)
    If onW >= 0.5 And wcut >= 0# Then th = th * ThetaWcut5(wcut, w95)
    ThetaOper5 = th
End Function

' QnomBlock, two accepted layouts (the knot row is shared by every stratum on purpose -
' complexity is controlled by the fit's ridge, never by moving knots per field, so the block
' stays a rectangle):
'   LEGACY  3 rows x 8 cols: row1 knots, row2 nonsour theta, row3 sour theta.
'   KEYED   N rows x 9 cols: row1 [label, knots...], rows 2.. [key, theta...].
Private Sub ReadQnomBlock(ByVal QnomBlock As Range, ByVal isSour As Boolean, _
    ByRef kn() As Double, ByRef th() As Double, Optional ByVal key As String = "")
    Dim v As Variant, i As Long, ri As Long, c0 As Long, kv As Double, tv As Double
    If QnomBlock Is Nothing Then Exit Sub
    On Error GoTo EH
    v = QnomBlock.Value2
    If Len(key) > 0 And QnomBlock.Columns.Count >= 9 Then
        ri = KeyRow4(QnomBlock, key)
        If ri = 0 Then Exit Sub
        c0 = 1
    Else
        If QnomBlock.Rows.Count < 3 Or QnomBlock.Columns.Count < 8 Then Exit Sub
        If isSour Then ri = 3 Else ri = 2
        c0 = 0
    End If
    For i = 0 To 7
        kv = SafeNum4(v(1, c0 + i + 1))
        tv = SafeNum4(v(ri, c0 + i + 1))
        If kv > 0# And tv > 0# Then
            kn(i) = kv
            th(i) = tv
        End If
    Next i
    Exit Sub
EH:
End Sub

' ---------------------------------------------------------------------------
'  Numerics (self-contained: this module must not depend on ModuleFailureV1)
' ---------------------------------------------------------------------------

Private Function RMST730v4(ByVal eta As Double, ByVal beta As Double) As Double
    Dim a As Double, x As Double
    If eta <= 0# Or beta <= 0# Then RMST730v4 = 0#: Exit Function
    a = 1# / beta
    x = (730# / eta) ^ beta
    RMST730v4 = eta * Exp(GammaLn4(a + 1#)) * GammP4(a, x) / 1#
    If RMST730v4 > 730# Then RMST730v4 = 730#
    If RMST730v4 < 0# Then RMST730v4 = 0#
End Function

Private Function EtaFromRmst4(ByVal targetRmst As Double, ByVal beta As Double) As Double
    Dim lo As Double, hi As Double, mid As Double, i As Long
    If targetRmst <= 0# Then EtaFromRmst4 = 1#: Exit Function
    lo = 1#: hi = 100000#
    For i = 1 To 200
        mid = 0.5 * (lo + hi)
        If RMST730v4(mid, beta) < targetRmst Then lo = mid Else hi = mid
    Next i
    EtaFromRmst4 = 0.5 * (lo + hi)
End Function

Private Function GammaLn4(ByVal xx As Double) As Double
    Dim cof(0 To 5) As Double, x As Double, y As Double, tmp As Double, ser As Double, j As Long
    cof(0) = 76.18009172947146: cof(1) = -86.50532032941677
    cof(2) = 24.01409824083091: cof(3) = -1.231739572450155
    cof(4) = 0.1208650973866179E-2: cof(5) = -0.5395239384953E-5
    x = xx: y = x
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * Log(tmp)
    ser = 1.000000000190015
    For j = 0 To 5
        y = y + 1#
        ser = ser + cof(j) / y
    Next j
    GammaLn4 = -tmp + Log(2.5066282746310005 * ser / x)
End Function

Private Function GammP4(ByVal a As Double, ByVal x As Double) As Double
    If x <= 0# Then GammP4 = 0#: Exit Function
    If x < a + 1# Then
        GammP4 = gser4(a, x)
    Else
        GammP4 = 1# - gcf4(a, x)
    End If
End Function

Private Function gser4(ByVal a As Double, ByVal x As Double) As Double
    Dim ap As Double, sum As Double, del As Double, n As Long
    ap = a: sum = 1# / a: del = sum
    For n = 1 To 500
        ap = ap + 1#
        del = del * x / ap
        sum = sum + del
        If Abs(del) < Abs(sum) * 0.0000000001 Then Exit For
    Next n
    gser4 = sum * Exp(-x + a * Log(x) - GammaLn4(a))
End Function

Private Function gcf4(ByVal a As Double, ByVal x As Double) As Double
    Dim b As Double, c As Double, d As Double, h As Double
    Dim an As Double, del As Double, i As Long
    b = x + 1# - a
    c = 1E+30
    d = 1# / b
    h = d
    For i = 1 To 500
        an = -i * (i - a)
        b = b + 2#
        d = an * d + b
        If Abs(d) < 0.0000000000000000000000000000001 Then d = 0.0000000000000000000000000000001
        c = b + an / c
        If Abs(c) < 0.0000000000000000000000000000001 Then c = 0.0000000000000000000000000000001
        d = 1# / d
        del = d * c
        h = h * del
        If Abs(del - 1#) < 0.0000000001 Then Exit For
    Next i
    gcf4 = Exp(-x + a * Log(x) - GammaLn4(a)) * h
End Function

Private Function Clamp4(ByVal x As Double, ByVal lo As Double, ByVal hi As Double) As Double
    If x < lo Then
        Clamp4 = lo
    ElseIf x > hi Then
        Clamp4 = hi
    Else
        Clamp4 = x
    End If
End Function

Private Function SafeNum4(ByVal v As Variant) As Double
    On Error GoTo EH
    If IsEmpty(v) Or IsNull(v) Then SafeNum4 = 0#: Exit Function
    If IsNumeric(v) Then SafeNum4 = CDbl(v) Else SafeNum4 = 0#
    Exit Function
EH:
    SafeNum4 = 0#
End Function

' Lowercase + strip everything that is not a latin/cyrillic letter or a digit.
' Identical to ModuleFailureV1.NormName, deliberately: the two modules must classify a
' contractor name the same way or their outputs are not comparable column against column.
Private Function NormName4(ByVal s As String) As String
    Dim i As Long, code As Long, ch As String, out As String
    s = LCase$(Trim$(s))
    out = ""
    For i = 1 To Len(s)
        ch = mid$(s, i, 1)
        code = AscW(ch)
        If (code >= 97 And code <= 122) Or (code >= 48 And code <= 57) _
           Or (code >= 1072 And code <= 1103) Or code = 1105 Then
            out = out & ch
        End If
    Next i
    NormName4 = out
End Function

' Borets (RU "Borec") / Borets / Borec / BRT -> brt
'
' The needles MUST be lowercase Cyrillic (bor = 1073/1086/1088), because NormName4 has
' already lowercased the input.  An earlier version of this module compared against capital
' capital "Bor" (1041/1086/1088) and so matched NOTHING in Cyrillic: every Cyrillic
' "Borec" fell through to `oth`, priced 1.91 instead of 1.0.  The plan sheet is "Borec" on all
' 1094 rows, so that single case error nearly doubled the hazard of the entire forecast
' without raising anything.
Private Function IsBrt4(ByVal nm As String) As Boolean
    Dim s As String
    s = NormName4(nm)
    IsBrt4 = (InStr(s, ChrW(1073) & ChrW(1086) & ChrW(1088)) > 0) _
        Or (InStr(s, "borets") > 0) Or (InStr(s, "borec") > 0) Or (InStr(s, "brt") > 0)
End Function

' Schlumberger (RU forms: shlyu / slayb / slb) / SLB -> slb
Private Function IsSlb4(ByVal nm As String) As Boolean
    Dim s As String
    s = NormName4(nm)
    IsSlb4 = (InStr(s, ChrW(1096) & ChrW(1083) & ChrW(1102)) > 0) _
        Or (InStr(s, ChrW(1089) & ChrW(1083) & ChrW(1072) & ChrW(1081)) > 0) _
        Or (InStr(s, ChrW(1089) & ChrW(1083) & ChrW(1073)) > 0) _
        Or (InStr(s, "slb") > 0) Or (InStr(s, "schlumberger") > 0)
End Function
