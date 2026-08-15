"""Derive ``vba/ModuleFailureV5.bas`` from V4 by replacing only the layer machinery.

    python scripts/deploy/build_module_v5.py

Why a transformation instead of a hand-written module: two thirds of V4 — key resolution,
TuneBlock/QnomBlock readers, the RMST/eta inversion, the incomplete-gamma helpers, the day-grid
renewal loop — is unchanged and already in production.  Retyping it would risk a silent
transcription error in code no one would re-read.  Every replacement below asserts its anchor
text is present, so a drift in V4 fails the build instead of producing a half-patched module.

What actually changes in V5 (the v5.2 fit):

* **Frequency and Kпод become fitted rationals, not operator closed forms.**  V4 built the
  frequency curve out of the operator's own anchors (a cubic through θ@40/50/60/70) and the
  Kпод curve out of a plateau plus two arm exponents.  Those were shapes the operator
  *defined*.  v5.2 fits both on Ya's counting-process frame, and the shipped object is the
  fitted curve.  So the shape moves into code and the sheet keeps only levels.

* **Two shipped shapes per layer, not one.**  «Базовый» and «Стресс» are separate fits — the
  stress curve is NOT the base curve raised to a power (checked: matching them at 35 Hz leaves
  a 0.27 gap at 41 Hz), so both coefficient sets live here and the scenario selects one.

* **The operator keeps two-sided control through an exponent.**  θ_deploy = θ_fit ^ γ with
  γ = Ln(target) / Ln(θ_fit(anchor)), applied independently below and above the reference.
  θ(ref) = 1 for any γ, and γ = 1 reproduces the fit exactly, so "no opinion" is the default.
  ⚠ It cannot move where a curve turns — only how deep it goes.

* **Water cut is a new layer** with an on/off switch, shipped OFF.  The sheet never carried a
  water-cut column: it derives cut from cumulative recovery through the ХВ curve inside the
  oil-rate formula.  Rather than add ~4 400 helper formulas across two workbooks, the renewal
  loop reproduces that same lookup internally from the oil-rate column, НИЗ and ТИЗ.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "vba" / "ModuleFailureV4.bas"
DST = REPO_ROOT / "vba" / "ModuleFailureV5.bas"
ENC = "cp1251"

HEADER = '''Attribute VB_Name = "ModuleFailureV5"
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
'  therefore agree under «Базовый» but not under «Стресс».  Left as is by explicit
'  decision - the deviation is far below the precision anything else is quoted at.
' ============================================================================
'''

CONSTS = '''
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
'''

LAYERS = '''
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
'''

CTRL_DOC = """' CtrlBlock = 9 cells (a column or a row), in order:
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
"""

READCTRL = '''Private Sub ReadCtrl5(ByVal CtrlBlock As Range, _
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

'''


def cut(text: str, start: str, end: str) -> str:
    i, j = text.index(start), text.index(end)
    return text[i:j]


def main() -> int:
    src = SRC.read_text(encoding=ENC)
    out = src

    # 1. header + constants ---------------------------------------------------
    out = HEADER + CONSTS + out[out.index("' --- unified v4 fitted constants"):]
    old_consts = cut(out, "' --- unified v4 fitted constants", "' -------------------------"
                     "--------------------------------------------------\n'  Public UDFs")
    out = out.replace(old_consts, "")

    # 2. layer functions ------------------------------------------------------
    old_freq = cut(out, "' Frequency layer: polynomial in LOG theta",
                   "' ---------------------------------------------------------------------------\n"
                   "'  Day-grid renewal schedule")
    out = out.replace(old_freq, LAYERS.lstrip("\n") + "\n")

    # 3. CtrlBlock reader -----------------------------------------------------
    # the doc comment above the reader describes the OLD 10-cell layout, so it is
    # replaced together with the reader rather than left to contradict the code
    old_ctrl = cut(out, "' CtrlBlock = 10 cells", "' QnomBlock, two accepted layouts")
    out = out.replace(old_ctrl, CTRL_DOC + READCTRL)

    # 4. call sites -----------------------------------------------------------
    for old, new in CALLSITES:
        assert old in out, old[:70]
        out = out.replace(old, new)

    assert "ReadCtrl4" not in out, "a call to the old CtrlBlock reader survived"
    assert "ThetaFreq4" not in out and "ThetaKpod4" not in out, "an old layer call survived"
    DST.write_text(out, encoding=ENC)
    print(f"{DST}  ({out.count(chr(10))} lines, {len(out)} bytes, {ENC})")
    return 0


# --- the public UDFs, rewritten around the new layer set ---------------------
CALLSITES = [
    # CM4_LifeDays: new optional water-cut argument, appended so existing sheet
    # formulas keep working unchanged.
    ("""    Optional ByVal FieldKey As String = "") As Double
    Dim isSour As Boolean, b0 As Double, e0 As Double, key As String
    Dim brtC As Double, slbC As Double, othC As Double
    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double
    Dim theta As Double, etaEff As Double
    On Error GoTo EH
    isSour = (Sour >= 0.5)
    key = CM4_ResolveKey(FieldKey, Sour, TuneBlock, QnomBlock)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, key)
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    theta = ContractorMult4(contractor, brtC, slbC, othC) _
          * ThetaQnom(isSour, qnom, QnomBlock, key) _
          * ThetaFreq4(freq, t40, t50, t60, t70) _
          * ThetaKpod4(kpod, k02, k12, kpLo, kpHi, kshLo, kshHi)""",
     """    Optional ByVal FieldKey As String = "", _
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
          * ThetaOper5(freq, kpod, Wcut, iSet, onF, t40, t60, onK, k02, k16, onW, w95)"""),

    # CM4_ThetaFreq / CM4_ThetaKpod, plus the new CM4_ThetaWcut.
    ("""Public Function CM4_ThetaFreq(ByVal freq As Double, _
    Optional ByVal CtrlBlock As Range = Nothing) As Double
    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    CM4_ThetaFreq = ThetaFreq4(freq, t40, t50, t60, t70)
End Function

Public Function CM4_ThetaKpod(ByVal kpod As Double, _
    Optional ByVal CtrlBlock As Range = Nothing) As Double
    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    CM4_ThetaKpod = ThetaKpod4(kpod, k02, k12, kpLo, kpHi, kshLo, kshHi)
End Function""",
     """Public Function CM4_ThetaFreq(ByVal freq As Double, _
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
End Function"""),

    # CM4_LifeAtRef: the reference point is now the fitted reference exactly, so
    # there is no plateau midpoint to compute and no CtrlBlock read needed for it.
    ("""    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double, kRef As Double
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    kRef = 0.5 * (kpLo + kpHi)                  ' anywhere in the plateau: theta = 1
    CM4_LifeAtRef = CM4_LifeDays(Sour, "brt", REF_QNOM, REF_FREQ, kRef, _
                                 TuneBlock, CtrlBlock, QnomBlock, FieldKey)""",
     """    ' The fitted layers are all pinned at the reference point, so it is a
    ' constant now instead of a plateau midpoint read out of the CtrlBlock.
    CM4_LifeAtRef = CM4_LifeDays(Sour, "brt", REF_QNOM, REF_FREQ, REF_KPOD, _
                                 TuneBlock, CtrlBlock, QnomBlock, FieldKey, REF_WCUT)"""),

    # The renewal loop: new optional oil / HV / NIZ / TIZ arguments feed the
    # water-cut layer without any new sheet column.
    ("""    Optional ByVal FieldKey As String = "") As Variant
    Dim vq As Variant, vc As Variant, vf As Variant, tbl As Variant
    Dim i As Long, n As Long, dd As Long, key As String
    Dim ql As Double, freq As Double, kpod As Double, contractor As String
    Dim isSour As Boolean, theta As Double, b0 As Double, e0 As Double
    Dim brtC As Double, slbC As Double, othC As Double
    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double
    Dim etaEff As Double, life As Double, consumed As Double, nominal As Double
    Dim res() As Variant, iv() As Long
    On Error GoTo CalcErr
    If StepDays <= 0 Then StepDays = 1#
    dd = CLng(DownDays): If dd < 1 Then dd = 1
    isSour = (Sour >= 0.5)
    key = CM4_ResolveKey(FieldKey, Sour, TuneBlock, QnomBlock)
    Call ReadTune4(TuneBlock, isSour, b0, e0, brtC, slbC, othC, key)
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    vq = QlRange.Value2: vc = ContractorRange.Value2: vf = FreqRange.Value2
    tbl = NominalTable.Value2""",
     """    Optional ByVal FieldKey As String = "", _
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
    cumOil = 0#"""),

    ("""        ql = SafeNum4(vq(i, 1))
        contractor = CStr(vc(i, 1))
        freq = SafeNum4(vf(i, 1))
        If nominal <= 0# Then nominal = InitNominal
        kpod = ql / nominal
        theta = ContractorMult4(contractor, brtC, slbC, othC) _
              * ThetaQnom(isSour, nominal, QnomBlock, key) _
              * ThetaFreq4(freq, t40, t50, t60, t70) _
              * ThetaKpod4(kpod, k02, k12, kpLo, kpHi, kshLo, kshHi)""",
     """        ql = SafeNum4(vq(i, 1))
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
              * ThetaOper5(freq, kpod, wcut, iSet, onF, t40, t60, onK, k02, k16, onW, w95)"""),
]

if __name__ == "__main__":
    raise SystemExit(main())
