Attribute VB_Name = "ModuleFailureV4"
Option Explicit

' ============================================================================
'  Vt operating-life model (deploy form; do NOT refit here).
'  Source of truth: backend/analysis/workflows/production_risk/unified_v4.py
'                   backend/analysis/workflows/production_risk/ya_freq_empirical.py
'
'    theta_total = level_contractor * ThetaQnom * ThetaFreq4 * ThetaKpod4
'    eta_eff     = eta0 * theta_total ^ (-1/beta0)
'    life (days) = RMST(0,730)
'
'  --- What changed at the 2026-07-28 rewire -------------------------------
'  1. FITTED LAYERS come from unified v4: one model structure shared by Ya,
'     Vt_nonsour and Vt_sour, with every layer fitted as a FREE polyline
'     (piecewise-linear in log theta, no shape assumed up front).
'     Baselines and the Qnom knots below are that fit's Vt strata.
'
'  2. KPOD IS NOT A NULL.  The previous header claimed "Kpod is the Ql residual
'     and is a documented null".  BOTH halves of that were wrong:
'       * kpod_run is telemetry-derived, NOT ql/qnom (that identity holds only
'         in the Svod panel), so Kpod is near-orthogonal to nameplate rate;
'       * every "theta_Kpod = 1" result came from a TENT shape pinned at 0.8
'         with theta >= 1 on both arms, which cannot represent a monotone
'         effect and so returns 1 whenever the truth is monotone.
'     Fitted free, Vt's Kpod runs 0.58 -> 1.27 across 0.2 -> 1.6 and is worth
'     +8.4 out-of-sample log-likelihood units.  It is a real layer.
'
'  3. FREQ is the reverse case: fitted free it is flat within noise in BOTH
'     fields (1 of 14 knots clears 1; CV Ya +1.1, Vt -0.8).  It is retained
'     here as an OPERATOR-SET prior, not as a measurement.
'
'  4. FREQ FORM is now a polynomial in LOG theta on 35..70 Hz, specified by
'     its values at 40 / 50 / 60 Hz plus an optional fourth anchor at 70 Hz.
'     Three anchors fix a parabola uniquely; the fourth adds the cubic term.
'     Taken in log theta because a plain polynomial through a valley crosses
'     zero before 35 Hz, i.e. a negative hazard multiplier.
'
'  5. KPOD FORM now has a PLATEAU (a band where theta = 1) instead of a single
'     optimum point, plus a curvature exponent per arm.
'
'  --- What changed at the 2026-08-04 rewire --------------------------------
'  6. EVERY FIELD, not just Vt.  TuneBlock and QnomBlock may now carry ONE ROW
'     PER STRATUM keyed by the workbook's UN / Uchastok nedr cell:
'     Ya, Vt_nonsour, Vt_sour, Az, Ic, Au, Mc, Mr.  Pass the field cell as the
'     new trailing FieldKey argument and the model follows the field selector.
'     Both old block layouts still work and an unknown key falls back to Vt, so
'     a sheet that has not been rewired behaves exactly as before.
'
'  7. Qnom LADDER FALLBACK fixed.  LookupNominal4 used to return 999999 when Ql
'     ran above the top of the catalogue, which drove Kpod to the 0.15 clamp and
'     cost ~40 % of the modelled life for a purely cosmetic reason.  It now
'     falls back to the LARGEST nominal, i.e. Kpod > 1 - an undersized pump.
'
'  8. SOUR Qnom CLAMPED FLAT above 1000 (fitted 1.2034 at 1600 -> 1.3668).  Free
'     -fitted the sour arm turns DOWN there on 20 runs / 14 events, which would
'     model a 1600 pump as longer-lived than a 1000 one.  Deployment guard.
'
'  9. CONTRACTOR LEVELS are now measured net of pump size (log Qnom + log Kpod
'     in the stage-1 window fit), so the level is the contractor effect AT the
'     reference point instead of one averaged over that contractor's own pump
'     mix.  Vt: slb 1.200 -> 1.196, oth 1.910 -> 1.750.
'
'  --- Reference point -------------------------------------------------------
'  WARNING: the shipped frequency anchors put theta(50 Hz) = 0.9, NOT 1.  So
'  "all theta = 1" is NOT the life at 50 Hz any more.  CM4_RmstRef still
'  reports the bare baseline (theta = 1); CM4_LifeAtRef reports the life at the
'  actual reference operating point (brt, Qnom 250, 50 Hz, Kpod in plateau),
'  which is what a sheet should compare against.  Do not confuse the two.
'
'  --- Known deployment assumption -------------------------------------------
'  theta_Kpod here is the OPERATOR BATHTUB and it disagrees with the fitted
'  layer at low Kpod: fitted free, Vt's Kpod is monotone RISING, so a lightly
'  loaded pump is protected, while the bathtub penalises it.  Kept by explicit
'  decision.  Consequence to keep in view: a typical slb pump (Qnom ~465,
'  Kpod ~0.57) composes to theta ~1.75 here against ~1.31 fit-consistent, i.e.
'  about 13 % shorter life.  It is an assumption, not a fit.
' ============================================================================

' --- unified v4 fitted constants (Vt strata, 2026-07-27) -------------------
Private Const NB_BETA As Double = 1.2511259     ' nonsour
Private Const NB_ETA  As Double = 517.84469
Private Const SB_BETA As Double = 1.3447649     ' sour
Private Const SB_ETA  As Double = 200.64525

' Contractor levels: fitted once on the Ql overlap window 200..500, where all
' three contractors actually coexist, then held fixed - otherwise the
' contractor term quietly absorbs the rate effect and low-rate slb wells get
' penalised for their rate.  Shared by both strata in unified v4.
Private Const NB_SLB As Double = 1.19994
Private Const NB_OTH As Double = 1.90976
Private Const SB_SLB As Double = 1.19994
Private Const SB_OTH As Double = 1.90976

Private Const QN_GUARD As Double = 60#          ' first knot; flat below
Private Const QN_CAP   As Double = 1600#        ' last knot; flat above

' --- operator-set defaults (overridden by CtrlBlock) -----------------------
Private Const D_T40 As Double = 1.8             ' theta_freq at 40 Hz
Private Const D_T50 As Double = 0.9             ' theta_freq at 50 Hz  (NOT 1)
Private Const D_T60 As Double = 1.3             ' theta_freq at 60 Hz
Private Const D_T70 As Double = 3.75            ' theta_freq at 70 Hz (4th anchor)

Private Const D_K02    As Double = 2#           ' theta_Kpod at 0.2
Private Const D_K12    As Double = 1.25         ' theta_Kpod at 1.2
Private Const D_KPL_LO As Double = 0.7          ' plateau start (theta = 1)
Private Const D_KPL_HI As Double = 0.95         ' plateau end
Private Const D_KSH_LO As Double = 2#           ' left-arm curvature exponent
Private Const D_KSH_HI As Double = 2#           ' right-arm curvature exponent

Private Const FREQ_LO As Double = 35#           ' clamp; load-bearing, not cosmetic
Private Const FREQ_HI As Double = 70#           ' a polynomial is unbounded
Private Const FREQ_A1 As Double = 40#           ' the three exact anchors
Private Const FREQ_A2 As Double = 50#
Private Const FREQ_A3 As Double = 60#
Private Const FREQ_A4 As Double = 70#           ' optional fourth anchor

Private Const KPOD_B2 As Double = 0.3
Private Const KPOD_LO As Double = 0.15
Private Const KPOD_HI As Double = 1.8
Private Const KPOD_A_LO As Double = 0.2         ' anchors the two arms are pinned to
Private Const KPOD_A_HI As Double = 1.2

Private Const REF_QNOM As Double = 250#
Private Const REF_FREQ As Double = 50#

' ---------------------------------------------------------------------------
'  Public UDFs
' ---------------------------------------------------------------------------

' Operating life in days, RMST(0,730).
Public Function CM4_LifeDays(ByVal Sour As Double, ByVal contractor As String, _
    ByVal qnom As Double, ByVal freq As Double, ByVal kpod As Double, _
    Optional ByVal TuneBlock As Range = Nothing, _
    Optional ByVal CtrlBlock As Range = Nothing, _
    Optional ByVal QnomBlock As Range = Nothing, _
    Optional ByVal FieldKey As String = "") As Double
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
          * ThetaKpod4(kpod, k02, k12, kpLo, kpHi, kshLo, kshHi)
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
    Dim t40 As Double, t50 As Double, t60 As Double, t70 As Double
    Dim k02 As Double, k12 As Double, kpLo As Double, kpHi As Double
    Dim kshLo As Double, kshHi As Double, kRef As Double
    Call ReadCtrl4(CtrlBlock, t40, t50, t60, t70, k02, k12, kpLo, kpHi, kshLo, kshHi)
    kRef = 0.5 * (kpLo + kpHi)                  ' anywhere in the plateau: theta = 1
    CM4_LifeAtRef = CM4_LifeDays(Sour, "brt", REF_QNOM, REF_FREQ, kRef, _
                                 TuneBlock, CtrlBlock, QnomBlock, FieldKey)
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

' Frequency layer: polynomial in LOG theta through theta(40), theta(50),
' theta(60), plus an optional fourth anchor at 70 Hz.
'
'   u = Clamp(f, 35, 70) - 50,   h = 10
'   Ln theta = a + b*u + c*u^2 + d*u^3
'     a = Ln(t50)
'     c = (Ln(t40) + Ln(t60) - 2*Ln(t50)) / (2*h^2)      independent of d
'     b = (Ln(t60) - Ln(t40)) / (2*h) - d*h^2            absorbs d, keeping the
'                                                        three anchors exact
'     d = (Ln(t70) - Ln(quad(70))) / (u4^3 - h^2*u4),  u4 = 20  => den = 6000
'
' Setting t70 to the parabola's own value at 70 gives d = 0, i.e. the pure
' 3-anchor form.  Never set d directly: it is violently non-linear (d = 1.5e-3
' sends theta(70) to ~2e4) and it moves the curve BETWEEN the anchors as well
' as outside them.
Private Function ThetaFreq4(ByVal f As Double, ByVal t40 As Double, _
    ByVal t50 As Double, ByVal t60 As Double, ByVal t70 As Double) As Double
    Dim la As Double, lb As Double, lc As Double
    Dim h As Double, a As Double, b As Double, c As Double, D As Double
    Dim u As Double, u4 As Double, den As Double, quad4 As Double
    If f <= 0# Then ThetaFreq4 = 1#: Exit Function
    If t40 <= 0# Or t50 <= 0# Or t60 <= 0# Then ThetaFreq4 = 1#: Exit Function
    h = 0.5 * (FREQ_A3 - FREQ_A1)
    la = Log(t40): lb = Log(t50): lc = Log(t60)
    a = lb
    c = (la + lc - 2# * lb) / (2# * h * h)
    b = (lc - la) / (2# * h)
    D = 0#
    If t70 > 0# Then
        u4 = FREQ_A4 - FREQ_A2
        den = u4 * u4 * u4 - h * h * u4
        If Abs(den) > 0.000000001 Then
            quad4 = Exp(a + b * u4 + c * u4 * u4)
            D = (Log(t70) - Log(quad4)) / den
        End If
    End If
    b = b - D * h * h
    u = Clamp4(f, FREQ_LO, FREQ_HI) - FREQ_A2
    ThetaFreq4 = Exp(a + b * u + c * u * u + D * u * u * u)
    If ThetaFreq4 < 0.000001 Then ThetaFreq4 = 0.000001
End Function

' Kpod layer: bathtub with a flat PLATEAU (theta = 1 across kpLo..kpHi) and a
' curvature exponent per arm, in v = |Log(k / plateau edge)|:
'
'   theta = 1 + c * v^p / (1 + b2 * v^p),   c set so the arm hits its anchor
'
' p = 1 is linear (a corner at the plateau edge), p = 2 parabolic (default,
' zero slope at the join), p = 3..4 flat-bottomed then steep.  The anchors at
' Kpod 0.2 and 1.2 are hit exactly for any p.
'
' Fitted free, this layer is monotone RISING on Vt (0.58 at 0.2 -> 1.27 at 1.6);
' the bathtub shipped here is the operator's constrained form, which is a
' different object.  See the header note 2.
Private Function ThetaKpod4(ByVal k As Double, ByVal k02 As Double, _
    ByVal k12 As Double, ByVal kpLo As Double, ByVal kpHi As Double, _
    ByVal shLo As Double, ByVal shHi As Double) As Double
    Dim kk As Double, pLo As Double, pHi As Double
    Dim pl As Double, ph As Double
    Dim vLo As Double, vHi As Double, cL As Double, cR As Double, v As Double
    If k <= 0# Then ThetaKpod4 = 1#: Exit Function
    pLo = kpLo: pHi = kpHi
    If pLo > pHi Then pLo = kpHi: pHi = kpLo
    If pLo <= 0# Or pHi <= 0# Then ThetaKpod4 = 1#: Exit Function
    pl = shLo: If pl < 1# Then pl = 1#
    ph = shHi: If ph < 1# Then ph = 1#

    kk = Clamp4(k, KPOD_LO, KPOD_HI)
    If kk >= pLo And kk <= pHi Then ThetaKpod4 = 1#: Exit Function

    vLo = Abs(Log(KPOD_A_LO / pLo))
    vHi = Abs(Log(KPOD_A_HI / pHi))
    If vLo < 0.000000001 Or vHi < 0.000000001 Then ThetaKpod4 = 1#: Exit Function
    cL = (k02 - 1#) * (1# + KPOD_B2 * vLo ^ pl) / (vLo ^ pl)
    cR = (k12 - 1#) * (1# + KPOD_B2 * vHi ^ ph) / (vHi ^ ph)

    If kk < pLo Then
        v = Log(pLo / kk)
        ThetaKpod4 = 1# + cL * v ^ pl / (1# + KPOD_B2 * v ^ pl)
    Else
        v = Log(kk / pHi)
        ThetaKpod4 = 1# + cR * v ^ ph / (1# + KPOD_B2 * v ^ ph)
    End If
    If ThetaKpod4 < 0.000001 Then ThetaKpod4 = 0.000001
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
    Optional ByVal FieldKey As String = "") As Variant
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
    tbl = NominalTable.Value2
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
        theta = ContractorMult4(contractor, brtC, slbC, othC) _
              * ThetaQnom(isSour, nominal, QnomBlock, key) _
              * ThetaFreq4(freq, t40, t50, t60, t70) _
              * ThetaKpod4(kpod, k02, k12, kpLo, kpHi, kshLo, kshHi)
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

' CtrlBlock = 10 cells (a column or a row), in order:
'   1  theta_freq @ 40 Hz
'   2  theta_freq @ 50 Hz          (may be < 1; shifts the layer's LEVEL)
'   3  theta_freq @ 60 Hz
'   4  theta_freq @ 70 Hz          (4th anchor; blank/0 -> pure 3-anchor form)
'   5  theta_Kpod @ 0.2
'   6  theta_Kpod @ 1.2
'   7  Kpod plateau start
'   8  Kpod plateau end
'   9  Kpod left-arm exponent      (1 linear, 2 parabolic, 3-4 flat-bottomed)
'  10  Kpod right-arm exponent
' A block shorter than 10 cells is ignored entirely rather than half-applied,
' so a sheet that still carries the OLD 6-cell layout falls back to the
' defaults above instead of silently reading m60/s/shift into t40/t50/t60.
Private Sub ReadCtrl4(ByVal CtrlBlock As Range, _
    ByRef t40 As Double, ByRef t50 As Double, ByRef t60 As Double, _
    ByRef t70 As Double, ByRef k02 As Double, ByRef k12 As Double, _
    ByRef kpLo As Double, ByRef kpHi As Double, _
    ByRef shLo As Double, ByRef shHi As Double)
    Dim flat() As Double, n As Long, c As Range
    t40 = D_T40: t50 = D_T50: t60 = D_T60: t70 = D_T70
    k02 = D_K02: k12 = D_K12: kpLo = D_KPL_LO: kpHi = D_KPL_HI
    shLo = D_KSH_LO: shHi = D_KSH_HI
    If CtrlBlock Is Nothing Then Exit Sub
    On Error GoTo EH
    If CtrlBlock.Cells.Count < 10 Then Exit Sub
    ReDim flat(1 To 10)
    n = 0
    For Each c In CtrlBlock.Cells
        n = n + 1
        If n > 10 Then Exit For
        flat(n) = SafeNum4(c.Value2)
    Next c
    If flat(1) > 0# Then t40 = flat(1)
    If flat(2) > 0# Then t50 = flat(2)
    If flat(3) > 0# Then t60 = flat(3)
    t70 = flat(4)                               ' 0 => 3-anchor form, legitimately
    If flat(5) > 0# Then k02 = flat(5)
    If flat(6) > 0# Then k12 = flat(6)
    If flat(7) > 0# Then kpLo = flat(7)
    If flat(8) > 0# Then kpHi = flat(8)
    If flat(9) >= 1# Then shLo = flat(9)
    If flat(10) >= 1# Then shHi = flat(10)
    Exit Sub
EH:
End Sub

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
