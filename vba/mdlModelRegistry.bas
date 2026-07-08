Attribute VB_Name = "mdlModelRegistry"
Option Explicit

' ============================================================
' mdlModelRegistry v2 -- Parameter cache and stratum lookup.
'
' Model parameters live in the ESP_Models sheet (schema v2 -- see below).
' Call EnsureRegistryLoaded(); VBA caches rows in mModels().
' LookupModel() / FindModelIndex() implement the priority cascade:
'   1. field + h2s_class + contractor_group   (most specific)
'   2. field + h2s_class + "Pooled"
'   3. "Global_Pooled"                         (domain-knowledge fallback)
'
' Contractor group codes: brt (Borec), slb (Schlumberger), oth (others),
'                         unk (blank/None -> Pooled), Pooled (field-level pooled)
' H2S class codes:        sour | nonsour
'
' v2 changes (agents/analyses/vba_model_v2.md):
'   * Fits are on the OPERATING-time clock (ttf_mix), full population.
'   * Header-driven load (no positional column assumptions).
'   * New columns: model_kind, b20/b50/b80, b50_lo/b50_hi, uptime_factor,
'     n_runs, pct_mixed_clock, clock.
'   * Single-Weibull / degenerate strata stored as w1=0 (§0.3); model_kind in
'     {k2, k1_aic, k1_degenerate}.
'   * CreateModelSheet moved to the GENERATED module mdlModelSeed.bas so the
'     hardcoded fallback can never drift from esp_models.csv.
'   * CreateCoxSheet removed -- the theta layer is decommissioned (see mdlCoxHR).
' ============================================================

Private Const MODEL_SHEET As String = "ESP_Models"

Private Type ModelRow
    stratum      As String
    field        As String
    h2sClass     As String   ' "sour" or "nonsour"
    ctrGroup     As String   ' "brt", "slb", "oth", "Pooled", "unk"
    modelKind    As String   ' "k2", "k1_aic", "k1_degenerate"
    w1           As Double
    b1           As Double
    e1           As Double
    b2           As Double
    e2           As Double
    b20          As Double
    b50          As Double
    b80          As Double
    b50Lo        As Double
    b50Hi        As Double
    uptimeFactor As Double
    nRuns        As Long
    nFail        As Long
    pctMixed     As Double
    clockName    As String
End Type

Private mModels()  As ModelRow
Private mCount     As Long
Private mLoaded    As Boolean

' ------------------------------------------------------------------
' Safe numeric parse: blank / non-numeric -> 0 (used for optional CI cells).
' ------------------------------------------------------------------
Private Function SafeDbl(ByVal v As Variant) As Double
    On Error Resume Next
    Dim s As String
    s = Trim(CStr(v))
    If s = "" Then
        SafeDbl = 0#
    Else
        SafeDbl = CDbl(Val(s))
    End If
    On Error GoTo 0
End Function

' ------------------------------------------------------------------
' Find a header column by name in row 1 (case-insensitive). 0 if absent.
' ------------------------------------------------------------------
' Strip a leading UTF-8 BOM that may survive as ANSI bytes (EF BB BF) or U+FEFF.
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
        If CleanHdr(CStr(ws.Cells(1, c).Value)) = LCase(name) Then
            HeaderCol = c
            Exit Function
        End If
    Next c
    HeaderCol = 0
End Function

' ------------------------------------------------------------------
' Load (or reload) the model table from the ESP_Models sheet (header-driven).
' ------------------------------------------------------------------
Public Sub LoadModelRegistry()
    Dim ws As Worksheet
    Dim lastRow As Long, i As Long

    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEL_SHEET)
    On Error GoTo 0

    If ws Is Nothing Then
        mLoaded = False   ' silent fail -- UDFs return #N/A. Run CreateModelSheet.
        Exit Sub
    End If

    ' Resolve columns by header name.
    Dim cStrt As Long, cFld As Long, cH2s As Long, cCtr As Long, cKind As Long
    Dim cW1 As Long, cB1 As Long, cE1 As Long, cB2 As Long, cE2 As Long
    Dim cB20 As Long, cB50 As Long, cB80 As Long, cLo As Long, cHi As Long
    Dim cUp As Long, cNr As Long, cNf As Long, cPm As Long, cClk As Long

    cStrt = HeaderCol(ws, "stratum")
    cFld = HeaderCol(ws, "field")
    cH2s = HeaderCol(ws, "h2s_class")
    cCtr = HeaderCol(ws, "contractor_group")
    cKind = HeaderCol(ws, "model_kind")
    cW1 = HeaderCol(ws, "w1")
    cB1 = HeaderCol(ws, "beta1")
    cE1 = HeaderCol(ws, "eta1")
    cB2 = HeaderCol(ws, "beta2")
    cE2 = HeaderCol(ws, "eta2")
    cB20 = HeaderCol(ws, "b20")
    cB50 = HeaderCol(ws, "b50")
    cB80 = HeaderCol(ws, "b80")
    cLo = HeaderCol(ws, "b50_lo")
    cHi = HeaderCol(ws, "b50_hi")
    cUp = HeaderCol(ws, "uptime_factor")
    cNr = HeaderCol(ws, "n_runs")
    cNf = HeaderCol(ws, "n_failures")
    cPm = HeaderCol(ws, "pct_mixed_clock")
    cClk = HeaderCol(ws, "clock")

    If cStrt = 0 Or cW1 = 0 Or cB1 = 0 Or cE1 = 0 Or cB2 = 0 Or cE2 = 0 Then
        mLoaded = False   ' schema too old / broken -- ImportModelCSV rebuilds it.
        Exit Sub
    End If

    lastRow = ws.Cells(ws.Rows.Count, cStrt).End(xlUp).Row
    If lastRow < 2 Then
        mLoaded = False
        Exit Sub
    End If

    ReDim mModels(1 To lastRow - 1)
    mCount = 0

    For i = 2 To lastRow
        If Trim(CStr(ws.Cells(i, cStrt).Value)) = "" Then GoTo NextRow
        mCount = mCount + 1
        With mModels(mCount)
            .stratum = Trim(CStr(ws.Cells(i, cStrt).Value))
            .field = IIf(cFld > 0, Trim(CStr(ws.Cells(i, cFld).Value)), "")
            .h2sClass = IIf(cH2s > 0, LCase(Trim(CStr(ws.Cells(i, cH2s).Value))), "")
            .ctrGroup = IIf(cCtr > 0, Trim(CStr(ws.Cells(i, cCtr).Value)), "Pooled")
            .modelKind = IIf(cKind > 0, LCase(Trim(CStr(ws.Cells(i, cKind).Value))), "k2")
            .w1 = SafeDbl(ws.Cells(i, cW1).Value)
            .b1 = SafeDbl(ws.Cells(i, cB1).Value)
            .e1 = SafeDbl(ws.Cells(i, cE1).Value)
            .b2 = SafeDbl(ws.Cells(i, cB2).Value)
            .e2 = SafeDbl(ws.Cells(i, cE2).Value)
            .b20 = IIf(cB20 > 0, SafeDbl(ws.Cells(i, cB20).Value), 0#)
            .b50 = IIf(cB50 > 0, SafeDbl(ws.Cells(i, cB50).Value), 0#)
            .b80 = IIf(cB80 > 0, SafeDbl(ws.Cells(i, cB80).Value), 0#)
            .b50Lo = IIf(cLo > 0, SafeDbl(ws.Cells(i, cLo).Value), 0#)
            .b50Hi = IIf(cHi > 0, SafeDbl(ws.Cells(i, cHi).Value), 0#)
            .uptimeFactor = IIf(cUp > 0, SafeDbl(ws.Cells(i, cUp).Value), 0#)
            .nRuns = IIf(cNr > 0, CLng(Val(ws.Cells(i, cNr).Value)), 0)
            .nFail = IIf(cNf > 0, CLng(Val(ws.Cells(i, cNf).Value)), 0)
            .pctMixed = IIf(cPm > 0, SafeDbl(ws.Cells(i, cPm).Value), 0#)
            .clockName = IIf(cClk > 0, LCase(Trim(CStr(ws.Cells(i, cClk).Value))), "")
        End With
NextRow:
    Next i

    mLoaded = (mCount > 0)
End Sub

' ------------------------------------------------------------------
' Map raw contractor string (col D) to group code.
' Returns "brt" (Borec), "slb" (Schlumberger), "oth" (others), "unk" (blank).
' Uses ChrW() to build Cyrillic comparison strings so the .bas source stays ASCII.
' ------------------------------------------------------------------
Public Function ContractorGroup(ByVal rawName As String) As String
    Dim s As String
    s = Trim(rawName)

    ' Accept the Latin group codes directly (manual use / validation spot-checks).
    Select Case LCase(s)
        Case "brt": ContractorGroup = "brt" : Exit Function
        Case "slb": ContractorGroup = "slb" : Exit Function
        Case "oth": ContractorGroup = "oth" : Exit Function
        Case "pooled": ContractorGroup = "Pooled" : Exit Function
    End Select

    Dim sBorec As String, sSLB As String
    sBorec = ChrW(1041) & ChrW(1086) & ChrW(1088) & ChrW(1077) & ChrW(1094)
    sSLB = ChrW(1064) & ChrW(1083) & ChrW(1102) & ChrW(1084) & ChrW(1073) & ChrW(1077) & ChrW(1088) & ChrW(1078) & ChrW(1077)

    If s = "" Or s = "None" Then
        ContractorGroup = "unk"
    ElseIf s = sBorec Then
        ContractorGroup = "brt"
    ElseIf s = sSLB Then
        ContractorGroup = "slb"
    Else
        ContractorGroup = "oth"   ' nt, novomet, ink, ...
    End If
End Function

' ------------------------------------------------------------------
' Map H2S column value to "sour" or "nonsour".
' Blank / other -> "nonsour" (only Vt field has sour wells).
' ------------------------------------------------------------------
Public Function H2SClass(ByVal rawValue As String) As String
    ' Accept the Latin codes directly (manual use / validation spot-checks).
    Select Case LCase(Trim(rawValue))
        Case "sour": H2SClass = "sour" : Exit Function
        Case "nonsour": H2SClass = "nonsour" : Exit Function
    End Select

    Dim sKisliy As String
    sKisliy = ChrW(1050) & ChrW(1080) & ChrW(1089) & ChrW(1083) & ChrW(1099) & ChrW(1081)
    If Trim(rawValue) = sKisliy Then
        H2SClass = "sour"
    Else
        H2SClass = "nonsour"
    End If
End Function

' ------------------------------------------------------------------
' Core index resolver with the priority cascade. Returns the mModels index
' (1..mCount) of the matched row, or 0 if nothing (not even Global) matched.
' ------------------------------------------------------------------
Public Function FindModelIndex(ByVal field As String, _
                               ByVal h2sCls As String, _
                               ByVal ctrGroup As String) As Long
    If Not mLoaded Then LoadModelRegistry
    If Not mLoaded Then FindModelIndex = 0 : Exit Function

    Dim priorities(0 To 2) As String
    priorities(0) = Trim(field) & "_" & h2sCls & "_" & ctrGroup
    priorities(1) = Trim(field) & "_" & h2sCls & "_Pooled"
    priorities(2) = "Global_Pooled"

    Dim p As Integer, i As Long
    For p = 0 To 2
        For i = 1 To mCount
            If mModels(i).stratum = priorities(p) Then
                FindModelIndex = i
                Exit Function
            End If
        Next i
    Next p
    FindModelIndex = 0
End Function

' ------------------------------------------------------------------
' Core lookup with priority cascade. Fills w1..e2 by reference and returns
' the stratum key used. isDegenerate = (model_kind = "k1_degenerate").
' Signature preserved for backward compatibility with existing callers.
' ------------------------------------------------------------------
Public Function LookupModel(ByVal field As String, _
                            ByVal h2sCls As String, _
                            ByVal ctrGroup As String, _
                            ByRef w1 As Double, _
                            ByRef b1 As Double, ByRef e1 As Double, _
                            ByRef b2 As Double, ByRef e2 As Double, _
                            ByRef stratumUsed As String, _
                            ByRef isDegenerate As Boolean) As Boolean
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    If idx = 0 Then
        LookupModel = False
        Exit Function
    End If
    With mModels(idx)
        w1 = .w1 : b1 = .b1 : e1 = .e1 : b2 = .b2 : e2 = .e2
        stratumUsed = .stratum
        isDegenerate = (.modelKind = "k1_degenerate")
    End With
    LookupModel = True
End Function

' ------------------------------------------------------------------
' Public accessors (used by mdlPublicFunctions / mdlBatchProcess / validation).
' ------------------------------------------------------------------
Public Function ResolveStratumKey(ByVal field As String, _
                                  ByVal h2sCls As String, _
                                  ByVal ctrGroup As String) As String
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    ResolveStratumKey = IIf(idx = 0, "", mModels(idx).stratum)
End Function

Public Function ResolveModelKind(ByVal field As String, _
                                 ByVal h2sCls As String, _
                                 ByVal ctrGroup As String) As String
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    ResolveModelKind = IIf(idx = 0, "", mModels(idx).modelKind)
End Function

' Per-stratum uptime factor u_s = mean(ttf_mix / run_days).  Exposed for the
' ESP_Uptime audit UDF; conversion is left to the user (§0.1 document-only).
Public Function UptimeFactor(ByVal field As String, _
                             ByVal h2sCls As String, _
                             ByVal ctrGroup As String) As Double
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    UptimeFactor = IIf(idx = 0, 0#, mModels(idx).uptimeFactor)
End Function

' Fill imported B50 [lo, hi] for a stratum by reference. Returns False if the
' interval is unavailable (0 sentinel).
Public Function ResolveB50CI(ByVal field As String, _
                             ByVal h2sCls As String, _
                             ByVal ctrGroup As String, _
                             ByRef lo As Double, ByRef hi As Double) As Boolean
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    If idx = 0 Then ResolveB50CI = False : Exit Function
    lo = mModels(idx).b50Lo
    hi = mModels(idx).b50Hi
    ResolveB50CI = (lo > 0 And hi > 0)
End Function

' Imported B50 point (as regenerated by Python) -- used by ValidateRegistry to
' compare against the VBA bisection recompute.
Public Function ImportedB50(ByVal field As String, _
                            ByVal h2sCls As String, _
                            ByVal ctrGroup As String) As Double
    Dim idx As Long
    idx = FindModelIndex(field, h2sCls, ctrGroup)
    ImportedB50 = IIf(idx = 0, 0#, mModels(idx).b50)
End Function

Public Function RegistryRowCount() As Long
    If Not mLoaded Then LoadModelRegistry
    RegistryRowCount = IIf(mLoaded, mCount, 0)
End Function

' Clock stamp of the loaded registry (first row). Empty if unloaded.
Public Function RegistryClock() As String
    If Not mLoaded Then LoadModelRegistry
    RegistryClock = IIf(mLoaded And mCount > 0, mModels(1).clockName, "")
End Function

' Stratum key of the i-th loaded row (1-based). For iteration in validation.
Public Function RegistryStratumAt(ByVal i As Long) As String
    If Not mLoaded Then LoadModelRegistry
    If i >= 1 And i <= mCount Then RegistryStratumAt = mModels(i).stratum Else RegistryStratumAt = ""
End Function

' Full params of the i-th loaded row (for validation bisection check).
Public Function RegistryParamsAt(ByVal i As Long, _
                                 ByRef w1 As Double, ByRef b1 As Double, _
                                 ByRef e1 As Double, ByRef b2 As Double, _
                                 ByRef e2 As Double, ByRef b50 As Double) As Boolean
    If Not mLoaded Then LoadModelRegistry
    If i < 1 Or i > mCount Then RegistryParamsAt = False : Exit Function
    With mModels(i)
        w1 = .w1 : b1 = .b1 : e1 = .e1 : b2 = .b2 : e2 = .e2 : b50 = .b50
    End With
    RegistryParamsAt = True
End Function

' ------------------------------------------------------------------
' Cache management.
' ------------------------------------------------------------------
Public Sub EnsureRegistryLoaded()
    If Not mLoaded Then LoadModelRegistry
End Sub

Public Sub RefreshRegistry()
    mLoaded = False
    LoadModelRegistry
End Sub
