Attribute VB_Name = "mdlModelRegistry"
Option Explicit

' ============================================================
' mdlModelRegistry -- Parameter cache and stratum lookup.
'
' Model parameters live in the ESP_Models sheet.
' Call LoadModelRegistry() once; VBA caches them in mModels().
' LookupModel() implements the priority cascade:
'   1. field + h2s_class + contractor_group   (most specific)
'   2. field + h2s_class + "Pooled"
'   3. "Global_Pooled"                        (domain-knowledge fallback)
'
' Contractor group codes: brt (Borec), slb (Schlumberger), oth (others),
'                         unk (blank/None), Pooled (field-level pooled)
' H2S class codes:        sour | nonsour
' ============================================================

Private Const MODEL_SHEET As String = "ESP_Models"

Private Type ModelRow
    stratum     As String
    field       As String
    h2sClass    As String   ' "sour" or "nonsour"
    ctrGroup    As String   ' "brt", "slb", "oth", "Pooled", "unk"
    w1          As Double
    b1          As Double
    e1          As Double
    b2          As Double
    e2          As Double
    degenerate  As Boolean
    fitMode     As String
    nFail       As Long
End Type

Private mModels()   As ModelRow
Private mCount      As Long
Private mLoaded     As Boolean

' ------------------------------------------------------------------
' Load (or reload) the model table from the ESP_Models sheet.
' Called lazily from every public function.
' ------------------------------------------------------------------
Public Sub LoadModelRegistry()
    Dim ws As Worksheet
    Dim lastRow As Long, i As Long

    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEL_SHEET)
    On Error GoTo 0

    If ws Is Nothing Then
        mLoaded = False   ' silent fail -- UDFs will return #N/A
        Exit Sub          ' run CreateModelSheet() to create the ESP_Models sheet
    End If

    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    If lastRow < 2 Then
        mLoaded = False
        Exit Sub
    End If

    ReDim mModels(1 To lastRow - 1)
    mCount = 0

    For i = 2 To lastRow
        If Trim(CStr(ws.Cells(i, 1).Value)) = "" Then GoTo NextRow
        mCount = mCount + 1
        With mModels(mCount)
            .stratum   = Trim(CStr(ws.Cells(i, 1).Value))
            .field     = Trim(CStr(ws.Cells(i, 2).Value))
            .h2sClass  = LCase(Trim(CStr(ws.Cells(i, 3).Value)))
            .ctrGroup  = Trim(CStr(ws.Cells(i, 4).Value))
            .nFail     = CLng(Val(ws.Cells(i, 5).Value))
            .fitMode   = Trim(CStr(ws.Cells(i, 6).Value))
            .w1        = CDbl(ws.Cells(i, 7).Value)
            .b1        = CDbl(ws.Cells(i, 8).Value)
            .e1        = CDbl(ws.Cells(i, 9).Value)
            .b2        = CDbl(ws.Cells(i, 10).Value)
            .e2        = CDbl(ws.Cells(i, 11).Value)
            .degenerate = CBool(ws.Cells(i, 12).Value)
        End With
NextRow:
    Next i

    mLoaded = True
End Sub

' ------------------------------------------------------------------
' Map raw contractor string (col D) to group code.
' Returns "brt" (Borec), "slb" (Schlumberger), "oth" (others), "unk" (blank).
' Uses Chr() to build Cyrillic comparison strings -- avoids encoding issues
' in the .bas source file while still matching Excel cell values at runtime.
' ------------------------------------------------------------------
Public Function ContractorGroup(ByVal rawName As String) As String
    Dim s As String
    s = Trim(rawName)

    ' Build Cyrillic literals via Unicode code points so the .bas file stays ASCII.
    ' ChrW() accepts 0-65535; Chr() only accepts 0-255 and would throw Error 5.
    Dim sBorec As String    ' ChrW(1041)&ChrW(1086)&ChrW(1088)&ChrW(1077)&ChrW(1094)
    Dim sSLB   As String    ' ChrW(1064)&ChrW(1083)&...
    sBorec = ChrW(1041) & ChrW(1086) & ChrW(1088) & ChrW(1077) & ChrW(1094)
    sSLB   = ChrW(1064) & ChrW(1083) & ChrW(1102) & ChrW(1084) & ChrW(1073) & ChrW(1077) & ChrW(1088) & ChrW(1078) & ChrW(1077)

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
    ' "sour" trigger word: ChrW(1050)&ChrW(1080)&ChrW(1089)&ChrW(1083)&ChrW(1099)&ChrW(1081)
    Dim sKisliy As String
    sKisliy = ChrW(1050) & ChrW(1080) & ChrW(1089) & ChrW(1083) & ChrW(1099) & ChrW(1081)
    If Trim(rawValue) = sKisliy Then
        H2SClass = "sour"
    Else
        H2SClass = "nonsour"
    End If
End Function

' ------------------------------------------------------------------
' Core lookup with priority cascade.
' Fills w1..e2 by reference and returns the stratum key used.
' Returns True if a model was found (even the global fallback).
' ------------------------------------------------------------------
Public Function LookupModel(ByVal field As String, _
                             ByVal h2sCls As String, _
                             ByVal ctrGroup As String, _
                             ByRef w1 As Double, _
                             ByRef b1 As Double, ByRef e1 As Double, _
                             ByRef b2 As Double, ByRef e2 As Double, _
                             ByRef stratumUsed As String, _
                             ByRef isDegenerate As Boolean) As Boolean
    If Not mLoaded Then LoadModelRegistry
    If Not mLoaded Then
        LookupModel = False
        Exit Function
    End If

    Dim i As Long
    Dim key1 As String, key2 As String, key3 As String

    key1 = field & "_" & h2sCls & "_" & ctrGroup   ' field_h2s_contractor
    key2 = field & "_" & h2sCls & "_Pooled"         ' field_h2s_Pooled
    key3 = "Global_Pooled"                             ' domain-knowledge fallback

    Dim priorities(0 To 2) As String
    priorities(0) = key1
    priorities(1) = key2
    priorities(2) = key3

    Dim p As Integer
    For p = 0 To 2
        For i = 1 To mCount
            If mModels(i).stratum = priorities(p) Then
                w1 = mModels(i).w1
                b1 = mModels(i).b1
                e1 = mModels(i).e1
                b2 = mModels(i).b2
                e2 = mModels(i).e2
                isDegenerate = mModels(i).degenerate
                stratumUsed = mModels(i).stratum
                LookupModel = True
                Exit Function
            End If
        Next i
    Next p

    LookupModel = False
End Function

' ------------------------------------------------------------------
' Return the stratum key that would be resolved (for audit column).
' ------------------------------------------------------------------
Public Function ResolveStratumKey(ByVal field As String, _
                                   ByVal h2sCls As String, _
                                   ByVal ctrGroup As String) As String
    Dim w1 As Double, b1 As Double, e1 As Double
    Dim b2 As Double, e2 As Double
    Dim stratumUsed As String, isDegen As Boolean
    Call LookupModel(field, h2sCls, ctrGroup, w1, b1, e1, b2, e2, stratumUsed, isDegen)
    ResolveStratumKey = stratumUsed
End Function

' ------------------------------------------------------------------
' Ensure the registry is loaded without re-reading if already cached.
' Call this from other modules instead of checking mLoaded directly.
' ------------------------------------------------------------------
Public Sub EnsureRegistryLoaded()
    If Not mLoaded Then LoadModelRegistry
End Sub

' ------------------------------------------------------------------
' Force reload (call after editing ESP_Models sheet).
' ------------------------------------------------------------------
Public Sub RefreshRegistry()
    mLoaded = False
    LoadModelRegistry
End Sub

' ------------------------------------------------------------------
' CreateModelSheet -- builds the ESP_Models sheet with default parameters.
' Safe to call multiple times; will prompt before overwriting.
' ------------------------------------------------------------------
Public Sub CreateModelSheet()
    Dim ws As Worksheet
    Dim exists As Boolean

    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEL_SHEET)
    exists = (Not ws Is Nothing)
    On Error GoTo 0

    If exists Then
        If MsgBox("Sheet '" & MODEL_SHEET & "' already exists. Overwrite?", _
                  vbQuestion + vbYesNo) = vbNo Then Exit Sub
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If

    Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    ws.Name = MODEL_SHEET

    ' --- Header row ---
    Dim hdr As Variant
    hdr = Array("stratum", "field", "h2s_class", "contractor_group", "n_failures", _
                "fit_mode", "w1", "beta1", "eta1", "beta2", "eta2", "degenerate")
    Dim c As Integer
    For c = 0 To 11
        ws.Cells(1, c + 1).Value = hdr(c)
    Next c
    ws.Rows(1).Font.Bold = True

    ' --- Model rows ---
    ' Stratum key format: {field}_{h2s}_{ctr}
    ' Contractor codes: brt=Borec  slb=Schlumberger  oth=other  Pooled=field-pooled
    ' Format: stratum, field, h2s, ctr, n, mode, w1, b1, e1, b2, e2, degen
    Dim rows As Variant
    rows = Array( _
        Array("Ya_nonsour_brt",     "Ya", "nonsour", "brt",   397, "independent_K2", 0.1951, 1.0715,  63.8, 1.2032,  657.0, False), _
        Array("Ya_nonsour_slb",     "Ya", "nonsour", "slb",   261, "independent_K2", 0.1269, 1.0670,  33.2, 1.0770,  465.3, False), _
        Array("Ya_nonsour_oth",     "Ya", "nonsour", "oth",    44, "independent_K2", 0.6177, 1.2854,  60.4, 1.2954,  315.6, False), _
        Array("Ya_nonsour_Pooled",  "Ya", "nonsour", "Pooled", 702, "independent_K2", 0.1552, 1.0812,  41.3, 1.0912,  539.5, False), _
        Array("Za_nonsour_brt",     "Za", "nonsour", "brt",    87, "independent_K2", 0.3197, 1.2853,  35.4, 1.2953,  310.0, False), _
        Array("Za_nonsour_slb",     "Za", "nonsour", "slb",    33, "two_stage_K2",   0.4871, 0.9000, 104.5, 1.6000,  463.7, False), _
        Array("Za_nonsour_Pooled",  "Za", "nonsour", "Pooled", 135, "independent_K2", 0.7994, 0.9309, 123.5, 2.4940,  597.1, True),  _
        Array("Vt_sour_brt",        "Vt", "sour",    "brt",    23, "two_stage_K2",   0.3490, 1.3470,  25.0, 1.3920,  117.0, False), _
        Array("Vt_sour_slb",        "Vt", "sour",    "slb",    33, "two_stage_K2",   0.1350, 1.3470,   9.0, 1.3920,  124.0, False), _
        Array("Vt_sour_oth",        "Vt", "sour",    "oth",    33, "two_stage_K2",   0.3660, 1.3470,  13.0, 1.3920,   99.0, False), _
        Array("Vt_sour_Pooled",     "Vt", "sour",    "Pooled", 89, "independent_K2", 0.2583, 1.3468,  14.4, 1.3924,  113.7, False), _
        Array("Vt_nonsour_brt",     "Vt", "nonsour", "brt",    65, "independent_K2", 0.3330, 0.9740,  63.0, 1.1390,  334.0, False), _
        Array("Vt_nonsour_slb",     "Vt", "nonsour", "slb",    84, "independent_K2", 0.2270, 1.2500,  18.0, 1.2600,  254.0, False), _
        Array("Vt_nonsour_Pooled",  "Vt", "nonsour", "Pooled", 167, "independent_K2", 0.9441, 0.8710, 168.6, 9.4156,  532.5, True),  _
        Array("Ic_nonsour_Pooled",  "Ic", "nonsour", "Pooled", 141, "independent_K2", 0.2542, 1.2087,  37.0, 1.4126,  518.7, False), _
        Array("Az_nonsour_Pooled",  "Az", "nonsour", "Pooled", 143, "independent_K2", 0.5284, 0.9915,  67.0, 1.8635,  479.6, False), _
        Array("Mc_nonsour_Pooled",  "Mc", "nonsour", "Pooled",  38, "two_stage_K2",   0.0463, 1.1450,  18.2, 1.4025,  254.3, False), _
        Array("Da_nonsour_Pooled",  "Da", "nonsour", "Pooled",  31, "two_stage_K2",   0.1535, 1.1450,  52.5, 1.4025,  734.9, False), _
        Array("Global_Pooled",      "",   "",         "Pooled",   0, "domain_default", 0.2000, 1.1000,  41.0, 1.4000,  500.0, False)  _
    )

    Dim rowNum As Long
    rowNum = 2
    Dim r As Long, col As Integer
    For r = 0 To UBound(rows)
        For col = 0 To 11
            ws.Cells(rowNum, col + 1).Value = rows(r)(col)
        Next col
        rowNum = rowNum + 1
    Next r

    ' Format numbers
    ws.Columns("G:K").NumberFormat = "0.0000"
    ws.Columns("A:F").EntireColumn.AutoFit

    ' Name the data range for easy reference
    Dim rng As Range
    Set rng = ws.Range(ws.Cells(1, 1), ws.Cells(rowNum - 1, 12))
    On Error Resume Next
    ThisWorkbook.Names("ESP_Models_Table").Delete
    On Error GoTo 0
    ThisWorkbook.Names.Add Name:="ESP_Models_Table", RefersTo:=rng

    MsgBox "ESP_Models sheet created with " & (rowNum - 2) & " model rows.", vbInformation

    ' Reload registry
    mLoaded = False
    LoadModelRegistry
End Sub

' ------------------------------------------------------------------
' CreateCoxSheet -- builds ESP_CoxCoeffs sheet with fitted beta values.
' Run once after importing mdlCoxHR.bas, or to reset coefficients.
'
' The betas here must match the constants in mdlCoxHR.bas.
' Source: phase5e extended Cox, 2026-07-02.
' ------------------------------------------------------------------
Public Sub CreateCoxSheet()
    Const COX_SHEET As String = "ESP_CoxCoeffs"
    Dim ws As Worksheet
    Dim exists As Boolean

    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(COX_SHEET)
    exists = (Not ws Is Nothing)
    On Error GoTo 0

    If exists Then
        If MsgBox("Sheet '" & COX_SHEET & "' already exists. Overwrite?", _
                  vbQuestion + vbYesNo) = vbNo Then Exit Sub
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If

    Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    ws.Name = COX_SHEET

    ' Header
    ws.Cells(1, 1).Value = "covariate"
    ws.Cells(1, 2).Value = "beta"
    ws.Cells(1, 3).Value = "hr"
    ws.Cells(1, 4).Value = "hr_lo95"
    ws.Cells(1, 5).Value = "hr_hi95"
    ws.Cells(1, 6).Value = "p"
    ws.Cells(1, 7).Value = "reference_value"
    ws.Cells(1, 8).Value = "interpretation"
    ws.Rows(1).Font.Bold = True

    ' Coefficient rows
    ' delta_bep = Q_actual/Q_nominal - 1; HR=1.43 per unit (1 unit = 100% BEP deviation)
    ws.Cells(2, 1).Value = "delta_bep"
    ws.Cells(2, 2).Value = 0.360455
    ws.Cells(2, 3).Value = 1.434
    ws.Cells(2, 4).Value = 1.2789
    ws.Cells(2, 5).Value = 1.6079
    ws.Cells(2, 6).Value = 0.000
    ws.Cells(2, 7).Value = -0.342
    ws.Cells(2, 8).Value = "BEP deviation; overloaded pump -> shorter life"

    ' p_bot = bottomhole pressure atm; HR=1.0005/atm (small but significant)
    ws.Cells(3, 1).Value = "p_bot"
    ws.Cells(3, 2).Value = 0.000466
    ws.Cells(3, 3).Value = 1.0005
    ws.Cells(3, 4).Value = 1.0001
    ws.Cells(3, 5).Value = 1.0008
    ws.Cells(3, 6).Value = 0.009
    ws.Cells(3, 7).Value = 110.3
    ws.Cells(3, 8).Value = "Bottomhole pressure atm; higher back-pressure -> shorter life"

    ' n_stages_ratio = n_stages / q_nominal; HR=0.949 (more stages/flow = conservative design)
    ws.Cells(4, 1).Value = "n_stages_ratio"
    ws.Cells(4, 2).Value = -0.052032
    ws.Cells(4, 3).Value = 0.9493
    ws.Cells(4, 4).Value = 0.9292
    ws.Cells(4, 5).Value = 0.9699
    ws.Cells(4, 6).Value = 0.000
    ws.Cells(4, 7).Value = 2.249
    ws.Cells(4, 8).Value = "Stages / nominal_flow (m3/day); over-staged pump -> longer life"

    ws.Columns("A:H").EntireColumn.AutoFit
    ws.Columns("B:G").NumberFormat = "0.000000"

    MsgBox "ESP_CoxCoeffs sheet created. Betas must match mdlCoxHR.bas constants.", vbInformation
End Sub
