Attribute VB_Name = "mdlBatchProcess"
Option Explicit

' ============================================================
' mdlBatchProcess v2 -- Batch macros and utilities.
'
' RunPredictions()      -- fills the ESP output columns in the data sheet.
' ImportModelCSV()      -- header-driven load of esp_models.csv (validates clock).
' ImportModeMixCSV()    -- loads esp_mode_mix.csv + mode_group_map.csv (Phase B).
' RefreshModels()       -- clears the registry caches.
' ExportSurvivalCurve() -- writes an S(t) table + chart for a stratum.
'
' v2 changes (agents/analyses/vba_model_v2.md):
'   * Operating-time clock (ttf_mix): every time output header carries _op_d.
'   * Cox columns REMOVED from RunPredictions (theta shelved -- see mdlCoxHR).
'   * New columns: ESP_ModelKind, ESP_B50_lo/hi_op_d, ESP_Uptime, ESP_RunSeq
'     (+caution), ESP_DomMode_90op.
'   * ImportModelCSV is header-driven and REFUSES a calendar-clock or
'     schema-mismatched file with a written cell message (never MsgBox).
' ============================================================

' Input column letters (1-based column numbers)
Private Const COL_FIELD      As Long = 1   ' A  Field
Private Const COL_CONTRACTOR As Long = 4   ' D  Contractor
Private Const COL_AGE        As Long = 9   ' I  Age in service (OPERATING days -- see §0.1)
Private Const COL_H2S        As Long = 75  ' BW H2S class (sour/nonsour)
Private Const COL_FLAG       As Long = 76  ' BX Failure Flag (0=running,1=failed,-1=suspended)

' Output columns (written after the last existing data column = 77)
Private Const OUT_FIRST     As Long = 80
Private Const OUT_STRATUM   As Long = 80  ' BY
Private Const OUT_MODELKIND As Long = 81  ' BZ
Private Const OUT_B20       As Long = 82  ' CA
Private Const OUT_B50       As Long = 83  ' CB
Private Const OUT_B50LO     As Long = 84  ' CC
Private Const OUT_B50HI     As Long = 85  ' CD
Private Const OUT_B80       As Long = 86  ' CE
Private Const OUT_RUL       As Long = 87  ' CF
Private Const OUT_TTFPRED   As Long = 88  ' CG
Private Const OUT_SFNOW     As Long = 89  ' CH
Private Const OUT_COMPONENT As Long = 90  ' CI
Private Const OUT_HAZARD    As Long = 91  ' CJ
Private Const OUT_UPTIME    As Long = 92  ' CK
Private Const OUT_RUNSEQ    As Long = 93  ' CL
Private Const OUT_CAUTION   As Long = 94  ' CM
Private Const OUT_DOMMODE   As Long = 95  ' CN
Private Const OUT_HAZFLAGS  As Long = 96  ' CO  advisory elevated modes (display-only)
Private Const OUT_HAZTHETA  As Long = 97  ' CP  hazard multiplier (1 unless enabled)
' Sensitivity overlay (USER-OVERRIDE, not OOS-validated) -- baseline never altered
Private Const OUT_RULSENS   As Long = 98  ' CQ  RUL with full-theta applied
Private Const OUT_B50SENS   As Long = 99  ' CR  B50 with full-theta applied
Private Const OUT_DRUL_GLF  As Long = 100 ' CS  per-group marginal delta_RUL (op-days)
Private Const OUT_DRUL_LOAD As Long = 101 ' CT
Private Const OUT_DRUL_KPOD As Long = 102 ' CU
Private Const OUT_DRUL_FREQ As Long = 103 ' CV
Private Const OUT_DRUL_CURV As Long = 104 ' CW
Private Const OUT_DRUL_HIST As Long = 105 ' CX
Private Const OUT_DRUL_VINT As Long = 106 ' CY
Private Const OUT_LAST      As Long = 106

' Returns the data sheet name via ChrW() (no Cyrillic literals in source).
' ChrW(1057)&ChrW(1074)&ChrW(1086)&ChrW(1076) = "Svod" (Russian sheet name).
Private Function DataSheetName() As String
    Static s As String
    If s = "" Then s = ChrW(1057) & ChrW(1074) & ChrW(1086) & ChrW(1076)
    DataSheetName = s
End Function

' Detect the well-name column by scanning row-1 headers for the Cyrillic stem
' "skv" (ChrW 1089,1082,1074 = "скв", as in "Скважина"). 0 if not found.
Private Function DetectWellCol(ByVal ws As Worksheet) As Long
    Dim stem As String
    stem = ChrW(1089) & ChrW(1082) & ChrW(1074)
    Dim lastCol As Long, c As Long, h As String
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column
    For c = 1 To lastCol
        h = LCase(Trim(CStr(ws.Cells(1, c).Value)))
        If InStr(h, stem) > 0 Then DetectWellCol = c : Exit Function
    Next c
    DetectWellCol = 0
End Function

' Prepare the workbook for writing: fail fast if it is read-only, otherwise clear
' workbook-structure protection and unprotect every sheet (no-op without a
' password). Returns False (with a status message) if the workbook cannot be made
' writable -- callers must abort. This is the single gate every writer goes through.
Public Function PrepWorkbook() As Boolean
    If ThisWorkbook.ReadOnly Then
        ' Can't write a status cell on a read-only book; MsgBox is safe here because
        ' PrepWorkbook is only reached from manually-run Subs, never from a UDF.
        MsgBox "ESP import BLOCKED: the workbook is READ-ONLY (Protected View, " & _
               "'Mark as Final', a network/OneDrive lock, or opened from Downloads)." & _
               vbCrLf & vbCrLf & "Enable editing (yellow bar at the top), or File > " & _
               "Save As a local .xlsm copy, then retry.", vbExclamation, "ESP v2"
        PrepWorkbook = False
        Exit Function
    End If
    On Error Resume Next
    ThisWorkbook.Unprotect                 ' structure protection (no password)
    Dim sh As Worksheet
    For Each sh In ThisWorkbook.Worksheets
        sh.Unprotect                        ' cell protection per sheet
    Next sh
    On Error GoTo 0
    PrepWorkbook = True
End Function

' Return an empty worksheet named sheetName, robust to a PROTECTED existing sheet
' (common in production workbooks): try Unprotect + Clear; if Clear still fails,
' delete and recreate (sheet-level protection does not block deleting the sheet).
' Returns Nothing only if the workbook STRUCTURE is protected (caller reports it).
Public Function FreshSheet(ByVal sheetName As String) As Worksheet
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(sheetName)
    On Error GoTo 0

    Application.DisplayAlerts = False
    If Not ws Is Nothing Then
        On Error Resume Next
        ws.Unprotect                     ' no-op if unprotected / no password
        Err.Clear
        ws.Cells.Clear
        If Err.Number <> 0 Then          ' still locked -> delete + recreate
            Err.Clear
            ws.Delete
            If Err.Number <> 0 Then       ' workbook structure protected
                Err.Clear
                On Error GoTo 0
                Application.DisplayAlerts = True
                Set FreshSheet = Nothing
                Exit Function
            End If
            Set ws = Nothing
        End If
        On Error GoTo 0
    End If

    If ws Is Nothing Then
        On Error Resume Next
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        If ws Is Nothing Then
            On Error GoTo 0
            Application.DisplayAlerts = True
            Set FreshSheet = Nothing
            Exit Function
        End If
        ws.Name = sheetName
        On Error GoTo 0
    End If
    Application.DisplayAlerts = True
    Set FreshSheet = ws
End Function

' Write one hazard group's marginal delta_RUL (op-days) for a running pump. Skips the
' LatentRUL integral when the group's theta is ~1 (no covariates for this run). Blank
' for non-running pumps (RUL undefined). Baseline is passed in (computed once/row).
Private Sub WriteGroupDelta(ByVal ws As Worksheet, ByVal i As Long, ByVal outCol As Long, _
                            ByVal group As String, ByVal wkey As String, ByVal runSeq As Variant, _
                            ByVal flag As Integer, ByVal baseRUL As Double, ByVal age As Double, _
                            ByVal w1 As Double, ByVal b1 As Double, ByVal e1 As Double, _
                            ByVal b2 As Double, ByVal e2 As Double)
    If flag <> 0 Then ws.Cells(i, outCol).Value = "" : Exit Sub
    Dim thg As Double
    thg = HazardThetaGroupForKey(wkey, runSeq, group)
    If Abs(thg - 1#) < 0.0000001 Then
        ws.Cells(i, outCol).Value = 0
    Else
        ws.Cells(i, outCol).Value = _
            LatentRUL(age, w1, b1, ApplyCoxEta(e1, b1, thg), b2, ApplyCoxEta(e2, b2, thg)) - baseRUL
    End If
End Sub

' Append a timestamped message to the ESP_Status sheet (cell output, no MsgBox).
' Fully guarded so it can never itself raise (e.g. on a read-only workbook).
Private Sub WriteStatus(ByVal msg As String)
    On Error Resume Next
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("ESP_Status")
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = "ESP_Status"
        ws.Cells(1, 1).Value = "timestamp"
        ws.Cells(1, 2).Value = "message"
        ws.Rows(1).Font.Bold = True
    End If
    Dim r As Long
    r = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row + 1
    If r < 2 Then r = 2
    ws.Cells(r, 1).Value = Format(Now, "yyyy-mm-dd hh:nn:ss")
    ws.Cells(r, 2).Value = msg
    ws.Columns("A:B").AutoFit
End Sub

' Read a file as UTF-8 (via ADODB.Stream, late-bound -- no reference needed) and
' return its lines as a Collection.  This correctly decodes the utf-8-sig BOM and
' Cyrillic content (e.g. the node->mode map) that a plain ANSI Line Input mangles.
Private Function ReadAllLines(ByVal path As String) As Collection
    Dim stream As Object, txt As String
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 2            ' adTypeText
    stream.Charset = "utf-8"
    stream.Open
    stream.LoadFromFile path
    txt = stream.ReadText(-1)  ' adReadAll
    stream.Close

    txt = Replace(txt, vbCrLf, vbLf)
    txt = Replace(txt, vbCr, vbLf)
    Dim parts() As String, i As Long
    parts = Split(txt, vbLf)
    Dim col As Collection
    Set col = New Collection
    For i = LBound(parts) To UBound(parts)
        col.Add parts(i)
    Next i
    Set ReadAllLines = col
End Function

' Minimal CSV line parser (handles double-quoted fields with embedded commas).
Private Function ParseCsvLine(ByVal line As String) As Variant
    Dim out() As String
    Dim n As Long, i As Long
    Dim ch As String, cur As String
    Dim inQ As Boolean
    ReDim out(0 To 0)
    n = 0
    inQ = False
    cur = ""
    For i = 1 To Len(line)
        ch = Mid(line, i, 1)
        If inQ Then
            If ch = """" Then
                If i < Len(line) And Mid(line, i + 1, 1) = """" Then
                    cur = cur & """" : i = i + 1
                Else
                    inQ = False
                End If
            Else
                cur = cur & ch
            End If
        Else
            If ch = """" Then
                inQ = True
            ElseIf ch = "," Then
                ReDim Preserve out(0 To n)
                out(n) = cur : n = n + 1 : cur = ""
            Else
                cur = cur & ch
            End If
        End If
    Next i
    ReDim Preserve out(0 To n)
    out(n) = cur
    ParseCsvLine = out
End Function

' Strip a UTF-8 BOM from the first header cell if present.  VBA's Line Input reads
' the file in the ANSI codepage, so a utf-8-sig BOM (EF BB BF) arrives as the three
' characters ChrW(239)&ChrW(187)&ChrW(191), NOT a single U+FEFF -- handle both.
Private Function StripBom(ByVal s As String) As String
    If Len(s) >= 3 Then
        If AscW(Mid(s, 1, 1)) = 239 And AscW(Mid(s, 2, 1)) = 187 And AscW(Mid(s, 3, 1)) = 191 Then
            s = Mid(s, 4)
        End If
    End If
    If Len(s) > 0 Then
        If AscW(Left(s, 1)) = 65279 Then s = Mid(s, 2)   ' U+FEFF
    End If
    StripBom = s
End Function

' ------------------------------------------------------------------
' RunPredictions -- Write all ESP output columns to the data sheet.
' Skips suspended runs (flag = -1). Progress in the status bar.
' ------------------------------------------------------------------
Public Sub RunPredictions()
    Dim ws As Worksheet
    Dim lastRow As Long, i As Long
    Dim field As String, h2sRaw As String, ctrRaw As String
    Dim ageDays As Double, flag As Integer
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean

    On Error GoTo ErrHandler

    If Not PrepWorkbook() Then Exit Sub

    Set ws = ThisWorkbook.Worksheets(DataSheetName())
    lastRow = ws.Cells(ws.Rows.Count, COL_FIELD).End(xlUp).Row

    ' Headers (clock suffix _op_d on every operating-time column)
    ws.Cells(1, OUT_STRATUM).Value   = "ESP_Stratum"
    ws.Cells(1, OUT_MODELKIND).Value = "ESP_ModelKind"
    ws.Cells(1, OUT_B20).Value       = "ESP_B20_op_d"
    ws.Cells(1, OUT_B50).Value       = "ESP_B50_op_d"
    ws.Cells(1, OUT_B50LO).Value     = "ESP_B50_lo_op_d"
    ws.Cells(1, OUT_B50HI).Value     = "ESP_B50_hi_op_d"
    ws.Cells(1, OUT_B80).Value       = "ESP_B80_op_d"
    ws.Cells(1, OUT_RUL).Value       = "ESP_RUL_op_d"
    ws.Cells(1, OUT_TTFPRED).Value   = "ESP_TTF_Pred_op_d"
    ws.Cells(1, OUT_SFNOW).Value     = "ESP_SF_Now"
    ws.Cells(1, OUT_COMPONENT).Value = "ESP_Component"
    ws.Cells(1, OUT_HAZARD).Value    = "ESP_Hazard_op"
    ws.Cells(1, OUT_UPTIME).Value    = "ESP_Uptime"
    ws.Cells(1, OUT_RUNSEQ).Value    = "ESP_RunSeq"
    ws.Cells(1, OUT_CAUTION).Value   = "ESP_RunSeq_Caution"
    ws.Cells(1, OUT_DOMMODE).Value   = "ESP_DomMode_90op"
    ws.Cells(1, OUT_HAZFLAGS).Value  = "ESP_HazardFlags"
    ws.Cells(1, OUT_HAZTHETA).Value  = "ESP_HazardTheta"
    ws.Cells(1, OUT_RULSENS).Value   = "ESP_RUL_sens_op_d"
    ws.Cells(1, OUT_B50SENS).Value   = "ESP_B50_sens_op_d"
    ws.Cells(1, OUT_DRUL_GLF).Value  = "ESP_dRUL_GLF"
    ws.Cells(1, OUT_DRUL_LOAD).Value = "ESP_dRUL_load"
    ws.Cells(1, OUT_DRUL_KPOD).Value = "ESP_dRUL_kpod"
    ws.Cells(1, OUT_DRUL_FREQ).Value = "ESP_dRUL_freq"
    ws.Cells(1, OUT_DRUL_CURV).Value = "ESP_dRUL_curv"
    ws.Cells(1, OUT_DRUL_HIST).Value = "ESP_dRUL_history"
    ws.Cells(1, OUT_DRUL_VINT).Value = "ESP_dRUL_vintage"
    ws.Rows(1).Font.Bold = True

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    EnsureRegistryLoaded
    EnsureModeMixLoaded
    EnsureHazardLoaded

    Dim wellCol As Long
    wellCol = DetectWellCol(ws)
    Dim seqDict As Object
    Set seqDict = CreateObject("Scripting.Dictionary")

    Dim t0 As Long
    t0 = Timer

    For i = 2 To lastRow
        If (i Mod 500) = 0 Then
            Application.StatusBar = "ESP predictions: row " & i & " of " & lastRow & _
                                    "  (" & Format((i - 1) / (lastRow - 1) * 100, "0") & "%)"
            DoEvents
        End If

        field  = Trim(CStr(ws.Cells(i, COL_FIELD).Value))
        ctrRaw = Trim(CStr(ws.Cells(i, COL_CONTRACTOR).Value))
        h2sRaw = Trim(CStr(ws.Cells(i, COL_H2S).Value))
        flag   = CInt(Val(CStr(ws.Cells(i, COL_FLAG).Value)))

        ' Run sequence (counts every run on the well, suspended included)
        Dim runSeq As Long
        runSeq = 0
        If wellCol > 0 Then
            Dim wkey As String
            wkey = LCase(Trim(CStr(ws.Cells(i, wellCol).Value)))
            If wkey <> "" Then
                If seqDict.Exists(wkey) Then
                    seqDict(wkey) = seqDict(wkey) + 1
                Else
                    seqDict.Add wkey, 1
                End If
                runSeq = seqDict(wkey)
            End If
        End If

        ' Skip suspended runs (still counted in runSeq above)
        If flag = -1 Then
            Dim col As Long
            For col = OUT_FIRST To OUT_LAST
                ws.Cells(i, col).Value = ""
            Next col
            GoTo NextRow
        End If

        ageDays = Val(CStr(ws.Cells(i, COL_AGE).Value))
        If ageDays < 0 Then ageDays = 0

        Dim h2sCls As String, ctrGrp As String
        h2sCls = H2SClass(h2sRaw)
        ctrGrp = ContractorGroup(ctrRaw)
        If ctrGrp = "unk" Then ctrGrp = "Pooled"

        Dim found As Boolean
        found = LookupModel(field, h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen)

        ws.Cells(i, OUT_STRATUM).Value   = IIf(found, stratumKey, "#NoModel")
        ws.Cells(i, OUT_MODELKIND).Value = IIf(found, ResolveModelKind(field, h2sCls, ctrGrp), "")

        If Not found Then
            For col = OUT_B20 To OUT_LAST
                ws.Cells(i, col).Value = ""
            Next col
            GoTo NextRow
        End If

        ws.Cells(i, OUT_B20).Value = LatentQuantile(0.2, w1, b1, e1, b2, e2)
        ws.Cells(i, OUT_B50).Value = LatentQuantile(0.5, w1, b1, e1, b2, e2)
        ws.Cells(i, OUT_B80).Value = LatentQuantile(0.8, w1, b1, e1, b2, e2)

        Dim lo As Double, hi As Double
        If ResolveB50CI(field, h2sCls, ctrGrp, lo, hi) Then
            ws.Cells(i, OUT_B50LO).Value = lo
            ws.Cells(i, OUT_B50HI).Value = hi
        Else
            ws.Cells(i, OUT_B50LO).Value = ""
            ws.Cells(i, OUT_B50HI).Value = ""
        End If

        ws.Cells(i, OUT_SFNOW).Value = LatentSF(ageDays, w1, b1, e1, b2, e2)

        If LatentC1Posterior(ageDays, w1, b1, e1, b2, e2) >= 0.5 Then
            ws.Cells(i, OUT_COMPONENT).Value = "C1 (early failure)"
        Else
            ws.Cells(i, OUT_COMPONENT).Value = "C2 (wear-out)"
        End If

        ws.Cells(i, OUT_HAZARD).Value = LatentHazard(ageDays, w1, b1, e1, b2, e2)

        If flag = 0 Then
            Dim rul As Double
            rul = LatentRUL(ageDays, w1, b1, e1, b2, e2)
            ws.Cells(i, OUT_RUL).Value     = IIf(rul > 0, rul, "")
            ws.Cells(i, OUT_TTFPRED).Value = ""
        Else
            ws.Cells(i, OUT_RUL).Value     = ""
            ws.Cells(i, OUT_TTFPRED).Value = LatentQuantile(0.5, w1, b1, e1, b2, e2)
        End If

        ws.Cells(i, OUT_UPTIME).Value = UptimeFactor(field, h2sCls, ctrGrp)

        If runSeq > 0 Then
            ws.Cells(i, OUT_RUNSEQ).Value   = runSeq
            ws.Cells(i, OUT_CAUTION).Value  = IIf(runSeq >= 3, "REPEAT>=3", "")
        Else
            ws.Cells(i, OUT_RUNSEQ).Value   = ""
            ws.Cells(i, OUT_CAUTION).Value  = ""
        End If

        Dim dm As Variant
        dm = ESP_DominantMode(field, h2sRaw, ctrRaw, ageDays)
        ws.Cells(i, OUT_DOMMODE).Value = IIf(IsError(dm), "", dm)

        ' Hazard overlay. Advisory flags always; SENSITIVITY columns (RUL/B50 with
        ' theta applied + per-group marginal deltas) when the layer is enabled. The
        ' baseline B50/RUL columns above are NEVER altered. Not OOS-validated.
        If wellCol > 0 And wkey <> "" Then
            ws.Cells(i, OUT_HAZFLAGS).Value = HazardFlagsForKey(wkey, runSeq)
            Dim thAll As Double
            thAll = HazardThetaForKey(wkey, runSeq)
            ws.Cells(i, OUT_HAZTHETA).Value = thAll

            Dim baseRUL As Double
            baseRUL = LatentRUL(ageDays, w1, b1, e1, b2, e2)
            ws.Cells(i, OUT_B50SENS).Value = LatentQuantile(0.5, w1, b1, _
                ApplyCoxEta(e1, b1, thAll), b2, ApplyCoxEta(e2, b2, thAll))
            If flag = 0 Then
                ws.Cells(i, OUT_RULSENS).Value = LatentRUL(ageDays, w1, b1, _
                    ApplyCoxEta(e1, b1, thAll), b2, ApplyCoxEta(e2, b2, thAll))
            Else
                ws.Cells(i, OUT_RULSENS).Value = ""
            End If

            WriteGroupDelta ws, i, OUT_DRUL_GLF,  "GLF",          wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_LOAD, "load",         wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_KPOD, "kpod",         wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_FREQ, "frequency",    wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_CURV, "curvature",    wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_HIST, "well_history", wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
            WriteGroupDelta ws, i, OUT_DRUL_VINT, "vintage",      wkey, runSeq, flag, baseRUL, ageDays, w1, b1, e1, b2, e2
        Else
            Dim hc As Long
            For hc = OUT_HAZFLAGS To OUT_DRUL_VINT
                ws.Cells(i, hc).Value = ""
            Next hc
        End If

NextRow:
    Next i

    ' Formats
    With ws.Range(ws.Cells(2, OUT_B20), ws.Cells(lastRow, OUT_TTFPRED))
        .NumberFormat = "0.0"
    End With
    ws.Columns(OUT_UPTIME).NumberFormat = "0.000"
    ws.Columns(OUT_HAZTHETA).NumberFormat = "0.000"
    With ws.Range(ws.Cells(2, OUT_RULSENS), ws.Cells(lastRow, OUT_DRUL_VINT))
        .NumberFormat = "0.0"
    End With
    With ws.Range(ws.Cells(2, OUT_HAZARD), ws.Cells(lastRow, OUT_HAZARD))
        .NumberFormat = "0.0000"
    End With

    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.StatusBar = False

    WriteStatus "RunPredictions complete: " & (lastRow - 1) & " rows in " & _
                Format(Timer - t0, "0.0") & "s (clock=ttf_mix, operating days)."
    Exit Sub

ErrHandler:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.StatusBar = False
    WriteStatus "RunPredictions ERROR at row " & i & ": " & Err.Description
End Sub

' ------------------------------------------------------------------
' RefreshModels -- Force reload of both registries.
' ------------------------------------------------------------------
Public Sub RefreshModels()
    RefreshRegistry
    RefreshModeMix
    WriteStatus "Registries reloaded (ESP_Models + ESP_ModeMix)."
End Sub

' ------------------------------------------------------------------
' ImportModelCSV v2 -- header-driven load of esp_models.csv into ESP_Models.
' Refuses (writes a status message, leaves the sheet untouched) if the file is
' schema-mismatched OR carries any row on a clock other than "ttf_mix".
' ------------------------------------------------------------------
Public Sub ImportModelCSV(Optional ByVal csvPath As String = "")
    If csvPath = "" Then
        Dim fd As FileDialog
        Set fd = Application.FileDialog(msoFileDialogFilePicker)
        fd.Title = "Select esp_models.csv"
        fd.Filters.Add "CSV files", "*.csv"
        If fd.Show = False Then Exit Sub
        csvPath = fd.SelectedItems(1)
    End If

    If Not PrepWorkbook() Then Exit Sub

    ' ── Pass 1: read + validate BEFORE touching the sheet ────────────────────
    Dim lineText As String
    Dim headers As Variant, cells As Variant
    Dim hIdx As Object
    Set hIdx = CreateObject("Scripting.Dictionary")
    Dim isHeader As Boolean
    isHeader = True

    Dim dataLines As Collection
    Set dataLines = New Collection

    Dim srcLines As Collection, srcLine As Variant
    Set srcLines = ReadAllLines(csvPath)
    For Each srcLine In srcLines
        lineText = CStr(srcLine)
        If Trim(lineText) = "" Then GoTo ContinueLoop
        If isHeader Then
            headers = ParseCsvLine(lineText)
            Dim k As Long, hn As String
            For k = LBound(headers) To UBound(headers)
                hn = LCase(Trim(StripBom(CStr(headers(k)))))
                If hn <> "" And Not hIdx.Exists(hn) Then hIdx.Add hn, k
            Next k
            isHeader = False
        Else
            dataLines.Add lineText
        End If
ContinueLoop:
    Next srcLine

    ' Required schema
    Dim req As Variant, rq As Variant
    req = Array("stratum", "w1", "beta1", "eta1", "beta2", "eta2", "clock")
    For Each rq In req
        If Not hIdx.Exists(CStr(rq)) Then
            WriteStatus "ImportModelCSV REFUSED: missing required column '" & rq & _
                        "' in " & csvPath & " (schema mismatch)."
            Exit Sub
        End If
    Next rq

    ' Clock guard -- every row must be ttf_mix
    Dim li As Long, badClock As String
    Dim clkCol As Long
    clkCol = hIdx("clock")
    For li = 1 To dataLines.Count
        cells = ParseCsvLine(CStr(dataLines(li)))
        If clkCol <= UBound(cells) Then
            If LCase(Trim(CStr(cells(clkCol)))) <> "ttf_mix" Then
                badClock = Trim(CStr(cells(clkCol)))
                WriteStatus "ImportModelCSV REFUSED: row " & li & " has clock='" & _
                            badClock & "' (expected 'ttf_mix'). Calendar-clock or " & _
                            "stale file rejected; sheet unchanged."
                Exit Sub
            End If
        End If
    Next li

    ' ── Pass 2: write ────────────────────────────────────────────────────────
    Dim ws As Worksheet
    Set ws = FreshSheet("ESP_Models")
    If ws Is Nothing Then
        WriteStatus "ImportModelCSV REFUSED: cannot reset the ESP_Models sheet " & _
                    "(workbook structure is protected). Unprotect the workbook " & _
                    "(Review > Protect Workbook) and retry."
        Exit Sub
    End If

    ' Header row = the file's header (preserves the v2 schema order)
    Dim c As Long
    For c = LBound(headers) To UBound(headers)
        ws.Cells(1, c + 1).Value = Trim(StripBom(CStr(headers(c))))
    Next c
    ws.Rows(1).Font.Bold = True

    Dim rowNum As Long
    rowNum = 2
    For li = 1 To dataLines.Count
        cells = ParseCsvLine(CStr(dataLines(li)))
        For c = LBound(cells) To UBound(cells)
            ws.Cells(rowNum, c + 1).Value = cells(c)
        Next c
        rowNum = rowNum + 1
    Next li

    ws.Columns("A:U").AutoFit
    RefreshRegistry
    WriteStatus "ImportModelCSV OK: " & (rowNum - 2) & " rows loaded (clock=ttf_mix)."
End Sub

' ------------------------------------------------------------------
' ImportModeMixCSV -- load esp_mode_mix.csv into ESP_ModeMix, and (optionally)
' mode_group_map.csv into the ESP_ModeMap audit sheet. Header-driven; refuses a
' non-ttf_mix mode-mix file the same way as ImportModelCSV.
' ------------------------------------------------------------------
Public Sub ImportModeMixCSV(Optional ByVal modeMixPath As String = "", _
                            Optional ByVal mapPath As String = "")
    If Not PrepWorkbook() Then Exit Sub
    If modeMixPath = "" Then
        Dim fd As FileDialog
        Set fd = Application.FileDialog(msoFileDialogFilePicker)
        fd.Title = "Select esp_mode_mix.csv"
        fd.Filters.Add "CSV files", "*.csv"
        If fd.Show = False Then Exit Sub
        modeMixPath = fd.SelectedItems(1)
    End If

    If Not ImportSimpleCsv(modeMixPath, "ESP_ModeMix", "ttf_mix") Then Exit Sub

    If mapPath <> "" Then
        Call ImportSimpleCsv(mapPath, "ESP_ModeMap", "")
    End If

    RefreshModeMix
    WriteStatus "ImportModeMixCSV OK: ESP_ModeMix loaded" & _
                IIf(mapPath <> "", " (+ ESP_ModeMap audit)", "") & "."
End Sub

' Generic header-driven CSV -> sheet loader. If requireClock <> "", every row's
' "clock" column must equal it or the import is refused (status message).
Private Function ImportSimpleCsv(ByVal csvPath As String, ByVal sheetName As String, _
                                 ByVal requireClock As String) As Boolean
    Dim lineText As String
    Dim headers As Variant, cells As Variant
    Dim isHeader As Boolean, clkCol As Long
    Dim dataLines As Collection
    Set dataLines = New Collection
    isHeader = True
    clkCol = -1

    Dim srcLines As Collection, srcLine As Variant
    Set srcLines = ReadAllLines(csvPath)
    For Each srcLine In srcLines
        lineText = CStr(srcLine)
        If Trim(lineText) = "" Then GoTo ContinueLoop
        If isHeader Then
            headers = ParseCsvLine(lineText)
            Dim k As Long
            For k = LBound(headers) To UBound(headers)
                If LCase(Trim(StripBom(CStr(headers(k))))) = "clock" Then clkCol = k
            Next k
            isHeader = False
        Else
            dataLines.Add lineText
        End If
ContinueLoop:
    Next srcLine

    If requireClock <> "" Then
        If clkCol < 0 Then
            WriteStatus "Import REFUSED (" & sheetName & "): no 'clock' column in " & csvPath
            ImportSimpleCsv = False : Exit Function
        End If
        Dim li As Long
        For li = 1 To dataLines.Count
            cells = ParseCsvLine(CStr(dataLines(li)))
            If clkCol <= UBound(cells) Then
                If LCase(Trim(CStr(cells(clkCol)))) <> LCase(requireClock) Then
                    WriteStatus "Import REFUSED (" & sheetName & "): row " & li & _
                                " clock <> '" & requireClock & "'; sheet unchanged."
                    ImportSimpleCsv = False : Exit Function
                End If
            End If
        Next li
    End If

    Dim ws As Worksheet
    Set ws = FreshSheet(sheetName)
    If ws Is Nothing Then
        WriteStatus "Import REFUSED (" & sheetName & "): cannot reset the sheet " & _
                    "(workbook structure protected). Unprotect and retry."
        ImportSimpleCsv = False : Exit Function
    End If

    Dim c As Long
    For c = LBound(headers) To UBound(headers)
        ws.Cells(1, c + 1).Value = Trim(StripBom(CStr(headers(c))))
    Next c
    ws.Rows(1).Font.Bold = True

    Dim rowNum As Long, j As Long
    rowNum = 2
    For j = 1 To dataLines.Count
        cells = ParseCsvLine(CStr(dataLines(j)))
        For c = LBound(cells) To UBound(cells)
            ws.Cells(rowNum, c + 1).Value = cells(c)
        Next c
        rowNum = rowNum + 1
    Next j
    ws.Columns("A:N").AutoFit
    ImportSimpleCsv = True
End Function

' ------------------------------------------------------------------
' ImportVersionManifest -- load bundle_manifest.txt into the hidden ESP_Version
' sheet (key : value rows) so any workbook can state exactly which build it runs.
' ------------------------------------------------------------------
Public Sub ImportVersionManifest(ByVal txtPath As String)
    Dim ws As Worksheet
    Set ws = FreshSheet("ESP_Version")
    If ws Is Nothing Then
        WriteStatus "ImportVersionManifest skipped: ESP_Version sheet locked (workbook protected)."
        Exit Sub
    End If

    ws.Cells(1, 1).Value = "key"
    ws.Cells(1, 2).Value = "value"
    ws.Rows(1).Font.Bold = True

    Dim fileNum As Integer, lineText As String, rowNum As Long, p As Long
    fileNum = FreeFile
    Open txtPath For Input As #fileNum
    rowNum = 2
    Do While Not EOF(fileNum)
        Line Input #fileNum, lineText
        If Trim(lineText) <> "" Then
            p = InStr(lineText, ":")
            If p > 0 Then
                ws.Cells(rowNum, 1).Value = Trim(Left(lineText, p - 1))
                ws.Cells(rowNum, 2).Value = Trim(Mid(lineText, p + 1))
            Else
                ws.Cells(rowNum, 1).Value = Trim(lineText)
            End If
            rowNum = rowNum + 1
        End If
    Loop
    Close #fileNum

    ws.Columns("A:B").AutoFit
    ws.Visible = xlSheetHidden
    WriteStatus "ImportVersionManifest OK: " & (rowNum - 2) & " lines -> ESP_Version (hidden)."
End Sub

' ------------------------------------------------------------------
' ImportBundle -- one-call import of a whole bundle folder:
'   <folder>\esp_models.csv, esp_mode_mix.csv, mode_group_map.csv, bundle_manifest.txt
' Refuses (per file) any calendar-clock / schema-mismatched artifact.
' ------------------------------------------------------------------
Public Sub ImportBundle(ByVal folder As String)
    If Not PrepWorkbook() Then Exit Sub
    If Right(folder, 1) <> "\" And Right(folder, 1) <> "/" Then folder = folder & "\"
    Call ImportModelCSV(folder & "esp_models.csv")
    Call ImportModeMixCSV(folder & "esp_mode_mix.csv", folder & "mode_group_map.csv")
    If Len(Dir(folder & "bundle_manifest.txt")) > 0 Then
        Call ImportVersionManifest(folder & "bundle_manifest.txt")
    End If

    ' Optional hazard layer (operational + completion). Both refuse non-ttf_mix.
    If Len(Dir(folder & "esp_cox_coeffs.csv")) > 0 Then
        Call ImportSimpleCsv(folder & "esp_cox_coeffs.csv", "ESP_CoxCoeffs", "ttf_mix")
    End If
    If Len(Dir(folder & "esp_run_covariates.csv")) > 0 Then
        Call ImportSimpleCsv(folder & "esp_run_covariates.csv", "ESP_RunCov", "ttf_mix")
    End If
    RefreshHazardLayer

    WriteStatus "ImportBundle complete from " & folder & _
                " (hazard layer: enabled=" & CBool(HazardEnabled()) & _
                ", " & HazardCovariateCount() & " covariates)"
End Sub

' ------------------------------------------------------------------
' ExportSurvivalCurve -- Write S(t) table + chart to a new sheet.
' ------------------------------------------------------------------
Public Sub ExportSurvivalCurve(ByVal field As String, _
                               ByVal h2s As String, _
                               ByVal contractor As String, _
                               Optional ByVal tMax As Double = 2000, _
                               Optional ByVal nPoints As Integer = 200)
    Dim w1 As Double, b1 As Double, e1 As Double, b2 As Double, e2 As Double
    Dim stratumKey As String, isDegen As Boolean
    Dim h2sCls As String, ctrGrp As String

    h2sCls = H2SClass(h2s)
    ctrGrp = ContractorGroup(contractor)
    If ctrGrp = "unk" Then ctrGrp = "Pooled"

    If Not LookupModel(Trim(field), h2sCls, ctrGrp, w1, b1, e1, b2, e2, stratumKey, isDegen) Then
        WriteStatus "ExportSurvivalCurve: no model for " & field & "/" & h2s & "/" & contractor
        Exit Sub
    End If

    Dim wsOut As Worksheet, shName As String
    shName = Left("Curve_" & stratumKey, 31)
    On Error Resume Next
    Set wsOut = ThisWorkbook.Worksheets(shName)
    On Error GoTo 0
    If wsOut Is Nothing Then
        Set wsOut = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        wsOut.Name = shName
    Else
        wsOut.Cells.Clear
    End If

    wsOut.Cells(1, 1).Value = "t_op_d"
    wsOut.Cells(1, 2).Value = "S(t)"
    wsOut.Cells(1, 3).Value = "C1_contrib"
    wsOut.Cells(1, 4).Value = "C2_contrib"
    wsOut.Cells(1, 5).Value = "Hazard_op"
    wsOut.Rows(1).Font.Bold = True

    Dim stepD As Double
    stepD = tMax / nPoints
    Dim i As Integer, t As Double
    For i = 0 To nPoints
        t = i * stepD
        wsOut.Cells(i + 2, 1).Value = t
        wsOut.Cells(i + 2, 2).Value = LatentSF(t, w1, b1, e1, b2, e2)
        wsOut.Cells(i + 2, 3).Value = w1 * WeibullSF(t, b1, e1)
        wsOut.Cells(i + 2, 4).Value = (1# - w1) * WeibullSF(t, b2, e2)
        wsOut.Cells(i + 2, 5).Value = LatentHazard(t, w1, b1, e1, b2, e2)
    Next i
    wsOut.Columns("A:E").AutoFit

    Dim chartObj As ChartObject
    Set chartObj = wsOut.ChartObjects.Add(Left:=wsOut.Columns("G").Left, _
                                          Top:=wsOut.Rows(1).Top, Width:=400, Height:=250)
    With chartObj.Chart
        .ChartType = xlLine
        .SetSourceData Source:=wsOut.Range(wsOut.Cells(1, 1), wsOut.Cells(nPoints + 2, 2))
        .HasTitle = True
        .ChartTitle.Text = "Survival: " & stratumKey & " (operating days)"
        .Axes(xlValue).MinimumScale = 0
        .Axes(xlValue).MaximumScale = 1
    End With

    WriteStatus "ExportSurvivalCurve: '" & stratumKey & "' -> sheet '" & shName & "'."
End Sub
