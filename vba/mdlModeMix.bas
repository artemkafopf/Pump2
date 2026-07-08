Attribute VB_Name = "mdlModeMix"
Option Explicit

' ============================================================
' mdlModeMix -- Phase B failure-mode planning layer (read-only).
'
' Reads the ESP_ModeMix sheet (imported from esp_mode_mix.csv) into a cache and
' exposes per-field-level-stratum x mode incidence functions.  Absolute risk is
' the Aalen-Johansen CIF (honest under competing modes), NOT 1-KM per cause.
'
' The node -> mode map is the frozen single source of truth in
' backend/analysis/data/failure_modes.py; it is imported to an ESP_ModeMap audit
' sheet.  VBA must NEVER re-hardcode node lists.
'
' Mode groups: "hydraulic", "electro-thermal", "protector", "other".
'
' UDFs:
'   ESP_ModeShare(field, h2s, ctr, mode, horizon_days)
'       share of incidence attributable to `mode` at 90 or 365 operating days.
'   ESP_ModeIncidenceTime(field, h2s, ctr, mode, pct)
'       operating days to `pct`% cumulative incidence (pct in {10, 25}).
'   ESP_DominantMode(field, h2s, ctr, age_op_d)
'       the mode with the largest near-term incidence share (planning heuristic).
'
' Mode-mix is FIELD-LEVEL (field x H2S); the contractor argument is accepted for
' call-site symmetry but folded to the field-level stratum.
' ============================================================

Private Const MODEMIX_SHEET As String = "ESP_ModeMix"

Private Type ModeRow
    stratum As String   ' field-level: {field}_{h2s}
    mode    As String
    share90 As Double
    share365 As Double
    t10     As Double
    t25     As Double
End Type

Private mRows()  As ModeRow
Private mNum     As Long
Private mMMLoaded As Boolean

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
            HeaderCol = c : Exit Function
        End If
    Next c
    HeaderCol = 0
End Function

Private Function SafeDbl(ByVal v As Variant) As Double
    On Error Resume Next
    Dim s As String
    s = Trim(CStr(v))
    If s = "" Then SafeDbl = 0# Else SafeDbl = CDbl(Val(s))
    On Error GoTo 0
End Function

Public Sub LoadModeMix()
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEMIX_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then mMMLoaded = False : Exit Sub

    Dim cStrt As Long, cMode As Long, c90 As Long, c365 As Long, c10 As Long, c25 As Long
    cStrt = HeaderCol(ws, "stratum")
    cMode = HeaderCol(ws, "mode_group")
    c90 = HeaderCol(ws, "share_90d")
    c365 = HeaderCol(ws, "share_365d")
    c10 = HeaderCol(ws, "t_to_10pct_op_d")
    c25 = HeaderCol(ws, "t_to_25pct_op_d")
    If cStrt = 0 Or cMode = 0 Then mMMLoaded = False : Exit Sub

    Dim lastRow As Long, i As Long
    lastRow = ws.Cells(ws.Rows.Count, cStrt).End(xlUp).Row
    If lastRow < 2 Then mMMLoaded = False : Exit Sub

    ReDim mRows(1 To lastRow - 1)
    mNum = 0
    For i = 2 To lastRow
        If Trim(CStr(ws.Cells(i, cStrt).Value)) = "" Then GoTo NextRow
        mNum = mNum + 1
        With mRows(mNum)
            .stratum = Trim(CStr(ws.Cells(i, cStrt).Value))
            .mode = LCase(Trim(CStr(ws.Cells(i, cMode).Value)))
            .share90 = IIf(c90 > 0, SafeDbl(ws.Cells(i, c90).Value), 0#)
            .share365 = IIf(c365 > 0, SafeDbl(ws.Cells(i, c365).Value), 0#)
            .t10 = IIf(c10 > 0, SafeDbl(ws.Cells(i, c10).Value), 0#)
            .t25 = IIf(c25 > 0, SafeDbl(ws.Cells(i, c25).Value), 0#)
        End With
NextRow:
    Next i
    mMMLoaded = (mNum > 0)
End Sub

Public Sub EnsureModeMixLoaded()
    If Not mMMLoaded Then LoadModeMix
End Sub

Public Sub RefreshModeMix()
    mMMLoaded = False
    LoadModeMix
End Sub

' Resolve the field-level stratum key for mode-mix: {field}_{h2s}, falling back
' to Other_{h2s} for small (pooled-to-Other) fields.
Private Function FieldLevelKey(ByVal field As String, ByVal h2sCls As String) As String
    Dim k As String, i As Long
    k = Trim(field) & "_" & h2sCls
    For i = 1 To mNum
        If mRows(i).stratum = k Then FieldLevelKey = k : Exit Function
    Next i
    FieldLevelKey = "Other_" & h2sCls
End Function

' Return the mRows index for (field-level stratum, mode); 0 if absent.
Private Function FindModeRow(ByVal field As String, ByVal h2sCls As String, _
                             ByVal modeCode As String) As Long
    If Not mMMLoaded Then LoadModeMix
    If Not mMMLoaded Then FindModeRow = 0 : Exit Function
    Dim key As String, i As Long
    key = FieldLevelKey(field, h2sCls)
    For i = 1 To mNum
        If mRows(i).stratum = key And mRows(i).mode = LCase(Trim(modeCode)) Then
            FindModeRow = i : Exit Function
        End If
    Next i
    FindModeRow = 0
End Function

' ------------------------------------------------------------------
' ESP_ModeShare -- share of incidence by `mode` at a 90 or 365 op-day horizon.
' Only 90d and 365d are tabulated; the nearer of the two is returned.
'   =ESP_ModeShare(A2, BW2, D2, "electro-thermal", 365)
' ------------------------------------------------------------------
Public Function ESP_ModeShare(ByVal field As String, ByVal h2s As String, _
                              ByVal contractor As String, ByVal mode As String, _
                              ByVal horizonDays As Double) As Variant
    On Error GoTo ErrHandler
    Dim h2sCls As String
    h2sCls = H2SClass(h2s)
    Dim idx As Long
    idx = FindModeRow(field, h2sCls, mode)
    If idx = 0 Then ESP_ModeShare = CVErr(xlErrNA) : Exit Function
    If Abs(horizonDays - 90#) <= Abs(horizonDays - 365#) Then
        ESP_ModeShare = mRows(idx).share90
    Else
        ESP_ModeShare = mRows(idx).share365
    End If
    Exit Function
ErrHandler:
    ESP_ModeShare = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_ModeIncidenceTime -- operating days to `pct`% cumulative incidence (CIF).
' pct accepts 10/25 or 0.10/0.25. Returns "" if that level is never reached.
'   =ESP_ModeIncidenceTime(A2, BW2, D2, "hydraulic", 10)
' ------------------------------------------------------------------
Public Function ESP_ModeIncidenceTime(ByVal field As String, ByVal h2s As String, _
                                      ByVal contractor As String, ByVal mode As String, _
                                      ByVal pct As Double) As Variant
    On Error GoTo ErrHandler
    Dim h2sCls As String
    h2sCls = H2SClass(h2s)
    Dim idx As Long
    idx = FindModeRow(field, h2sCls, mode)
    If idx = 0 Then ESP_ModeIncidenceTime = CVErr(xlErrNA) : Exit Function

    Dim p As Double
    p = pct
    If p < 1# Then p = p * 100#   ' accept fraction form

    Dim v As Double
    If Abs(p - 10#) <= Abs(p - 25#) Then v = mRows(idx).t10 Else v = mRows(idx).t25
    ESP_ModeIncidenceTime = IIf(v > 0, v, "")
    Exit Function
ErrHandler:
    ESP_ModeIncidenceTime = CVErr(xlErrValue)
End Function

' ------------------------------------------------------------------
' ESP_DominantMode -- planning heuristic: the mode with the largest near-term
' incidence share for a pump at age_op_d.  Uses the 90d share for young pumps
' and the 365d share otherwise (the two tabulated horizons).  This is a
' share-based approximation of "which failure mode dominates the next window",
' NOT an age-conditional CIF increment.
'   =ESP_DominantMode(A2, BW2, D2, I2)
' ------------------------------------------------------------------
Public Function ESP_DominantMode(ByVal field As String, ByVal h2s As String, _
                                 ByVal contractor As String, _
                                 ByVal ageOpDays As Double) As Variant
    On Error GoTo ErrHandler
    If Not mMMLoaded Then LoadModeMix
    If Not mMMLoaded Then ESP_DominantMode = CVErr(xlErrNA) : Exit Function

    Dim h2sCls As String, key As String
    h2sCls = H2SClass(h2s)
    key = FieldLevelKey(field, h2sCls)

    Dim useEarly As Boolean
    useEarly = (ageOpDays + 90# <= 227#)   ' midpoint of 90 and 365

    Dim bestMode As String, bestShare As Double, i As Long
    bestMode = "" : bestShare = -1#
    For i = 1 To mNum
        If mRows(i).stratum = key Then
            Dim sh As Double
            sh = IIf(useEarly, mRows(i).share90, mRows(i).share365)
            If sh > bestShare Then bestShare = sh : bestMode = mRows(i).mode
        End If
    Next i
    ESP_DominantMode = IIf(bestMode = "", CVErr(xlErrNA), bestMode)
    Exit Function
ErrHandler:
    ESP_DominantMode = CVErr(xlErrValue)
End Function
