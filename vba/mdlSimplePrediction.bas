Attribute VB_Name = "mdlSimplePrediction"
Option Explicit

' ============================================================
' mdlSimplePrediction -- refresh / calculate helper for the
' one-row simplified prediction sheet.
'
' The sheet layout itself is created externally (Python/Excel COM)
' so this module stays free of Cyrillic literals.  This macro:
'   * scans the source data sheet ("Свод") for matching well history,
'   * fills the history grid on the prediction sheet,
'   * derives prior-install count and days since previous failure,
'   * recalculates the workbook so the UDF-driven outputs refresh.
' ============================================================

Private Const SRC_FIELD_COL As Long = 1
Private Const SRC_WELL_COL As Long = 3
Private Const SRC_CTR_COL As Long = 4
Private Const SRC_ESP_COL As Long = 5
Private Const SRC_RUN_COL As Long = 6
Private Const SRC_MOUNT_COL As Long = 7
Private Const SRC_STOP_COL As Long = 8
Private Const SRC_AGE_COL As Long = 9
Private Const SRC_FLAG_COL As Long = 76
Private Const SRC_FAILNODE_COL As Long = 13
Private Const SRC_FAILCAUSE_COL As Long = 16

Private Const IN_FIELD As String = "B4"
Private Const IN_WELL As String = "B5"
Private Const IN_PAD As String = "B6"
Private Const IN_CTR As String = "B7"
Private Const IN_ESP As String = "B8"
Private Const IN_MOUNT As String = "B9"

Private Const OUT_PRIOR As String = "B30"
Private Const OUT_DAYS_PREV As String = "B32"
Private Const OUT_STATUS As String = "H18"
Private Const HIST_HEADER_ROW As Long = 40
Private Const HIST_FIRST_ROW As Long = 41

Private Const M_ROW_FIRST As Long = 3
Private Const M_COL_FIELD As Long = 1   ' A
Private Const M_COL_WELL As Long = 2    ' B
Private Const M_COL_MOUNT As Long = 6   ' F
Private Const M_COL_PRIOR As Long = 33  ' AG
Private Const M_COL_RUNSEQ As Long = 34 ' AH
Private Const M_COL_DAYS As Long = 35   ' AI
Private Const M_STATUS_CELL As String = "AW1"

Private Function PredictionSheetName() As String
    Static s As String
    If s = "" Then
        s = ChrW(1055) & ChrW(1088) & ChrW(1086) & ChrW(1075) & ChrW(1085) & ChrW(1086) & ChrW(1079)
    End If
    PredictionSheetName = s
End Function

Private Function SourceSheetName() As String
    Static s As String
    If s = "" Then
        s = ChrW(1057) & ChrW(1074) & ChrW(1086) & ChrW(1076)
    End If
    SourceSheetName = s
End Function

Private Function MultiSheetName() As String
    Static s As String
    If s = "" Then
        s = ChrW(1055) & ChrW(1088) & ChrW(1086) & ChrW(1075) & ChrW(1085) & ChrW(1086) & ChrW(1079) & "_" & _
            ChrW(1084) & ChrW(1091) & ChrW(1083) & ChrW(1100) & ChrW(1090) & ChrW(1080)
    End If
    MultiSheetName = s
End Function

Private Function TrimKey(ByVal v As Variant) As String
    TrimKey = LCase(Trim(CStr(v)))
End Function

Private Function DateDiffSafe(ByVal d1 As Variant, ByVal d2 As Variant) As Variant
    On Error GoTo FailDiff
    If IsDate(d1) And IsDate(d2) Then
        DateDiffSafe = CLng(CDate(d1) - CDate(d2))
        Exit Function
    End If
FailDiff:
    DateDiffSafe = ""
End Function

Private Function MaxDate(ByVal a As Variant, ByVal b As Variant) As Variant
    If IsDate(a) And IsDate(b) Then
        If CDate(a) >= CDate(b) Then MaxDate = a Else MaxDate = b
    ElseIf IsDate(a) Then
        MaxDate = a
    ElseIf IsDate(b) Then
        MaxDate = b
    Else
        MaxDate = ""
    End If
End Function

Private Sub WriteHistoryHeaders(ByVal ws As Worksheet)
    ws.Cells(HIST_HEADER_ROW, 1).Value = "#"
    ws.Cells(HIST_HEADER_ROW, 2).Value = "Field"
    ws.Cells(HIST_HEADER_ROW, 3).Value = "Well"
    ws.Cells(HIST_HEADER_ROW, 4).Value = "MountDate"
    ws.Cells(HIST_HEADER_ROW, 5).Value = "StopDate"
    ws.Cells(HIST_HEADER_ROW, 6).Value = "RunDays"
    ws.Cells(HIST_HEADER_ROW, 7).Value = "FailureFlag"
    ws.Cells(HIST_HEADER_ROW, 8).Value = "FailedNode"
    ws.Cells(HIST_HEADER_ROW, 9).Value = "FailureCause"
    ws.Cells(HIST_HEADER_ROW, 10).Value = "Contractor"
    ws.Cells(HIST_HEADER_ROW, 11).Value = "ESPType"
    ws.Rows(HIST_HEADER_ROW).Font.Bold = True
End Sub

Private Sub ClearHistoryArea(ByVal ws As Worksheet)
    ws.Range(ws.Cells(HIST_FIRST_ROW, 1), ws.Cells(ws.Rows.Count, 11)).ClearContents
    WriteHistoryHeaders ws
End Sub

Private Sub SetStatus(ByVal ws As Worksheet, ByVal msg As String)
    ws.Range(OUT_STATUS).Value = msg
End Sub

Private Sub SetMultiStatus(ByVal ws As Worksheet, ByVal msg As String)
    ws.Range(M_STATUS_CELL).Value = msg
End Sub

Private Sub HistoryStats(ByVal src As Worksheet, ByVal fieldKey As String, ByVal wellKey As String, _
                         ByVal currentMount As Variant, ByRef nAll As Long, ByRef nPrev As Long, _
                         ByRef latestPrevEnd As Variant)
    Dim lastRow As Long, r As Long
    Dim srcMount As Variant, srcEnd As Variant

    nAll = 0
    nPrev = 0
    latestPrevEnd = ""
    lastRow = src.Cells(src.Rows.Count, SRC_FIELD_COL).End(xlUp).Row
    For r = 2 To lastRow
        If TrimKey(src.Cells(r, SRC_FIELD_COL).Value) = fieldKey And _
           TrimKey(src.Cells(r, SRC_WELL_COL).Value) = wellKey Then
            nAll = nAll + 1
            srcMount = src.Cells(r, SRC_MOUNT_COL).Value
            If IsDate(currentMount) And IsDate(srcMount) Then
                If CDate(srcMount) < CDate(currentMount) Then
                    nPrev = nPrev + 1
                    srcEnd = MaxDate(src.Cells(r, SRC_STOP_COL).Value, src.Cells(r, 11).Value)
                    latestPrevEnd = MaxDate(latestPrevEnd, srcEnd)
                End If
            Else
                nPrev = nAll
                srcEnd = MaxDate(src.Cells(r, SRC_STOP_COL).Value, src.Cells(r, 11).Value)
                latestPrevEnd = MaxDate(latestPrevEnd, srcEnd)
            End If
        End If
    Next r
End Sub

Public Sub CalculateSimplePrediction()
    Dim ws As Worksheet, src As Worksheet
    Dim fieldKey As String, wellKey As String
    Dim currentMount As Variant
    Dim lastRow As Long, r As Long, outRow As Long
    Dim nAll As Long, nPrev As Long
    Dim latestPrevEnd As Variant, srcMount As Variant, srcEnd As Variant

    On Error GoTo FailHandler
    Set ws = ThisWorkbook.Worksheets(PredictionSheetName())
    Set src = ThisWorkbook.Worksheets(SourceSheetName())

    fieldKey = TrimKey(ws.Range(IN_FIELD).Value)
    wellKey = TrimKey(ws.Range(IN_WELL).Value)
    currentMount = ws.Range(IN_MOUNT).Value

    ClearHistoryArea ws
    ws.Range(OUT_PRIOR).Value = 0
    ws.Range(OUT_DAYS_PREV).Value = ""

    If fieldKey = "" Or wellKey = "" Then
        SetStatus ws, "Enter field and well, then click Calculate."
        Exit Sub
    End If

    lastRow = src.Cells(src.Rows.Count, SRC_FIELD_COL).End(xlUp).Row
    outRow = HIST_FIRST_ROW
    nAll = 0
    nPrev = 0

    For r = 2 To lastRow
        If TrimKey(src.Cells(r, SRC_FIELD_COL).Value) = fieldKey And _
           TrimKey(src.Cells(r, SRC_WELL_COL).Value) = wellKey Then

            nAll = nAll + 1
            ws.Cells(outRow, 1).Value = nAll
            ws.Cells(outRow, 2).Value = src.Cells(r, SRC_FIELD_COL).Value
            ws.Cells(outRow, 3).Value = src.Cells(r, SRC_WELL_COL).Value
            ws.Cells(outRow, 4).Value = src.Cells(r, SRC_MOUNT_COL).Value
            ws.Cells(outRow, 5).Value = src.Cells(r, SRC_STOP_COL).Value
            ws.Cells(outRow, 6).Value = src.Cells(r, SRC_AGE_COL).Value
            ws.Cells(outRow, 7).Value = src.Cells(r, SRC_FLAG_COL).Value
            ws.Cells(outRow, 8).Value = src.Cells(r, SRC_FAILNODE_COL).Value
            ws.Cells(outRow, 9).Value = src.Cells(r, SRC_FAILCAUSE_COL).Value
            ws.Cells(outRow, 10).Value = src.Cells(r, SRC_CTR_COL).Value
            ws.Cells(outRow, 11).Value = src.Cells(r, SRC_ESP_COL).Value
            outRow = outRow + 1

            srcMount = src.Cells(r, SRC_MOUNT_COL).Value
            If IsDate(currentMount) And IsDate(srcMount) Then
                If CDate(srcMount) < CDate(currentMount) Then
                    nPrev = nPrev + 1
                    srcEnd = MaxDate(src.Cells(r, SRC_STOP_COL).Value, src.Cells(r, 11).Value)
                    latestPrevEnd = MaxDate(latestPrevEnd, srcEnd)
                End If
            Else
                nPrev = nAll
                srcEnd = MaxDate(src.Cells(r, SRC_STOP_COL).Value, src.Cells(r, 11).Value)
                latestPrevEnd = MaxDate(latestPrevEnd, srcEnd)
            End If
        End If
    Next r

    Call HistoryStats(src, fieldKey, wellKey, currentMount, nAll, nPrev, latestPrevEnd)
    ws.Range(OUT_PRIOR).Value = nPrev
    If IsDate(currentMount) And IsDate(latestPrevEnd) Then
        ws.Range(OUT_DAYS_PREV).Value = DateDiffSafe(currentMount, latestPrevEnd)
    Else
        ws.Range(OUT_DAYS_PREV).Value = ""
    End If

    ws.Columns("A:K").AutoFit
    Application.CalculateFull

    SetStatus ws, "Calculated " & Format(Now, "yyyy-mm-dd hh:nn:ss") & _
                  " | matches in history: " & nAll & " | prior installs: " & nPrev
    Exit Sub

FailHandler:
    SetStatus ws, "Error " & Err.Number & ": " & Err.Description
End Sub

Public Sub CalculateMultiPrediction()
    Dim ws As Worksheet, src As Worksheet
    Dim lastRow As Long, r As Long
    Dim fieldKey As String, wellKey As String
    Dim currentMount As Variant
    Dim nAll As Long, nPrev As Long, nTouched As Long
    Dim latestPrevEnd As Variant

    On Error GoTo FailHandler
    Set ws = ThisWorkbook.Worksheets(MultiSheetName())
    Set src = ThisWorkbook.Worksheets(SourceSheetName())

    lastRow = ws.Cells(ws.Rows.Count, M_COL_FIELD).End(xlUp).Row
    If lastRow < M_ROW_FIRST Then lastRow = M_ROW_FIRST
    nTouched = 0

    For r = M_ROW_FIRST To lastRow
        fieldKey = TrimKey(ws.Cells(r, M_COL_FIELD).Value)
        wellKey = TrimKey(ws.Cells(r, M_COL_WELL).Value)
        currentMount = ws.Cells(r, M_COL_MOUNT).Value
        If fieldKey = "" Or wellKey = "" Then
            ws.Cells(r, M_COL_PRIOR).Value = ""
            ws.Cells(r, M_COL_DAYS).Value = ""
            If Trim(CStr(ws.Cells(r, M_COL_RUNSEQ).Formula)) = "" Then ws.Cells(r, M_COL_RUNSEQ).Value = ""
        Else
            Call HistoryStats(src, fieldKey, wellKey, currentMount, nAll, nPrev, latestPrevEnd)
            ws.Cells(r, M_COL_PRIOR).Value = nPrev
            If IsDate(currentMount) And IsDate(latestPrevEnd) Then
                ws.Cells(r, M_COL_DAYS).Value = DateDiffSafe(currentMount, latestPrevEnd)
            Else
                ws.Cells(r, M_COL_DAYS).Value = ""
            End If
            nTouched = nTouched + 1
        End If
    Next r

    Application.CalculateFull
    SetMultiStatus ws, "Рассчитано " & Format(Now, "yyyy-mm-dd hh:nn:ss") & _
                       " | строк с данными: " & nTouched
    Exit Sub

FailHandler:
    SetMultiStatus ws, "Ошибка " & Err.Number & ": " & Err.Description
End Sub
