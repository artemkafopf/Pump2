Attribute VB_Name = "mdlProductionRiskLauncher"
Option Explicit

Private Function QuoteArg(ByVal value As String) As String
    QuoteArg = Chr$(34) & Replace(value, Chr$(34), Chr$(34) & Chr$(34)) & Chr$(34)
End Function

Private Function RequiredPath(ByVal ws As Worksheet, ByVal address As String, ByVal label As String) As String
    Dim fso As Object
    Dim value As String

    value = Trim$(CStr(ws.Range(address).Value2))
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Len(value) = 0 Then
        Err.Raise vbObjectError + 2101, "Pump2", label & " path is empty."
    End If
    If Not fso.FileExists(value) Then
        Err.Raise vbObjectError + 2101, "Pump2", label & " file was not found:" & vbCrLf & value
    End If
    RequiredPath = value
End Function

Public Sub RunProductionRisk()
    On Error GoTo Failed

    Dim ws As Worksheet
    Dim shell As Object
    Dim fso As Object
    Dim exePath As String
    Dim command As String
    Dim outputFolder As String
    Dim exitCode As Long
    Dim downtimeValue As Variant
    Dim errorMessage As String

    Set ws = ThisWorkbook.Worksheets(1)
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")

    exePath = RequiredPath(ws, "B8", "Executable")
    command = QuoteArg(exePath)
    command = command & " --pp-master " & QuoteArg(RequiredPath(ws, "B1", "Production plan"))
    command = command & " --gtm-schedule " & QuoteArg(RequiredPath(ws, "B2", "GTM schedule"))
    command = command & " --prediction-workbook " & QuoteArg(RequiredPath(ws, "B3", "Prediction workbook"))
    command = command & " --techregime-workbook " & QuoteArg(RequiredPath(ws, "B4", "Techregime workbook"))

    If Not IsDate(ws.Range("B6").Value) Or Not IsDate(ws.Range("B7").Value) Then
        Err.Raise vbObjectError + 2102, "Pump2", "Forecast dates in B6:B7 are invalid."
    End If
    command = command & " --forecast-start " & Format$(CDate(ws.Range("B6").Value), "yyyy-mm-dd")
    command = command & " --horizon-end " & Format$(CDate(ws.Range("B7").Value), "yyyy-mm-dd")

    downtimeValue = ws.Range("B5").Value2
    If Len(Trim$(CStr(downtimeValue))) > 0 Then
        If Not IsNumeric(downtimeValue) Or CLng(downtimeValue) < 1 Then
            Err.Raise vbObjectError + 2103, "Pump2", "Downtime in B5 must be blank or a positive whole number."
        End If
        command = command & " --downtime-days " & CStr(CLng(downtimeValue))
    End If

    ws.Range("B10").Value = "Running..."
    ws.Range("B11").Value = Now
    ws.Range("B12").Value = command
    ThisWorkbook.Save
    DoEvents

    exitCode = shell.Run(command, 1, True)
    ws.Range("B11").Value = Now
    If exitCode <> 0 Then
        Err.Raise vbObjectError + 2104, "Pump2", "Forecast process returned exit code " & CStr(exitCode) & "."
    End If

    outputFolder = fso.BuildPath(fso.GetParentFolderName(exePath), "results")
    ws.Range("B10").Value = "Completed"
    ws.Range("B13").Value = outputFolder
    ThisWorkbook.Save

    If fso.FolderExists(outputFolder) Then shell.Run "explorer.exe " & QuoteArg(outputFolder), 1, False
    MsgBox "Forecast completed successfully." & vbCrLf & outputFolder, vbInformation, "Pump2"
    Exit Sub

Failed:
    errorMessage = Err.Description
    On Error Resume Next
    If Not ws Is Nothing Then
        ws.Range("B10").Value = "Error: " & errorMessage
        ws.Range("B11").Value = Now
    End If
    ThisWorkbook.Save
    MsgBox errorMessage, vbCritical, "Pump2 forecast"
End Sub
