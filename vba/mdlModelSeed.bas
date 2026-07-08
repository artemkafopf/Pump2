Attribute VB_Name = "mdlModelSeed"
Option Explicit

' ============================================================
' mdlModelSeed -- GENERATED, DO NOT EDIT BY HAND.
' Source : esp_models.csv  (fit_date 2026-07-08, git 8f29331, clock ttf_mix)
' Regenerate: python scripts/run/export_model_csv_for_vba.py
'
' CreateModelSheet builds the ESP_Models sheet from the same numbers the
' CSV importer would load, so the hardcoded fallback cannot drift from the
' bundle.  Prefer ImportModelCSV for a live update; use this only to bootstrap
' a fresh workbook offline.
' ============================================================

Private Const MODEL_SHEET As String = "ESP_Models"

Public Sub CreateModelSheet()
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Worksheets(MODEL_SHEET)
    On Error GoTo 0
    If Not ws Is Nothing Then
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
    End If
    Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    ws.Name = MODEL_SHEET

    Dim hdr As Variant
    hdr = Array("stratum", "field", "h2s_class", "contractor_group", "model_kind", "w1", "beta1", "eta1", "beta2", "eta2", "b20", "b50", "b80", "b50_lo", "b50_hi", "uptime_factor", "n_runs", "n_failures", "pct_mixed_clock", "clock", "fit_date")
    Dim c As Long
    For c = 0 To 20
        ws.Cells(1, c + 1).Value = hdr(c)
    Next c
    ws.Rows(1).Font.Bold = True

    Dim rows(25) As Variant
    rows(0) = Array("Ya_nonsour_brt", "Ya", "nonsour", "brt", "k2", 0.626716, 0.739665, 202.96, 2.119616, 981.69, 55.3, 312.5, 883.4, 253.8, 376.1, 0.5324, 809, 410, 0.1335, "ttf_mix", "2026-07-08")
    rows(1) = Array("Ya_nonsour_slb", "Ya", "nonsour", "slb", "k1_aic", 0.0, 0.737091, 325.25, 0.737091, 325.25, 42.5, 197.8, 620.3, 172.8, 332.8, 0.6179, 414, 274, 0.1208, "ttf_mix", "2026-07-08")
    rows(2) = Array("Vt_nonsour_slb", "Vt", "nonsour", "slb", "k2", 0.213108, 1.012732, 18.79, 1.299287, 263.05, 26.4, 143.2, 335.1, 104.1, 180.9, 0.8056, 135, 92, 0.2, "ttf_mix", "2026-07-08")
    rows(3) = Array("Za_nonsour_brt", "Za", "nonsour", "brt", "k2", 0.289679, 1.142207, 24.04, 1.523035, 336.59, 24.8, 169.3, 393.3, 118.7, 236.0, 0.7889, 135, 92, 0.1704, "ttf_mix", "2026-07-08")
    rows(4) = Array("Ic_nonsour_brt", "Ic", "nonsour", "brt", "k2", 0.359714, 1.030019, 38.49, 1.738087, 538.81, 30.4, 241.7, 587.9, 135.3, 342.4, 0.6409, 129, 89, 0.2403, "ttf_mix", "2026-07-08")
    rows(5) = Array("Az_nonsour_brt", "Az", "nonsour", "brt", "k2", 0.244328, 1.16262, 39.29, 1.41678, 482.29, 46.9, 258.4, 589.6, 180.7, 365.1, 0.7957, 135, 77, 0.2148, "ttf_mix", "2026-07-08")
    rows(6) = Array("Vt_nonsour_brt", "Vt", "nonsour", "brt", "k1_aic", 0.0, 0.85372, 336.35, 0.85372, 336.35, 58.0, 219.0, 587.3, 165.7, 3017.6, 0.831, 131, 65, 0.2824, "ttf_mix", "2026-07-08")
    rows(7) = Array("Ic_nonsour_slb", "Ic", "nonsour", "slb", "k1_aic", 0.0, 0.693913, 477.75, 0.693913, 477.75, 55.0, 281.7, 948.5, 184.8, 502.3, 0.7841, 95, 61, 0.2421, "ttf_mix", "2026-07-08")
    rows(8) = Array("Az_nonsour_slb", "Az", "nonsour", "slb", "k2", 0.416652, 1.323733, 50.61, 2.285452, 576.87, 36.5, 254.7, 594.3, 87.8, 413.0, 0.8544, 79, 53, 0.2405, "ttf_mix", "2026-07-08")
    rows(9) = Array("Ya_nonsour_oth", "Ya", "nonsour", "oth", "k1_degenerate", 0.0, 0.810848, 110.83, 0.810848, 110.83, 17.4, 70.5, 199.3, 47.2, 161.8, 0.7837, 58, 47, 0.2414, "ttf_mix", "2026-07-08")
    rows(10) = Array("Mc_nonsour_brt", "Mc", "nonsour", "brt", "k1_degenerate", 0.0, 0.876686, 584.85, 0.876686, 584.85, 105.7, 385.0, 1006.4, 291.7, 4162.2, 0.7381, 111, 41, 0.2523, "ttf_mix", "2026-07-08")
    rows(11) = Array("Za_nonsour_slb", "Za", "nonsour", "slb", "k2", 0.246517, 1.012732, 46.04, 1.523035, 483.53, 55.4, 269.8, 582.1, 172.3, 363.5, 0.7888, 64, 35, 0.1875, "ttf_mix", "2026-07-08")
    rows(12) = Array("Da_nonsour_brt", "Da", "nonsour", "brt", "k1_aic", 0.0, 0.726388, 845.02, 0.726388, 845.02, 107.2, 510.2, 1627.0, 324.0, 1159.5, 0.467, 71, 31, 0.169, "ttf_mix", "2026-07-08")
    rows(13) = Array("Vt_sour_slb", "Vt", "sour", "slb", "k1_aic", 0.0, 1.107747, 106.6, 1.107747, 106.6, 27.5, 76.6, 163.8, 51.4, 119.3, 0.8628, 35, 30, 0.2857, "ttf_mix", "2026-07-08")
    rows(14) = Array("Vt_sour_oth", "Vt", "sour", "oth", "k2", 0.379373, 1.012732, 10.1, 1.523035, 98.63, 7.0, 37.9, 107.0, 17.0, 67.2, 0.7613, 33, 28, 0.2727, "ttf_mix", "2026-07-08")
    rows(15) = Array("Vt_nonsour_oth", "Vt", "nonsour", "oth", "k1_aic", 0.0, 0.867646, 135.53, 0.867646, 135.53, 24.1, 88.8, 234.6, 54.0, 139.5, 0.8321, 43, 27, 0.3256, "ttf_mix", "2026-07-08")
    rows(16) = Array("Vt_sour_brt", "Vt", "sour", "brt", "k2", 0.303828, 1.012732, 15.28, 1.523035, 139.87, 13.7, 68.5, 161.7, 33.0, 101.2, 0.6987, 33, 26, 0.0909, "ttf_mix", "2026-07-08")
    rows(17) = Array("Ya_nonsour_Pooled", "Ya", "nonsour", "Pooled", "k2", 0.5584, 0.735116, 143.82, 1.557376, 812.53, 45.6, 251.0, 754.8, 217.3, 295.0, 0.5714, 1281, 731, 0.1343, "ttf_mix", "2026-07-08")
    rows(18) = Array("Vt_nonsour_Pooled", "Vt", "nonsour", "Pooled", "k1_degenerate", 0.0, 0.823367, 234.5, 0.823367, 234.5, 37.9, 150.3, 418.0, 126.6, 238.3, 0.8201, 309, 184, 0.2524, "ttf_mix", "2026-07-08")
    rows(19) = Array("Ic_nonsour_Pooled", "Ic", "nonsour", "Pooled", "k2", 0.319528, 0.873796, 43.08, 1.466329, 603.13, 37.7, 272.8, 692.5, 206.1, 351.7, 0.7018, 228, 153, 0.2368, "ttf_mix", "2026-07-08")
    rows(20) = Array("Az_nonsour_Pooled", "Az", "nonsour", "Pooled", "k2", 0.336561, 1.141845, 48.9, 1.665032, 510.78, 41.6, 240.0, 569.6, 177.9, 326.4, 0.8211, 235, 144, 0.2255, "ttf_mix", "2026-07-08")
    rows(21) = Array("Za_nonsour_Pooled", "Za", "nonsour", "Pooled", "k2", 0.28647, 1.015514, 29.06, 1.419015, 371.59, 29.1, 179.7, 440.2, 140.6, 235.1, 0.7922, 222, 143, 0.1892, "ttf_mix", "2026-07-08")
    rows(22) = Array("Vt_sour_Pooled", "Vt", "sour", "Pooled", "k2", 0.312599, 1.43291, 13.19, 1.790852, 132.62, 12.6, 70.0, 149.2, 43.2, 97.0, 0.776, 101, 84, 0.2178, "ttf_mix", "2026-07-08")
    rows(23) = Array("Mc_nonsour_Pooled", "Mc", "nonsour", "Pooled", "k1_degenerate", 0.0, 0.881914, 585.58, 0.881914, 585.58, 106.9, 386.5, 1004.5, 283.5, 889.1, 0.745, 114, 41, 0.2719, "ttf_mix", "2026-07-08")
    rows(24) = Array("Da_nonsour_Pooled", "Da", "nonsour", "Pooled", "k1_aic", 0.0, 0.707763, 826.29, 0.707763, 826.29, 99.3, 492.3, 1618.6, 303.2, 1138.7, 0.4671, 74, 32, 0.1757, "ttf_mix", "2026-07-08")
    rows(25) = Array("Global_Pooled", "", "", "Pooled", "k2", 0.195534, 0.845927, 33.63, 1.01, 470.79, 40.6, 226.8, 653.1, 208.4, 246.2, 0.6718, 2634, 1527, 0.1936, "ttf_mix", "2026-07-08")

    Dim r As Long, col As Long
    For r = 0 To UBound(rows)
        For col = 0 To 20
            ws.Cells(r + 2, col + 1).Value = rows(r)(col)
        Next col
    Next r

    ws.Columns("A:U").EntireColumn.AutoFit
    ' Registry reload is the caller's responsibility (RefreshModels).
End Sub
