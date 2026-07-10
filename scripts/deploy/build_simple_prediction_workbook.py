"""Build simplified single-well and multi-well ESP prediction sheets in an .xlsm."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_WORKBOOK = Path(
    r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all_esp_models_2026-07-08.xlsm"
)
SHEET_SINGLE = "Прогноз"
SHEET_MULTI = "Прогноз_мульти"
KEEP_VISIBLE = {"Свод", SHEET_SINGLE, SHEET_MULTI}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simplified ESP prediction sheets.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    return parser.parse_args()


def require_windows():
    if sys.platform != "win32":
        raise RuntimeError("This script only works on Windows with desktop Excel.")


def imports():
    import pythoncom  # type: ignore
    import win32com.client as win32  # type: ignore
    return pythoncom, win32


def set_cell(ws, addr: str, value, *, bold=False, fill=None, fmt=None, wrap=False, size=None):
    cell = ws.Range(addr)
    cell.Value = value
    if bold:
        cell.Font.Bold = True
    if size is not None:
        cell.Font.Size = size
    if fill is not None:
        cell.Interior.Color = fill
    if fmt is not None:
        cell.NumberFormat = fmt
    cell.WrapText = bool(wrap)


def set_formula(ws, addr: str, formula: str, *, fmt=None):
    cell = ws.Range(addr)
    cell.Formula = formula
    if fmt is not None:
        cell.NumberFormat = fmt


def set_comment(ws, addr: str, text: str):
    cell = ws.Range(addr)
    try:
        if cell.Comment is not None:
            cell.Comment.Delete()
    except Exception:
        pass
    cell.AddComment(text)
    cell.Comment.Visible = False


def add_result_comments_single(ws):
    comments = {
        "G4": "Страта модели: какая модельная группа выбрана по месторождению, среде и принадлежности.",
        "G5": "Тип модели: служебное имя выбранной базовой модели.",
        "G6": "ННО_баз: базовая медианная наработка до отказа без учета инд. осложнений.",
        "G7": "ОР_баз: базовый остаточный ресурс на текущую наработку без учета инд. осложнений.",
        "G8": "Множитель риска: во сколько раз текущие условия повышают или снижают риск относительно базового.",
        "G9": "Флаги риска: какие группы осложнений сейчас выглядят неблагоприятными.",
        "G10": "ННО: медианная наработка до отказа с учетом инд. осложнений.",
        "G11": "ОР: остаточный ресурс с учетом инд. осложнений.",
        "G13": "Категория отказа: наиболее вероятная группа причины отказа.",
        "G14": "Режим отказа: ранний отказ или износной отказ.",
        "G15": "Вероятность дожития SF(t): вероятность доработать до текущей наработки без отказа.",
        "G16": "Вероятность отказа за 90 дней: условная вероятность отказа в ближайшие 90 суток с учетом инд. осложнений.",
        "G17": "Коэф. эксплуатации: доля календарного времени, когда установка реально работала.",
        "G18": "Статус расчета и служебная информация по примеру.",
    }
    for addr, text in comments.items():
        set_comment(ws, addr, text)


def add_result_comments_multi(ws):
    comments = {
        "AH4": "Номер спуска, который используется в расчете: ручной ввод или число прошлых установок + 1.",
        "AI4": "Сколько дней прошло после предыдущего отказа на этой скважине.",
        "AJ4": "ННО_баз: базовая медианная наработка до отказа без учета инд. осложнений.",
        "AK4": "ОР_баз: базовый остаточный ресурс на текущую наработку без учета инд. осложнений.",
        "AL4": "Множитель риска: во сколько раз текущие условия повышают или снижают риск относительно базового.",
        "AM4": "Флаги риска: какие группы осложнений сейчас выглядят неблагоприятными.",
        "AN4": "ННО: медианная наработка до отказа с учетом инд. осложнений.",
        "AO4": "ОР: остаточный ресурс с учетом инд. осложнений.",
        "AQ4": "Категория отказа: наиболее вероятная группа причины отказа.",
        "AR4": 'Режим отказа: "Ранний" или "Износ".',
        "AS4": "Вероятность дожития SF(t): вероятность доработать до текущей наработки без отказа.",
        "AT4": "Вероятность отказа за 90 дней: условная вероятность отказа в ближайшие 90 суток с учетом инд. осложнений.",
        "AU4": "Коэф. эксплуатации: доля календарного времени, когда установка реально работала.",
    }
    for addr, text in comments.items():
        set_comment(ws, addr, text)


def add_input_comments_single(ws):
    comments = {
        "D11": "Количество заметных изменений частоты: сколько раз частота менялась более чем на 1 Гц, в пересчете на 100 дней работы.",
        "D12": "Если есть фактическая доля дней с Кпод_freq < 0.7, можно ввести ее вручную. Если пусто, ниже будет использована простая оценка из текущего Кпод_freq.",
        "A27": "Кпод_freq: частотно-скорректированный коэффициент подачи. Считается как Qж / Qном * (50 / f).",
        "A28": "Доля дней, когда Кпод_freq был ниже 0.7. Если ручной ввод сверху пустой, здесь будет упрощенная оценка.",
        "A29": "Коэф. эксплуатации: отношение текущей наработки к календарному времени с даты монтажа, ограниченное сверху 1.",
        "A30": "Сколько предыдущих установок найдено по этой скважине в листе Свод.",
        "A31": "Номер спуска, используемый в расчете: ручной ввод или число прошлых установок + 1.",
        "A32": "Сколько дней прошло между текущим монтажом и окончанием предыдущей установки.",
    }
    for addr, text in comments.items():
        set_comment(ws, addr, text)


def add_input_comments_multi(ws):
    comments = {
        "R2": "Количество заметных изменений частоты: сколько раз частота менялась более чем на 1 Гц, в пересчете на 100 дней работы.",
        "S2": "Если есть фактическая доля дней с Кпод_freq < 0.7, можно ввести ее вручную. Если пусто, будет использована простая оценка из текущего Кпод_freq.",
        "AD2": "Кпод_freq: частотно-скорректированный коэффициент подачи. Считается как Qж / Qном * (50 / f).",
        "AE2": "Доля дней, когда Кпод_freq был ниже 0.7. Если ручной ввод в колонке S пустой, здесь будет упрощенная оценка.",
        "AF2": "Коэф. эксплуатации: отношение текущей наработки к календарному времени с даты монтажа, ограниченное сверху 1.",
        "AG2": "Сколько предыдущих установок найдено по этой скважине в листе Свод.",
        "AH2": "Номер спуска, используемый в расчете: ручной ввод или число прошлых установок + 1.",
        "AI2": "Сколько дней прошло между текущим монтажом и окончанием предыдущей установки.",
    }
    for addr, text in comments.items():
        set_comment(ws, addr, text)


def excel_date_serial(dt_obj):
    if dt_obj in (None, ""):
        return ""
    try:
        year = dt_obj.year
        month = dt_obj.month
        day = dt_obj.day
    except Exception:
        return dt_obj
    import datetime as _dt
    return (_dt.datetime(year, month, day) - _dt.datetime(1899, 12, 30)).days


def place_button(ws, name: str, cell_addr: str, width_cells: str, caption: str, macro: str):
    for shape in list(ws.Shapes):
        if shape.Name == name:
            shape.Delete()
    cell = ws.Range(cell_addr)
    width_rng = ws.Range(width_cells)
    button = ws.Shapes.AddShape(1, cell.Left, cell.Top + 1, width_rng.Width, cell.Height - 2)
    button.Name = name
    button.TextFrame.Characters().Text = caption
    button.OnAction = macro
    button.Fill.ForeColor.RGB = 0x4F81BD
    button.Line.ForeColor.RGB = 0x385D8A
    button.TextFrame.Characters().Font.Color = 0xFFFFFF
    button.TextFrame.HorizontalAlignment = -4108


def style_block_borders(ws, ranges: list[str]):
    xl_edge_bottom = 9
    xl_edge_top = 8
    xl_edge_left = 7
    xl_edge_right = 10
    xl_continuous = 1
    for rng in ranges:
        block = ws.Range(rng)
        for border_id in [xl_edge_left, xl_edge_right, xl_edge_top, xl_edge_bottom]:
            border = block.Borders(border_id)
            border.LineStyle = xl_continuous


def build_single_sheet(ws):
    xl_center = -4108
    xl_left = -4131

    ws.Cells.Clear()
    ws.Name = SHEET_SINGLE
    ws.Activate()
    ws.Range("A1:K60").Font.Name = "Calibri"
    ws.Range("A1:K60").Font.Size = 11

    ws.Range("A1:K1").Merge()
    set_cell(
        ws,
        "A1",
        "ESP прогноз по одной установке",
        bold=True,
        fill=0xD9EAF7,
        size=14,
    )
    ws.Range("A1").HorizontalAlignment = xl_center

    ws.Range("A2:K2").Merge()
    set_cell(
        ws,
        "A2",
        "Вводите значения из Свод или вручную. Принадлежность принимает сырые значения "
        "из Свод (Борец, Шлюмберже, Новые Технологии и др.); модель сама сведет их к группам "
        "brt/slb/oth/Pooled. Пустой 'Кислый/Некислый' считается как Некислый. "
        "Пустые hazard-параметры игнорируются и не ухудшают прогноз.",
        fill=0xFCE4D6,
        wrap=True,
    )

    # General
    set_cell(ws, "A3", "Общее", bold=True, fill=0xE2F0D9)
    labels_general = [
        ("A4", "Месторождение"),
        ("A5", "Скважина"),
        ("A6", "Куст / Pad"),
        ("A7", "Принадлежность"),
        ("A8", "Тип УЭЦН"),
        ("A9", "Дата монтажа"),
        ("A10", "Тип ствола"),
        ("A11", "Кислый / Некислый"),
        ("A12", "Наработка текущая, оп.сут"),
        ("A13", "Номер спуска (ручной, опц.)"),
    ]
    for addr, label in labels_general:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("A", "B")).Interior.Color = 0xFFFFFF
    ws.Range("B9").NumberFormat = "dd.mm.yyyy"

    # Current regime
    set_cell(ws, "D3", "Текущий режим работы", bold=True, fill=0xE2F0D9)
    labels_regime = [
        ("D4", "Qж, м3/сут"),
        ("D5", "Qном, м3/сут"),
        ("D6", "Частота, Гц"),
        ("D7", "Загрузка, %"),
        ("D8", "Разброс загрузки"),
        ("D9", "Газовый фактор"),
        ("D10", "% времени >55 Гц"),
        ("D11", "Изменений частоты >1 Гц на 100 дней"),
        ("D12", "Доля дней kpod<0.7 (ручн., опц.)"),
    ]
    for addr, label in labels_regime:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("D", "E")).Interior.Color = 0xFFFFFF

    # Chemistry
    set_cell(ws, "A17", "Химия / среда", bold=True, fill=0xE2F0D9)
    labels_chem = [
        ("A18", "H2S, мг/дм3"),
        ("A19", "Ca, мг/л"),
        ("A20", "Cl, мг/л"),
        ("A21", "SO4, мг/л"),
        ("A22", "КВЧ, мг/дм3"),
    ]
    for addr, label in labels_chem:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("A", "B")).Interior.Color = 0xFFFFFF

    # Completion
    set_cell(ws, "D17", "Компоновка / паспорт", bold=True, fill=0xE2F0D9)
    labels_completion = [
        ("D18", "Кривизна, °/10м"),
        ("D19", "Номинальная частота, Гц"),
        ("D20", "Мощность ПЭД, кВт"),
        ("D21", "Габарит УЭЦН"),
        ("D22", "Глубина спуска, м"),
    ]
    for addr, label in labels_completion:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("D", "E")).Interior.Color = 0xFFFFFF

    # Derived / history
    set_cell(ws, "A26", "Производные / история", bold=True, fill=0xFFF2CC)
    labels_derived = [
        ("A27", "Кпод_freq"),
        ("A28", "Доля дней kpod<0.7 (Свод)"),
        ("A29", "Кэкспл"),
        ("A30", "Прошлых установок (Свод)"),
        ("A31", "Номер спуска (Свод)"),
        ("A32", "Дней после пред. отказа"),
    ]
    for addr, label in labels_derived:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("A", "B")).Interior.Color = 0xF7F7F7

    set_formula(ws, "E10", '=IF(E6="","",IF(E6>55,100,0))', fmt="0.0")
    set_formula(ws, "B27", '=IF(OR(E4="",E5="",E6=""),"",IF(E5=0,"",E4/E5*(50/E6)))', fmt="0.000")
    set_formula(ws, "B28", '=IF(E12<>"",E12,IF(B27<>"",IF(B27<0.7,1,0),""))', fmt="0.000")
    set_formula(ws, "B29", '=IF(AND(B9<>"",B12<>"",TODAY()>B9),MIN(1,B12/(TODAY()-B9)),"")', fmt="0.000")
    set_cell(ws, "B30", 0, fmt="0")
    set_formula(ws, "B31", '=IF(B13<>"",B13,B30+1)', fmt="0")
    set_cell(ws, "B32", "", fmt="0")

    # Outputs
    ws.Range("G3:G3").Merge()
    set_cell(ws, "G3", "Расчетные показатели", bold=True, fill=0xD9EAF7)
    labels_out = [
        ("G4", "Страта"),
        ("G5", "Тип модели"),
        ("G6", "ННО_баз, оп.сут"),
        ("G7", "ОР_баз, оп.сут"),
        ("G8", "Множитель риска"),
        ("G9", "Флаги риска"),
        ("G10", "ННО, оп.сут"),
        ("G11", "ОР, оп.сут"),
        ("G12", ""),
        ("G13", "Категория отказа"),
        ("G14", "Режим отказа"),
        ("G15", "Вероятность дожития SF(t)"),
        ("G16", "Вероятность отказа за 90 дней"),
        ("G17", "Коэф. эксплуатации"),
        ("G18", "Статус"),
    ]
    for addr, label in labels_out:
        set_cell(ws, addr, label, bold=True)
        ws.Range(addr.replace("G", "H")).Interior.Color = 0xF7F7F7

    direct_args = "E9,E8,E7,B28,B27,E10,E11,E18,B31,B32,B9"
    set_formula(ws, "H4", '=IF(B4="","",ESP_Stratum(B4,B11,B7))')
    set_formula(ws, "H5", '=IF(B4="","",ESP_ModelKind(B4,B11,B7))')
    set_formula(ws, "H6", '=IF(B4="","",ESP_B50(B4,B11,B7))', fmt="0.0")
    set_formula(ws, "H7", '=IF(OR(B4="",B12=""),"",IF(B12>0,ESP_RUL(B12,B4,B11,B7),""))', fmt="0.0")
    set_formula(ws, "H8", f'=IF(B4="","",ESP_HazardTheta_Direct({direct_args}))', fmt="0.000")
    set_formula(ws, "H9", f'=IF(B4="","",ESP_HazardFlags_RU_Direct({direct_args}))')
    set_formula(ws, "H10", f'=IF(B4="","",ESP_B50_Sens_Direct(B4,B11,B7,{direct_args}))', fmt="0.0")
    set_formula(ws, "H11", f'=IF(OR(B4="",B12=""),"",IF(B12>0,ESP_RUL_Sens_Direct(B12,B4,B11,B7,{direct_args}),""))', fmt="0.0")
    set_formula(ws, "H12", '=""')
    set_formula(ws, "H13", '=IF(B4="","",IF(ESP_DominantMode(B4,B11,B7,MAX(0,B12))="hydraulic","Гидравлический",IF(ESP_DominantMode(B4,B11,B7,MAX(0,B12))="electro-thermal","Электротепловой",IF(ESP_DominantMode(B4,B11,B7,MAX(0,B12))="protector","Гидрозащита",IF(ESP_DominantMode(B4,B11,B7,MAX(0,B12))="other","Прочее",ESP_DominantMode(B4,B11,B7,MAX(0,B12)))))))')
    set_formula(ws, "H14", '=IF(B4="","",IF(ESP_Component(MAX(0,B12),B4,B11,B7)="C1 (early failure)","Ранний",IF(ESP_Component(MAX(0,B12),B4,B11,B7)="C2 (wear-out)","Износ",ESP_Component(MAX(0,B12),B4,B11,B7))))')
    set_formula(ws, "H15", '=IF(B4="","",ESP_SF(MAX(0,B12),B4,B11,B7))', fmt="0.000")
    set_formula(ws, "H16", f'=IF(OR(B4="",B12=""),"",ESP_FailProbX_Sens_Direct(MAX(0,B12),90,B4,B11,B7,{direct_args}))', fmt="0.0%")
    set_formula(ws, "H17", '=IF(B4="","",ESP_Uptime(B4,B11,B7))', fmt="0.000")

    # Notes / history
    ws.Range("G20:K22").Merge()
    set_cell(
        ws,
        "G20",
        "ННО_баз/ОР_баз = без учета инд. осложнений. ННО/ОР и вероятность отказа за 90 дней "
        "учитывают режим, историю и производные параметры.",
        fill=0xFFF2CC,
        wrap=True,
    )
    ws.Range("A24:E24").Merge()
    set_cell(
        ws,
        "A24",
        "Ввод по Принадлежности и Кислый/Некислый ожидается в тех же значениях, как в Свод. "
        "Пустой Кислый/Некислый трактуется как Некислый.",
        fill=0xFCE4D6,
        wrap=True,
    )
    set_cell(ws, "A38", "История по скважине из листа Свод", bold=True, fill=0xD9EAD3)
    ws.Range("A40:K40").Interior.Color = 0xD9EAD3

    place_button(ws, "btnCalcPrediction", "H3", "H3:J3", "Рассчитать прогноз", "CalculateSimplePrediction")

    style_block_borders(ws, ["A3:B13", "D3:E12", "A17:B22", "D17:E22", "A26:B32", "G3:H18", "A38:K40"])

    widths = {"A": 30, "B": 18, "C": 4, "D": 30, "E": 16, "F": 4, "G": 30, "H": 22, "I": 12, "J": 14, "K": 16}
    for col, width in widths.items():
        ws.Columns(col).ColumnWidth = width
    add_input_comments_single(ws)
    add_result_comments_single(ws)
    ws.Range("A40:K200").HorizontalAlignment = xl_left
    ws.Range("A4").Select()


def build_multi_sheet(ws):
    xl_center = -4108

    ws.Cells.Clear()
    ws.Name = SHEET_MULTI
    ws.Activate()
    ws.Range("A1:AW300").Font.Name = "Calibri"
    ws.Range("A1:AW300").Font.Size = 10

    # Section headers
    ws.Range("A1:J1").Merge()
    set_cell(ws, "A1", "Общее", bold=True, fill=0xE2F0D9)
    ws.Range("K1:S1").Merge()
    set_cell(ws, "K1", "Текущий режим работы", bold=True, fill=0xE2F0D9)
    ws.Range("T1:X1").Merge()
    set_cell(ws, "T1", "Химия / среда", bold=True, fill=0xE2F0D9)
    ws.Range("Y1:AC1").Merge()
    set_cell(ws, "Y1", "Компоновка", bold=True, fill=0xE2F0D9)
    ws.Range("AD1:AU1").Merge()
    set_cell(ws, "AD1", "Расчетные показатели", bold=True, fill=0xD9EAF7)
    set_cell(ws, "AW1", "Статус", bold=True, fill=0xF7F7F7)

    headers = [
        "Месторождение", "Скважина", "Куст", "Принадлежность", "Тип УЭЦН", "Дата монтажа",
        "Тип ствола", "Кислый/Некислый", "Наработка, оп.сут", "Номер спуска ручной",
        "Qж", "Qном", "Частота", "Загрузка, %", "Разброс загрузки", "Газовый фактор", "% >55 Гц",
        "Изменений частоты >1 Гц на 100 дней", "Доля дней kpod<0.7 ручн.", "H2S", "Ca", "Cl", "SO4", "КВЧ",
        "Кривизна °/10м", "Ном. частота", "Мощность кВт", "Габарит", "Глубина спуска",
        "Кпод_freq", "Доля дней kpod<0.7 (Свод)", "Кэкспл", "Прошлых установок (Свод)", "Номер спуска",
        "Дней после пред. отказа", "ННО_баз", "ОР_баз", "Множитель риска", "Флаги риска",
        "ННО", "ОР", "", "Категория отказа", "Режим отказа",
        "Вероятность дожития SF(t)", "Вероятность отказа за 90 дней", "Коэф. эксплуатации"
    ]
    for idx, header in enumerate(headers, start=1):
        set_cell(ws, ws.Cells(2, idx).Address, header, bold=True, fill=0xF3F3F3, wrap=True)

    start_row = 3
    end_row = 205
    for r in range(start_row, end_row + 1):
        set_formula(ws, f"AD{r}", f'=IF(OR(K{r}="",L{r}="",M{r}=""),"",IF(L{r}=0,"",K{r}/L{r}*(50/M{r})))', fmt="0.000")
        set_formula(ws, f"AE{r}", f'=IF(S{r}<>"",S{r},IF(AD{r}<>"",IF(AD{r}<0.7,1,0),""))', fmt="0.000")
        set_formula(ws, f"AF{r}", f'=IF(AND(F{r}<>"",I{r}<>"",TODAY()>F{r}),MIN(1,I{r}/(TODAY()-F{r})),"")', fmt="0.000")
        set_formula(ws, f"Q{r}", f'=IF(M{r}="","",IF(M{r}>55,100,0))', fmt="0.0")
        set_formula(ws, f"AH{r}", f'=IF(OR(A{r}="",B{r}=""),"",IF(J{r}<>"",J{r},IF(AG{r}<>"",AG{r}+1,"")))', fmt="0")
        direct = f'P{r},O{r},N{r},AE{r},AD{r},Q{r},R{r},Y{r},AH{r},AI{r},F{r}'
        set_formula(ws, f"AJ{r}", f'=IF(A{r}="","",ESP_B50(A{r},H{r},D{r}))', fmt="0.0")
        set_formula(ws, f"AK{r}", f'=IF(OR(A{r}="",I{r}=""),"",IF(I{r}>0,ESP_RUL(I{r},A{r},H{r},D{r}),""))', fmt="0.0")
        set_formula(ws, f"AL{r}", f'=IF(A{r}="","",ESP_HazardTheta_Direct({direct}))', fmt="0.000")
        set_formula(ws, f"AM{r}", f'=IF(A{r}="","",ESP_HazardFlags_RU_Direct({direct}))')
        set_formula(ws, f"AN{r}", f'=IF(A{r}="","",ESP_B50_Sens_Direct(A{r},H{r},D{r},{direct}))', fmt="0.0")
        set_formula(ws, f"AO{r}", f'=IF(OR(A{r}="",I{r}=""),"",IF(I{r}>0,ESP_RUL_Sens_Direct(I{r},A{r},H{r},D{r},{direct}),""))', fmt="0.0")
        set_formula(ws, f"AP{r}", '=""')
        set_formula(ws, f"AQ{r}", f'=IF(A{r}="","",IF(ESP_DominantMode(A{r},H{r},D{r},MAX(0,I{r}))="hydraulic","Гидравлический",IF(ESP_DominantMode(A{r},H{r},D{r},MAX(0,I{r}))="electro-thermal","Электротепловой",IF(ESP_DominantMode(A{r},H{r},D{r},MAX(0,I{r}))="protector","Гидрозащита",IF(ESP_DominantMode(A{r},H{r},D{r},MAX(0,I{r}))="other","Прочее",ESP_DominantMode(A{r},H{r},D{r},MAX(0,I{r})))))))')
        set_formula(ws, f"AR{r}", f'=IF(A{r}="","",IF(ESP_Component(MAX(0,I{r}),A{r},H{r},D{r})="C1 (early failure)","Ранний",IF(ESP_Component(MAX(0,I{r}),A{r},H{r},D{r})="C2 (wear-out)","Износ",ESP_Component(MAX(0,I{r}),A{r},H{r},D{r}))))')
        set_formula(ws, f"AS{r}", f'=IF(A{r}="","",ESP_SF(MAX(0,I{r}),A{r},H{r},D{r}))', fmt="0.000")
        set_formula(ws, f"AT{r}", f'=IF(OR(A{r}="",I{r}=""),"",ESP_FailProbX_Sens_Direct(MAX(0,I{r}),90,A{r},H{r},D{r},{direct}))', fmt="0.0%")
        set_formula(ws, f"AU{r}", f'=IF(A{r}="","",ESP_Uptime(A{r},H{r},D{r}))', fmt="0.000")

    place_button(ws, "btnCalcPredictionMulti", "AV1", "AV1:AZ1", "Рассчитать прогнозы", "CalculateMultiPrediction")

    widths = {
        "A": 16, "B": 16, "C": 14, "D": 18, "E": 20, "F": 12, "G": 14, "H": 14,
        "I": 12, "J": 10, "K": 10, "L": 10, "M": 10, "N": 10, "O": 10, "P": 12,
        "Q": 10, "R": 12, "S": 14, "T": 10, "U": 8, "V": 8, "W": 8, "X": 8,
        "Y": 10, "Z": 10, "AA": 10, "AB": 10, "AC": 12, "AD": 10, "AE": 14, "AF": 10,
        "AG": 12, "AH": 10, "AI": 12, "AJ": 12, "AK": 12, "AL": 12, "AM": 16, "AN": 12,
        "AO": 12, "AP": 12, "AQ": 14, "AR": 14, "AS": 18, "AT": 14, "AU": 12, "AV": 10, "AW": 10,
        "AX": 4, "AY": 4, "AZ": 4
    }
    for col, width in widths.items():
        ws.Columns(col).ColumnWidth = width

    ws.Range("A2:AW2").AutoFilter()
    add_input_comments_multi(ws)
    add_result_comments_multi(ws)
    window = ws.Application.ActiveWindow
    window.SplitColumn = 4
    window.SplitRow = 2
    window.FreezePanes = True
    ws.Range("A3").Select()


def configure_visibility(wb):
    xl_sheet_visible = -1
    xl_sheet_hidden = 0
    for ws in wb.Worksheets:
        ws.Visible = xl_sheet_visible if ws.Name in KEEP_VISIBLE else xl_sheet_hidden


def source_headers(ws):
    headers = {}
    last_col = ws.Cells(1, ws.Columns.Count).End(-4159).Column
    for c in range(1, last_col + 1):
        val = ws.Cells(1, c).Value
        if val not in (None, ""):
            headers[str(val).strip()] = c
    return headers


def source_row_dict(ws, row_idx: int, headers: dict[str, int]) -> dict[str, object]:
    def g(name: str):
        col = headers.get(name)
        return ws.Cells(row_idx, col).Value if col else None

    return {
        "row": row_idx,
        "field": g("Месторождение"),
        "pad": g("Куст"),
        "well": g("Скв."),
        "owner": g("Принадлежность"),
        "esp_type": g("Тип УЭЦН"),
        "mount_date": g("Дата монтажа"),
        "age_days": g("Наработка (сут)"),
        "well_type": g("Тип ствола скв"),
        "qliq": g("Дебит жидк."),
        "glf": g("Газовый фактор"),
        "freq": g("Частота"),
        "load_mean": g("Загр, Двиг,"),
        "h2s_mgdm3": g("Массовая доля сероводорода, мг/дм³"),
        "cl_mgl": g("Cl⁻, мг/л"),
        "so4_mgl": g("SO₄²⁻, мг/л"),
        "ca_mgl": g("Ca₂⁺, мг/л"),
        "kvch_mgdm3": g("Механические примеси (КВЧ), мг/дм³"),
        "gabarit": g("Габарит УЭЦН"),
        "curvature": g("Работа в кривизне"),
        "qnom": g("Ном. Произв. м₃/сут"),
        "nom_freq": g("Номинальная частота, Гц"),
        "power_kw": g("Мощность, кВт"),
        "depth_m": g("Глубина спуска УЭЦН, по НКТ"),
        "sour_class": g("Кислый/Некислый"),
        "failure_flag": g("Failure Flag"),
    }


def valid_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def clean_source_value(v):
    if v in (None, "", "-"):
        return ""
    return v


def same_date_key(v):
    if v in (None, ""):
        return None
    try:
        return (v.year, v.month, v.day)
    except Exception:
        return None


def pick_sample_rows(src_ws):
    headers = source_headers(src_ws)
    last_row = src_ws.Cells(src_ws.Rows.Count, 1).End(-4162).Row
    nonfail = None
    failed = None
    for row_idx in range(2, last_row + 1):
        row = source_row_dict(src_ws, row_idx, headers)
        ff = row["failure_flag"]
        qliq_ok = valid_number(row["qliq"])
        qnom_ok = valid_number(row["qnom"])
        freq_ok = valid_number(row["freq"])
        load_ok = valid_number(row["load_mean"])
        age_ok = valid_number(row["age_days"]) and row["age_days"] > 0
        if not age_ok:
            continue
        if ff in (0, 0.0, "0") and nonfail is None and qliq_ok and qnom_ok and freq_ok and load_ok:
            nonfail = row
        if ff in (1, 1.0, "1", True) and failed is None and qliq_ok and qnom_ok and freq_ok:
            failed = row
        if nonfail is not None and failed is not None:
            break
    return nonfail, failed


def history_stats(src_ws, sample: dict[str, object]):
    headers = source_headers(src_ws)
    last_row = src_ws.Cells(src_ws.Rows.Count, 1).End(-4162).Row
    field_key = str(sample["field"]).strip().lower()
    well_key = str(sample["well"]).strip().lower()
    current_mount = same_date_key(sample["mount_date"])
    n_all = 0
    n_prev = 0
    latest_prev_end = None
    stop_col = headers.get("Дата остановки")
    demount_col = headers.get("Дата демонтажа")

    for row_idx in range(2, last_row + 1):
        row_field = str(src_ws.Cells(row_idx, headers["Месторождение"]).Value).strip().lower()
        row_well = str(src_ws.Cells(row_idx, headers["Скв."]).Value).strip().lower()
        if row_field != field_key or row_well != well_key:
            continue
        n_all += 1
        row_mount = same_date_key(src_ws.Cells(row_idx, headers["Дата монтажа"]).Value)
        if current_mount is not None and row_mount is not None:
            if row_mount < current_mount:
                n_prev += 1
                stop_dt = src_ws.Cells(row_idx, stop_col).Value if stop_col else None
                demount_dt = src_ws.Cells(row_idx, demount_col).Value if demount_col else None
                best_end = demount_dt if same_date_key(demount_dt) and same_date_key(demount_dt) >= same_date_key(stop_dt) else stop_dt
                if same_date_key(best_end) is not None and (latest_prev_end is None or same_date_key(best_end) > same_date_key(latest_prev_end)):
                    latest_prev_end = best_end
        else:
            n_prev = n_all
    return n_all, n_prev, latest_prev_end


def fill_single_sample(ws, sample: dict[str, object]):
    ws.Range("B4").Value = clean_source_value(sample["field"])
    ws.Range("B5").Value = clean_source_value(sample["well"])
    ws.Range("B6").Value = clean_source_value(sample["pad"])
    ws.Range("B7").Value = clean_source_value(sample["owner"])
    ws.Range("B8").Value = clean_source_value(sample["esp_type"])
    ws.Range("B9").Value = excel_date_serial(sample["mount_date"])
    ws.Range("B10").Value = clean_source_value(sample["well_type"])
    ws.Range("B11").Value = clean_source_value(sample["sour_class"]) or "Некислый"
    ws.Range("B12").Value = clean_source_value(sample["age_days"])
    ws.Range("B13").Value = ""

    ws.Range("E4").Value = clean_source_value(sample["qliq"])
    ws.Range("E5").Value = clean_source_value(sample["qnom"])
    ws.Range("E6").Value = clean_source_value(sample["freq"])
    ws.Range("E7").Value = clean_source_value(sample["load_mean"])
    ws.Range("E8").Value = ""
    ws.Range("E9").Value = clean_source_value(sample["glf"])
    ws.Range("E10").Value = ""
    ws.Range("E11").Value = ""
    ws.Range("E12").Value = ""

    ws.Range("B18").Value = clean_source_value(sample["h2s_mgdm3"])
    ws.Range("B19").Value = clean_source_value(sample["ca_mgl"])
    ws.Range("B20").Value = clean_source_value(sample["cl_mgl"])
    ws.Range("B21").Value = clean_source_value(sample["so4_mgl"])
    ws.Range("B22").Value = clean_source_value(sample["kvch_mgdm3"])

    ws.Range("E18").Value = clean_source_value(sample["curvature"])
    ws.Range("E19").Value = clean_source_value(sample["nom_freq"])
    ws.Range("E20").Value = clean_source_value(sample["power_kw"])
    ws.Range("E21").Value = clean_source_value(sample["gabarit"])
    ws.Range("E22").Value = clean_source_value(sample["depth_m"])


def fill_multi_sample_row(ws, row_idx: int, sample: dict[str, object], *, age_override=None):
    ws.Cells(row_idx, 1).Value = clean_source_value(sample["field"])
    ws.Cells(row_idx, 2).Value = clean_source_value(sample["well"])
    ws.Cells(row_idx, 3).Value = clean_source_value(sample["pad"])
    ws.Cells(row_idx, 4).Value = clean_source_value(sample["owner"])
    ws.Cells(row_idx, 5).Value = clean_source_value(sample["esp_type"])
    ws.Cells(row_idx, 6).Value = excel_date_serial(sample["mount_date"])
    ws.Cells(row_idx, 7).Value = clean_source_value(sample["well_type"])
    ws.Cells(row_idx, 8).Value = clean_source_value(sample["sour_class"]) or "Некислый"
    ws.Cells(row_idx, 9).Value = clean_source_value(sample["age_days"] if age_override is None else age_override)
    ws.Cells(row_idx, 10).Value = ""
    ws.Cells(row_idx, 11).Value = clean_source_value(sample["qliq"])
    ws.Cells(row_idx, 12).Value = clean_source_value(sample["qnom"])
    ws.Cells(row_idx, 13).Value = clean_source_value(sample["freq"])
    ws.Cells(row_idx, 14).Value = clean_source_value(sample["load_mean"])
    ws.Cells(row_idx, 15).Value = ""
    ws.Cells(row_idx, 16).Value = clean_source_value(sample["glf"])
    ws.Cells(row_idx, 17).Value = ""
    ws.Cells(row_idx, 18).Value = ""
    ws.Cells(row_idx, 19).Value = ""
    ws.Cells(row_idx, 20).Value = clean_source_value(sample["h2s_mgdm3"])
    ws.Cells(row_idx, 21).Value = clean_source_value(sample["ca_mgl"])
    ws.Cells(row_idx, 22).Value = clean_source_value(sample["cl_mgl"])
    ws.Cells(row_idx, 23).Value = clean_source_value(sample["so4_mgl"])
    ws.Cells(row_idx, 24).Value = clean_source_value(sample["kvch_mgdm3"])
    ws.Cells(row_idx, 25).Value = clean_source_value(sample["curvature"])
    ws.Cells(row_idx, 26).Value = clean_source_value(sample["nom_freq"])
    ws.Cells(row_idx, 27).Value = clean_source_value(sample["power_kw"])
    ws.Cells(row_idx, 28).Value = clean_source_value(sample["gabarit"])
    ws.Cells(row_idx, 29).Value = clean_source_value(sample["depth_m"])


def populate_sample_inputs(wb):
    try:
        src_ws = wb.Worksheets("Свод")
        ws_single = wb.Worksheets(SHEET_SINGLE)
        ws_multi = wb.Worksheets(SHEET_MULTI)
    except Exception:
        return

    nonfail, failed = pick_sample_rows(src_ws)
    if nonfail is None:
        return

    fill_single_sample(ws_single, nonfail)
    n_all, n_prev, latest_prev_end = history_stats(src_ws, nonfail)
    ws_single.Range("B30").Value = n_prev
    if latest_prev_end not in (None, "") and same_date_key(nonfail["mount_date"]) is not None:
        ws_single.Range("B32").Value = excel_date_serial(nonfail["mount_date"]) - excel_date_serial(latest_prev_end)
    else:
        ws_single.Range("B32").Value = ""
    ws_single.Range("H18").Value = f"Пример загружен из Свод: строка {nonfail['row']}; историй={n_all}; прошлых установок={n_prev}"
    ws_single.Range("A36").Value = f"Пример: действующая неотказная установка из Свод (строка {nonfail['row']})"

    ws_multi.Range("A3:AC6").ClearContents
    ws_multi.Range("A208:A209").ClearContents
    fill_multi_sample_row(ws_multi, 3, nonfail)
    ws_multi.Range("A208").Value = f"Строка 3: неотказная установка из Свод (строка {nonfail['row']})"
    ws_multi.Cells(3, 33).Value = n_prev
    if latest_prev_end not in (None, "") and same_date_key(nonfail["mount_date"]) is not None:
        ws_multi.Cells(3, 35).Value = excel_date_serial(nonfail["mount_date"]) - excel_date_serial(latest_prev_end)
    else:
        ws_multi.Cells(3, 35).Value = ""

    if failed is not None:
        _, n_prev_failed, latest_prev_end_failed = history_stats(src_ws, failed)
        fill_multi_sample_row(ws_multi, 4, failed, age_override=0)
        ws_multi.Cells(4, 33).Value = n_prev_failed
        if latest_prev_end_failed not in (None, "") and same_date_key(failed["mount_date"]) is not None:
            ws_multi.Cells(4, 35).Value = excel_date_serial(failed["mount_date"]) - excel_date_serial(latest_prev_end_failed)
        else:
            ws_multi.Cells(4, 35).Value = ""
        ws_multi.Range("A209").Value = f"Строка 4: отказавшая установка из Свод (строка {failed['row']}), но Наработка=0 для примера TTF"
    ws_multi.Range("AW1").Value = "Примеры загружены из Свод"


def main() -> int:
    args = parse_args()
    require_windows()
    pythoncom, win32 = imports()
    workbook_path = args.workbook.expanduser().resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(workbook_path)

    pythoncom.CoInitialize()
    app = None
    wb = None
    try:
        app = win32.DispatchEx("Excel.Application")
        app.Visible = False
        app.DisplayAlerts = False
        app.EnableEvents = False
        wb = app.Workbooks.Open(str(workbook_path), ReadOnly=False, IgnoreReadOnlyRecommended=True)
        try:
            app.Calculation = -4135
            app.CalculateBeforeSave = False
        except Exception:
            pass
        try:
            ws_single = wb.Worksheets(SHEET_SINGLE)
        except Exception:
            ws_single = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
            ws_single.Name = SHEET_SINGLE
        build_single_sheet(ws_single)

        try:
            ws_multi = wb.Worksheets(SHEET_MULTI)
        except Exception:
            ws_multi = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
            ws_multi.Name = SHEET_MULTI
        build_multi_sheet(ws_multi)

        configure_visibility(wb)
        populate_sample_inputs(wb)
        wb.Worksheets(SHEET_SINGLE).Activate()
        wb.Save()
        print(f"Built simplified prediction sheets in: {workbook_path}")
        return 0
    finally:
        if wb is not None:
            try:
                wb.Close(SaveChanges=True)
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    raise SystemExit(main())
