<template>
  <section class="dashboard-grid">
    <section class="panel overview-panel">
      <div class="panel-header">
        <h2>Сводка</h2>
        <p>Ключевые метрики по таблице и выбранной целевой переменной.</p>
      </div>

      <div class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Целевая переменная</span>
          <strong>{{ analysis.dataset.target_column || "Не указана" }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Строк</span>
          <strong>{{ analysis.overview.total_rows }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Колонок в анализе</span>
          <strong>{{ analysis.overview.numeric_column_count + analysis.overview.categorical_column_count + analysis.overview.datetime_column_count }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Валидный target</span>
          <strong>{{ analysis.overview.valid_target_rows }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Среднее target</span>
          <strong>{{ formatNumber(analysis.overview.target_mean) }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Пропуски</span>
          <strong>{{ formatPercent(analysis.overview.missing_values_share) }}</strong>
        </div>
      </div>

      <div v-if="analysis.notes.length" class="notes-list">
        <span v-for="note in analysis.notes" :key="note" class="note-chip">{{ note }}</span>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>CatBoost Feature Importance</h2>
        <p>Важность зависимых переменных относительно target.</p>
      </div>
      <PlotlyChart :data="featureImportanceData" :layout="featureImportanceLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Корреляционная матрица</h2>
        <p>Корреляция признаков с целевой переменной.</p>
      </div>
      <PlotlyChart :data="correlationMatrixData" :layout="correlationMatrixLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Сводная таблица</h2>
        <p>Среднее по целевой переменной в разрезе участков недр по годам.</p>
      </div>

      <div v-if="pivotTable.headers.length" class="table-wrap compact-table">
        <table class="data-table">
          <thead>
            <tr>
              <th>Участок недр</th>
              <th v-for="year in pivotTable.headers" :key="year">{{ year }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in pivotTable.rows" :key="row.label">
              <td>{{ row.label }}</td>
              <td v-for="year in pivotTable.headers" :key="`${row.label}-${year}`">
                {{ row.values[year] ?? "—" }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="empty-state">
        Не удалось автоматически определить колонки участка недр и даты для построения сводной таблицы.
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Целевая переменная во времени</h2>
        <p>Выбирайте любую переменную из списка временных колонок. Даты на оси X форматируются автоматически.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Колонка по оси X</span>
          <div class="button-group">
            <button
              v-for="column in availableTimeColumns"
              :key="column"
              type="button"
              class="choice-button"
              :class="{ active: column === selectedTimeColumn }"
              @click="selectedTimeColumn = column"
            >
              {{ column }}
            </button>
          </div>
        </div>

        <div class="control-group">
          <span class="control-title">Группировка</span>
          <div class="button-group">
            <button
              type="button"
              class="choice-button"
              :class="{ active: !selectedGroupColumn }"
              @click="selectedGroupColumn = ''"
            >
              Без группировки
            </button>
            <button
              v-for="column in groupableColumns"
              :key="column"
              type="button"
              class="choice-button"
              :class="{ active: column === selectedGroupColumn }"
              @click="selectedGroupColumn = column"
            >
              {{ column }}
            </button>
          </div>
        </div>

        <div v-if="selectedGroupColumn" class="control-group">
          <span class="control-title">Срезы</span>
          <div class="button-group">
            <button
              v-for="value in groupValues"
              :key="value"
              type="button"
              class="choice-button"
              :class="{ active: selectedGroupValues.includes(value) }"
              @click="toggleGroupValue(value)"
            >
              {{ value }}
            </button>
          </div>
        </div>
      </div>

      <PlotlyChart :data="timeSeriesData" :layout="timeSeriesLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Scatter Plot</h2>
        <p>Выбирайте любые числовые или календарные переменные кнопками.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Ось X</span>
          <div class="button-group">
            <button
              v-for="column in scatterColumns"
              :key="`scatter-x-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: column === selectedScatterX }"
              @click="selectedScatterX = column"
            >
              {{ column }}
            </button>
          </div>
        </div>

        <div class="control-group">
          <span class="control-title">Ось Y</span>
          <div class="button-group">
            <button
              v-for="column in scatterColumns"
              :key="`scatter-y-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: column === selectedScatterY }"
              @click="selectedScatterY = column"
            >
              {{ column }}
            </button>
          </div>
        </div>
      </div>

      <PlotlyChart :data="scatterData" :layout="scatterLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Гистограммы</h2>
        <p>Для колонок-дат значения группируются по времени, для числовых строится histogram.</p>
      </div>

      <div class="control-group">
        <span class="control-title">Переменная</span>
        <div class="button-group">
          <button
            v-for="column in allColumns"
            :key="`hist-${column}`"
            type="button"
            class="choice-button"
            :class="{ active: column === selectedHistogramColumn }"
            @click="selectedHistogramColumn = column"
          >
            {{ column }}
          </button>
        </div>
      </div>

      <PlotlyChart :data="histogramData" :layout="histogramLayout" />
    </section>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import PlotlyChart from "./PlotlyChart.vue";

const props = defineProps({
  analysis: {
    type: Object,
    required: true,
  },
  rows: {
    type: Array,
    default: () => [],
  },
  columns: {
    type: Array,
    default: () => [],
  },
});

const viewportWidth = ref(typeof window === "undefined" ? 1440 : window.innerWidth);

function handleResize() {
  viewportWidth.value = window.innerWidth;
}

onMounted(() => {
  window.addEventListener("resize", handleResize);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
});

const chartHeight = computed(() => {
  if (viewportWidth.value < 640) {
    return 300;
  }
  if (viewportWidth.value < 1024) {
    return 360;
  }
  return 430;
});

const compactLeftMargin = computed(() => (viewportWidth.value < 720 ? 80 : 180));

const datetimeColumnSet = computed(() => new Set(props.analysis.datetime_columns || []));

function excelSerialToDate(value) {
  const epoch = new Date(Date.UTC(1899, 11, 30));
  epoch.setUTCDate(epoch.getUTCDate() + Number(value));
  return epoch;
}

function toNumeric(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  const parsed = Number(String(value).trim().replace(",", "."));
  return Number.isFinite(parsed) ? parsed : null;
}

function toDate(value, column = "") {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = toNumeric(value);
  if (numeric !== null && datetimeColumnSet.value.has(column)) {
    return excelSerialToDate(numeric);
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatDateLabel(date) {
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function formatMonthLabel(date) {
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric",
    month: "short",
  }).format(date);
}

function normalizeCategory(value) {
  return value === null || value === undefined || value === "" ? "Пусто" : String(value);
}

const inferredNumericColumns = computed(() =>
  props.columns.filter((column) => {
    const valid = props.rows.map((row) => toNumeric(row[column])).filter((value) => value !== null);
    return valid.length >= 3 && valid.length / Math.max(props.rows.length, 1) >= 0.5;
  }),
);

const inferredDateColumns = computed(() =>
  props.columns.filter((column) => {
    const valid = props.rows.map((row) => toDate(row[column], column)).filter(Boolean);
    return valid.length >= 3 && valid.length / Math.max(props.rows.length, 1) >= 0.4;
  }),
);

const availableTimeColumns = computed(() => {
  const merged = new Set([...(props.analysis.datetime_columns || []), ...inferredDateColumns.value]);
  return Array.from(merged);
});

const scatterColumns = computed(() => {
  const merged = new Set([
    ...(props.analysis.numeric_columns || []),
    ...(props.analysis.datetime_columns || []),
    ...inferredNumericColumns.value,
  ]);
  return Array.from(merged);
});

const allColumns = computed(() => props.columns);

const selectedTimeColumn = ref("");
const selectedGroupColumn = ref("");
const selectedGroupValues = ref([]);
const selectedScatterX = ref("");
const selectedScatterY = ref("");
const selectedHistogramColumn = ref("");

watch(
  availableTimeColumns,
  (columns) => {
    if (!columns.includes(selectedTimeColumn.value)) {
      selectedTimeColumn.value = columns[0] || "";
    }
  },
  { immediate: true },
);

watch(
  scatterColumns,
  (columns) => {
    if (!columns.includes(selectedScatterX.value)) {
      selectedScatterX.value = columns[0] || "";
    }
    if (!columns.includes(selectedScatterY.value)) {
      selectedScatterY.value = columns[1] || columns[0] || "";
    }
  },
  { immediate: true },
);

watch(
  allColumns,
  (columns) => {
    if (!columns.includes(selectedHistogramColumn.value)) {
      selectedHistogramColumn.value = columns[0] || "";
    }
  },
  { immediate: true },
);

watch(selectedGroupColumn, () => {
  selectedGroupValues.value = [];
});

const groupableColumns = computed(() =>
  props.columns.filter(
    (column) => column !== props.analysis.dataset.target_column && column !== selectedTimeColumn.value,
  ),
);

const groupValues = computed(() => {
  if (!selectedGroupColumn.value) {
    return [];
  }
  const values = new Set();
  for (const row of props.rows) {
    values.add(normalizeCategory(row[selectedGroupColumn.value]));
  }
  return Array.from(values).slice(0, 30);
});

function toggleGroupValue(value) {
  if (selectedGroupValues.value.includes(value)) {
    selectedGroupValues.value = selectedGroupValues.value.filter((item) => item !== value);
  } else {
    selectedGroupValues.value = [...selectedGroupValues.value, value];
  }
}

const featureImportanceData = computed(() => [
  {
    type: "bar",
    orientation: "h",
    x: [...(props.analysis.feature_importance || [])].map((item) => item.importance).reverse(),
    y: [...(props.analysis.feature_importance || [])].map((item) => item.feature).reverse(),
    marker: { color: "#0f766e" },
  },
]);

const featureImportanceLayout = computed(() => ({
  margin: { l: compactLeftMargin.value, r: 20, t: 10, b: 40 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: "Importance" },
}));

const correlationMatrixData = computed(() => [
  {
    type: "heatmap",
    x: (props.analysis.correlations || []).map((item) => item.feature),
    y: [props.analysis.dataset.target_column || "Target"],
    z: [(props.analysis.correlations || []).map((item) => item.correlation)],
    zmid: 0,
    colorscale: [
      [0, "#7f1d1d"],
      [0.5, "#f8fafc"],
      [1, "#0f766e"],
    ],
    hovertemplate: "Признак: %{x}<br>Корреляция: %{z:.3f}<extra></extra>",
  },
]);

const correlationMatrixLayout = computed(() => ({
  margin: { l: 70, r: 20, t: 10, b: viewportWidth.value < 720 ? 150 : 120 },
  height: Math.max(260, chartHeight.value - 60),
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
}));

const timeSeriesData = computed(() => {
  const targetColumn = props.analysis.dataset.target_column;
  if (!targetColumn || !selectedTimeColumn.value) {
    return [];
  }

  const aggregateMap = new Map();
  for (const row of props.rows) {
    const timeValue = toDate(row[selectedTimeColumn.value], selectedTimeColumn.value);
    const targetValue = toNumeric(row[targetColumn]);
    if (!timeValue || targetValue === null) {
      continue;
    }

    const groupValue = selectedGroupColumn.value ? normalizeCategory(row[selectedGroupColumn.value]) : "Все";
    if (
      selectedGroupColumn.value &&
      selectedGroupValues.value.length > 0 &&
      !selectedGroupValues.value.includes(groupValue)
    ) {
      continue;
    }

    const bucketDate = new Date(Date.UTC(timeValue.getUTCFullYear(), timeValue.getUTCMonth(), 1));
    const key = `${groupValue}__${bucketDate.toISOString()}`;
    const current = aggregateMap.get(key) || { group: groupValue, date: bucketDate, values: [] };
    current.values.push(targetValue);
    aggregateMap.set(key, current);
  }

  const byGroup = new Map();
  for (const item of aggregateMap.values()) {
    const line = byGroup.get(item.group) || [];
    line.push({
      x: item.date,
      y: item.values.reduce((sum, value) => sum + value, 0) / item.values.length,
    });
    byGroup.set(item.group, line);
  }

  return Array.from(byGroup.entries()).map(([group, points]) => {
    const sortedPoints = points.sort((a, b) => a.x - b.x);
    return {
      type: "scatter",
      mode: "lines+markers",
      name: group,
      x: sortedPoints.map((point) => point.x.toISOString()),
      y: sortedPoints.map((point) => point.y),
      hovertemplate: "%{x|%d.%m.%Y}<br>Среднее: %{y:.2f}<extra>" + group + "</extra>",
    };
  });
});

const timeSeriesLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 60 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: selectedTimeColumn.value || "Дата",
    type: "date",
    tickformat: "%d.%m.%Y",
  },
  yaxis: { title: props.analysis.dataset.target_column || "Target" },
}));

function valueForScatter(row, column) {
  if (datetimeColumnSet.value.has(column)) {
    const date = toDate(row[column], column);
    return date ? date.getTime() : null;
  }
  return toNumeric(row[column]);
}

const scatterData = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) {
    return [];
  }

  const points = props.rows
    .map((row) => ({
      x: valueForScatter(row, selectedScatterX.value),
      y: valueForScatter(row, selectedScatterY.value),
      xLabel: datetimeColumnSet.value.has(selectedScatterX.value)
        ? formatDateLabel(toDate(row[selectedScatterX.value], selectedScatterX.value))
        : null,
      yLabel: datetimeColumnSet.value.has(selectedScatterY.value)
        ? formatDateLabel(toDate(row[selectedScatterY.value], selectedScatterY.value))
        : null,
    }))
    .filter((point) => point.x !== null && point.y !== null);

  return [
    {
      type: "scatter",
      mode: "markers",
      x: points.map((point) => point.x),
      y: points.map((point) => point.y),
      text: points.map((point) => {
        const parts = [];
        if (point.xLabel) {
          parts.push(`${selectedScatterX.value}: ${point.xLabel}`);
        }
        if (point.yLabel) {
          parts.push(`${selectedScatterY.value}: ${point.yLabel}`);
        }
        return parts.join("<br>");
      }),
      hovertemplate: "%{text}<br>X=%{x}<br>Y=%{y}<extra></extra>",
      marker: {
        color: "#ea580c",
        size: viewportWidth.value < 720 ? 7 : 8,
        opacity: 0.72,
      },
    },
  ];
});

const scatterLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 50 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: selectedScatterX.value || "X",
    type: datetimeColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
  },
  yaxis: {
    title: selectedScatterY.value || "Y",
    type: datetimeColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
  },
}));

const histogramData = computed(() => {
  if (!selectedHistogramColumn.value) {
    return [];
  }

  if (datetimeColumnSet.value.has(selectedHistogramColumn.value)) {
    const counts = new Map();
    for (const row of props.rows) {
      const date = toDate(row[selectedHistogramColumn.value], selectedHistogramColumn.value);
      if (!date) {
        continue;
      }
      const label = formatMonthLabel(new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), 1)));
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    const entries = Array.from(counts.entries());
    return [
      {
        type: "bar",
        x: entries.map((item) => item[0]),
        y: entries.map((item) => item[1]),
        marker: { color: "#1d4ed8" },
      },
    ];
  }

  const numericValues = props.rows
    .map((row) => toNumeric(row[selectedHistogramColumn.value]))
    .filter((value) => value !== null);

  if (numericValues.length >= Math.max(5, props.rows.length * 0.5)) {
    return [
      {
        type: "histogram",
        x: numericValues,
        marker: { color: "#1d4ed8" },
      },
    ];
  }

  const counts = new Map();
  for (const row of props.rows) {
    const key = normalizeCategory(row[selectedHistogramColumn.value]);
    counts.set(key, (counts.get(key) || 0) + 1);
  }

  const topValues = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 30);

  return [
    {
      type: "bar",
      x: topValues.map((item) => item[0]),
      y: topValues.map((item) => item[1]),
      marker: { color: "#9333ea" },
    },
  ];
});

const histogramLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: viewportWidth.value < 720 ? 120 : 100 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: selectedHistogramColumn.value || "Column" },
  yaxis: { title: "Count" },
}));

function findSubsoilColumn() {
  return props.columns.find((column) => {
    const normalized = column.toLowerCase();
    return normalized.includes("участ") || normalized.includes("недр");
  }) || "";
}

const pivotTable = computed(() => {
  const targetColumn = props.analysis.dataset.target_column;
  const dateColumn = selectedTimeColumn.value || availableTimeColumns.value[0] || "";
  const subsoilColumn = findSubsoilColumn();

  if (!targetColumn || !dateColumn || !subsoilColumn) {
    return { headers: [], rows: [] };
  }

  const grouped = new Map();
  const years = new Set();

  for (const row of props.rows) {
    const date = toDate(row[dateColumn], dateColumn);
    const targetValue = toNumeric(row[targetColumn]);
    if (!date || targetValue === null) {
      continue;
    }
    const year = String(date.getUTCFullYear());
    const area = normalizeCategory(row[subsoilColumn]);
    years.add(year);

    const key = `${area}__${year}`;
    const current = grouped.get(key) || { label: area, year, values: [] };
    current.values.push(targetValue);
    grouped.set(key, current);
  }

  const headers = Array.from(years).sort();
  const byArea = new Map();
  for (const item of grouped.values()) {
    const areaRow = byArea.get(item.label) || {};
    const avg = item.values.reduce((sum, value) => sum + value, 0) / item.values.length;
    areaRow[item.year] = formatNumber(avg);
    byArea.set(item.label, areaRow);
  }

  return {
    headers,
    rows: Array.from(byArea.entries())
      .map(([label, values]) => ({ label, values }))
      .sort((a, b) => a.label.localeCompare(b.label)),
  };
});
</script>
