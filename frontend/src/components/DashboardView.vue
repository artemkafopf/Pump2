<template>
  <section class="dashboard-grid">
    <section class="panel overview-panel">
      <div class="panel-header">
        <h2>Сводка</h2>
        <p>Ключевые метрики по таблице и целевой переменной.</p>
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
          <span class="metric-label">Колонок</span>
          <strong>{{ analysis.overview.total_columns }}</strong>
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
        <p>Какие признаки сильнее всего связаны с целевой переменной в модели.</p>
      </div>
      <PlotlyChart :data="featureImportanceData" :layout="featureImportanceLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Корреляционная матрица</h2>
        <p>Значения корреляции признаков с target.</p>
      </div>
      <PlotlyChart :data="correlationMatrixData" :layout="correlationMatrixLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Целевая переменная во времени</h2>
        <p>Можно выбрать временную колонку и добавить срезы по другим переменным.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Временная колонка</span>
          <div class="button-group">
            <button
              v-for="column in availableTimeColumns"
              :key="column"
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
              class="choice-button"
              :class="{ active: !selectedGroupColumn }"
              @click="selectedGroupColumn = ''"
            >
              Без группировки
            </button>
            <button
              v-for="column in groupableColumns"
              :key="column"
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
        <p>Любые числовые переменные можно выбрать кнопками для осей X и Y.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Ось X</span>
          <div class="button-group">
            <button
              v-for="column in scatterColumns"
              :key="`scatter-x-${column}`"
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
        <p>Для числовых колонок строится histogram, для категориальных считается частота значений.</p>
      </div>

      <div class="control-group">
        <span class="control-title">Переменная</span>
        <div class="button-group">
          <button
            v-for="column in allColumns"
            :key="`hist-${column}`"
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
import { computed, ref, watch } from "vue";
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

function toDate(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
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

const inferredNumericColumns = computed(() =>
  props.columns.filter((column) => {
    const valid = props.rows.map((row) => toNumeric(row[column])).filter((value) => value !== null);
    return valid.length >= 3 && valid.length / Math.max(props.rows.length, 1) >= 0.5;
  }),
);

const inferredDateColumns = computed(() =>
  props.columns.filter((column) => {
    const valid = props.rows.map((row) => toDate(row[column])).filter(Boolean);
    return valid.length >= 3 && valid.length / Math.max(props.rows.length, 1) >= 0.5;
  }),
);

const availableTimeColumns = computed(() => {
  const merged = new Set([...(props.analysis.datetime_columns || []), ...inferredDateColumns.value]);
  return Array.from(merged);
});

const groupableColumns = computed(() =>
  props.columns.filter(
    (column) => column !== props.analysis.dataset.target_column && column !== selectedTimeColumn.value,
  ),
);

const scatterColumns = computed(() => {
  const merged = new Set([...(props.analysis.numeric_columns || []), ...inferredNumericColumns.value]);
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

const groupValues = computed(() => {
  if (!selectedGroupColumn.value) {
    return [];
  }
  const values = new Set();
  for (const row of props.rows) {
    const raw = row[selectedGroupColumn.value];
    if (raw !== null && raw !== undefined && raw !== "") {
      values.add(String(raw));
    }
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
  margin: { l: 180, r: 20, t: 10, b: 40 },
  height: 420,
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
  margin: { l: 80, r: 20, t: 10, b: 120 },
  height: 320,
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
    const timeValue = toDate(row[selectedTimeColumn.value]);
    const targetValue = toNumeric(row[targetColumn]);
    if (!timeValue || targetValue === null) {
      continue;
    }

    const groupValue = selectedGroupColumn.value ? String(row[selectedGroupColumn.value] ?? "Пусто") : "Все";
    if (
      selectedGroupColumn.value &&
      selectedGroupValues.value.length > 0 &&
      !selectedGroupValues.value.includes(groupValue)
    ) {
      continue;
    }

    const dateKey = timeValue.toISOString().slice(0, 10);
    const key = `${groupValue}__${dateKey}`;
    const current = aggregateMap.get(key) || { group: groupValue, date: dateKey, values: [] };
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
    const sortedPoints = points.sort((a, b) => a.x.localeCompare(b.x));
    return {
      type: "scatter",
      mode: "lines+markers",
      name: group,
      x: sortedPoints.map((point) => point.x),
      y: sortedPoints.map((point) => point.y),
    };
  });
});

const timeSeriesLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 50 },
  height: 420,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: selectedTimeColumn.value || "Time" },
  yaxis: { title: props.analysis.dataset.target_column || "Target" },
}));

const scatterData = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) {
    return [];
  }

  const points = props.rows
    .map((row) => ({
      x: toNumeric(row[selectedScatterX.value]),
      y: toNumeric(row[selectedScatterY.value]),
    }))
    .filter((point) => point.x !== null && point.y !== null);

  return [
    {
      type: "scatter",
      mode: "markers",
      x: points.map((point) => point.x),
      y: points.map((point) => point.y),
      marker: {
        color: "#ea580c",
        size: 8,
        opacity: 0.72,
      },
    },
  ];
});

const scatterLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 50 },
  height: 420,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: selectedScatterX.value || "X" },
  yaxis: { title: selectedScatterY.value || "Y" },
}));

const histogramData = computed(() => {
  if (!selectedHistogramColumn.value) {
    return [];
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
    const value = row[selectedHistogramColumn.value];
    const key = value === null || value === undefined || value === "" ? "Пусто" : String(value);
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
  margin: { l: 60, r: 20, t: 10, b: 100 },
  height: 420,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: selectedHistogramColumn.value || "Column" },
  yaxis: { title: "Count" },
}));
</script>
