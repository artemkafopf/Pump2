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
                {{ row.values[year] ?? "-" }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="empty-state">
        Не удалось автоматически определить колонки участка недр и даты для сводной таблицы.
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Целевая переменная во времени</h2>
        <p>Выбирайте любую временную колонку. Усреднение по Y считается без нулей и пустых значений.</p>
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
            <button type="button" class="choice-button" :class="{ active: !selectedGroupColumn }" @click="selectedGroupColumn = ''">
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
        <p>Настраивайте оси, диапазоны и смотрите точки вместе с осреднением по окнам X с шагом 10%.</p>
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

        <div v-if="scatterRangeReady" class="range-grid">
          <label class="field">
            <span>Минимум X: {{ formatAxisValue(selectedScatterX, scatterXMinValue) }}</span>
            <input v-model.number="scatterXMinPercent" type="range" min="0" max="100" step="1" />
          </label>
          <label class="field">
            <span>Максимум X: {{ formatAxisValue(selectedScatterX, scatterXMaxValue) }}</span>
            <input v-model.number="scatterXMaxPercent" type="range" min="0" max="100" step="1" />
          </label>
          <label class="field">
            <span>Минимум Y: {{ formatAxisValue(selectedScatterY, scatterYMinValue) }}</span>
            <input v-model.number="scatterYMinPercent" type="range" min="0" max="100" step="1" />
          </label>
          <label class="field">
            <span>Максимум Y: {{ formatAxisValue(selectedScatterY, scatterYMaxValue) }}</span>
            <input v-model.number="scatterYMaxPercent" type="range" min="0" max="100" step="1" />
          </label>
        </div>
      </div>

      <PlotlyChart :data="scatterData" :layout="scatterLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Таблица осреднения Scatter</h2>
        <p>Окна по X с шагом 10%, среднее по Y без нулей и пустых значений.</p>
      </div>

      <div v-if="averagedScatterTable.length" class="table-wrap compact-table">
        <table class="data-table">
          <thead>
            <tr>
              <th>Окно X</th>
              <th>Средний X</th>
              <th>Средний Y</th>
              <th>Точек</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in averagedScatterTable" :key="row.windowLabel">
              <td>{{ row.windowLabel }}</td>
              <td>{{ row.avgX }}</td>
              <td>{{ row.avgY }}</td>
              <td>{{ row.count }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="empty-state">
        Недостаточно данных для расчета осреднения в окнах по X.
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Тепловая карта</h2>
        <p>Плотность или среднее значение выбранной переменной по выбранным осям scatter plot.</p>
      </div>

      <div class="control-group">
        <span class="control-title">Раскраска</span>
        <div class="button-group">
          <button
            v-for="option in heatmapColorOptions"
            :key="option.value"
            type="button"
            class="choice-button"
            :class="{ active: option.value === selectedHeatmapColorBy }"
            @click="selectedHeatmapColorBy = option.value"
          >
            {{ option.label }}
          </button>
        </div>
      </div>

      <PlotlyChart :data="scatterHeatmapData" :layout="scatterHeatmapLayout" />
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
  analysis: { type: Object, required: true },
  rows: { type: Array, default: () => [] },
  columns: { type: Array, default: () => [] },
});

const viewportWidth = ref(typeof window === "undefined" ? 1440 : window.innerWidth);
const selectedTimeColumn = ref("");
const selectedGroupColumn = ref("");
const selectedGroupValues = ref([]);
const selectedScatterX = ref("");
const selectedScatterY = ref("");
const selectedHistogramColumn = ref("");
const selectedHeatmapColorBy = ref("__count__");
const scatterXMinPercent = ref(0);
const scatterXMaxPercent = ref(100);
const scatterYMinPercent = ref(0);
const scatterYMaxPercent = ref(100);

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
  if (viewportWidth.value < 640) return 300;
  if (viewportWidth.value < 1024) return 360;
  return 430;
});

const compactLeftMargin = computed(() => (viewportWidth.value < 720 ? 80 : 180));
const datetimeColumnSet = computed(() => new Set(props.analysis.datetime_columns || []));

function excelSerialToDate(value) {
  const epoch = new Date(Date.UTC(1899, 11, 30));
  epoch.setUTCDate(epoch.getUTCDate() + Number(value));
  return epoch;
}

function parseDateFromString(value) {
  const trimmed = String(value).trim();
  const iso = new Date(trimmed);
  if (!Number.isNaN(iso.getTime())) return iso;

  const ruMatch = trimmed.match(/^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$/);
  if (ruMatch) {
    const day = Number(ruMatch[1]);
    const month = Number(ruMatch[2]) - 1;
    const yearRaw = Number(ruMatch[3]);
    const year = yearRaw < 100 ? 2000 + yearRaw : yearRaw;
    const hour = Number(ruMatch[4] || 0);
    const minute = Number(ruMatch[5] || 0);
    const second = Number(ruMatch[6] || 0);
    const parsed = new Date(Date.UTC(year, month, day, hour, minute, second));
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  return null;
}

function toNumeric(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const normalized = String(value).trim().replace(/\s/g, "").replace(",", ".");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function toDate(value, column = "") {
  if (value === null || value === undefined || value === "") return null;
  const numeric = toNumeric(value);
  if (numeric !== null && datetimeColumnSet.value.has(column)) return excelSerialToDate(numeric);
  return parseDateFromString(value);
}

function formatNumber(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "n/a";
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

function formatAxisValue(column, value) {
  if (value === null || value === undefined) return "n/a";
  if (datetimeColumnSet.value.has(column)) return formatDateLabel(new Date(value));
  return formatNumber(value);
}

function interpolateByPercent(min, max, percent) {
  return min + ((max - min) * percent) / 100;
}

function formatWindowLabel(start, end, column) {
  if (datetimeColumnSet.value.has(column)) {
    return `${formatDateLabel(new Date(start))} - ${formatDateLabel(new Date(end))}`;
  }
  return `${formatNumber(start)} - ${formatNumber(end)}`;
}

function axisValueFromRow(row, column) {
  if (datetimeColumnSet.value.has(column)) {
    const date = toDate(row[column], column);
    return date ? date.getTime() : null;
  }
  return toNumeric(row[column]);
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

const heatmapColorOptions = computed(() => {
  const options = [{ value: "__count__", label: "Плотность точек" }];
  const seen = new Set(["__count__"]);
  const candidates = [
    props.analysis.dataset.target_column,
    ...(props.analysis.numeric_columns || []),
    ...(props.analysis.datetime_columns || []),
    ...inferredNumericColumns.value,
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (!seen.has(candidate)) {
      options.push({ value: candidate, label: candidate });
      seen.add(candidate);
    }
  }
  return options;
});

watch(
  availableTimeColumns,
  (columns) => {
    if (!columns.includes(selectedTimeColumn.value)) selectedTimeColumn.value = columns[0] || "";
  },
  { immediate: true },
);

watch(
  scatterColumns,
  (columns) => {
    if (!columns.includes(selectedScatterX.value)) selectedScatterX.value = columns[0] || "";
    if (!columns.includes(selectedScatterY.value)) selectedScatterY.value = columns[1] || columns[0] || "";
  },
  { immediate: true },
);

watch(
  allColumns,
  (columns) => {
    if (!columns.includes(selectedHistogramColumn.value)) selectedHistogramColumn.value = columns[0] || "";
  },
  { immediate: true },
);

watch(
  heatmapColorOptions,
  (options) => {
    if (!options.some((option) => option.value === selectedHeatmapColorBy.value)) {
      selectedHeatmapColorBy.value = "__count__";
    }
  },
  { immediate: true },
);

watch(selectedGroupColumn, () => {
  selectedGroupValues.value = [];
});

const groupableColumns = computed(() =>
  props.columns.filter((column) => column !== props.analysis.dataset.target_column && column !== selectedTimeColumn.value),
);

const groupValues = computed(() => {
  if (!selectedGroupColumn.value) return [];
  const values = new Set();
  for (const row of props.rows) values.add(normalizeCategory(row[selectedGroupColumn.value]));
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
  if (!targetColumn || !selectedTimeColumn.value) return [];

  const aggregateMap = new Map();
  for (const row of props.rows) {
    const timeValue = toDate(row[selectedTimeColumn.value], selectedTimeColumn.value);
    const targetValue = toNumeric(row[targetColumn]);
    if (!timeValue || targetValue === null || targetValue === 0) continue;

    const groupValue = selectedGroupColumn.value ? normalizeCategory(row[selectedGroupColumn.value]) : "Все";
    if (selectedGroupColumn.value && selectedGroupValues.value.length > 0 && !selectedGroupValues.value.includes(groupValue)) continue;

    const bucketDate = new Date(Date.UTC(timeValue.getUTCFullYear(), timeValue.getUTCMonth(), 1));
    const key = `${groupValue}__${bucketDate.toISOString()}`;
    const current = aggregateMap.get(key) || { group: groupValue, date: bucketDate, values: [] };
    current.values.push(targetValue);
    aggregateMap.set(key, current);
  }

  const byGroup = new Map();
  for (const item of aggregateMap.values()) {
    const line = byGroup.get(item.group) || [];
    line.push({ x: item.date, y: item.values.reduce((sum, value) => sum + value, 0) / item.values.length });
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
  yaxis: {
    title: props.analysis.dataset.target_column || "Target",
  },
}));

const scatterBasePoints = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) return [];
  return props.rows
    .map((row) => {
      const x = axisValueFromRow(row, selectedScatterX.value);
      const y = axisValueFromRow(row, selectedScatterY.value);
      if (x === null || y === null) return null;

      const colorValue =
        selectedHeatmapColorBy.value === "__count__"
          ? 1
          : axisValueFromRow(row, selectedHeatmapColorBy.value);

      return {
        x,
        y,
        xLabel: datetimeColumnSet.value.has(selectedScatterX.value) ? formatDateLabel(new Date(x)) : null,
        yLabel: datetimeColumnSet.value.has(selectedScatterY.value) ? formatDateLabel(new Date(y)) : null,
        colorValue,
      };
    })
    .filter(Boolean);
});

const scatterStats = computed(() => {
  const points = scatterBasePoints.value;
  if (!points.length) return null;
  return {
    minX: Math.min(...points.map((point) => point.x)),
    maxX: Math.max(...points.map((point) => point.x)),
    minY: Math.min(...points.map((point) => point.y)),
    maxY: Math.max(...points.map((point) => point.y)),
  };
});

watch(
  () => [
    selectedScatterX.value,
    selectedScatterY.value,
    scatterStats.value?.minX,
    scatterStats.value?.maxX,
    scatterStats.value?.minY,
    scatterStats.value?.maxY,
  ],
  () => {
    scatterXMinPercent.value = 0;
    scatterXMaxPercent.value = 100;
    scatterYMinPercent.value = 0;
    scatterYMaxPercent.value = 100;
  },
);

const scatterXMinValue = computed(() => {
  if (!scatterStats.value) return null;
  return interpolateByPercent(scatterStats.value.minX, scatterStats.value.maxX, Math.min(scatterXMinPercent.value, scatterXMaxPercent.value));
});

const scatterXMaxValue = computed(() => {
  if (!scatterStats.value) return null;
  return interpolateByPercent(scatterStats.value.minX, scatterStats.value.maxX, Math.max(scatterXMinPercent.value, scatterXMaxPercent.value));
});

const scatterYMinValue = computed(() => {
  if (!scatterStats.value) return null;
  return interpolateByPercent(scatterStats.value.minY, scatterStats.value.maxY, Math.min(scatterYMinPercent.value, scatterYMaxPercent.value));
});

const scatterYMaxValue = computed(() => {
  if (!scatterStats.value) return null;
  return interpolateByPercent(scatterStats.value.minY, scatterStats.value.maxY, Math.max(scatterYMinPercent.value, scatterYMaxPercent.value));
});

const scatterRangeReady = computed(() => Boolean(scatterStats.value));

const filteredScatterPoints = computed(() => {
  if (!scatterRangeReady.value) return [];
  return scatterBasePoints.value.filter(
    (point) =>
      point.x >= scatterXMinValue.value &&
      point.x <= scatterXMaxValue.value &&
      point.y >= scatterYMinValue.value &&
      point.y <= scatterYMaxValue.value,
  );
});

const averagedScatterBuckets = computed(() => {
  const points = filteredScatterPoints.value
    .filter((point) => point.y !== null && point.y !== 0)
    .sort((a, b) => a.x - b.x);

  if (points.length < 2) return [];
  const minX = scatterXMinValue.value;
  const maxX = scatterXMaxValue.value;
  const span = maxX - minX;
  if (span <= 0) return [];

  const step = span * 0.1;
  if (step <= 0) return [];

  const buckets = [];
  for (let start = minX; start < maxX; start += step) {
    const end = Math.min(start + step, maxX);
    const bucket = points.filter((point) => point.x >= start && (point.x < end || end === maxX));
    if (!bucket.length) continue;

    const avgX = bucket.reduce((sum, point) => sum + point.x, 0) / bucket.length;
    const avgY = bucket.reduce((sum, point) => sum + point.y, 0) / bucket.length;
    buckets.push({
      start,
      end,
      avgX,
      avgY,
      count: bucket.length,
      windowLabel: formatWindowLabel(start, end, selectedScatterX.value),
      avgXLabel: formatAxisValue(selectedScatterX.value, avgX),
      avgYLabel: formatAxisValue(selectedScatterY.value, avgY),
    });
  }
  return buckets;
});

const averagedScatterTable = computed(() =>
  averagedScatterBuckets.value.map((bucket) => ({
    windowLabel: bucket.windowLabel,
    avgX: bucket.avgXLabel,
    avgY: bucket.avgYLabel,
    count: bucket.count,
  })),
);

const scatterData = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) return [];

  const traces = [
    {
      type: "scatter",
      mode: "markers",
      name: "Наблюдения",
      x: filteredScatterPoints.value.map((point) => point.x),
      y: filteredScatterPoints.value.map((point) => point.y),
      text: filteredScatterPoints.value.map((point) => {
        const parts = [
          `${selectedScatterX.value}: ${point.xLabel || formatNumber(point.x)}`,
          `${selectedScatterY.value}: ${point.yLabel || formatNumber(point.y)}`,
        ];
        return parts.join("<br>");
      }),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        color: "#ea580c",
        size: viewportWidth.value < 720 ? 7 : 8,
        opacity: 0.72,
      },
    },
  ];

  if (averagedScatterBuckets.value.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers+text",
      name: "Среднее Y по окнам X (10%)",
      x: averagedScatterBuckets.value.map((bucket) => bucket.avgX),
      y: averagedScatterBuckets.value.map((bucket) => bucket.avgY),
      text: averagedScatterBuckets.value.map((bucket) => `X: ${bucket.avgXLabel}<br>Y: ${bucket.avgYLabel}`),
      textposition: "top center",
      line: { color: "#0f766e", width: 3 },
      marker: { color: "#0f766e", size: 8 },
      hovertemplate: "%{text}<extra></extra>",
    });
  }

  return traces;
});

const scatterLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 60 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: { text: selectedScatterX.value || "Ось X" },
    type: datetimeColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
    range: scatterRangeReady.value ? [scatterXMinValue.value, scatterXMaxValue.value] : undefined,
  },
  yaxis: {
    title: { text: selectedScatterY.value || "Ось Y" },
    type: datetimeColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
    range: scatterRangeReady.value ? [scatterYMinValue.value, scatterYMaxValue.value] : undefined,
  },
  legend: { orientation: "h", y: 1.1 },
}));

const scatterHeatmapData = computed(() => {
  if (!filteredScatterPoints.value.length) return [];

  if (selectedHeatmapColorBy.value === "__count__") {
    return [
      {
        type: "histogram2d",
        x: filteredScatterPoints.value.map((point) => point.x),
        y: filteredScatterPoints.value.map((point) => point.y),
        colorscale: [
          [0, "#eff6ff"],
          [0.35, "#60a5fa"],
          [0.7, "#2563eb"],
          [1, "#1e3a8a"],
        ],
        hovertemplate: "X=%{x}<br>Y=%{y}<br>Count=%{z}<extra></extra>",
      },
    ];
  }

  const pointsWithColor = filteredScatterPoints.value.filter((point) => point.colorValue !== null && point.colorValue !== undefined);
  if (!pointsWithColor.length) return [];

  return [
    {
      type: "histogram2d",
      x: pointsWithColor.map((point) => point.x),
      y: pointsWithColor.map((point) => point.y),
      z: pointsWithColor.map((point) => point.colorValue),
      histfunc: "avg",
      colorscale: [
        [0, "#fff7ed"],
        [0.35, "#fdba74"],
        [0.7, "#f97316"],
        [1, "#9a3412"],
      ],
      hovertemplate: "X=%{x}<br>Y=%{y}<br>Среднее=%{z}<extra></extra>",
    },
  ];
});

const scatterHeatmapLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 60 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: { text: selectedScatterX.value || "Ось X" },
    type: datetimeColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
  },
  yaxis: {
    title: { text: selectedScatterY.value || "Ось Y" },
    type: datetimeColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
  },
}));

const histogramData = computed(() => {
  if (!selectedHistogramColumn.value) return [];

  if (datetimeColumnSet.value.has(selectedHistogramColumn.value)) {
    const counts = new Map();
    for (const row of props.rows) {
      const date = toDate(row[selectedHistogramColumn.value], selectedHistogramColumn.value);
      if (!date) continue;
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

  const numericValues = props.rows.map((row) => toNumeric(row[selectedHistogramColumn.value])).filter((value) => value !== null);
  if (numericValues.length >= Math.max(5, props.rows.length * 0.5)) {
    return [{ type: "histogram", x: numericValues, marker: { color: "#1d4ed8" } }];
  }

  const counts = new Map();
  for (const row of props.rows) {
    const key = normalizeCategory(row[selectedHistogramColumn.value]);
    counts.set(key, (counts.get(key) || 0) + 1);
  }

  const topValues = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 30);
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
  return (
    props.columns.find((column) => {
      const normalized = column.toLowerCase();
      return normalized.includes("участ") || normalized.includes("недр");
    }) || ""
  );
}

const pivotTable = computed(() => {
  const targetColumn = props.analysis.dataset.target_column;
  const dateColumn = selectedTimeColumn.value || availableTimeColumns.value[0] || "";
  const subsoilColumn = findSubsoilColumn();
  if (!targetColumn || !dateColumn || !subsoilColumn) return { headers: [], rows: [] };

  const grouped = new Map();
  const years = new Set();

  for (const row of props.rows) {
    const date = toDate(row[dateColumn], dateColumn);
    const targetValue = toNumeric(row[targetColumn]);
    if (!date || targetValue === null) continue;
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
