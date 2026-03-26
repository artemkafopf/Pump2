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

        <div v-if="availableYears.length" class="range-grid">
          <label class="field">
            <span>Год от: {{ selectedYearMin ?? "-" }}</span>
            <input v-model.number="selectedYearMin" type="range" :min="availableYears[0]" :max="availableYears[availableYears.length - 1]" step="1" />
          </label>
          <label class="field">
            <span>Год до: {{ selectedYearMax ?? "-" }}</span>
            <input v-model.number="selectedYearMax" type="range" :min="availableYears[0]" :max="availableYears[availableYears.length - 1]" step="1" />
          </label>
        </div>
      </div>

      <div class="time-summary-grid">
        <div class="time-summary-plot">
          <PlotlyChart :data="timeSeriesData" :layout="timeSeriesLayout" />
        </div>

        <div v-if="timeSummaryTable.headers.length" class="table-wrap compact-table">
          <table class="data-table">
            <thead>
              <tr>
                <th>{{ selectedGroupColumn || "Группа" }}</th>
                <th v-for="year in timeSummaryTable.headers" :key="`time-table-${year}`">{{ year }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in timeSummaryTable.rows" :key="row.label">
                <td>{{ row.label }}</td>
                <td v-for="year in timeSummaryTable.headers" :key="`${row.label}-${year}`">{{ row.values[year] ?? "-" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
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

        <div class="control-group">
          <span class="control-title">Раскраска точек</span>
          <div class="button-group">
            <button
              v-for="option in scatterColorOptions"
              :key="`scatter-color-${option.value}`"
              type="button"
              class="choice-button"
              :class="{ active: option.value === selectedScatterColorBy }"
              @click="selectedScatterColorBy = option.value"
            >
              {{ option.label }}
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

      <div class="averaged-scatter-grid">
        <div class="averaged-scatter-plot">
          <PlotlyChart :data="averagedScatterData" :layout="averagedScatterLayout" />
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
      </div>
      <div v-if="!averagedScatterTable.length" class="empty-state">
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

      <div class="control-group">
        <span class="control-title">Срезы</span>
        <div class="button-group">
          <button
            v-for="option in histogramGroupOptions"
            :key="`hist-group-${option.value}`"
            type="button"
            class="choice-button"
            :class="{ active: option.value === selectedHistogramGroupColumn }"
            @click="selectedHistogramGroupColumn = option.value"
          >
            {{ option.label }}
          </button>
        </div>
      </div>

      <div v-if="selectedHistogramGroupColumn" class="control-group">
        <span class="control-title">Значения срезов</span>
        <div class="button-group">
          <button
            type="button"
            class="choice-button"
            :class="{ active: includeHistogramAggregate }"
            @click="includeHistogramAggregate = !includeHistogramAggregate"
          >
            Все вместе
          </button>
          <button
            v-for="value in histogramGroupValues"
            :key="`hist-group-value-${value}`"
            type="button"
            class="choice-button"
            :class="{ active: selectedHistogramGroupValues.includes(value) }"
            @click="selectedHistogramGroupValues = selectedHistogramGroupValues.includes(value) ? selectedHistogramGroupValues.filter((item) => item !== value) : [...selectedHistogramGroupValues, value]"
          >
            {{ value }}
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

const analysis = computed(() => props.analysis);

const viewportWidth = ref(typeof window === "undefined" ? 1440 : window.innerWidth);
const selectedTimeColumn = ref("");
const selectedGroupColumn = ref("");
const selectedGroupValues = ref([]);
const selectedYearMin = ref(null);
const selectedYearMax = ref(null);
const selectedScatterX = ref("");
const selectedScatterY = ref("");
const selectedHistogramColumn = ref("");
const selectedHistogramGroupColumn = ref("");
const selectedHistogramGroupValues = ref([]);
const includeHistogramAggregate = ref(true);
const selectedHeatmapColorBy = ref("__count__");
const selectedScatterColorBy = ref("__none__");
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

function isPossibleExcelDate(value) {
  return Number.isFinite(value) && value > 20000 && value < 80000;
}

function excelSerialToDate(value) {
  const epoch = new Date(Date.UTC(1899, 11, 30));
  const wholeDays = Math.trunc(Number(value));
  const milliseconds = Math.round((Number(value) - wholeDays) * 24 * 60 * 60 * 1000);
  epoch.setUTCDate(epoch.getUTCDate() + wholeDays);
  epoch.setTime(epoch.getTime() + milliseconds);
  return epoch;
}

function parseDateFromString(value) {
  const trimmed = String(value).trim();
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

  const isoMatch = trimmed.match(/^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s](\d{1,2}):(\d{2})(?::(\d{2}))?)?$/);
  if (isoMatch) {
    const year = Number(isoMatch[1]);
    const month = Number(isoMatch[2]) - 1;
    const day = Number(isoMatch[3]);
    const hour = Number(isoMatch[4] || 0);
    const minute = Number(isoMatch[5] || 0);
    const second = Number(isoMatch[6] || 0);
    const parsed = new Date(Date.UTC(year, month, day, hour, minute, second));
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  const iso = new Date(trimmed);
  if (!Number.isNaN(iso.getTime())) return iso;

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
  if (numeric !== null && (dateColumnSet.value.has(column) || isPossibleExcelDate(numeric))) return excelSerialToDate(numeric);
  return parseDateFromString(value);
}

function formatNumber(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(value);
}

function formatOneDecimal(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
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

function extractYearValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = toNumeric(value);
  if (numeric !== null) {
    const rounded = Math.trunc(numeric);
    if (rounded >= 1900 && rounded <= 2100) return rounded;
  }

  const match = String(value).match(/(19|20)\d{2}/);
  if (match) return Number(match[0]);

  const parsed = parseDateFromString(value);
  if (parsed) return parsed.getUTCFullYear();
  return null;
}

function formatAxisValue(column, value) {
  if (value === null || value === undefined) return "n/a";
  if (dateColumnSet.value.has(column)) return formatDateLabel(new Date(value));
  return formatNumber(value);
}

function interpolateByPercent(min, max, percent) {
  return min + ((max - min) * percent) / 100;
}

function formatWindowLabel(start, end, column) {
  if (dateColumnSet.value.has(column)) {
    return `${formatDateLabel(new Date(start))} - ${formatDateLabel(new Date(end))}`;
  }
  return `${formatNumber(start)} - ${formatNumber(end)}`;
}

function axisValueFromRow(row, column) {
  if (dateColumnSet.value.has(column)) {
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

const dateColumnSet = computed(() => new Set());

const availableTimeColumns = computed(() =>
  props.columns.filter((column) => column !== props.analysis.dataset.target_column),
);

const scatterColumns = computed(() => {
  const merged = new Set([
    ...(props.analysis.numeric_columns || []),
    ...dateColumnSet.value,
    ...inferredNumericColumns.value,
  ]);
  return Array.from(merged);
});

const allColumns = computed(() => props.columns);

const histogramGroupOptions = computed(() => {
  const options = [{ value: "", label: "Без срезов" }];
  for (const column of props.columns) {
    if (column !== selectedHistogramColumn.value) {
      options.push({ value: column, label: column });
    }
  }
  if (selectedTimeColumn.value) {
    options.push({ value: "__year__", label: "Год" });
  }
  return options;
});

const heatmapColorOptions = computed(() => {
  const options = [{ value: "__count__", label: "Плотность точек" }];
  const seen = new Set(["__count__"]);
  const candidates = [
    props.analysis.dataset.target_column,
    ...(props.analysis.numeric_columns || []),
    ...dateColumnSet.value,
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

const scatterColorOptions = computed(() => {
  const options = [{ value: "__none__", label: "Один цвет" }];
  const seen = new Set(["__none__"]);
  for (const column of props.columns) {
    if (!seen.has(column) && column !== selectedScatterX.value && column !== selectedScatterY.value) {
      options.push({ value: column, label: column });
      seen.add(column);
    }
  }
  return options;
});

const histogramGroupValues = computed(() => {
  if (!selectedHistogramGroupColumn.value) return [];
  if (selectedHistogramGroupColumn.value === "__year__") {
    return availableYears.value.map((year) => String(year));
  }
  const values = new Set();
  for (const row of props.rows) values.add(normalizeCategory(row[selectedHistogramGroupColumn.value]));
  return Array.from(values).slice(0, 30);
});

const availableYears = computed(() => {
  if (!selectedTimeColumn.value) return [];
  const years = props.rows
    .map((row) => extractYearValue(row[selectedTimeColumn.value]))
    .filter((value) => value !== null);
  return Array.from(new Set(years)).sort((a, b) => a - b);
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

watch(
  scatterColorOptions,
  (options) => {
    if (!options.some((option) => option.value === selectedScatterColorBy.value)) {
      selectedScatterColorBy.value = "__none__";
    }
  },
  { immediate: true },
);

watch(selectedGroupColumn, () => {
  selectedGroupValues.value = [];
});

watch(
  availableYears,
  (years) => {
    if (!years.length) {
      selectedYearMin.value = null;
      selectedYearMax.value = null;
      return;
    }
    if (selectedYearMin.value === null || !years.includes(selectedYearMin.value)) {
      selectedYearMin.value = years[0];
    }
    if (selectedYearMax.value === null || !years.includes(selectedYearMax.value)) {
      selectedYearMax.value = years[years.length - 1];
    }
    if (selectedYearMin.value > selectedYearMax.value) {
      selectedYearMax.value = selectedYearMin.value;
    }
  },
  { immediate: true },
);

watch(selectedYearMin, (value) => {
  if (selectedYearMax.value !== null && value !== null && value > selectedYearMax.value) {
    selectedYearMax.value = value;
  }
});

watch(selectedYearMax, (value) => {
  if (selectedYearMin.value !== null && value !== null && value < selectedYearMin.value) {
    selectedYearMin.value = value;
  }
});

watch(selectedHistogramGroupColumn, () => {
  selectedHistogramGroupValues.value = [];
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
    text: [(props.analysis.correlations || []).map((item) => item.correlation.toFixed(2))],
    texttemplate: "%{text}",
    textfont: {
      color: "#0f172a",
      size: viewportWidth.value < 720 ? 10 : 12,
    },
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
    const year = extractYearValue(row[selectedTimeColumn.value]);
    const targetValue = toNumeric(row[targetColumn]);
    if (year === null || targetValue === null || targetValue === 0) continue;
    if (selectedYearMin.value !== null && year < selectedYearMin.value) continue;
    if (selectedYearMax.value !== null && year > selectedYearMax.value) continue;

    const groupValue = selectedGroupColumn.value ? normalizeCategory(row[selectedGroupColumn.value]) : "Все";
    if (selectedGroupColumn.value && selectedGroupValues.value.length > 0 && !selectedGroupValues.value.includes(groupValue)) continue;

    const key = `${groupValue}__${year}`;
    const current = aggregateMap.get(key) || { group: groupValue, x: year, xSort: year, values: [] };
    current.values.push(targetValue);
    aggregateMap.set(key, current);
  }

  const byGroup = new Map();
  for (const item of aggregateMap.values()) {
    const line = byGroup.get(item.group) || [];
    if (!item.values.length) continue;
    line.push({ x: item.x, xSort: item.xSort, y: item.values.reduce((sum, value) => sum + value, 0) / item.values.length });
    byGroup.set(item.group, line);
  }

  return Array.from(byGroup.entries()).map(([group, points]) => {
    const sortedPoints = points.sort((a, b) => {
      if (typeof a.xSort === "number" && typeof b.xSort === "number") return a.xSort - b.xSort;
      return String(a.xSort).localeCompare(String(b.xSort), "ru");
    });
    return {
      type: "scatter",
      mode: "lines+markers",
      name: group,
      x: sortedPoints.map((point) => point.x),
      y: sortedPoints.map((point) => point.y),
      hovertemplate: "Год: %{x}<br>Среднее: %{y:.1f}<extra>" + group + "</extra>",
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
    title: "Год",
  },
  yaxis: {
    title: props.analysis.dataset.target_column || "Target",
  },
}));

const timeSummaryTable = computed(() => {
  const targetColumn = props.analysis.dataset.target_column;
  if (!targetColumn || !selectedTimeColumn.value) return { headers: [], rows: [] };

  const grouped = new Map();
  const years = new Set();

  for (const row of props.rows) {
    const year = extractYearValue(row[selectedTimeColumn.value]);
    const targetValue = toNumeric(row[targetColumn]);
    if (year === null || targetValue === null || targetValue === 0) continue;
    if (selectedYearMin.value !== null && year < selectedYearMin.value) continue;
    if (selectedYearMax.value !== null && year > selectedYearMax.value) continue;

    const label = selectedGroupColumn.value ? normalizeCategory(row[selectedGroupColumn.value]) : "Все";
    if (selectedGroupColumn.value && selectedGroupValues.value.length > 0 && !selectedGroupValues.value.includes(label)) continue;

    years.add(String(year));
    const key = `${label}__${year}`;
    const current = grouped.get(key) || { label, year: String(year), values: [] };
    current.values.push(targetValue);
    grouped.set(key, current);
  }

  const headers = Array.from(years).sort();
  const rowMap = new Map();

  for (const item of grouped.values()) {
    const currentRow = rowMap.get(item.label) || {};
    const avg = item.values.reduce((sum, value) => sum + value, 0) / item.values.length;
    currentRow[item.year] = formatOneDecimal(avg);
    rowMap.set(item.label, currentRow);
  }

  return {
    headers,
    rows: Array.from(rowMap.entries()).map(([label, values]) => ({ label, values })),
  };
});

const scatterBasePoints = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) return [];
  return props.rows
    .map((row) => {
      const x = axisValueFromRow(row, selectedScatterX.value);
      const y = axisValueFromRow(row, selectedScatterY.value);
      if (x === null || y === null) return null;
      const scatterColorRaw =
        selectedScatterColorBy.value && selectedScatterColorBy.value !== "__none__"
          ? row[selectedScatterColorBy.value]
          : null;
      const scatterColorNumeric =
        selectedScatterColorBy.value && selectedScatterColorBy.value !== "__none__"
          ? axisValueFromRow(row, selectedScatterColorBy.value)
          : null;
      const heatmapColorValue =
        selectedHeatmapColorBy.value === "__count__"
          ? 1
          : axisValueFromRow(row, selectedHeatmapColorBy.value);

      return {
        x,
        y,
        xLabel: dateColumnSet.value.has(selectedScatterX.value) ? formatDateLabel(new Date(x)) : null,
        yLabel: dateColumnSet.value.has(selectedScatterY.value) ? formatDateLabel(new Date(y)) : null,
        scatterColorRaw,
        scatterColorNumeric,
        scatterColorLabel:
          selectedScatterColorBy.value && selectedScatterColorBy.value !== "__none__"
            ? normalizeCategory(row[selectedScatterColorBy.value])
            : null,
        heatmapColorValue,
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
    avgX: dateColumnSet.value.has(selectedScatterX.value) ? bucket.avgXLabel : formatOneDecimal(bucket.avgX),
    avgY: dateColumnSet.value.has(selectedScatterY.value) ? bucket.avgYLabel : formatOneDecimal(bucket.avgY),
    count: bucket.count,
  })),
);

const scatterColorMeta = computed(() => {
  if (selectedScatterColorBy.value === "__none__") return { mode: "none" };

  const numericValues = filteredScatterPoints.value
    .map((point) => point.scatterColorNumeric)
    .filter((value) => value !== null && value !== undefined);

  if (numericValues.length >= Math.max(5, filteredScatterPoints.value.length * 0.4)) {
    return { mode: "numeric" };
  }

  const palette = ["#ea580c", "#1d4ed8", "#0f766e", "#7c3aed", "#dc2626", "#0891b2", "#65a30d", "#d97706"];
  const categories = Array.from(new Set(filteredScatterPoints.value.map((point) => point.scatterColorLabel)));
  const colorMap = new Map(categories.map((category, index) => [category, palette[index % palette.length]]));
  return { mode: "categorical", colorMap };
});

const averagedScatterData = computed(() => {
  if (!averagedScatterBuckets.value.length) return [];

  return [
    {
      type: "scatter",
      mode: "lines+markers+text",
      name: "Усредненные точки",
      x: averagedScatterBuckets.value.map((bucket) => bucket.avgX),
      y: averagedScatterBuckets.value.map((bucket) => bucket.avgY),
      text: averagedScatterBuckets.value.map((bucket) => `X: ${bucket.avgXLabel}<br>Y: ${bucket.avgYLabel}`),
      textposition: "top center",
      line: { color: "#0f766e", width: 3 },
      marker: { color: "#0f766e", size: 8 },
      hovertemplate: "%{text}<extra></extra>",
    },
  ];
});

const averagedScatterLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 60 },
  height: Math.max(280, chartHeight.value - 40),
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: { text: selectedScatterX.value || "X" },
    type: dateColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
  },
  yaxis: {
    title: { text: selectedScatterY.value || "Y" },
    type: dateColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
  },
}));

const scatterData = computed(() => {
  if (!selectedScatterX.value || !selectedScatterY.value) return [];

  const marker = {
    size: viewportWidth.value < 720 ? 7 : 8,
    opacity: 0.72,
  };

  if (scatterColorMeta.value.mode === "numeric") {
    marker.color = filteredScatterPoints.value.map((point) => point.scatterColorNumeric);
    marker.colorscale = [
      [0, "#eff6ff"],
      [0.35, "#60a5fa"],
      [0.7, "#2563eb"],
      [1, "#1e3a8a"],
    ];
    marker.colorbar = { title: selectedScatterColorBy.value };
  } else if (scatterColorMeta.value.mode === "categorical") {
    marker.color = filteredScatterPoints.value.map(
      (point) => scatterColorMeta.value.colorMap.get(point.scatterColorLabel) || "#ea580c",
    );
  } else {
    marker.color = "#ea580c";
  }

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
        if (selectedScatterColorBy.value !== "__none__") {
          parts.push(`${selectedScatterColorBy.value}: ${point.scatterColorLabel}`);
        }
        return parts.join("<br>");
      }),
      hovertemplate: "%{text}<extra></extra>",
      marker,
    },
  ];

  if (averagedScatterData.value.length) {
    traces.push({
      ...averagedScatterData.value[0],
      name: "Среднее Y по окнам X (10%)",
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
    type: dateColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
    range: scatterRangeReady.value ? [scatterXMinValue.value, scatterXMaxValue.value] : undefined,
  },
  yaxis: {
    title: { text: selectedScatterY.value || "Ось Y" },
    type: dateColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
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

  const pointsWithColor = filteredScatterPoints.value.filter(
    (point) => point.heatmapColorValue !== null && point.heatmapColorValue !== undefined,
  );
  if (!pointsWithColor.length) return [];

  return [
    {
      type: "histogram2d",
      x: pointsWithColor.map((point) => point.x),
      y: pointsWithColor.map((point) => point.y),
      z: pointsWithColor.map((point) => point.heatmapColorValue),
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
    type: dateColumnSet.value.has(selectedScatterX.value) ? "date" : "linear",
  },
  yaxis: {
    title: { text: selectedScatterY.value || "Ось Y" },
    type: dateColumnSet.value.has(selectedScatterY.value) ? "date" : "linear",
  },
}));

function resolveHistogramSliceValue(row) {
  if (!selectedHistogramGroupColumn.value) return null;
  if (selectedHistogramGroupColumn.value === "__year__") {
    const year = selectedTimeColumn.value ? extractYearValue(row[selectedTimeColumn.value]) : null;
    return year === null ? null : String(year);
  }
  return normalizeCategory(row[selectedHistogramGroupColumn.value]);
}

const histogramData = computed(() => {
  if (!selectedHistogramColumn.value) return [];

  const palette = ["#1d4ed8", "#0f766e", "#ea580c", "#7c3aed", "#dc2626", "#0891b2", "#65a30d"];
  const isNumericHistogram =
    props.rows.map((row) => toNumeric(row[selectedHistogramColumn.value])).filter((value) => value !== null).length >=
    Math.max(5, props.rows.length * 0.5);

  const selectedSlices =
    selectedHistogramGroupValues.value.length > 0
      ? selectedHistogramGroupValues.value
      : histogramGroupValues.value.slice(0, 6);

  const traceGroups = [];
  if (includeHistogramAggregate.value) {
    traceGroups.push({ label: "Все вместе", rows: props.rows });
  }

  if (selectedHistogramGroupColumn.value) {
    for (const value of selectedSlices) {
      traceGroups.push({
        label: value,
        rows: props.rows.filter((row) => resolveHistogramSliceValue(row) === value),
      });
    }
  }

  if (!traceGroups.length) {
    traceGroups.push({ label: selectedHistogramColumn.value, rows: props.rows });
  }

  if (isNumericHistogram) {
    return traceGroups.map((group, index) => ({
      type: "histogram",
      name: group.label,
      x: group.rows.map((row) => toNumeric(row[selectedHistogramColumn.value])).filter((value) => value !== null),
      marker: { color: palette[index % palette.length] },
      opacity: 0.4,
    }));
  }

  return traceGroups.map((group, index) => {
    const counts = new Map();
    for (const row of group.rows) {
      const key = normalizeCategory(row[selectedHistogramColumn.value]);
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const topValues = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 30);
    return {
      type: "bar",
      name: group.label,
      x: topValues.map((item) => item[0]),
      y: topValues.map((item) => item[1]),
      marker: { color: palette[index % palette.length] },
      opacity: 0.4,
    };
  });
});

const histogramLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: viewportWidth.value < 720 ? 120 : 100 },
  height: chartHeight.value,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  barmode: "overlay",
  xaxis: { title: selectedHistogramColumn.value || "Column" },
  yaxis: { title: "Count" },
  legend: { orientation: "h", y: 1.1 },
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




