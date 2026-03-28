<template>
  <section class="prediction-grid">
    <section class="panel">
      <div class="panel-header">
        <h2>Настройка прогноза</h2>
        <p>Используем текущий датасет и CatBoost для оценки риска отказов и прогноза целевой переменной.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Целевая переменная</span>
          <div class="note-chip">{{ dataset.target_column || "Не выбрана" }}</div>
        </div>

        <div class="control-group">
          <span class="control-title">Зависимые переменные для модели</span>
          <div class="button-group">
            <button
              v-for="column in featureCandidates"
              :key="`forecast-feature-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: enabledFeatureColumns.includes(column) }"
              @click="toggleFeature(column)"
            >
              {{ column }}
            </button>
          </div>
          <p class="dataset-meta">CatBoost использует все включенные переменные. Ниже можно отдельно выбрать до 4 числовых признаков для быстрых сценариев и визуализаций.</p>
        </div>

        <div class="control-group">
          <span class="control-title">Дополнительные числовые признаки</span>
          <div class="feature-slot-grid">
            <label v-for="(_, index) in selectedFeatureSlots" :key="`feature-slot-${index}`" class="field">
              <span>Признак {{ index + 1 }}</span>
              <select v-model="selectedFeatureSlots[index]">
                <option value="">Не выбран</option>
                <option v-for="column in numericScenarioCandidates" :key="`${index}-${column}`" :value="column">
                  {{ column }}
                </option>
              </select>
            </label>
          </div>
        </div>

        <div class="range-grid">
          <label class="field">
            <span>Тестовая выборка: {{ Math.round(testFraction * 100) }}%</span>
            <input v-model.number="testFractionPercent" type="range" min="10" max="40" step="5" />
          </label>
          <label class="field">
            <span>Random seed</span>
            <input v-model.number="randomSeed" type="number" min="1" step="1" />
          </label>
        </div>

        <div class="prediction-actions">
          <button type="button" class="primary-button" :disabled="training || !enabledFeatureColumns.length" @click="handleTrain">
            {{ training ? "Обучение..." : "Обучить модель прогноза" }}
          </button>
          <button type="button" class="file-button" :disabled="savingModel || !modelInfo" @click="handleSaveModel">
            {{ savingModel ? "Сохранение..." : "Сохранить модель" }}
          </button>
          <button type="button" class="file-button" :disabled="predicting || !enabledFeatureColumns.length" @click="handlePredict">
            {{ predicting ? "Прогноз..." : "Обновить прогноз" }}
          </button>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Сохранённые модели</h2>
        <p>Выберите сохранённую модель CatBoost, которую нужно использовать для прогноза.</p>
      </div>

      <label class="field">
        <span>Модель для прогноза</span>
        <select v-model="selectedSavedModelId">
          <option :value="null">Текущая обученная модель</option>
          <option v-for="model in savedModels" :key="model.id" :value="model.id">
            {{ model.name }} · {{ formatMetric(model.metrics.rmse) }} RMSE
          </option>
        </select>
      </label>
    </section>

    <section v-if="modelInfo" class="panel">
      <div class="panel-header">
        <h2>Качество модели</h2>
        <p>Модель обучается по всем включенным зависимым переменным текущего датасета.</p>
      </div>

      <div class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Train rows</span>
          <strong>{{ modelInfo.metrics.train_rows }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Test rows</span>
          <strong>{{ modelInfo.metrics.test_rows }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">RMSE</span>
          <strong>{{ formatMetric(modelInfo.metrics.rmse) }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">MAE</span>
          <strong>{{ formatMetric(modelInfo.metrics.mae) }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">R²</span>
          <strong>{{ formatMetric(modelInfo.metrics.r2) }}</strong>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Целевая переменная во времени</h2>
        <p>Факт и прогноз показываются отдельно. Осреднение идёт по годам без учёта нулей и пустых значений.</p>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Колонка по оси X</span>
          <div class="button-group">
            <button
              v-for="column in timeAxisColumns"
              :key="`forecast-time-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: selectedTimeColumn === column }"
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
              v-for="column in timeGroupColumns"
              :key="`forecast-group-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: selectedGroupColumn === column }"
              @click="selectedGroupColumn = column"
            >
              {{ column }}
            </button>
          </div>
        </div>

        <div v-if="selectedGroupColumn && timeGroupValues.length" class="control-group">
          <span class="control-title">Срезы</span>
          <div class="button-group">
            <button
              v-for="value in timeGroupValues"
              :key="`forecast-group-value-${value}`"
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

      <PlotlyChart :data="timeSeriesData" :layout="timeSeriesLayout" />
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Номограмма</h2>
        <p>Изолинии показывают прогнозное значение целевой переменной по двум выбранным числовым признакам.</p>
      </div>

      <div class="control-block">
        <div class="feature-slot-grid">
          <label class="field">
            <span>Ось X</span>
            <select v-model="contourXFeature">
              <option value="">Не выбрана</option>
              <option v-for="column in contourAxisColumns" :key="`contour-x-${column}`" :value="column">{{ column }}</option>
            </select>
          </label>
          <label class="field">
            <span>Ось Y</span>
            <select v-model="contourYFeature">
              <option value="">Не выбрана</option>
              <option v-for="column in contourAxisColumns" :key="`contour-y-${column}`" :value="column">{{ column }}</option>
            </select>
          </label>
        </div>

        <div v-if="contourSliceColumns.length" class="control-group">
          <span class="control-title">Срез для номограммы</span>
          <div class="button-group">
            <button type="button" class="choice-button" :class="{ active: !selectedContourSliceColumn }" @click="selectedContourSliceColumn = ''">
              Без среза
            </button>
            <button
              v-for="column in contourSliceColumns"
              :key="`contour-slice-column-${column}`"
              type="button"
              class="choice-button"
              :class="{ active: selectedContourSliceColumn === column }"
              @click="selectedContourSliceColumn = column"
            >
              {{ column }}
            </button>
          </div>
        </div>

        <div v-if="selectedContourSliceColumn && contourSliceValues.length" class="control-group">
          <span class="control-title">Значение среза</span>
          <div class="button-group">
            <button type="button" class="choice-button" :class="{ active: selectedContourSliceValue === '' }" @click="selectedContourSliceValue = ''">
              Базовое значение
            </button>
            <button
              v-for="value in contourSliceValues"
              :key="`contour-slice-value-${value}`"
              type="button"
              class="choice-button"
              :class="{ active: selectedContourSliceValue === value }"
              @click="selectedContourSliceValue = value"
            >
              {{ value }}
            </button>
          </div>
        </div>

        <div class="control-group">
          <span class="control-title">Палитра номограммы</span>
          <div class="button-group">
            <button
              v-for="option in contourPaletteOptions"
              :key="`contour-palette-${option.value}`"
              type="button"
              class="choice-button"
              :class="{ active: selectedContourPalette === option.value }"
              @click="selectedContourPalette = option.value"
            >
              {{ option.label }}
            </button>
          </div>
        </div>
      </div>

      <PlotlyChart :data="contourData" :layout="contourLayout" />
    </section>

  </section>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import PlotlyChart from "./PlotlyChart.vue";
import { listSavedForecastModels, predictForecast, saveForecastModel, trainForecastModel } from "../services/api";

const props = defineProps({
  dataset: { type: Object, required: true },
  analysis: { type: Object, required: true },
  rows: { type: Array, default: () => [] },
});

const training = ref(false);
const predicting = ref(false);
const testFractionPercent = ref(20);
const randomSeed = ref(42);
const modelInfo = ref(null);
const savedModels = ref([]);
const selectedSavedModelId = ref(null);
const savingModel = ref(false);
const predictionResponse = ref(null);
const datasetPredictionResponse = ref(null);
const enabledFeatureColumns = ref([]);
const selectedFeatureSlots = ref(["", "", "", ""]);
const contourXFeature = ref("");
const contourYFeature = ref("");
const selectedContourSliceColumn = ref("");
const selectedContourSliceValue = ref("");
const selectedContourPalette = ref("softBlue");
const selectedTimeColumn = ref("");
const selectedGroupColumn = ref("");
const selectedGroupValues = ref([]);
const selectedYearMin = ref(null);
const selectedYearMax = ref(null);
const testFraction = computed(() => testFractionPercent.value / 100);
const targetColumn = computed(() => props.dataset.target_column || "");
const allColumns = computed(() => props.dataset.columns || []);

const featureCandidates = computed(() => {
  const selected = (props.dataset.selected_features || []).filter((column) => column !== props.dataset.target_column);
  if (selected.length) return selected;
  return allColumns.value.filter((column) => column !== props.dataset.target_column);
});

const numericScenarioCandidates = computed(() => {
  const numeric = new Set([...(props.analysis.numeric_columns || []), ...(props.analysis.datetime_columns || [])]);
  return enabledFeatureColumns.value.filter((column) => numeric.has(column));
});

const activeFeatureColumns = computed(() => selectedFeatureSlots.value.filter((column, index, array) => column && array.indexOf(column) === index));

const contourAxisColumns = computed(() => {
  const numeric = new Set([...(props.analysis.numeric_columns || []), ...(props.analysis.datetime_columns || [])]);
  return allColumns.value.filter((column) => numeric.has(column));
});

const contourSliceColumns = computed(() =>
  allColumns.value.filter((column) => column !== contourXFeature.value && column !== contourYFeature.value),
);

const timeAxisColumns = computed(() => {
  const columns = [...allColumns.value];
  if (targetColumn.value && !columns.includes(targetColumn.value)) {
    columns.push(targetColumn.value);
  }
  return columns;
});

const timeGroupColumns = computed(() => allColumns.value.filter((column) => column !== selectedTimeColumn.value));

function normalizeCategory(value) {
  if (value === null || value === undefined || value === "") return "Пусто";
  return String(value).trim() || "Пусто";
}

const contourSliceValues = computed(() => {
  if (!selectedContourSliceColumn.value) return [];
  return Array.from(
    new Set(
      props.rows
        .map((row) => normalizeCategory(row[selectedContourSliceColumn.value]))
        .filter((value) => value && value !== "Пусто"),
    ),
  ).slice(0, 20);
});

const contourSliceOverrides = computed(() => {
  if (!selectedContourSliceColumn.value || !selectedContourSliceValue.value) return {};
  return { [selectedContourSliceColumn.value]: selectedContourSliceValue.value };
});

const modelFeatureColumns = computed(() => {
  const next = [...enabledFeatureColumns.value];
  [contourXFeature.value, contourYFeature.value, selectedContourSliceColumn.value].forEach((column) => {
    if (column && !next.includes(column)) {
      next.push(column);
    }
  });
  return next;
});

const contourPaletteOptions = [
  { value: "softBlue", label: "Светло-голубая" },
  { value: "contrast", label: "Контрастная" },
  { value: "warm", label: "Тёплая" },
  { value: "green", label: "Бирюзовая" },
];

const contourColorScales = {
  softBlue: [
    [0, "#dbeafe"],
    [0.25, "#bfdbfe"],
    [0.5, "#93c5fd"],
    [0.75, "#60a5fa"],
    [1, "#1d4ed8"],
  ],
  contrast: [
    [0, "#e0f2fe"],
    [0.2, "#7dd3fc"],
    [0.45, "#38bdf8"],
    [0.7, "#0ea5e9"],
    [1, "#1e3a8a"],
  ],
  warm: [
    [0, "#fff7ed"],
    [0.25, "#fed7aa"],
    [0.5, "#fdba74"],
    [0.75, "#fb923c"],
    [1, "#c2410c"],
  ],
  green: [
    [0, "#ecfeff"],
    [0.25, "#a5f3fc"],
    [0.5, "#67e8f9"],
    [0.75, "#2dd4bf"],
    [1, "#0f766e"],
  ],
};

const timeGroupValues = computed(() => {
  if (!selectedGroupColumn.value) return [];
  return Array.from(
    new Set(
      props.rows
        .map((row) => normalizeCategory(row[selectedGroupColumn.value]))
        .filter((value) => value && value !== "Пусто"),
    ),
  ).slice(0, 20);
});

watch(
  featureCandidates,
  (columns) => {
    enabledFeatureColumns.value = enabledFeatureColumns.value.filter((column) => columns.includes(column));
    if (!enabledFeatureColumns.value.length) {
      enabledFeatureColumns.value = [...columns];
    }
  },
  { immediate: true },
);

watch(
  numericScenarioCandidates,
  (columns) => {
    const next = [...selectedFeatureSlots.value];
    for (let index = 0; index < 4; index += 1) {
      if (!next[index] || !columns.includes(next[index])) {
        next[index] = columns[index] || "";
      }
    }
    selectedFeatureSlots.value = next;
  },
  { immediate: true },
);

watch(
  contourAxisColumns,
  (columns) => {
    if (!columns.includes(contourXFeature.value)) contourXFeature.value = columns[0] || "";
    if (!columns.includes(contourYFeature.value) || contourYFeature.value === contourXFeature.value) {
      contourYFeature.value = columns[1] || columns[0] || "";
    }
  },
  { immediate: true },
);

watch(
  contourSliceColumns,
  (columns) => {
    if (!columns.includes(selectedContourSliceColumn.value)) {
      selectedContourSliceColumn.value = "";
      selectedContourSliceValue.value = "";
    }
  },
  { immediate: true },
);

watch(
  contourSliceValues,
  (values) => {
    if (!values.includes(selectedContourSliceValue.value)) {
      selectedContourSliceValue.value = "";
    }
  },
  { immediate: true },
);

watch(
  timeAxisColumns,
  (columns) => {
    if (!columns.includes(selectedTimeColumn.value)) {
      selectedTimeColumn.value = columns[0] || "";
    }
  },
  { immediate: true },
);

watch(
  timeGroupColumns,
  (columns) => {
    if (!columns.includes(selectedGroupColumn.value)) {
      selectedGroupColumn.value = "";
      selectedGroupValues.value = [];
    }
  },
  { immediate: true },
);

watch(
  timeGroupValues,
  (values) => {
    selectedGroupValues.value = selectedGroupValues.value.filter((value) => values.includes(value));
  },
  { immediate: true },
);

async function loadSavedModels() {
  savedModels.value = await listSavedForecastModels(props.dataset.id);
  const activeModel = savedModels.value.find((model) => model.is_active);
  if (activeModel) {
    selectedSavedModelId.value = activeModel.id;
  } else if (!savedModels.value.some((model) => model.id === selectedSavedModelId.value)) {
    selectedSavedModelId.value = null;
  }
}

watch(
  () => props.dataset.id,
  async () => {
    await loadSavedModels();
  },
  { immediate: true },
);

function formatMetric(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 }).format(value);
}

function formatOneDecimal(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function toNumeric(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const normalized = String(value).trim().replace(/\s/g, "").replace(",", ".");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

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
  const matchRu = trimmed.match(/^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})/);
  if (matchRu) {
    const day = Number(matchRu[1]);
    const month = Number(matchRu[2]) - 1;
    const yearRaw = Number(matchRu[3]);
    const year = yearRaw < 100 ? 2000 + yearRaw : yearRaw;
    const parsed = new Date(Date.UTC(year, month, day));
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  const matchIso = trimmed.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (matchIso) {
    const parsed = new Date(`${matchIso[1]}-${matchIso[2]}-${matchIso[3]}T00:00:00Z`);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  const parsed = new Date(trimmed);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function extractYearValue(value) {
  const numeric = toNumeric(value);
  if (numeric !== null) {
    if (numeric >= 1900 && numeric <= 2500 && Number.isInteger(numeric)) return numeric;
    if (isPossibleExcelDate(numeric)) return excelSerialToDate(numeric).getUTCFullYear();
  }
  const parsedDate = parseDateFromString(value);
  return parsedDate ? parsedDate.getUTCFullYear() : null;
}

function toggleFeature(column) {
  if (enabledFeatureColumns.value.includes(column)) {
    enabledFeatureColumns.value = enabledFeatureColumns.value.filter((item) => item !== column);
    return;
  }
  enabledFeatureColumns.value = [...enabledFeatureColumns.value, column];
}

function toggleGroupValue(value) {
  if (selectedGroupValues.value.includes(value)) {
    selectedGroupValues.value = selectedGroupValues.value.filter((item) => item !== value);
    return;
  }
  selectedGroupValues.value = [...selectedGroupValues.value, value];
}

async function loadDatasetPredictions() {
  if (!modelFeatureColumns.value.length) return;
  datasetPredictionResponse.value = await predictForecast(props.dataset.id, {
    feature_columns: modelFeatureColumns.value,
    rows: props.rows,
    test_fraction: testFraction.value,
    random_seed: randomSeed.value,
  });
}

async function handleTrain() {
  if (!modelFeatureColumns.value.length) return;
  training.value = true;
  try {
    modelInfo.value = await trainForecastModel(props.dataset.id, {
      feature_columns: modelFeatureColumns.value,
      test_fraction: testFraction.value,
      random_seed: randomSeed.value,
    });
    await loadDatasetPredictions();
  } finally {
    training.value = false;
  }
}

async function handleSaveModel() {
  if (!modelInfo.value || !modelFeatureColumns.value.length) return;
  savingModel.value = true;
  try {
    const model = await saveForecastModel(props.dataset.id, {
      name: `${props.dataset.name} v${props.dataset.storage_version}`,
      feature_columns: modelFeatureColumns.value,
      test_fraction: testFraction.value,
      random_seed: randomSeed.value,
    });
    await loadSavedModels();
    selectedSavedModelId.value = model.id;
  } finally {
    savingModel.value = false;
  }
}

async function handlePredict() {
  if (!modelFeatureColumns.value.length && !selectedSavedModelId.value) return;
  predicting.value = true;
  try {
    const response = await predictForecast(props.dataset.id, {
      model_id: selectedSavedModelId.value || null,
      feature_columns: modelFeatureColumns.value,
      rows: props.rows,
      x_feature: contourXFeature.value || null,
      y_feature: contourYFeature.value || null,
      slice_overrides: contourSliceOverrides.value,
      contour_resolution: 22,
      test_fraction: testFraction.value,
      random_seed: randomSeed.value,
    });

    predictionResponse.value = response;
    datasetPredictionResponse.value = response;
    modelInfo.value = {
      dataset: response.dataset,
      target_column: response.target_column,
      feature_columns: response.feature_columns,
      metrics: response.metrics,
      numeric_feature_columns: [...numericScenarioCandidates.value],
    };
  } finally {
    predicting.value = false;
  }
}

async function refreshContour() {
  if (!modelInfo.value && !selectedSavedModelId.value) return;
  const response = await predictForecast(props.dataset.id, {
    model_id: selectedSavedModelId.value || null,
    feature_columns: modelFeatureColumns.value,
    rows: props.rows,
    x_feature: contourXFeature.value || null,
    y_feature: contourYFeature.value || null,
    slice_overrides: contourSliceOverrides.value,
    contour_resolution: 22,
    test_fraction: testFraction.value,
    random_seed: randomSeed.value,
  });
  predictionResponse.value = response;
}

watch(
  () => [
    contourXFeature.value,
    contourYFeature.value,
    selectedContourSliceColumn.value,
    selectedContourSliceValue.value,
    testFraction.value,
    randomSeed.value,
  ],
  async () => {
    await refreshContour();
  },
);

const contourData = computed(() => {
  if (!predictionResponse.value?.contour) return [];
  const colorscale = contourColorScales[selectedContourPalette.value] || contourColorScales.softBlue;
  return [
    {
      type: "contour",
      x: predictionResponse.value.contour.x_values,
      y: predictionResponse.value.contour.y_values,
      z: predictionResponse.value.contour.z_values,
      colorscale,
      contours: {
        coloring: "heatmap",
        showlabels: true,
        labelfont: {
          size: 11,
          color: "#e5e7eb",
        },
      },
      line: { width: 1.1, color: "rgba(229, 231, 235, 0.22)" },
      colorbar: {
        title: targetColumn.value || "Target",
        tickcolor: "rgba(255,255,255,0.12)",
        tickfont: {
          color: "#cbd5e1",
        },
        titlefont: {
          color: "#e5e7eb",
        },
      },
      hovertemplate: `${contourXFeature.value || "X"}: %{x}<br>${contourYFeature.value || "Y"}: %{y}<br>${targetColumn.value || "Target"}: %{z:.2f}<extra></extra>`,
    },
  ];
});

const contourLayout = computed(() => ({
  height: 420,
  margin: { l: 60, r: 20, t: 10, b: 60 },
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { title: contourXFeature.value || "Ось X" },
  yaxis: { title: contourYFeature.value || "Ось Y" },
}));

const datasetRowsWithPredictions = computed(() =>
  props.rows.map((row, index) => ({
    row,
    predicted: datasetPredictionResponse.value?.predictions?.[index] ?? null,
  })),
);

const availableYears = computed(() => {
  if (!selectedTimeColumn.value) return [];
  return Array.from(
    new Set(
      props.rows
        .map((row) => extractYearValue(row[selectedTimeColumn.value]))
        .filter((year) => year !== null),
    ),
  ).sort((a, b) => a - b);
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
      selectedYearMin.value = years[0];
      selectedYearMax.value = years[years.length - 1];
    }
  },
  { immediate: true },
);

const timeSeriesGroups = computed(() => {
  if (!targetColumn.value || !selectedTimeColumn.value) return [];

  const aggregateMap = new Map();
  for (const item of datasetRowsWithPredictions.value) {
    const year = extractYearValue(item.row[selectedTimeColumn.value]);
    const actual = toNumeric(item.row[targetColumn.value]);
    const predicted = toNumeric(item.predicted);
    if (year === null) continue;
    if (selectedYearMin.value !== null && year < selectedYearMin.value) continue;
    if (selectedYearMax.value !== null && year > selectedYearMax.value) continue;

    const group = selectedGroupColumn.value ? normalizeCategory(item.row[selectedGroupColumn.value]) : "Все";
    if (selectedGroupColumn.value && selectedGroupValues.value.length && !selectedGroupValues.value.includes(group)) continue;

    const key = `${group}__${year}`;
    const current = aggregateMap.get(key) || { group, year, actualValues: [], predictedValues: [] };
    if (actual !== null && actual !== 0) current.actualValues.push(actual);
    if (predicted !== null && predicted !== 0) current.predictedValues.push(predicted);
    aggregateMap.set(key, current);
  }

  const groupMap = new Map();
  for (const item of aggregateMap.values()) {
    const current = groupMap.get(item.group) || [];
    current.push({
      year: item.year,
      actual: item.actualValues.length ? item.actualValues.reduce((sum, value) => sum + value, 0) / item.actualValues.length : null,
      predicted: item.predictedValues.length ? item.predictedValues.reduce((sum, value) => sum + value, 0) / item.predictedValues.length : null,
    });
    groupMap.set(item.group, current);
  }

  return Array.from(groupMap.entries()).map(([group, points]) => ({
    group,
    points: points.sort((a, b) => a.year - b.year),
  }));
});

const timeSeriesData = computed(() => {
  const palette = ["#1d4ed8", "#0f766e", "#ea580c", "#7c3aed", "#dc2626", "#0891b2"];
  const traces = [];

  timeSeriesGroups.value.forEach((groupItem, index) => {
    const color = palette[index % palette.length];
    const actualPoints = groupItem.points.filter((point) => point.actual !== null);
    const predictedPoints = groupItem.points.filter((point) => point.predicted !== null);

    if (actualPoints.length) {
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: selectedGroupColumn.value ? `${groupItem.group}: факт` : "Факт",
        x: actualPoints.map((point) => point.year),
        y: actualPoints.map((point) => point.actual),
        line: { color, width: 3 },
        marker: { color, size: 7 },
        hovertemplate: "Год: %{x}<br>Среднее: %{y:.1f}<extra></extra>",
      });
    }

    if (predictedPoints.length) {
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: selectedGroupColumn.value ? `${groupItem.group}: прогноз` : "Прогноз модели",
        x: predictedPoints.map((point) => point.year),
        y: predictedPoints.map((point) => point.predicted),
        line: { color, width: 2, dash: "dash" },
        marker: { color, size: 6, symbol: "diamond" },
        hovertemplate: "Год: %{x}<br>Прогноз: %{y:.1f}<extra></extra>",
      });
    }
  });

  return traces;
});

const timeSeriesLayout = computed(() => ({
  margin: { l: 60, r: 20, t: 10, b: 60 },
  height: 430,
  autosize: true,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    title: selectedTimeColumn.value || "Ось X",
    type: "linear",
  },
  yaxis: {
    title: targetColumn.value || "Target",
  },
  legend: { orientation: "h", y: 1.1 },
}));

const timeSummaryTable = computed(() => {
  const headers = availableYears.value
    .filter((year) => (selectedYearMin.value === null || year >= selectedYearMin.value) && (selectedYearMax.value === null || year <= selectedYearMax.value))
    .map(String);

  const rows = timeSeriesGroups.value.map((groupItem) => {
    const values = {};
    groupItem.points.forEach((point) => {
      if (point.actual !== null) {
        values[String(point.year)] = formatOneDecimal(point.actual);
      }
    });
    return { label: groupItem.group, values };
  });

  return { headers, rows };
});
</script>
