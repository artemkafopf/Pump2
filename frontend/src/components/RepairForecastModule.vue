<template>
  <section class="repair-forecast-grid">
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Прогноз ремонтов</h2>
          <p>
            Модуль использует текущую настроенную модель CatBoost из <strong>Факт ЭПУ</strong> и выбранный набор
            <strong>Сводпрогноз</strong> как источник входных данных.
          </p>
        </div>
      </div>

      <div class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Модельный датасет</span>
          <strong>{{ modelDataset.name }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Источник сводпрогноза</span>
          <strong>{{ selectedSourceDataset ? sourceDatasetLabel : "Не выбран" }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Последний сохранённый расчёт</span>
          <strong>{{ latestSavedCalculationLabel }}</strong>
        </div>
        <div v-if="selectedScopeLabel" class="metric-card">
          <span class="metric-label">Выбранная категория</span>
          <strong>{{ selectedScopeLabel }}</strong>
        </div>
      </div>

      <div class="note-list">
        <p>
          Источник сводпрогноза:
          <strong>{{ sourceDatasetDetails }}</strong>
        </p>
        <p>
          Факт ЭПУ для хвостов:
          <strong>{{ tailFactDatasetLabel }}</strong>
        </p>
      </div>

      <div v-if="requestError" class="note-list">
        <p>{{ requestError }}</p>
      </div>

      <div class="control-block">
        <div class="feature-slot-grid">
          <label class="field">
            <span>Набор исходных данных</span>
            <select v-model.number="selectedSourceDatasetId">
              <option :value="null">Не выбран</option>
              <option v-for="item in sourceDatasetOptions" :key="item.id" :value="item.id">
                {{ item.name }} В· v{{ item.storage_version }}
              </option>
            </select>
          </label>

          <label class="field">
            <span>Сохранённая модель CatBoost</span>
            <select v-model="selectedModelId">
              <option :value="null">Активная / текущая конфигурация</option>
              <option v-for="model in savedModels" :key="model.id" :value="model.id">
                {{ model.name }} В· RMSE {{ formatMetric(model.metrics?.rmse) }}
              </option>
            </select>
          </label>

          <label class="field">
            <span>Базовый коэффициент до отказа</span>
            <input v-model.number="baseFailureCoefficient" type="number" step="0.1" />
          </label>
        </div>

        <div class="control-group">
          <span class="control-title">Настройка хвостов распределения</span>
          <div class="feature-slot-grid">
            <label class="field">
              <span>Функция распределения</span>
              <select v-model="tailDistribution">
                <option value="kde">KDE по фактическим данным</option>
                <option value="spline">Сплайновая аппроксимация</option>
                <option value="normal">Нормальное распределение</option>
              </select>
            </label>

            <label class="field">
              <span>Набор Факт ЭПУ для сравнения</span>
              <select v-model.number="selectedTailFactDatasetId">
                <option :value="null">Текущий модельный Факт ЭПУ</option>
                <option v-for="item in tailFactDatasetOptions" :key="`tail-fact-${item.id}`" :value="item.id">
                  {{ item.name }} В· v{{ item.storage_version }}
                </option>
              </select>
            </label>

            <label class="field">
              <span>Min обрезка</span>
              <input v-model="tailClipMinInput" type="number" step="0.1" placeholder="Авто" />
            </label>

            <label class="field">
              <span>Max обрезка</span>
              <input v-model="tailClipMaxInput" type="number" step="0.1" placeholder="Авто" />
            </label>

            <label class="field">
              <span>Мин. размер группы</span>
              <input v-model.number="minGroupSize" type="number" min="2" step="1" />
            </label>

            <label class="field">
              <span>Max iter</span>
              <input v-model.number="maxSamplingIter" type="number" min="50" step="50" />
            </label>

            <label class="field">
              <span>Random state</span>
              <input v-model="randomStateInput" type="number" step="1" placeholder="42" />
            </label>
          </div>

          <label class="field checkbox-field">
            <span>Подгонять под фактические данные Факт ЭПУ</span>
            <input v-model="tailFitToFact" type="checkbox" />
          </label>

          <div class="prediction-actions">
            <button type="button" class="file-button" :disabled="tailPreviewLoading" @click="refreshTailPreview">
              {{ tailPreviewLoading ? "Обновление..." : "Обновить диаграммы" }}
            </button>
          </div>

          <div v-if="tailPreview?.image" class="repair-diagnostic-image-wrap">
            <img :src="tailPreview.image" alt="Диагностика хвостового распределения" class="repair-diagnostic-image" />
            <p class="plot-caption">
              Гистограмма показывает фактические значения ННО из выбранного набора «Факт ЭПУ», синяя линия — выбранную функцию
              распределения с текущими настройками, полупрозрачные столбцы — сгенерированные хвостовые значения, зелёные кресты —
              фактически сэмплированные точки хвоста, которые использует алгоритм.
            </p>
            <p v-if="tailPreview?.notes?.length" class="plot-caption">{{ tailPreview.notes.join(" ") }}</p>
          </div>
        </div>

        <div v-if="repairForecast?.missing_feature_columns?.length" class="control-group">
          <span class="control-title">Ручные значения для недостающих признаков</span>
          <div class="feature-slot-grid">
            <label
              v-for="column in repairForecast.missing_feature_columns"
              :key="`manual-feature-${column}`"
              class="field"
            >
              <span>{{ column }}</span>
              <input v-model="manualFeatureValues[column]" type="text" :placeholder="`Значение для ${column}`" />
            </label>
          </div>
        </div>

        <div class="prediction-actions">
          <button type="button" class="primary-button" :disabled="loading || !selectedSourceDatasetId" @click="runForecast">
            {{ loading ? "Расчёт..." : "Запустить новый расчёт" }}
          </button>
          <button type="button" class="file-button" :disabled="savingCalculation || !repairForecast" @click="handleSaveCalculation">
            {{ savingCalculation ? "Сохранение..." : "Сохранить расчёт" }}
          </button>
          <button
            v-if="selectedScopeKey"
            type="button"
            class="file-button"
            :disabled="loading || savingCalculation"
            @click="selectedScopeKey = ''"
          >
            Сбросить категорию
          </button>
        </div>

        <div v-if="savedCalculations.length" class="control-group">
          <span class="control-title">Сохранённые версии расчёта</span>
          <label class="field">
            <span>Выберите сохранённый расчёт</span>
            <select v-model.number="selectedSavedCalculationId" @change="applySavedCalculation">
              <option :value="null">Последняя сохранённая версия</option>
              <option v-for="item in savedCalculations" :key="item.id" :value="item.id">
                {{ item.name }} В· {{ formatDateTime(item.created_at) }}
              </option>
            </select>
          </label>
        </div>
      </div>
    </section>

    <section v-if="repairForecast" class="panel">
      <div class="panel-header">
        <div>
          <h2>Параметры расчёта</h2>
          <p>Отображается текущий результат: либо только что рассчитанный, либо последняя сохранённая версия.</p>
        </div>
      </div>

      <div class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Период</span>
          <strong>{{ forecastStartLabel }} - {{ forecastEndLabel }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Использовано признаков</span>
          <strong>{{ repairForecast.used_feature_columns.length }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Скважин</span>
          <strong>{{ repairForecast.rows.length }}</strong>
        </div>
      </div>
    </section>

    <section v-if="repairForecast" class="panel">
      <div class="panel-header">
        <div>
          <h2>Отказы по месяцам</h2>
          <p>При клике по строке таблицы диаграмма показывает сумму отказов для выбранного узла иерархии.</p>
        </div>
      </div>

      <PlotlyChart :data="monthlySummaryChartData" :layout="monthlySummaryChartLayout" />
    </section>

    <section v-if="repairForecast" class="panel">
      <div class="panel-header">
        <div>
          <h2>Таблица прогноза</h2>
          <p>Коды: 0 - отказ, 1 - в работе. Столбцы можно свернуть по дням, месяцам и годам.</p>
        </div>
      </div>

      <div class="control-block">
        <div class="control-group">
          <span class="control-title">Свернуть столбцы</span>
          <div class="button-group">
            <button
              v-for="option in groupingOptions"
              :key="option.value"
              type="button"
              class="choice-button"
              :class="{ active: groupingMode === option.value }"
              @click="groupingMode = option.value"
            >
              {{ option.label }}
            </button>
          </div>
        </div>
      </div>

      <div class="repair-scrollbar-bar">
        <span class="repair-scrollbar-label">Горизонтальная прокрутка</span>
        <input
          v-model.number="horizontalScrollValue"
          class="repair-scrollbar-range"
          type="range"
          min="0"
          :max="horizontalScrollMax"
          step="1"
          @input="handleHorizontalSlider"
          @change="handleHorizontalSlider"
        />
      </div>

      <div ref="tableViewportRef" class="table-wrap repair-table-viewport" @scroll="handleTableScroll">
        <div ref="tableContentRef" class="repair-table-content">
          <table ref="tableRef" class="data-table repair-table">
            <thead>
              <tr>
                <th>Категория</th>
                <th>Участок недр</th>
                <th>Куст</th>
                <th>Скважина</th>
                <th>Уровень</th>
                <th>Отказов</th>
                <th>Средний прогноз ННО</th>
                <th>Факт ННО</th>
                <th v-for="column in groupedDateColumns" :key="`repair-date-${column.key}`" class="repair-date-column">
                  {{ column.label }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in hierarchyRows"
                :key="row.key"
                class="repair-hierarchy-row"
                :class="{
                  'repair-row-selected': selectedScopeKey === row.key,
                  'repair-row-group': row.nodeType !== 'well',
                }"
                @click="selectScope(row.key)"
              >
                <td>{{ row.categoryLabel }}</td>
                <td>
                  <div class="repair-tree-cell" :style="{ paddingLeft: `${row.depth * 18}px` }">
                    <button
                      v-if="row.nodeType !== 'well'"
                      type="button"
                      class="repair-tree-toggle"
                      @click.stop="toggleExpanded(row.key)"
                    >
                      {{ expandedKeys.has(row.key) ? "−" : "+" }}
                    </button>
                    <span>{{ row.license_area || "-" }}</span>
                  </div>
                </td>
                <td>{{ row.nodeType === "well" ? row.cluster_name || "-" : row.nodeType === "cluster" ? row.cluster_name || "-" : "-" }}</td>
                <td>{{ row.nodeType === "well" ? row.well_name : "-" }}</td>
                <td>{{ levelLabel(row.nodeType) }}</td>
                <td>{{ row.failureCount }}</td>
                <td>{{ formatOneDecimal(row.predicted_nno) }}</td>
                <td>{{ formatOneDecimal(row.actual_nno) }}</td>
                <td
                  v-for="column in groupedDateColumns"
                  :key="`repair-cell-${row.key}-${column.key}`"
                  class="repair-status-cell"
                  :class="repairStatusClass(row.groupedStatuses[column.key])"
                >
                  {{ row.groupedStatuses[column.key] }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import PlotlyChart from "./PlotlyChart.vue";
import {
  calculateRepairForecast,
  fetchRepairForecastCalculation,
  fetchLatestRepairForecastCalculation,
  listRepairForecastCalculations,
  listSavedForecastModels,
  previewRepairForecastTail,
  saveRepairForecastCalculation,
} from "../services/api";

const props = defineProps({
  modelDataset: { type: Object, required: true },
  sourceDataset: { type: Object, default: null },
  sourceDatasetOptions: { type: Array, default: () => [] },
  tailFactDatasetOptions: { type: Array, default: () => [] },
});

const loading = ref(false);
const savingCalculation = ref(false);
const repairForecast = ref(null);
const savedModels = ref([]);
const savedCalculations = ref([]);
const latestSavedCalculation = ref(null);
const selectedSavedCalculationId = ref(null);
const selectedModelId = ref(null);
const selectedSourceDatasetId = ref(null);
const selectedTailFactDatasetId = ref(null);
const baseFailureCoefficient = ref(1.1);
const tailDistribution = ref("kde");
const tailClipMinInput = ref("");
const tailClipMaxInput = ref("");
const tailFitToFact = ref(true);
const minGroupSize = ref(20);
const maxSamplingIter = ref(1000);
const randomStateInput = ref("42");
const manualFeatureValues = ref({});
const groupingMode = ref("month");
const requestError = ref("");
const selectedScopeKey = ref("");
const expandedKeys = ref(new Set());
const tableViewportRef = ref(null);
const tableContentRef = ref(null);
const tableRef = ref(null);
const horizontalScrollValue = ref(0);
const horizontalScrollMax = ref(0);
const tailPreview = ref(null);
const tailPreviewLoading = ref(false);

const groupingOptions = [
  { value: "day", label: "День" },
  { value: "month", label: "Месяц" },
  { value: "year", label: "Год" },
];

const sourceDatasetOptions = computed(() => props.sourceDatasetOptions || []);
const tailFactDatasetOptions = computed(() => props.tailFactDatasetOptions || []);
const selectedSourceDataset = computed(
  () => sourceDatasetOptions.value.find((item) => item.id === selectedSourceDatasetId.value) || null,
);
const selectedTailFactDataset = computed(
  () => tailFactDatasetOptions.value.find((item) => item.id === selectedTailFactDatasetId.value) || null,
);

const sourceDatasetLabel = computed(() => {
  if (!selectedSourceDataset.value) return "Не выбран";
  return `${selectedSourceDataset.value.name} В· v${selectedSourceDataset.value.storage_version}`;
});

const sourceDatasetDetails = computed(() => {
  if (!selectedSourceDataset.value) {
    return "выберите датасет в разделе «Сводпрогноз» модуля «Хранение данных»";
  }
  return `${selectedSourceDataset.value.name} (раздел: ${selectedSourceDataset.value.storage_section}, версия: ${selectedSourceDataset.value.storage_version})`;
});

const tailFactDatasetLabel = computed(() => {
  if (!selectedTailFactDataset.value) return `${props.modelDataset.name} В· v${props.modelDataset.storage_version}`;
  return `${selectedTailFactDataset.value.name} В· v${selectedTailFactDataset.value.storage_version}`;
});

const latestSavedCalculationLabel = computed(() => {
  if (!latestSavedCalculation.value?.summary) return "Нет сохранённых версий";
  return `${latestSavedCalculation.value.summary.name} В· ${formatDateTime(latestSavedCalculation.value.summary.created_at)}`;
});

const forecastStartLabel = computed(() => {
  if (!repairForecast.value?.start_date) return "-";
  return new Intl.DateTimeFormat("ru-RU").format(new Date(`${repairForecast.value.start_date}T00:00:00`));
});

const forecastEndLabel = computed(() => {
  if (!repairForecast.value?.end_date) return "-";
  return new Intl.DateTimeFormat("ru-RU").format(new Date(`${repairForecast.value.end_date}T00:00:00`));
});

function formatMetric(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 }).format(value);
}

function formatOneDecimal(value) {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function formatDateTime(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function parseOptionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseOptionalInteger(value, fallback = 42) {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function applySavedSettings(settings) {
  if (!settings) return;
  selectedTailFactDatasetId.value = settings.tail_fact_dataset_id ?? selectedTailFactDatasetId.value;
  baseFailureCoefficient.value = settings.base_failure_coefficient ?? 1.1;
  tailDistribution.value = settings.tail_distribution || "kde";
  tailClipMinInput.value = settings.tail_clip_min ?? "";
  tailClipMaxInput.value = settings.tail_clip_max ?? "";
  tailFitToFact.value = settings.tail_fit_to_fact ?? true;
  minGroupSize.value = settings.min_group_size ?? 20;
  maxSamplingIter.value = settings.max_sampling_iter ?? 1000;
  randomStateInput.value = settings.random_state ?? "42";
  manualFeatureValues.value = { ...(settings.manual_feature_values || {}) };
}

async function refreshTailPreview() {
  if (!props.modelDataset?.id) return;
  tailPreviewLoading.value = true;
  try {
    tailPreview.value = await previewRepairForecastTail(props.modelDataset.id, {
      tail_fact_dataset_id: selectedTailFactDatasetId.value,
      random_state: parseOptionalInteger(randomStateInput.value, 42),
      min_group_size: minGroupSize.value,
      max_sampling_iter: maxSamplingIter.value,
      tail_distribution: tailDistribution.value,
      tail_clip_min: parseOptionalNumber(tailClipMinInput.value),
      tail_clip_max: parseOptionalNumber(tailClipMaxInput.value),
      tail_fit_to_fact: tailFitToFact.value,
    });
  } catch (error) {
    tailPreview.value = {
      image: null,
      notes: [error?.message || "Не удалось построить превью хвостового распределения."],
    };
  } finally {
    tailPreviewLoading.value = false;
  }
}

function repairStatusClass(value) {
  return {
    "repair-status--failure": value === 0,
    "repair-status--working": value === 1,
  };
}

function levelLabel(nodeType) {
  return {
    license: "Участок недр",
    cluster: "Куст",
    well: "Скважина",
  }[nodeType] || nodeType;
}

function aggregateStatuses(rows, columns) {
  const groupedStatuses = {};
  columns.forEach((column) => {
    const values = rows.map((row) => row.groupedStatuses[column.key] ?? 1);
    groupedStatuses[column.key] = values.includes(0) ? 0 : 1;
  });
  return groupedStatuses;
}

async function loadSavedModels() {
  savedModels.value = await listSavedForecastModels(props.modelDataset.id);
  const active = savedModels.value.find((item) => item.is_active);
  selectedModelId.value = active?.id ?? null;
}

async function loadSavedCalculations() {
  savedCalculations.value = await listRepairForecastCalculations(props.modelDataset.id);
  latestSavedCalculation.value = await fetchLatestRepairForecastCalculation(props.modelDataset.id);
  if (latestSavedCalculation.value?.result) {
    selectedSourceDatasetId.value = latestSavedCalculation.value.summary.source_dataset_id ?? selectedSourceDatasetId.value;
    selectedModelId.value = latestSavedCalculation.value.summary.trained_model_id ?? selectedModelId.value;
    applySavedSettings(latestSavedCalculation.value.settings);
    repairForecast.value = latestSavedCalculation.value.result;
    syncForecastStateFromResult();
  }
}

function syncForecastStateFromResult() {
  if (!repairForecast.value) return;
  const nextManual = {};
  (repairForecast.value.missing_feature_columns || []).forEach((column) => {
    nextManual[column] = manualFeatureValues.value[column] ?? "";
  });
  manualFeatureValues.value = nextManual;

  const initialExpanded = new Set();
  groupedRows.value.forEach((row) => {
    if (row.license_area) {
      initialExpanded.add(`license:${row.license_area || "Без УН"}`);
    }
  });
  expandedKeys.value = initialExpanded;
  selectedScopeKey.value = "";
}

const groupedDateColumns = computed(() => {
  const sourceDates = repairForecast.value?.dates || [];
  const groups = new Map();

  sourceDates.forEach((iso) => {
    const value = new Date(`${iso}T00:00:00`);
    let key = iso;
    let label = new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "2-digit" }).format(value);

    if (groupingMode.value === "month") {
      key = `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}`;
      label = new Intl.DateTimeFormat("ru-RU", { month: "short", year: "numeric" }).format(value);
    } else if (groupingMode.value === "year") {
      key = String(value.getFullYear());
      label = String(value.getFullYear());
    }

    if (!groups.has(key)) {
      groups.set(key, { key, label, dates: [] });
    }
    groups.get(key).dates.push(iso);
  });

  return Array.from(groups.values());
});

const groupedRows = computed(() =>
  (repairForecast.value?.rows || []).map((row) => {
    const statusByDate = new Map();
    (repairForecast.value?.dates || []).forEach((dateValue, index) => {
      statusByDate.set(dateValue, row.statuses[index] ?? 1);
    });

    const groupedStatuses = {};
    groupedDateColumns.value.forEach((column) => {
      const values = column.dates.map((item) => statusByDate.get(item) ?? 1);
      groupedStatuses[column.key] = values.includes(0) ? 0 : 1;
    });

    return {
      ...row,
      failureCount: (row.event_dates || []).length,
      groupedStatuses,
    };
  }),
);

const hierarchyRows = computed(() => {
  const rows = [];
  const grouped = groupedRows.value;
  const columns = groupedDateColumns.value;
  const byLicense = new Map();

  grouped.forEach((row) => {
    const licenseKey = row.license_area || "Без УН";
    const clusterKey = row.cluster_name || "Без куста";
    if (!byLicense.has(licenseKey)) {
      byLicense.set(licenseKey, new Map());
    }
    const clusterMap = byLicense.get(licenseKey);
    if (!clusterMap.has(clusterKey)) {
      clusterMap.set(clusterKey, []);
    }
    clusterMap.get(clusterKey).push(row);
  });

  Array.from(byLicense.entries())
    .sort((a, b) => a[0].localeCompare(b[0], "ru"))
    .forEach(([licenseKey, clusterMap]) => {
      const licenseRows = Array.from(clusterMap.values()).flat();
      const licenseNodeKey = `license:${licenseKey}`;
      rows.push({
        key: licenseNodeKey,
        nodeType: "license",
        depth: 0,
        categoryLabel: summarizeCategories(licenseRows),
        license_area: licenseKey,
        cluster_name: null,
        well_name: null,
        predicted_nno: averageValue(licenseRows.map((item) => item.predicted_nno)),
        actual_nno: averageValue(licenseRows.map((item) => item.actual_nno)),
        failureCount: licenseRows.reduce((sum, item) => sum + item.failureCount, 0),
        groupedStatuses: aggregateStatuses(licenseRows, columns),
      });

      if (!expandedKeys.value.has(licenseNodeKey)) {
        return;
      }

      Array.from(clusterMap.entries())
        .sort((a, b) => a[0].localeCompare(b[0], "ru"))
        .forEach(([clusterKey, clusterRows]) => {
          const clusterNodeKey = `cluster:${licenseKey}:${clusterKey}`;
          rows.push({
            key: clusterNodeKey,
            nodeType: "cluster",
            depth: 1,
            categoryLabel: summarizeCategories(clusterRows),
            license_area: licenseKey,
            cluster_name: clusterKey,
            well_name: null,
            predicted_nno: averageValue(clusterRows.map((item) => item.predicted_nno)),
            actual_nno: averageValue(clusterRows.map((item) => item.actual_nno)),
            failureCount: clusterRows.reduce((sum, item) => sum + item.failureCount, 0),
            groupedStatuses: aggregateStatuses(clusterRows, columns),
          });

          if (!expandedKeys.value.has(clusterNodeKey)) {
            return;
          }

          clusterRows
            .slice()
            .sort((a, b) => (a.well_name || "").localeCompare(b.well_name || "", "ru"))
            .forEach((row) => {
              rows.push({
                ...row,
                key: `well:${licenseKey}:${clusterKey}:${row.well_name}`,
                nodeType: "well",
                depth: 2,
                categoryLabel: row.category,
              });
            });
        });
    });

  return rows;
});

function averageValue(values) {
  const filtered = values.filter((value) => value !== null && value !== undefined);
  if (!filtered.length) return null;
  return filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
}

function summarizeCategories(rows) {
  const categories = Array.from(new Set(rows.map((row) => row.category).filter(Boolean)));
  if (!categories.length) return "-";
  return categories.join(", ");
}

const selectedScopeRows = computed(() => {
  if (!selectedScopeKey.value) {
    return groupedRows.value;
  }
  const [nodeType, licenseArea = "", clusterName = "", wellName = ""] = selectedScopeKey.value.split(":");
  if (nodeType === "license") {
    return groupedRows.value.filter((row) => (row.license_area || "Без УН") === licenseArea);
  }
  if (nodeType === "cluster") {
    return groupedRows.value.filter(
      (row) =>
        (row.license_area || "Без УН") === licenseArea &&
        (row.cluster_name || "Без куста") === clusterName,
    );
  }
  if (nodeType === "well") {
    return groupedRows.value.filter(
      (row) =>
        (row.license_area || "Без УН") === licenseArea &&
        (row.cluster_name || "Без куста") === clusterName &&
        (row.well_name || "") === wellName,
    );
  }
  return groupedRows.value;
});

const selectedScopeLabel = computed(() => {
  if (!selectedScopeKey.value) return "";
  const row = hierarchyRows.value.find((item) => item.key === selectedScopeKey.value);
  if (!row) return "";
  if (row.nodeType === "license") return `УН: ${row.license_area}`;
  if (row.nodeType === "cluster") return `УН ${row.license_area} / куст ${row.cluster_name}`;
  return `УН ${row.license_area} / куст ${row.cluster_name} / скважина ${row.well_name}`;
});

const monthlySummaryChartData = computed(() => {
  const bucket = new Map();
  selectedScopeRows.value.forEach((row) => {
    (row.event_dates || []).forEach((isoDate) => {
      const value = new Date(`${isoDate}T00:00:00`);
      if (Number.isNaN(value.getTime())) return;
      const monthKey = `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}`;
      bucket.set(monthKey, (bucket.get(monthKey) || 0) + 1);
    });
  });

  const months = Array.from(bucket.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  return [
    {
      type: "bar",
      x: months.map(([month]) => month),
      y: months.map(([, count]) => count),
      text: months.map(([, count]) => String(count)),
      textposition: "outside",
      cliponaxis: false,
      marker: {
        color: "#60a5fa",
        line: { color: "#93c5fd", width: 1 },
      },
      hovertemplate: "Месяц: %{x}<br>Отказов: %{y}<extra></extra>",
    },
  ];
});

const monthlySummaryChartLayout = computed(() => ({
  height: 320,
  margin: { l: 60, r: 20, t: 16, b: 80 },
  xaxis: { title: "Месяц", tickangle: -35 },
  yaxis: { title: "Отказы, шт" },
}));

async function syncScrollWidths() {
  await nextTick();
  if (!tableViewportRef.value || !tableContentRef.value || !tableRef.value) return;
  const width = Math.max(tableRef.value.scrollWidth || 0, tableContentRef.value.scrollWidth || 0);
  horizontalScrollMax.value = Math.max(width - tableViewportRef.value.clientWidth, 0);
  horizontalScrollValue.value = Math.min(tableViewportRef.value.scrollLeft || horizontalScrollValue.value, horizontalScrollMax.value);
}

function handleHorizontalSlider() {
  horizontalScrollValue.value = Math.min(Number(horizontalScrollValue.value) || 0, horizontalScrollMax.value);
  if (tableViewportRef.value) {
    tableViewportRef.value.scrollLeft = horizontalScrollValue.value;
  }
}

function handleTableScroll() {
  if (!tableViewportRef.value) return;
  horizontalScrollValue.value = Math.min(tableViewportRef.value.scrollLeft || 0, horizontalScrollMax.value);
}

function toggleExpanded(key) {
  const next = new Set(expandedKeys.value);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  expandedKeys.value = next;
}

function selectScope(key) {
  selectedScopeKey.value = selectedScopeKey.value === key ? "" : key;
}

async function runForecast() {
  if (!selectedSourceDatasetId.value || loading.value) return;
  loading.value = true;
  requestError.value = "";

  try {
    repairForecast.value = await calculateRepairForecast(props.modelDataset.id, {
      source_dataset_id: selectedSourceDatasetId.value,
      model_id: selectedModelId.value,
      tail_fact_dataset_id: selectedTailFactDatasetId.value,
      base_failure_coefficient: baseFailureCoefficient.value,
      manual_feature_values: manualFeatureValues.value,
      random_state: parseOptionalInteger(randomStateInput.value, 42),
      min_group_size: minGroupSize.value,
      max_sampling_iter: maxSamplingIter.value,
      tail_distribution: tailDistribution.value,
      tail_clip_min: parseOptionalNumber(tailClipMinInput.value),
      tail_clip_max: parseOptionalNumber(tailClipMaxInput.value),
      tail_fit_to_fact: tailFitToFact.value,
    });
    syncForecastStateFromResult();
    await syncScrollWidths();
  } catch (error) {
    requestError.value = error?.message || "Не удалось рассчитать прогноз ремонтов.";
  } finally {
    loading.value = false;
  }
}

async function handleSaveCalculation() {
  if (!repairForecast.value) return;
  savingCalculation.value = true;
  requestError.value = "";

  try {
    const saved = await saveRepairForecastCalculation(props.modelDataset.id, {
      source_dataset_id: selectedSourceDatasetId.value,
      model_id: selectedModelId.value,
      tail_fact_dataset_id: selectedTailFactDatasetId.value,
      name: `Расчёт ремонтов ${sourceDatasetLabel.value}`,
      base_failure_coefficient: baseFailureCoefficient.value,
      manual_feature_values: manualFeatureValues.value,
      random_state: parseOptionalInteger(randomStateInput.value, 42),
      min_group_size: minGroupSize.value,
      max_sampling_iter: maxSamplingIter.value,
      tail_distribution: tailDistribution.value,
      tail_clip_min: parseOptionalNumber(tailClipMinInput.value),
      tail_clip_max: parseOptionalNumber(tailClipMaxInput.value),
      tail_fit_to_fact: tailFitToFact.value,
      result: repairForecast.value,
    });
    latestSavedCalculation.value = saved;
    selectedSavedCalculationId.value = saved.summary.id;
    await loadSavedCalculations();
  } catch (error) {
    requestError.value = error?.message || "Не удалось сохранить расчёт.";
  } finally {
    savingCalculation.value = false;
  }
}

async function applySavedCalculation() {
  if (!selectedSavedCalculationId.value) {
    if (latestSavedCalculation.value?.result) {
      applySavedSettings(latestSavedCalculation.value.settings);
      selectedModelId.value = latestSavedCalculation.value.summary.trained_model_id ?? selectedModelId.value;
      repairForecast.value = latestSavedCalculation.value.result;
      syncForecastStateFromResult();
      await syncScrollWidths();
    }
    return;
  }

  const detail = await fetchRepairForecastCalculation(props.modelDataset.id, selectedSavedCalculationId.value);
  selectedSourceDatasetId.value = detail.summary.source_dataset_id ?? selectedSourceDatasetId.value;
  selectedModelId.value = detail.summary.trained_model_id ?? selectedModelId.value;
  applySavedSettings(detail.settings);
  repairForecast.value = detail.result;
  latestSavedCalculation.value = latestSavedCalculation.value || detail;
  syncForecastStateFromResult();
  await syncScrollWidths();
}

watch(groupedDateColumns, async () => {
  await syncScrollWidths();
});

watch(hierarchyRows, async () => {
  await syncScrollWidths();
});

watch(
  () => props.modelDataset.id,
  async () => {
    repairForecast.value = null;
    requestError.value = "";
    selectedSavedCalculationId.value = null;
    await loadSavedModels();
    await loadSavedCalculations();
  },
);

watch(
  sourceDatasetOptions,
  (options) => {
    if (!options.length) {
      selectedSourceDatasetId.value = null;
      return;
    }
    if (!options.some((item) => item.id === selectedSourceDatasetId.value)) {
      selectedSourceDatasetId.value = props.sourceDataset?.id || options[0].id;
    }
  },
  { immediate: true },
);

watch(
  tailFactDatasetOptions,
  (options) => {
    if (!options.length) {
      selectedTailFactDatasetId.value = null;
      return;
    }
    if (selectedTailFactDatasetId.value === null) {
      return;
    }
    if (!options.some((item) => item.id === selectedTailFactDatasetId.value)) {
      selectedTailFactDatasetId.value = props.modelDataset?.id || null;
    }
  },
  { immediate: true },
);

watch(
  [
    selectedTailFactDatasetId,
    tailDistribution,
    tailClipMinInput,
    tailClipMaxInput,
    tailFitToFact,
    minGroupSize,
    maxSamplingIter,
    randomStateInput,
  ],
  async () => {
    await refreshTailPreview();
  },
);

function handleResize() {
  syncScrollWidths();
}

onMounted(async () => {
  await loadSavedModels();
  await loadSavedCalculations();
  await refreshTailPreview();
  window.addEventListener("resize", handleResize);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
});
</script>


