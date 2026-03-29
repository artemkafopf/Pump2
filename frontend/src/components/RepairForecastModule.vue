<template>
  <section class="repair-forecast-grid">
    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Прогноз ремонтов</h2>
          <p>
            Модуль рассчитывает прогноз отказов ННО до {{ forecastEndLabel }} по данным из
            <strong>Факт ЭПУ</strong>, <strong>Добыча</strong> и <strong>ГТМ</strong>.
          </p>
        </div>
      </div>

      <div class="control-block">
        <div class="feature-slot-grid">
          <label class="field">
            <span>Модель CatBoost</span>
            <select v-model="selectedModelId">
              <option :value="null">Активная / текущая конфигурация</option>
              <option v-for="model in savedModels" :key="model.id" :value="model.id">
                {{ model.name }} · RMSE {{ formatMetric(model.metrics.rmse) }}
              </option>
            </select>
          </label>

          <label class="field">
            <span>База коэффициент до отказа</span>
            <input v-model.number="baseFailureCoefficient" type="number" step="0.1" />
          </label>

          <label class="field">
            <span>Коэффициент недостижения номинала</span>
            <input v-model.number="nominalGapCoefficient" type="number" step="0.1" />
          </label>
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
          <button type="button" class="primary-button" :disabled="loading" @click="runForecast">
            {{ loading ? "Расчет..." : "Рассчитать прогноз ремонтов" }}
          </button>
        </div>
      </div>
    </section>

    <section v-if="repairForecast" class="panel">
      <div class="panel-header">
        <div>
          <h2>Источники расчета</h2>
          <p>Модуль автоматически берет активный датасет Факт ЭПУ и последние версии разделов Добыча и ГТМ.</p>
        </div>
      </div>

      <div class="overview-grid">
        <div v-for="source in repairForecast.source_datasets" :key="source.label" class="metric-card">
          <span class="metric-label">{{ source.label }}</span>
          <strong>{{ source.dataset ? `v${source.dataset.storage_version} · ${source.dataset.name}` : "Не найден" }}</strong>
        </div>
      </div>

      <div v-if="repairForecast.notes?.length" class="note-list">
        <p v-for="note in repairForecast.notes" :key="note">{{ note }}</p>
      </div>
    </section>

    <section v-if="repairForecast" class="panel">
      <div class="panel-header">
        <div>
          <h2>Таблица прогноза</h2>
          <p>Коды: 0 — отказ, 1 — в работе, 2 — событие ГТМ / смена насоса.</p>
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

      <div ref="tableViewportRef" class="table-wrap repair-table-viewport">
        <div ref="tableContentRef" class="repair-table-content" :style="tableContentStyle">
        <table ref="tableRef" class="data-table repair-table">
          <thead>
            <tr>
              <th>Категория</th>
              <th>Месторождение</th>
              <th>Участок недр</th>
              <th>Куст</th>
              <th>Скважина</th>
              <th>Прогноз ННО</th>
              <th>Факт ННО</th>
              <th>Наработка</th>
              <th
                v-for="column in groupedDateColumns"
                :key="`repair-date-${column.key}`"
                class="repair-date-column"
              >
                {{ column.label }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in groupedRows" :key="`${row.category}-${row.well_name}`">
              <td>{{ row.category }}</td>
              <td>{{ row.field_name || "—" }}</td>
              <td>{{ row.license_area || "—" }}</td>
              <td>{{ row.cluster_name || "—" }}</td>
              <td>{{ row.well_name }}</td>
              <td>{{ formatOneDecimal(row.predicted_nno) }}</td>
              <td>{{ formatOneDecimal(row.actual_nno) }}</td>
              <td>{{ formatOneDecimal(row.runtime_days) }}</td>
              <td
                v-for="column in groupedDateColumns"
                :key="`repair-cell-${row.category}-${row.well_name}-${column.key}`"
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
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { calculateRepairForecast, listSavedForecastModels } from "../services/api";

const props = defineProps({
  dataset: { type: Object, required: true },
});

const loading = ref(false);
const repairForecast = ref(null);
const savedModels = ref([]);
const selectedModelId = ref(null);
const baseFailureCoefficient = ref(1.1);
const nominalGapCoefficient = ref(-0.8);
const manualFeatureValues = ref({});
const groupingMode = ref("month");
const tableViewportRef = ref(null);
const tableContentRef = ref(null);
const tableRef = ref(null);
const horizontalScrollValue = ref(0);
const horizontalScrollMax = ref(0);

const groupingOptions = [
  { value: "day", label: "День" },
  { value: "month", label: "Месяц" },
  { value: "year", label: "Год" },
];

async function loadSavedModels() {
  savedModels.value = await listSavedForecastModels(props.dataset.id);
  const active = savedModels.value.find((item) => item.is_active);
  selectedModelId.value = active?.id ?? null;
}

function formatMetric(value) {
  if (value === null || value === undefined) return "n/a";
  return new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 }).format(value);
}

function formatOneDecimal(value) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function repairStatusClass(value) {
  return {
    "repair-status--failure": value === 0,
    "repair-status--working": value === 1,
    "repair-status--event": value === 2,
  };
}

const forecastEndLabel = computed(() => {
  if (!repairForecast.value?.end_date) return "31.12." + (new Date().getFullYear() + 1);
  const value = new Date(repairForecast.value.end_date);
  return new Intl.DateTimeFormat("ru-RU").format(value);
});

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
      groupedStatuses[column.key] = values.includes(2) ? 2 : values.includes(0) ? 0 : 1;
    });

    return {
      ...row,
      groupedStatuses,
    };
  }),
);

const tableContentStyle = computed(() => ({
  transform: `translateX(-${horizontalScrollValue.value}px)`,
}));

async function syncScrollWidths() {
  await nextTick();
  if (!tableViewportRef.value || !tableContentRef.value || !tableRef.value) {
    return;
  }
  const width = Math.max(tableRef.value.scrollWidth || 0, tableContentRef.value.scrollWidth || 0);
  horizontalScrollMax.value = Math.max(width - tableViewportRef.value.clientWidth, 0);
  horizontalScrollValue.value = Math.min(horizontalScrollValue.value, horizontalScrollMax.value);
}

function handleHorizontalSlider() {
  if (!tableViewportRef.value) {
    return;
  }
  horizontalScrollValue.value = Math.min(Number(horizontalScrollValue.value) || 0, horizontalScrollMax.value);
}

async function runForecast() {
  loading.value = true;
  try {
    repairForecast.value = await calculateRepairForecast(props.dataset.id, {
      model_id: selectedModelId.value,
      base_failure_coefficient: baseFailureCoefficient.value,
      nominal_gap_coefficient: nominalGapCoefficient.value,
      manual_feature_values: manualFeatureValues.value,
    });

    const nextManual = {};
    (repairForecast.value.missing_feature_columns || []).forEach((column) => {
      nextManual[column] = manualFeatureValues.value[column] ?? "";
    });
    manualFeatureValues.value = nextManual;
    await syncScrollWidths();
  } finally {
    loading.value = false;
  }
}

watch(groupedDateColumns, async () => {
  await syncScrollWidths();
});

watch(groupedRows, async () => {
  await syncScrollWidths();
});

onMounted(async () => {
  await loadSavedModels();
  window.addEventListener("resize", syncScrollWidths);
});
</script>
