<template>
  <section class="panel">
    <div class="panel-header">
      <h2>Хранение данных</h2>
      <p>Выберите раздел хранения и версию набора данных, с которой будем работать в анализе и прогнозе.</p>
    </div>

    <div class="control-block">
      <div class="control-group">
        <span class="control-title">Разделы</span>
        <div class="button-group">
          <button
            v-for="section in storageSections"
            :key="section.value"
            type="button"
            class="choice-button"
            :class="{ active: section.value === activeSection }"
            @click="setSection(section.value)"
          >
            {{ section.label }}
          </button>
        </div>
      </div>

      <label class="field">
        <span>Версия набора данных</span>
        <select :value="selectedVersionId" @change="handleVersionChange">
          <option v-for="dataset in sectionDatasets" :key="dataset.id" :value="dataset.id">
            Версия {{ dataset.storage_version }} · {{ dataset.name }} · {{ formatDate(dataset.created_at) }}
          </option>
        </select>
      </label>

      <div v-if="currentDataset" class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Раздел</span>
          <strong>{{ activeSectionLabel }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Версия</span>
          <strong>{{ currentDataset.storage_version }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Файл</span>
          <strong>{{ currentDataset.original_filename }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Строк</span>
          <strong>{{ currentDataset.row_count }}</strong>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const STORAGE_SECTIONS = [
  { value: "fact_epu", label: "Факт ЭПУ" },
  { value: "plan_epu", label: "План ЭПУ" },
  { value: "production", label: "Добыча" },
  { value: "plan_gtm", label: "План ГТМ" },
  { value: "fact_krs", label: "Факт КРС" },
  { value: "plan_krs", label: "План КРС" },
  { value: "gtm_rating", label: "Рейтинг ГТМ" },
];

const props = defineProps({
  datasets: {
    type: Array,
    default: () => [],
  },
  activeSection: {
    type: String,
    required: true,
  },
  selectedDatasetId: {
    type: Number,
    default: null,
  },
});

const emit = defineEmits(["section-change", "select"]);

const storageSections = STORAGE_SECTIONS;

const sectionDatasets = computed(() =>
  props.datasets
    .filter((dataset) => dataset.storage_section === props.activeSection)
    .slice()
    .sort((a, b) => b.storage_version - a.storage_version),
);

const selectedVersionId = computed(() => {
  if (sectionDatasets.value.some((dataset) => dataset.id === props.selectedDatasetId)) {
    return props.selectedDatasetId;
  }
  return sectionDatasets.value[0]?.id ?? "";
});

const currentDataset = computed(
  () => sectionDatasets.value.find((dataset) => dataset.id === Number(selectedVersionId.value)) || null,
);

const activeSectionLabel = computed(
  () => storageSections.find((section) => section.value === props.activeSection)?.label || props.activeSection,
);

function setSection(section) {
  emit("section-change", section);
}

function handleVersionChange(event) {
  emit("select", Number(event.target.value));
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
</script>
