<template>
  <div class="app-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">Excel + CatBoost + PostgreSQL</p>
        <h1>Анализ связей с целевой переменной</h1>
        <p class="hero-copy">
          Загружайте Excel без изменения исходной таблицы, храните ее в Postgres и исследуйте связи
          через CatBoost, корреляции и интерактивные графики.
        </p>
      </div>
      <div class="hero-badge">
        <span>Backend: FastAPI</span>
        <span>Frontend: Vue 3</span>
        <span>DB: Postgres</span>
      </div>
    </header>

    <main class="layout">
      <aside class="sidebar">
        <UploadPanel :loading="uploading" @upload="handleUpload" />
        <DatasetList :datasets="datasets" :selected-dataset-id="selectedDatasetId" @select="loadDataset" />
      </aside>

      <section class="content">
        <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>
        <div v-if="loadingData" class="loading-panel">Загрузка данных...</div>

        <template v-else-if="selectedDataset && selectedAnalysis">
          <section class="panel dataset-headline">
            <div class="panel-header">
              <h2>{{ selectedDataset.dataset.name }}</h2>
              <p>{{ selectedDataset.dataset.original_filename }}</p>
            </div>
            <div class="table-meta">
              <span>Целевая переменная: {{ selectedDataset.dataset.target_column || "Не указана" }}</span>
              <span>Зависимых переменных: {{ selectedDataset.dataset.selected_features.length }}</span>
              <span>Строк: {{ selectedDataset.dataset.row_count }}</span>
              <span>Колонок: {{ selectedDataset.dataset.columns.length }}</span>
            </div>
          </section>

          <SelectionPanel
            :columns="selectedDataset.dataset.columns"
            :target-column="selectedDataset.dataset.target_column || ''"
            :selected-features="selectedDataset.dataset.selected_features"
            :saving="savingSelection"
            @save="handleSelectionSave"
          />

          <DashboardView
            :analysis="selectedAnalysis"
            :rows="selectedDataset.rows"
            :columns="selectedDataset.dataset.columns"
          />

          <RawTable :columns="selectedDataset.dataset.columns" :rows="selectedDataset.rows" />
        </template>

        <section v-else class="panel empty-big">
          <h2>Проект готов к работе</h2>
          <p>Загрузите первый Excel-файл, чтобы увидеть таблицу, связи с target и интерактивные графики.</p>
        </section>
      </section>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import DashboardView from "./components/DashboardView.vue";
import DatasetList from "./components/DatasetList.vue";
import RawTable from "./components/RawTable.vue";
import SelectionPanel from "./components/SelectionPanel.vue";
import UploadPanel from "./components/UploadPanel.vue";
import {
  fetchAnalysis,
  fetchDataset,
  listDatasets,
  updateDatasetSelection,
  uploadDataset,
} from "./services/api";

const datasets = ref([]);
const selectedDatasetId = ref(null);
const selectedDataset = ref(null);
const selectedAnalysis = ref(null);
const uploading = ref(false);
const loadingData = ref(false);
const savingSelection = ref(false);
const errorMessage = ref("");

async function refreshDatasets() {
  datasets.value = await listDatasets();
  if (!selectedDatasetId.value && datasets.value.length) {
    await loadDataset(datasets.value[0].id);
  }
}

async function loadDataset(datasetId) {
  try {
    loadingData.value = true;
    errorMessage.value = "";
    selectedDatasetId.value = datasetId;
    const [dataset, analysis] = await Promise.all([fetchDataset(datasetId), fetchAnalysis(datasetId)]);
    selectedDataset.value = dataset;
    selectedAnalysis.value = analysis;
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    loadingData.value = false;
  }
}

async function handleUpload(payload) {
  try {
    uploading.value = true;
    errorMessage.value = "";
    const response = await uploadDataset(payload);
    await refreshDatasets();
    await loadDataset(response.dataset.id);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    uploading.value = false;
  }
}

async function handleSelectionSave(payload) {
  if (!selectedDatasetId.value) {
    return;
  }

  try {
    savingSelection.value = true;
    errorMessage.value = "";
    const updatedDataset = await updateDatasetSelection(selectedDatasetId.value, payload);
    if (selectedDataset.value) {
      selectedDataset.value = {
        ...selectedDataset.value,
        dataset: updatedDataset,
      };
    }
    selectedAnalysis.value = await fetchAnalysis(selectedDatasetId.value);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    savingSelection.value = false;
  }
}

onMounted(async () => {
  try {
    await refreshDatasets();
  } catch (error) {
    errorMessage.value = error.message;
  }
});
</script>
