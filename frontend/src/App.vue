<template>
  <div class="app-shell">
    <div class="logo-banner" aria-label="WOWPUMP">
      <img :src="logoAsset" alt="WOWPUMP" class="logo-image" />
    </div>

    <main class="layout">
      <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
        <section class="panel module-sidebar">
          <div class="module-sidebar-header">
            <h2>Модули</h2>
            <button type="button" class="choice-button sidebar-toggle" @click="sidebarCollapsed = !sidebarCollapsed">
              {{ sidebarCollapsed ? "Развернуть" : "Свернуть" }}
            </button>
          </div>

          <div class="module-nav">
            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'storage' }"
              @click="activeModule = 'storage'"
            >
              <span class="dataset-name">Хранение данных</span>
              <span class="dataset-meta">Разделы хранения и версии наборов данных</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'analysis' }"
              @click="activeModule = 'analysis'"
            >
              <span class="dataset-name">Анализ работы насосов</span>
              <span class="dataset-meta">Анализ зависимостей, CatBoost и визуализация</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'semantic' }"
              @click="activeModule = 'semantic'"
            >
              <span class="dataset-name">Словарь и отчеты</span>
              <span class="dataset-meta">LLaMA, согласование заголовков и генерация отчетов</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'forecast' }"
              @click="activeModule = 'forecast'"
            >
              <span class="dataset-name">Прогноз отказов</span>
              <span class="dataset-meta">Обучение, сохранение моделей и прогнозные сценарии</span>
            </button>
          </div>
        </section>
      </aside>

      <section class="content">
        <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>
        <div v-if="loadingData" class="loading-panel">Загрузка данных...</div>

        <template v-else-if="selectedDataset && selectedAnalysis">
          <section class="panel dataset-headline">
            <div class="panel-header">
              <div>
                <h2>{{ selectedDataset.dataset.name }}</h2>
                <p>{{ selectedDataset.dataset.original_filename }}</p>
              </div>
            </div>

            <div class="table-meta">
              <span>Раздел: {{ selectedDataset.dataset.storage_section }}</span>
              <span>Версия: {{ selectedDataset.dataset.storage_version }}</span>
              <span>Целевая переменная: {{ selectedDataset.dataset.target_column || "Не указана" }}</span>
              <span>Зависимых переменных: {{ selectedDataset.dataset.selected_features.length }}</span>
              <span>Строк: {{ selectedDataset.dataset.row_count }}</span>
              <span>Колонок: {{ selectedDataset.dataset.columns.length }}</span>
            </div>
          </section>

          <template v-if="activeModule === 'storage'">
            <UploadPanel :loading="uploading" @upload="handleUpload" />
            <StoragePanel
              :datasets="datasets"
              :active-section="storageSection"
              :selected-dataset-id="selectedDatasetId"
              @section-change="handleStorageSectionChange"
              @select="loadDataset"
            />
          </template>

          <template v-else-if="activeModule === 'analysis'">
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

          <AIWorkbench
            v-else-if="activeModule === 'semantic'"
            :dataset="selectedDataset.dataset"
          />

          <PredictionModule
            v-else
            :dataset="selectedDataset.dataset"
            :analysis="selectedAnalysis"
            :rows="selectedDataset.rows"
          />
        </template>

        <section v-else class="panel empty-big">
          <h2>Проект готов к работе</h2>
          <p>Загрузите первый Excel-файл, чтобы увидеть таблицу, аналитику, словарь переменных и прогнозные сценарии.</p>
        </section>
      </section>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import logoAsset from "./assets/wowpumpLOGO.png";
import AIWorkbench from "./components/AIWorkbench.vue";
import DashboardView from "./components/DashboardView.vue";
import PredictionModule from "./components/PredictionModule.vue";
import RawTable from "./components/RawTable.vue";
import SelectionPanel from "./components/SelectionPanel.vue";
import StoragePanel from "./components/StoragePanel.vue";
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
const storageSection = ref("fact_epu");
const activeModule = ref("analysis");
const sidebarCollapsed = ref(false);

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
    storageSection.value = dataset.dataset.storage_section || "fact_epu";
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

async function handleStorageSectionChange(section) {
  storageSection.value = section;
  const nextDataset = datasets.value
    .filter((dataset) => dataset.storage_section === section)
    .sort((a, b) => b.storage_version - a.storage_version)[0];
  if (nextDataset) {
    await loadDataset(nextDataset.id);
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
