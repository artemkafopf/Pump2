<template>
  <div class="app-shell">
    <main class="layout">
      <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
        <section class="panel module-sidebar">
          <div class="sidebar-brand" aria-label="WOWPUMP">
            <img :src="logoAsset" alt="WOWPUMP" class="logo-image" />
          </div>

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
              :class="{ active: activeModule === 'semantic' }"
              @click="activeModule = 'semantic'"
            >
              <span class="dataset-name">Словарь и отчеты</span>
              <span class="dataset-meta">LLaMA, согласование заголовков и генерация отчетов</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'analysis' }"
              @click="activeModule = 'analysis'"
            >
              <span class="dataset-name">Анализ работы насосов</span>
              <span class="dataset-meta">Использует только датасет из раздела "Факт ЭПУ"</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'forecast' }"
              @click="activeModule = 'forecast'"
            >
              <span class="dataset-name">Настройка модели отказов</span>
              <span class="dataset-meta">Использует только датасет из раздела "Факт ЭПУ"</span>
            </button>

            <button
              type="button"
              class="dataset-item module-item"
              :class="{ active: activeModule === 'repairForecast' }"
              @click="activeModule = 'repairForecast'"
            >
              <span class="dataset-name">Прогноз ремонтов</span>
              <span class="dataset-meta">Текущая модель CatBoost + выбранный сводпрогноз из хранения данных</span>
            </button>
          </div>
        </section>
      </aside>

      <section class="content">
        <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>
        <div v-if="loadingData" class="loading-panel">Загрузка данных...</div>

        <div v-if="mappingDialog.open" class="modal-overlay" @click.self="closeMappingDialog">
          <section class="modal-card">
            <div class="panel-header">
              <div>
                <h2>
                  {{ mappingDialog.stage === "columns" ? "Распознавание заголовков" : "Распознавание строк" }}
                </h2>
                <p>
                  <template v-if="mappingDialog.stage === 'columns'">
                    На этом подэтапе заголовки нового файла сопоставляются со словарем канонических переменных.
                    Нераспознанные строки подсвечены красным. Для них можно выбрать существующую переменную,
                    создать новую или отдельно запустить распознавание через LLM.
                  </template>
                  <template v-else>
                    После заголовков система предлагает сопоставить значения строк для сущностей
                    <strong>Участок недр</strong>, <strong>Куст</strong> и <strong>Скважина</strong>.
                    Нераспознанные значения подсвечены красным: их можно связать с уже известным значением
                    из словаря или создать новое каноническое значение.
                  </template>
                </p>
              </div>
            </div>

            <div class="table-meta recognition-stage-meta">
              <span :class="{ active: mappingDialog.stage === 'columns' }">1. Заголовки</span>
              <span :class="{ active: mappingDialog.stage === 'entities' }">2. Строки</span>
            </div>

            <div v-if="mappingDialog.stage === 'columns'" class="table-wrap compact-table">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Заголовок файла</th>
                    <th>Каноническая переменная</th>
                    <th>Уверенность</th>
                    <th>Комментарий</th>
                    <th>Действия</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="row in mappingDialog.rows"
                    :key="row.source_column"
                    :class="{ 'manual-row-unresolved': row.unresolved }"
                  >
                    <td>{{ row.source_column }}</td>
                    <td>
                      <select
                        v-model="row.selectionMode"
                        class="manual-match-input"
                        @change="handleColumnSelectionModeChange(row)"
                      >
                        <option value="dictionary">Выбрать из словаря</option>
                        <option value="new">Создать новую каноническую переменную</option>
                      </select>

                      <select
                        v-if="row.selectionMode === 'dictionary'"
                        v-model="row.canonical_name"
                        class="manual-match-input manual-match-secondary"
                      >
                        <option value="">Не выбрано</option>
                        <option
                          v-for="item in mappingDialog.dictionary"
                          :key="`${row.source_column}-${item.id}`"
                          :value="item.canonical_name"
                        >
                          {{ item.canonical_name }}
                        </option>
                      </select>

                      <input
                        v-else
                        v-model.trim="row.new_canonical_name"
                        type="text"
                        class="manual-match-input manual-match-secondary"
                        placeholder="Введите новую каноническую переменную"
                      />
                    </td>
                    <td>{{ formatConfidence(row.confidence) }}</td>
                    <td>{{ row.reasoning || (row.unresolved ? "Словарь не дал уверенного совпадения" : "Совпадение по словарю") }}</td>
                    <td>
                      <div class="inline-actions">
                        <button
                          type="button"
                          class="file-button"
                          :disabled="row.llmLoading || mappingDialog.applying"
                          @click="recognizeColumnWithLlm(row)"
                        >
                          {{ row.llmLoading ? "LLM..." : "Распознать с LLM" }}
                        </button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div v-else class="table-wrap compact-table">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Тип сущности</th>
                    <th>Значение в файле</th>
                    <th>Каноническое значение</th>
                    <th>Уверенность</th>
                    <th>Комментарий</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="row in mappingDialog.entityRows"
                    :key="`${row.entity_type}-${row.source_value}`"
                    :class="{ 'manual-row-unresolved': row.unresolved }"
                  >
                    <td>{{ entityTypeLabel(row.entity_type) }}</td>
                    <td>{{ row.source_value }}</td>
                    <td>
                      <select v-model="row.selectionMode" class="manual-match-input">
                        <option value="dictionary">Выбрать из словаря</option>
                        <option value="new">Создать новое каноническое значение</option>
                      </select>

                      <select
                        v-if="row.selectionMode === 'dictionary'"
                        v-model="row.canonical_value"
                        class="manual-match-input manual-match-secondary"
                      >
                        <option value="">Не выбрано</option>
                        <option
                          v-for="item in entityOptionsByType(row.entity_type)"
                          :key="`${row.entity_type}-${item.id}`"
                          :value="item.canonical_value"
                        >
                          {{ item.canonical_value }}
                        </option>
                      </select>

                      <input
                        v-else
                        v-model.trim="row.new_canonical_value"
                        type="text"
                        class="manual-match-input manual-match-secondary"
                        placeholder="Введите новое каноническое значение"
                      />
                    </td>
                    <td>{{ formatConfidence(row.confidence) }}</td>
                    <td>{{ row.reasoning || (row.unresolved ? "Словарь не дал уверенного совпадения" : "Совпадение по словарю") }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="prediction-actions">
              <button
                type="button"
                class="primary-button"
                :disabled="mappingDialog.applying"
                @click="handleMappingPrimaryAction"
              >
                {{
                  mappingDialog.applying
                    ? "Сохранение..."
                    : mappingDialog.stage === "columns"
                      ? "Продолжить к сопоставлению строк"
                      : "Сохранить распознанные строки"
                }}
              </button>
              <button type="button" class="file-button" :disabled="mappingDialog.applying" @click="closeMappingDialog">
                Закрыть
              </button>
            </div>
          </section>
        </div>

        <template v-else-if="displayDataset && displayAnalysis">
          <section class="panel dataset-headline">
            <div class="panel-header">
              <div>
                <h2>{{ displayDataset.dataset.name }}</h2>
                <p>{{ displayDataset.dataset.original_filename }}</p>
              </div>
            </div>

            <div class="table-meta">
              <span>Раздел: {{ displayDataset.dataset.storage_section }}</span>
              <span>Версия: {{ displayDataset.dataset.storage_version }}</span>
              <span>Целевая переменная: {{ displayDataset.dataset.target_column || "Не указана" }}</span>
              <span>Зависимых переменных: {{ displayDataset.dataset.selected_features.length }}</span>
              <span>Строк: {{ displayDataset.dataset.row_count }}</span>
              <span>Колонок: {{ displayDataset.dataset.columns.length }}</span>
            </div>
          </section>

          <template v-if="activeModule === 'storage'">
            <UploadPanel :loading="uploading" @upload="handleUpload" />
            <StoragePanel
              :datasets="datasets"
              :active-section="storageSection"
              :selected-dataset-id="storageDatasetId"
              @section-change="handleStorageSectionChange"
              @select="loadStorageDataset"
              @edit-mappings="handleEditMappings"
              @clear-dictionary="handleClearDictionary"
            />
          </template>

          <template v-else-if="activeModule === 'analysis' && factDataset && factAnalysis">
            <SelectionPanel
              :columns="factDataset.dataset.columns"
              :target-column="factDataset.dataset.target_column || ''"
              :selected-features="factDataset.dataset.selected_features"
              :saving="savingSelection"
              @save="handleSelectionSave"
            />

            <DashboardView
              :analysis="factAnalysis"
              :rows="factDataset.rows"
              :columns="factDataset.dataset.columns"
            />

            <RawTable :columns="factDataset.dataset.columns" :rows="factDataset.rows" />
          </template>

          <AIWorkbench
            v-else-if="activeModule === 'semantic'"
            :dataset="displayDataset.dataset"
          />

          <PredictionModule
            v-else-if="activeModule === 'forecast' && factDataset && factAnalysis"
            :dataset="factDataset.dataset"
            :analysis="factAnalysis"
            :rows="factDataset.rows"
          />

          <RepairForecastModule
            v-else-if="activeModule === 'repairForecast' && factDataset"
            :model-dataset="factDataset.dataset"
            :source-dataset="repairSourceDataset"
          />
        </template>

        <section v-else class="panel empty-big">
          <h2>Проект готов к работе</h2>
          <p>
            Загрузите первый Excel-файл, чтобы увидеть таблицу, аналитику, словарь переменных и прогнозные сценарии.
          </p>
        </section>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import logoAsset from "./assets/wowpumpLOGO.png";
import AIWorkbench from "./components/AIWorkbench.vue";
import DashboardView from "./components/DashboardView.vue";
import PredictionModule from "./components/PredictionModule.vue";
import RawTable from "./components/RawTable.vue";
import RepairForecastModule from "./components/RepairForecastModule.vue";
import SelectionPanel from "./components/SelectionPanel.vue";
import StoragePanel from "./components/StoragePanel.vue";
import UploadPanel from "./components/UploadPanel.vue";
import {
  clearVariableDictionary,
  fetchEntityDictionary,
  fetchEntityMatches,
  fetchAnalysis,
  fetchDataset,
  fetchVariableDictionary,
  fetchVariableMatches,
  listDatasets,
  reconcileEntities,
  reconcileVariables,
  saveManualEntityMatches,
  saveManualVariableMatches,
  updateDatasetSelection,
  uploadDataset,
} from "./services/api";

const FACT_EPU_SECTION = "fact_epu";
const SVODPROGNOZ_SECTION = "svodprognoz";

const datasets = ref([]);
const storageDatasetId = ref(null);
const storageDataset = ref(null);
const storageAnalysis = ref(null);
const factDatasetId = ref(null);
const factDataset = ref(null);
const factAnalysis = ref(null);
const uploading = ref(false);
const loadingData = ref(false);
const savingSelection = ref(false);
const errorMessage = ref("");
const storageSection = ref(FACT_EPU_SECTION);
const activeModule = ref("analysis");
const sidebarCollapsed = ref(false);
const mappingDialog = ref({
  open: false,
  datasetId: null,
  applying: false,
  stage: "columns",
  dictionary: [],
  rows: [],
  entityDictionary: [],
  entityRows: [],
});

const displayDataset = computed(() => (activeModule.value === "storage" ? storageDataset.value : factDataset.value));
const displayAnalysis = computed(() => (activeModule.value === "storage" ? storageAnalysis.value : factAnalysis.value));
const latestSvodprognozDataset = computed(() =>
  datasets.value
    .filter((dataset) => dataset.storage_section === SVODPROGNOZ_SECTION)
    .slice()
    .sort((a, b) => b.storage_version - a.storage_version)[0] || null,
);
const repairSourceDataset = computed(() =>
  storageDataset.value?.dataset?.storage_section === SVODPROGNOZ_SECTION
    ? storageDataset.value.dataset
    : latestSvodprognozDataset.value,
);

function getLatestDatasetIdBySection(section) {
  return datasets.value
    .filter((dataset) => dataset.storage_section === section)
    .sort((a, b) => b.storage_version - a.storage_version)[0]?.id ?? null;
}

function mapSuggestionRows(matches, dictionary) {
  const dictionaryNames = new Set(dictionary.map((item) => item.canonical_name));
  return matches.map((item) => {
    const recognized = item.confidence >= 0.8 && Boolean(item.canonical_name);
    return {
      source_column: item.source_column,
      canonical_name: recognized ? item.canonical_name : "",
      suggested_name: item.canonical_name || "",
      new_canonical_name: recognized ? "" : item.canonical_name || item.source_column,
      confidence: item.confidence ?? 0,
      reasoning: item.reasoning || "",
      unresolved: !recognized,
      selectionMode: recognized && dictionaryNames.has(item.canonical_name) ? "dictionary" : "new",
      llmLoading: false,
    };
  });
}

function mapEntitySuggestionRows(matches, dictionary, unresolvedValues = {}) {
  return matches.map((item) => {
    const entityOptions = dictionary.filter((entry) => entry.entity_type === item.entity_type);
    const hasDictionaryValue = entityOptions.some((entry) => entry.canonical_value === item.canonical_value);
    const unresolved = (unresolvedValues[item.entity_type] || []).includes(item.source_value) || item.confidence < 0.8;
    return {
      entity_type: item.entity_type,
      source_value: item.source_value,
      canonical_value: hasDictionaryValue && !unresolved ? item.canonical_value : "",
      suggested_value: item.canonical_value || "",
      new_canonical_value: unresolved ? item.canonical_value || item.source_value : item.canonical_value || "",
      confidence: item.confidence ?? 0,
      reasoning: item.reasoning || "",
      unresolved,
      selectionMode: hasDictionaryValue && !unresolved ? "dictionary" : "new",
    };
  });
}

async function loadDatasetBundle(datasetId) {
  const [dataset, analysis] = await Promise.all([fetchDataset(datasetId), fetchAnalysis(datasetId)]);
  return { dataset, analysis };
}

async function loadStorageDataset(datasetId) {
  if (!datasetId) {
    storageDatasetId.value = null;
    storageDataset.value = null;
    storageAnalysis.value = null;
    return;
  }

  const bundle = await loadDatasetBundle(datasetId);
  storageDatasetId.value = datasetId;
  storageDataset.value = bundle.dataset;
  storageAnalysis.value = bundle.analysis;
  storageSection.value = bundle.dataset.dataset.storage_section || FACT_EPU_SECTION;
}

async function loadFactDataset(datasetId) {
  if (!datasetId) {
    factDatasetId.value = null;
    factDataset.value = null;
    factAnalysis.value = null;
    return;
  }

  const bundle = await loadDatasetBundle(datasetId);
  factDatasetId.value = datasetId;
  factDataset.value = bundle.dataset;
  factAnalysis.value = bundle.analysis;
}

async function ensureFactDatasetLoaded() {
  const latestFactDatasetId = getLatestDatasetIdBySection(FACT_EPU_SECTION);
  if (!latestFactDatasetId) {
    await loadFactDataset(null);
    return;
  }

  if (factDatasetId.value !== latestFactDatasetId) {
    await loadFactDataset(latestFactDatasetId);
  }
}

async function refreshDatasets() {
  datasets.value = await listDatasets();

  if (!storageDatasetId.value) {
    const initialStorageDatasetId = getLatestDatasetIdBySection(storageSection.value) || datasets.value[0]?.id || null;
    await loadStorageDataset(initialStorageDatasetId);
  } else {
    const stillExists = datasets.value.some((dataset) => dataset.id === storageDatasetId.value);
    await loadStorageDataset(stillExists ? storageDatasetId.value : getLatestDatasetIdBySection(storageSection.value));
  }

  await ensureFactDatasetLoaded();
}

async function openRecognitionStage(datasetId, options = {}) {
  const { preferSaved = false } = options;
  const dictionary = await fetchVariableDictionary();
  let response;

  if (preferSaved) {
    const existingMatches = await fetchVariableMatches(datasetId);
    if (existingMatches.length) {
      response = {
        matches: existingMatches,
      };
    }
  }

  if (!response) {
    response = await reconcileVariables(datasetId, {
      persist: false,
      use_llm: false,
    });
  }

  mappingDialog.value = {
    open: true,
    datasetId,
    applying: false,
    stage: "columns",
    dictionary,
    rows: mapSuggestionRows(response.matches, dictionary),
    entityDictionary: [],
    entityRows: [],
  };
}

async function openEntityRecognitionStage(datasetId, options = {}) {
  const { preferSaved = false } = options;
  const entityDictionary = await fetchEntityDictionary();
  let response;

  if (preferSaved) {
    const existingMatches = await fetchEntityMatches(datasetId);
    if (existingMatches.length) {
      response = {
        matches: existingMatches,
        unresolved_values: {},
      };
    }
  }

  if (!response) {
    response = await reconcileEntities(datasetId, {
      persist: false,
    });
  }

  mappingDialog.value = {
    ...mappingDialog.value,
    open: true,
    datasetId,
    applying: false,
    stage: "entities",
    entityDictionary,
    entityRows: mapEntitySuggestionRows(response.matches, entityDictionary, response.unresolved_values || {}),
  };
}

async function handleUpload(payload) {
  try {
    uploading.value = true;
    errorMessage.value = "";
    const response = await uploadDataset(payload);
    await refreshDatasets();
    await loadStorageDataset(response.dataset.id);
    if ((response.dataset.storage_section || FACT_EPU_SECTION) === FACT_EPU_SECTION) {
      await loadFactDataset(response.dataset.id);
    }
    await openRecognitionStage(response.dataset.id);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    uploading.value = false;
  }
}

async function recognizeColumnWithLlm(row) {
  if (!mappingDialog.value.datasetId) {
    return;
  }

  row.llmLoading = true;
  try {
    const response = await reconcileVariables(mappingDialog.value.datasetId, {
      persist: false,
      use_llm: true,
      columns: [row.source_column],
    });
    const suggestion = response.matches[0];
    if (suggestion) {
      row.canonical_name = suggestion.canonical_name || "";
      row.suggested_name = suggestion.canonical_name || "";
      row.new_canonical_name = suggestion.canonical_name || row.source_column;
      row.confidence = suggestion.confidence ?? 0;
      row.reasoning = suggestion.reasoning || "";
      row.unresolved = !(suggestion.confidence >= 0.8 && suggestion.canonical_name);
      row.selectionMode = row.unresolved ? "new" : "dictionary";
    }
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    row.llmLoading = false;
  }
}

async function saveRecognizedMappings() {
  if (!mappingDialog.value.datasetId) {
    return;
  }

  const invalidRows = mappingDialog.value.rows.filter((row) => {
    if (row.selectionMode === "dictionary") {
      return !row.canonical_name;
    }
    return !row.new_canonical_name.trim();
  });

  if (invalidRows.length) {
    errorMessage.value = "Для всех красных строк нужно выбрать каноническую переменную, создать новую или распознать через LLM.";
    return;
  }

  try {
    mappingDialog.value = {
      ...mappingDialog.value,
      applying: true,
    };

    await saveManualVariableMatches(mappingDialog.value.datasetId, {
      matches: mappingDialog.value.rows.map((row) => ({
        source_column: row.source_column,
        canonical_name: row.selectionMode === "dictionary" ? row.canonical_name : row.new_canonical_name.trim(),
      })),
    });

    await openEntityRecognitionStage(mappingDialog.value.datasetId);
  } catch (error) {
    errorMessage.value = error.message;
    mappingDialog.value = {
      ...mappingDialog.value,
      applying: false,
    };
  }
}

async function saveRecognizedEntities() {
  if (!mappingDialog.value.datasetId) {
    return;
  }

  const invalidRows = mappingDialog.value.entityRows.filter((row) => {
    if (row.selectionMode === "dictionary") {
      return !row.canonical_value;
    }
    return !row.new_canonical_value.trim();
  });

  if (invalidRows.length) {
    errorMessage.value = "Для всех красных строк нужно выбрать каноническое значение или создать новое.";
    return;
  }

  try {
    mappingDialog.value = {
      ...mappingDialog.value,
      applying: true,
    };

    await saveManualEntityMatches(mappingDialog.value.datasetId, {
      matches: mappingDialog.value.entityRows.map((row) => ({
        entity_type: row.entity_type,
        source_value: row.source_value,
        canonical_value: row.selectionMode === "dictionary" ? row.canonical_value : row.new_canonical_value.trim(),
      })),
    });

    closeMappingDialog();
  } catch (error) {
    errorMessage.value = error.message;
    mappingDialog.value = {
      ...mappingDialog.value,
      applying: false,
    };
  }
}

async function handleMappingPrimaryAction() {
  if (mappingDialog.value.stage === "columns") {
    await saveRecognizedMappings();
    return;
  }
  await saveRecognizedEntities();
}

function closeMappingDialog() {
  mappingDialog.value = {
    open: false,
    datasetId: null,
    applying: false,
    stage: "columns",
    dictionary: [],
    rows: [],
    entityDictionary: [],
    entityRows: [],
  };
}

async function handleSelectionSave(payload) {
  if (!factDatasetId.value) {
    return;
  }

  try {
    savingSelection.value = true;
    errorMessage.value = "";
    const updatedDataset = await updateDatasetSelection(factDatasetId.value, payload);
    if (factDataset.value) {
      factDataset.value = {
        ...factDataset.value,
        dataset: updatedDataset,
      };
    }
    factAnalysis.value = await fetchAnalysis(factDatasetId.value);

    if (storageDatasetId.value === factDatasetId.value && storageDataset.value) {
      storageDataset.value = {
        ...storageDataset.value,
        dataset: updatedDataset,
      };
      storageAnalysis.value = factAnalysis.value;
    }
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    savingSelection.value = false;
  }
}

async function handleStorageSectionChange(section) {
  storageSection.value = section;
  const nextDatasetId = getLatestDatasetIdBySection(section);
  await loadStorageDataset(nextDatasetId);
}

async function handleEditMappings(datasetId) {
  if (!datasetId) {
    return;
  }

  try {
    errorMessage.value = "";
    await openRecognitionStage(datasetId, { preferSaved: true });
  } catch (error) {
    errorMessage.value = error.message;
  }
}

async function handleClearDictionary() {
  const confirmed = window.confirm(
    "Очистить словарь канонических переменных и все сохраненные сопоставления столбцов? Загруженные датасеты останутся на месте.",
  );
  if (!confirmed) {
    return;
  }

  try {
    errorMessage.value = "";
    await clearVariableDictionary();
  } catch (error) {
    errorMessage.value = error.message;
  }
}

function formatConfidence(value) {
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value ?? 0);
}

function entityTypeLabel(entityType) {
  return (
    {
      license_area: "Участок недр",
      cluster: "Куст",
      well: "Скважина",
    }[entityType] || entityType
  );
}

function entityOptionsByType(entityType) {
  return mappingDialog.value.entityDictionary.filter((item) => item.entity_type === entityType);
}

function handleColumnSelectionModeChange(row) {
  if (row.selectionMode === "new" && !row.new_canonical_name?.trim()) {
    row.new_canonical_name = row.source_column || "";
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
