<template>
  <section class="dashboard-grid">
    <section class="panel">
      <div class="panel-header">
        <h2>Локальная LLaMA</h2>
        <p>Сервис используется для семантического согласования заголовков и подготовки текстовых отчётов.</p>
      </div>

      <div class="overview-grid">
        <div class="metric-card">
          <span class="metric-label">Статус</span>
          <strong>{{ llmStatus?.available ? "Доступна" : "Недоступна" }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Модель</span>
          <strong>{{ llmStatus?.model || "n/a" }}</strong>
        </div>
        <div class="metric-card">
          <span class="metric-label">Скачана</span>
          <strong>{{ llmStatus?.installed ? "Да" : "Нет" }}</strong>
        </div>
      </div>

      <div v-if="llmStatus?.error" class="notes-list">
        <span class="note-chip">{{ llmStatus.error }}</span>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Согласование переменных</h2>
        <p>Похожим заголовкам будет предложено единое каноническое имя с сохранением в словаре алиасов.</p>
      </div>

      <div class="prediction-actions">
        <button type="button" class="primary-button" :disabled="reconciling" @click="handleReconcile(true)">
          {{ reconciling ? "Согласование..." : "Согласовать через LLaMA" }}
        </button>
        <button type="button" class="file-button" :disabled="reconciling" @click="handleReconcile(false)">
          Эвристики без LLaMA
        </button>
      </div>

      <div v-if="matches.length" class="table-wrap compact-table">
        <table class="data-table">
          <thead>
            <tr>
              <th>Исходный заголовок</th>
              <th>Каноническая переменная</th>
              <th>Confidence</th>
              <th>Статус</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in matches" :key="`${item.source_column}-${item.canonical_name}`">
              <td>{{ item.source_column }}</td>
              <td>{{ item.canonical_name }}</td>
              <td>{{ formatConfidence(item.confidence) }}</td>
              <td>{{ item.status }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="unresolvedColumns.length" class="notes-list">
        <span v-for="column in unresolvedColumns" :key="column" class="note-chip">{{ column }}</span>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Словарь канонических переменных</h2>
        <p>Справочник накапливает канонические имена и все подтвержденные алиасы.</p>
      </div>

      <div v-if="dictionary.length" class="table-wrap compact-table">
        <table class="data-table">
          <thead>
            <tr>
              <th>Каноническое имя</th>
              <th>Алиасы</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in dictionary" :key="item.id">
              <td>{{ item.canonical_name }}</td>
              <td>{{ item.aliases.join(", ") }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Генерация отчётов</h2>
        <p>Отчёт собирается по текущему анализу, словарю переменных и, при наличии, модели прогноза.</p>
      </div>

      <div class="feature-slot-grid">
        <label class="field">
          <span>Тип отчёта</span>
          <select v-model="reportType">
            <option value="analysis_forecast">Анализ и прогноз</option>
            <option value="analysis_only">Только анализ</option>
            <option value="semantic_dictionary">Словарь переменных</option>
          </select>
        </label>
      </div>

      <div class="prediction-actions">
        <button type="button" class="primary-button" :disabled="generatingReport" @click="handleGenerateReport(true)">
          {{ generatingReport ? "Генерация..." : "Сформировать через LLaMA" }}
        </button>
        <button type="button" class="file-button" :disabled="generatingReport" @click="handleGenerateReport(false)">
          Шаблонный отчет
        </button>
      </div>

      <div class="report-list">
        <article v-for="report in reports" :key="report.id" class="report-card">
          <div class="panel-header">
            <div>
              <h2>{{ report.title }}</h2>
              <p>{{ formatDate(report.created_at) }} · {{ report.llm_used ? "LLaMA" : "Template" }}</p>
            </div>
          </div>
          <pre class="report-content">{{ report.content }}</pre>
        </article>
      </div>
    </section>
  </section>
</template>

<script setup>
import { ref, watch } from "vue";
import {
  fetchLLMStatus,
  fetchReports,
  fetchVariableDictionary,
  fetchVariableMatches,
  generateReport,
  reconcileVariables,
} from "../services/api";

const props = defineProps({
  dataset: { type: Object, required: true },
});

const llmStatus = ref(null);
const dictionary = ref([]);
const matches = ref([]);
const reports = ref([]);
const unresolvedColumns = ref([]);
const reconciling = ref(false);
const generatingReport = ref(false);
const reportType = ref("analysis_forecast");

function formatConfidence(value) {
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value ?? 0);
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : new Intl.DateTimeFormat("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

async function refreshAll() {
  if (!props.dataset?.id) return;
  const [status, dict, currentMatches, currentReports] = await Promise.all([
    fetchLLMStatus(),
    fetchVariableDictionary(),
    fetchVariableMatches(props.dataset.id),
    fetchReports(props.dataset.id),
  ]);
  llmStatus.value = status;
  dictionary.value = dict;
  matches.value = currentMatches;
  reports.value = currentReports;
  unresolvedColumns.value = currentMatches
    .filter((item) => item.confidence < 0.65)
    .map((item) => item.source_column);
}

async function handleReconcile(useLLM) {
  reconciling.value = true;
  try {
    const response = await reconcileVariables(props.dataset.id, { persist: true, use_llm: useLLM });
    matches.value = response.matches;
    unresolvedColumns.value = response.unresolved_columns;
    dictionary.value = await fetchVariableDictionary();
    llmStatus.value = await fetchLLMStatus();
  } finally {
    reconciling.value = false;
  }
}

async function handleGenerateReport(useLLM) {
  generatingReport.value = true;
  try {
    await generateReport(props.dataset.id, {
      report_type: reportType.value,
      use_llm: useLLM,
    });
    reports.value = await fetchReports(props.dataset.id);
  } finally {
    generatingReport.value = false;
  }
}

watch(
  () => props.dataset?.id,
  async () => {
    await refreshAll();
  },
  { immediate: true },
);
</script>
