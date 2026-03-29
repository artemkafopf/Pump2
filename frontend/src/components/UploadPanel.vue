<template>
  <section class="panel upload-panel">
    <div class="panel-header">
      <h2>Загрузка Excel</h2>
      <p>
        После загрузки сервис сохранит таблицу и откроет окно распознавания заголовков:
        сначала по словарю, а для спорных колонок можно включить LLaMA или задать соответствие вручную.
      </p>
    </div>

    <form class="upload-form" @submit.prevent="handleSubmit">
      <label class="field">
        <span>Название набора</span>
        <input v-model.trim="datasetName" type="text" placeholder="Например, Скважины март" required />
      </label>

      <label class="field">
        <span>Раздел хранения</span>
        <select v-model="storageSection">
          <option v-for="section in STORAGE_SECTIONS" :key="section.value" :value="section.value">
            {{ section.label }}
          </option>
        </select>
      </label>

      <label class="file-picker">
        <input type="file" accept=".xlsx,.xls" @change="handleFileChange" />
        <span class="file-button">Выбрать Excel</span>
        <span class="file-name">{{ selectedFileName }}</span>
      </label>

      <button class="primary-button" type="submit" :disabled="loading">
        {{ loading ? "Загрузка..." : "Загрузить и открыть сопоставление" }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";

const STORAGE_SECTIONS = [
  { value: "fact_epu", label: "Факт ЭПУ" },
  { value: "plan_epu", label: "План ЭПУ" },
  { value: "production", label: "Добыча" },
  { value: "plan_gtm", label: "ГТМ" },
  { value: "fact_krs", label: "Факт КРС" },
  { value: "plan_krs", label: "План КРС" },
  { value: "gtm_rating", label: "Рейтинг ГТМ" },
];

const emit = defineEmits(["upload"]);

defineProps({
  loading: {
    type: Boolean,
    default: false,
  },
});

const file = ref(null);
const datasetName = ref("");
const storageSection = ref("fact_epu");

const selectedFileName = computed(() => file.value?.name || "Файл не выбран");

function handleFileChange(event) {
  file.value = event.target.files?.[0] || null;
}

function handleSubmit() {
  if (!file.value) {
    window.alert("Выберите Excel файл.");
    return;
  }

  emit("upload", {
    datasetName: datasetName.value,
    storageSection: storageSection.value,
    file: file.value,
  });
}
</script>
