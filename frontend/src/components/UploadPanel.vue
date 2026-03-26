<template>
  <section class="panel upload-panel">
    <div class="panel-header">
      <h2>Загрузка Excel</h2>
      <p>Файл выбирается через проводник, целевая переменная вводится отдельной строкой.</p>
    </div>

    <form class="upload-form" @submit.prevent="handleSubmit">
      <label class="field">
        <span>Название набора</span>
        <input v-model.trim="datasetName" type="text" placeholder="Например, Скважины март" required />
      </label>

      <label class="field">
        <span>Целевая переменная</span>
        <input v-model.trim="targetColumnName" type="text" placeholder="Введите имя колонки" />
      </label>

      <label class="file-picker">
        <input type="file" accept=".xlsx,.xls" @change="handleFileChange" />
        <span class="file-button">Выбрать Excel</span>
        <span class="file-name">{{ selectedFileName }}</span>
      </label>

      <button class="primary-button" type="submit" :disabled="loading">
        {{ loading ? "Загрузка..." : "Загрузить и проанализировать" }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";

const emit = defineEmits(["upload"]);

defineProps({
  loading: {
    type: Boolean,
    default: false,
  },
});

const file = ref(null);
const datasetName = ref("");
const targetColumnName = ref("");

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
    targetColumnName: targetColumnName.value,
    file: file.value,
  });
}
</script>
