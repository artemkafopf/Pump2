<template>
  <section class="panel">
    <div class="panel-header">
      <h2>Выбор переменных</h2>
      <p>После распознавания заголовков выберите целевую и зависимые переменные кнопками.</p>
    </div>

    <div class="control-group">
      <span class="control-title">Целевая переменная</span>
      <div class="button-group">
        <button
          v-for="column in columns"
          :key="`target-${column}`"
          type="button"
          class="choice-button"
          :class="{ active: column === localTargetColumn }"
          @click="selectTarget(column)"
        >
          {{ column }}
        </button>
      </div>
    </div>

    <div class="control-group">
      <div class="selection-actions">
        <span class="control-title">Зависимые переменные</span>
        <div class="button-group">
          <button type="button" class="choice-button" @click="selectAllFeatures">Выбрать все</button>
          <button type="button" class="choice-button" @click="clearFeatures">Очистить</button>
        </div>
      </div>
      <div class="button-group">
        <button
          v-for="column in featureCandidates"
          :key="`feature-${column}`"
          type="button"
          class="choice-button"
          :class="{ active: localSelectedFeatures.includes(column) }"
          @click="toggleFeature(column)"
        >
          {{ column }}
        </button>
      </div>
    </div>

    <button type="button" class="primary-button" :disabled="saving || !localTargetColumn" @click="saveSelection">
      {{ saving ? "Сохранение..." : "Запустить аналитику CatBoost" }}
    </button>
  </section>
</template>

<script setup>
import { computed, ref, watch } from "vue";

const props = defineProps({
  columns: {
    type: Array,
    default: () => [],
  },
  targetColumn: {
    type: String,
    default: "",
  },
  selectedFeatures: {
    type: Array,
    default: () => [],
  },
  saving: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["save"]);

const localTargetColumn = ref("");
const localSelectedFeatures = ref([]);

watch(
  () => props.targetColumn,
  (value) => {
    localTargetColumn.value = value || "";
  },
  { immediate: true },
);

watch(
  () => props.selectedFeatures,
  (value) => {
    localSelectedFeatures.value = [...value];
  },
  { immediate: true },
);

const featureCandidates = computed(() =>
  props.columns.filter((column) => column !== localTargetColumn.value),
);

function selectTarget(column) {
  localTargetColumn.value = column;
  localSelectedFeatures.value = localSelectedFeatures.value.filter((item) => item !== column);
}

function toggleFeature(column) {
  if (localSelectedFeatures.value.includes(column)) {
    localSelectedFeatures.value = localSelectedFeatures.value.filter((item) => item !== column);
  } else {
    localSelectedFeatures.value = [...localSelectedFeatures.value, column];
  }
}

function selectAllFeatures() {
  localSelectedFeatures.value = featureCandidates.value.slice();
}

function clearFeatures() {
  localSelectedFeatures.value = [];
}

function saveSelection() {
  emit("save", {
    targetColumn: localTargetColumn.value || null,
    selectedFeatures: localSelectedFeatures.value,
  });
}
</script>
