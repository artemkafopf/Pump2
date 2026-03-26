<template>
  <section class="panel dataset-list">
    <div class="panel-header">
      <h2>Наборы данных</h2>
      <p>Выберите загруженный набор, чтобы открыть таблицу и аналитику.</p>
    </div>

    <div v-if="datasets.length" class="dataset-items">
      <button
        v-for="dataset in datasets"
        :key="dataset.id"
        class="dataset-item"
        :class="{ active: dataset.id === selectedDatasetId }"
        @click="$emit('select', dataset.id)"
      >
        <span class="dataset-name">{{ dataset.name }}</span>
        <span class="dataset-meta">{{ dataset.original_filename }}</span>
        <span class="dataset-meta">{{ dataset.row_count }} строк</span>
      </button>
    </div>

    <div v-else class="empty-state">
      Пока нет загруженных таблиц.
    </div>
  </section>
</template>

<script setup>
defineProps({
  datasets: {
    type: Array,
    default: () => [],
  },
  selectedDatasetId: {
    type: Number,
    default: null,
  },
});
</script>
