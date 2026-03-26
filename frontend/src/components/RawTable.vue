<template>
  <section class="panel">
    <div class="panel-header">
      <h2>Исходная таблица</h2>
      <p>Показываются все заголовки и строки в сохраненном виде.</p>
    </div>

    <div class="table-meta">
      <span>Колонок: {{ columns.length }}</span>
      <span>Строк: {{ rows.length }}</span>
    </div>

    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th v-for="column in columns" :key="column">{{ column }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in rows" :key="index">
            <td v-for="column in columns" :key="`${index}-${column}`">{{ row[column] ?? "" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script setup>
defineProps({
  columns: {
    type: Array,
    default: () => [],
  },
  rows: {
    type: Array,
    default: () => [],
  },
});
</script>
