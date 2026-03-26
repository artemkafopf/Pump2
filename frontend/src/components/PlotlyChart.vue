<template>
  <div ref="chartRef" class="plotly-chart"></div>
</template>

<script setup>
import Plotly from "plotly.js-dist-min";
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

const props = defineProps({
  data: {
    type: Array,
    default: () => [],
  },
  layout: {
    type: Object,
    default: () => ({}),
  },
  config: {
    type: Object,
    default: () => ({}),
  },
});

const chartRef = ref(null);

async function renderChart() {
  await nextTick();
  if (!chartRef.value) {
    return;
  }
  await Plotly.react(chartRef.value, props.data, props.layout, {
    responsive: true,
    displaylogo: false,
    ...props.config,
  });
}

onMounted(renderChart);

watch(
  () => [props.data, props.layout, props.config],
  renderChart,
  { deep: true },
);

onBeforeUnmount(() => {
  if (chartRef.value) {
    Plotly.purge(chartRef.value);
  }
});
</script>
