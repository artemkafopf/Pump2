<template>
  <div ref="chartRef" class="plotly-chart"></div>
</template>

<script setup>
import Plotly from "plotly.js-dist-min";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

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

function mergeObjects(base, override) {
  if (!override) {
    return { ...base };
  }

  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      base[key] &&
      typeof base[key] === "object" &&
      !Array.isArray(base[key])
    ) {
      result[key] = mergeObjects(base[key], value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

const themedLayout = computed(() => {
  const baseLayout = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.02)",
    colorway: ["#8b5cf6", "#60a5fa", "#22c55e", "#f59e0b", "#f97316", "#38bdf8", "#f472b6"],
    font: {
      family: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      size: 13,
      color: "#e5e7eb",
    },
    title: {
      font: {
        family: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        size: 16,
        color: "#e5e7eb",
      },
    },
    margin: { l: 60, r: 24, t: 18, b: 56 },
    hoverlabel: {
      bgcolor: "rgba(18, 18, 18, 0.94)",
      bordercolor: "rgba(255,255,255,0.08)",
      font: {
        family: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        size: 12,
        color: "#e5e7eb",
      },
    },
    legend: {
      orientation: "h",
      y: 1.1,
      x: 0,
      bgcolor: "rgba(21,24,33,0.72)",
      bordercolor: "rgba(255,255,255,0.06)",
      borderwidth: 1,
      font: {
        color: "#cbd5e1",
        size: 12,
      },
    },
    xaxis: {
      automargin: true,
      zeroline: false,
      showline: false,
      gridcolor: "rgba(255,255,255,0.08)",
      tickcolor: "rgba(255,255,255,0.12)",
      tickfont: { color: "#cbd5e1", size: 12 },
      title: { font: { color: "#e5e7eb", size: 13 } },
    },
    yaxis: {
      automargin: true,
      zeroline: false,
      showline: false,
      gridcolor: "rgba(255,255,255,0.08)",
      tickcolor: "rgba(255,255,255,0.12)",
      tickfont: { color: "#cbd5e1", size: 12 },
      title: { font: { color: "#e5e7eb", size: 13 } },
    },
  };

  return mergeObjects(baseLayout, props.layout || {});
});

const themedConfig = computed(() => ({
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: {
    format: "png",
    filename: "chart",
    scale: 2,
  },
  ...props.config,
}));

async function renderChart() {
  await nextTick();
  if (!chartRef.value) {
    return;
  }
  await Plotly.react(chartRef.value, props.data, themedLayout.value, themedConfig.value);
}

onMounted(renderChart);

watch(
  () => [props.data, themedLayout.value, themedConfig.value],
  renderChart,
  { deep: true },
);

onBeforeUnmount(() => {
  if (chartRef.value) {
    Plotly.purge(chartRef.value);
  }
});
</script>
