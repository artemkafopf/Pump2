const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001/api";

async function parseResponse(response) {
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function listDatasets() {
  const response = await fetch(`${API_BASE_URL}/datasets`);
  return parseResponse(response);
}

export async function uploadDataset({ datasetName, file, storageSection }) {
  const formData = new FormData();
  formData.append("dataset_name", datasetName);
  formData.append("storage_section", storageSection || "fact_epu");
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

export async function fetchDataset(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`);
  return parseResponse(response);
}

export async function fetchAnalysis(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/analysis`);
  return parseResponse(response);
}

export async function updateDatasetSelection(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/selection`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      target_column: payload.targetColumn,
      selected_features: payload.selectedFeatures,
    }),
  });
  return parseResponse(response);
}

export async function trainForecastModel(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/forecast/train`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function listSavedForecastModels(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/forecast/models`);
  return parseResponse(response);
}

export async function saveForecastModel(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/forecast/models`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function predictForecast(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/forecast/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function exportForecastContour(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/forecast/contour-export`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  return response.blob();
}

export async function fetchLLMStatus() {
  const response = await fetch(`${API_BASE_URL}/llm/status`);
  return parseResponse(response);
}

export async function fetchVariableDictionary() {
  const response = await fetch(`${API_BASE_URL}/variables/dictionary`);
  return parseResponse(response);
}

export async function clearVariableDictionary() {
  const response = await fetch(`${API_BASE_URL}/variables/dictionary`, {
    method: "DELETE",
  });
  return parseResponse(response);
}

export async function fetchEntityDictionary() {
  const response = await fetch(`${API_BASE_URL}/entities/dictionary`);
  return parseResponse(response);
}

export async function fetchVariableMatches(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/variables/matches`);
  return parseResponse(response);
}

export async function reconcileVariables(datasetId, payload = {}) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/variables/reconcile`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function saveManualVariableMatches(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/variables/manual`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function fetchEntityMatches(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/entities/matches`);
  return parseResponse(response);
}

export async function reconcileEntities(datasetId, payload = {}) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/entities/reconcile`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function saveManualEntityMatches(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/entities/manual`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function fetchReports(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/reports`);
  return parseResponse(response);
}

export async function generateReport(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/reports/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function calculateRepairForecast(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function previewRepairForecastTail(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/tail-preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function listRepairForecastCalculations(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/calculations`);
  return parseResponse(response);
}

export async function fetchLatestRepairForecastCalculation(datasetId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/calculations/latest`);
  return parseResponse(response);
}

export async function fetchRepairForecastCalculation(datasetId, calculationId) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/calculations/${calculationId}`);
  return parseResponse(response);
}

export async function saveRepairForecastCalculation(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/calculations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function exportRepairForecast(datasetId, payload) {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/repair-forecast/export`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.blob();
}
