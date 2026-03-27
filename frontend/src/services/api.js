const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

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
