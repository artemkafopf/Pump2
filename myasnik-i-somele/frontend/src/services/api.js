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

function withAuth(token, options = {}) {
  return {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
      Authorization: `Bearer ${token}`,
    },
  };
}

export async function fetchPublicEvents() {
  const response = await fetch(`${API_BASE_URL}/public/events`);
  return parseResponse(response);
}

export async function fetchCalendar(month) {
  const response = await fetch(`${API_BASE_URL}/public/events/calendar?month=${month}`);
  return parseResponse(response);
}

export async function fetchPublicEvent(slug) {
  const response = await fetch(`${API_BASE_URL}/public/events/${slug}`);
  return parseResponse(response);
}

export async function createBooking(eventId, payload) {
  const response = await fetch(`${API_BASE_URL}/public/events/${eventId}/bookings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function fetchBooking(token) {
  const response = await fetch(`${API_BASE_URL}/public/bookings/${token}`);
  return parseResponse(response);
}

export async function adminLogin(payload) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function fetchAdminOverview(token) {
  const response = await fetch(`${API_BASE_URL}/admin/overview`, withAuth(token));
  return parseResponse(response);
}

export async function fetchAdminEvents(token) {
  const response = await fetch(`${API_BASE_URL}/admin/events`, withAuth(token));
  return parseResponse(response);
}

export async function createAdminEvent(token, payload) {
  const response = await fetch(
    `${API_BASE_URL}/admin/events`,
    withAuth(token, { method: "POST", body: JSON.stringify(payload) }),
  );
  return parseResponse(response);
}

export async function updateAdminEvent(token, eventId, payload) {
  const response = await fetch(
    `${API_BASE_URL}/admin/events/${eventId}`,
    withAuth(token, { method: "PATCH", body: JSON.stringify(payload) }),
  );
  return parseResponse(response);
}

export async function fetchAdminBookings(token) {
  const response = await fetch(`${API_BASE_URL}/admin/bookings`, withAuth(token));
  return parseResponse(response);
}

export async function updateAdminBooking(token, bookingId, payload) {
  const response = await fetch(
    `${API_BASE_URL}/admin/bookings/${bookingId}`,
    withAuth(token, { method: "PATCH", body: JSON.stringify(payload) }),
  );
  return parseResponse(response);
}
