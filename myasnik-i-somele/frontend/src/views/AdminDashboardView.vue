<template>
  <section class="page-stack admin-page">
    <section class="section-heading">
      <div>
        <span class="kicker">Админ-панель</span>
        <h1>Управление событиями и бронями</h1>
      </div>
      <button class="ghost-chip" @click="logout">Выйти</button>
    </section>

    <div class="stats-grid">
      <article class="stat-card">
        <span>Событий</span>
        <strong>{{ overview?.total_events || 0 }}</strong>
      </article>
      <article class="stat-card">
        <span>Опубликовано</span>
        <strong>{{ overview?.published_events || 0 }}</strong>
      </article>
      <article class="stat-card">
        <span>Ожидают оплаты</span>
        <strong>{{ overview?.pending_bookings || 0 }}</strong>
      </article>
      <article class="stat-card">
        <span>Оплачено</span>
        <strong>{{ overview?.paid_bookings || 0 }}</strong>
      </article>
    </div>

    <form class="booking-form" @submit.prevent="saveEvent">
      <h2>{{ editingId ? "Редактировать событие" : "Быстрое создание" }}</h2>
      <input v-model.trim="form.title" class="field-input" type="text" placeholder="Название" required />
      <input v-model.trim="form.short_description" class="field-input" type="text" placeholder="Короткое описание" required />
      <textarea v-model.trim="form.description" class="field-input field-input--area" placeholder="Полное описание" required />
      <input v-model.trim="form.venue" class="field-input" type="text" placeholder="Локация" required />
      <input v-model="form.starts_at" class="field-input" type="datetime-local" required />
      <input v-model.number="form.capacity" class="field-input" type="number" min="1" placeholder="Мест" required />
      <input v-model.number="form.price_amount" class="field-input" type="number" min="0" placeholder="Цена" required />
      <select v-model="form.status" class="field-input" required>
        <option value="draft">Черновик</option>
        <option value="published">Опубликовано</option>
        <option value="archived">Архив</option>
      </select>
      <input v-model.trim="form.hero_note" class="field-input" type="text" placeholder="Короткий ярлык" />
      <button class="primary-button" :disabled="saving">{{ saving ? "Сохраняем…" : "Сохранить событие" }}</button>
    </form>

    <section class="section-heading">
      <div>
        <span class="kicker">Афиша</span>
        <h2>События</h2>
      </div>
    </section>

    <div class="card-list">
      <article v-for="event in events" :key="event.id" class="event-card">
        <div class="event-card__badge">{{ event.status }}</div>
        <h3>{{ event.title }}</h3>
        <p>{{ event.short_description }}</p>
        <div class="event-meta">
          <span>{{ formatDate(event.starts_at) }}</span>
          <span>{{ event.spots_left }} мест</span>
        </div>
        <div class="event-actions">
          <button class="ghost-chip" @click="startEdit(event)">Изменить</button>
          <button class="ghost-chip" @click="setStatus(event, 'published')">Публиковать</button>
          <button class="ghost-chip" @click="setStatus(event, 'archived')">Архив</button>
        </div>
      </article>
    </div>

    <section class="section-heading">
      <div>
        <span class="kicker">Журнал</span>
        <h2>Брони и оплаты</h2>
      </div>
    </section>

    <div class="soft-panel booking-log">
      <article v-for="booking in bookings" :key="booking.id" class="booking-log__item">
        <div>
          <strong>{{ booking.customer_name }}</strong>
          <p>{{ booking.event_title }}</p>
          <p>{{ booking.phone }} · {{ booking.email }}</p>
        </div>
        <div class="booking-log__actions">
          <span>{{ booking.status }}</span>
          <button class="ghost-chip" @click="markPaid(booking.id)">Оплачено</button>
          <button class="ghost-chip" @click="markCancelled(booking.id)">Отменить</button>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";

import { createAdminEvent, fetchAdminBookings, fetchAdminEvents, fetchAdminOverview, updateAdminBooking, updateAdminEvent } from "../services/api";
import { useAuthStore } from "../stores/auth";

const router = useRouter();
const auth = useAuthStore();
const overview = ref(null);
const events = ref([]);
const bookings = ref([]);
const saving = ref(false);
const editingId = ref(null);
const form = reactive({
  title: "",
  short_description: "",
  description: "",
  venue: "Иркутск, Карла Маркса, 5",
  starts_at: "",
  capacity: 18,
  price_amount: 4500,
  status: "draft",
  hero_note: "",
});

function resetForm() {
  editingId.value = null;
  form.title = "";
  form.short_description = "";
  form.description = "";
  form.venue = "Иркутск, Карла Маркса, 5";
  form.starts_at = "";
  form.capacity = 18;
  form.price_amount = 4500;
  form.status = "draft";
  form.hero_note = "";
}

function toPayload() {
  return {
    ...form,
    starts_at: new Date(form.starts_at).toISOString(),
  };
}

function formatDate(value) {
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

async function loadAll() {
  if (!auth.isAuthenticated) {
    router.push("/admin");
    return;
  }
  overview.value = await fetchAdminOverview(auth.token);
  events.value = await fetchAdminEvents(auth.token);
  bookings.value = await fetchAdminBookings(auth.token);
}

function startEdit(event) {
  editingId.value = event.id;
  form.title = event.title;
  form.short_description = event.short_description;
  form.description = event.description || "";
  form.venue = event.venue;
  form.starts_at = new Date(event.starts_at).toISOString().slice(0, 16);
  form.capacity = event.capacity;
  form.price_amount = Number(event.price_amount);
  form.status = event.status;
  form.hero_note = event.hero_note || "";
}

async function saveEvent() {
  saving.value = true;
  try {
    if (editingId.value) {
      await updateAdminEvent(auth.token, editingId.value, toPayload());
    } else {
      await createAdminEvent(auth.token, toPayload());
    }
    resetForm();
    await loadAll();
  } finally {
    saving.value = false;
  }
}

async function setStatus(event, status) {
  await updateAdminEvent(auth.token, event.id, { status });
  await loadAll();
}

async function markPaid(bookingId) {
  await updateAdminBooking(auth.token, bookingId, { payment_status: "paid" });
  await loadAll();
}

async function markCancelled(bookingId) {
  await updateAdminBooking(auth.token, bookingId, { payment_status: "cancelled" });
  await loadAll();
}

function logout() {
  auth.logout();
  router.push("/admin");
}

onMounted(loadAll);
</script>
