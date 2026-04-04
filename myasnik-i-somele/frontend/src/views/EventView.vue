<template>
  <section v-if="store.activeEvent" class="page-stack">
    <section class="hero-panel hero-panel--detail">
      <span class="hero-panel__eyebrow">{{ store.activeEvent.hero_note || "Событие" }}</span>
      <h1>{{ store.activeEvent.title }}</h1>
      <p>{{ store.activeEvent.description }}</p>
    </section>

    <section class="soft-panel">
      <div class="info-row"><span>Когда</span><strong>{{ formatDate(store.activeEvent.starts_at) }}</strong></div>
      <div class="info-row"><span>Где</span><strong>{{ store.activeEvent.venue }}</strong></div>
      <div class="info-row"><span>Цена</span><strong>{{ formatMoney(store.activeEvent.price_amount) }}</strong></div>
      <div class="info-row"><span>Свободно</span><strong>{{ store.activeEvent.spots_left }} мест</strong></div>
    </section>

    <form class="booking-form" @submit.prevent="submitBooking">
      <h2>Записаться</h2>
      <input v-model.trim="form.customer_name" class="field-input" type="text" placeholder="Имя" required />
      <input v-model.trim="form.phone" class="field-input" type="tel" placeholder="Телефон" required />
      <input v-model.trim="form.email" class="field-input" type="email" placeholder="Email" required />
      <input v-model.number="form.seats" class="field-input" type="number" min="1" max="8" placeholder="Места" />
      <textarea
        v-model.trim="form.customer_comment"
        class="field-input field-input--area"
        placeholder="Комментарий, если нужен"
      />
      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
      <button class="primary-button" :disabled="submitting">
        {{ submitting ? "Создаём бронь…" : "Получить QR для оплаты" }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";

import { createBooking } from "../services/api";
import { useEventsStore } from "../stores/events";

const route = useRoute();
const router = useRouter();
const store = useEventsStore();
const submitting = ref(false);
const errorMessage = ref("");
const form = reactive({
  customer_name: "",
  phone: "",
  email: "",
  seats: 1,
  customer_comment: "",
});

onMounted(() => {
  store.loadEvent(route.params.slug);
});

function formatDate(value) {
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatMoney(value) {
  return new Intl.NumberFormat("ru-RU", {
    style: "currency",
    currency: "RUB",
    maximumFractionDigits: 0,
  }).format(value);
}

async function submitBooking() {
  if (!store.activeEvent) {
    return;
  }

  submitting.value = true;
  errorMessage.value = "";
  try {
    const result = await createBooking(store.activeEvent.id, form);
    if (result.status === "sold_out") {
      errorMessage.value = result.message;
      await store.loadEvent(route.params.slug);
      return;
    }
    await router.push(`/booking/${result.booking_token}/payment`);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    submitting.value = false;
  }
}
</script>
