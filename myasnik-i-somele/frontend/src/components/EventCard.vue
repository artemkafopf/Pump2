<template>
  <article class="event-card">
    <div class="event-card__badge">{{ event.hero_note || "Мероприятие" }}</div>
    <h3>{{ event.title }}</h3>
    <p>{{ event.short_description }}</p>
    <div class="event-meta">
      <span>{{ formatDate(event.starts_at) }}</span>
      <span>{{ event.spots_left }} мест</span>
    </div>
    <div class="event-meta">
      <span>{{ event.venue }}</span>
      <strong>{{ formatMoney(event.price_amount) }}</strong>
    </div>
    <router-link :to="`/event/${event.slug}`" class="primary-button">Открыть</router-link>
  </article>
</template>

<script setup>
defineProps({
  event: {
    type: Object,
    required: true,
  },
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
</script>
