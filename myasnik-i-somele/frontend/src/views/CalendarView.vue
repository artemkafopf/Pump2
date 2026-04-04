<template>
  <section class="page-stack">
    <section class="section-heading">
      <div>
        <span class="kicker">Календарь</span>
        <h2>Мероприятия по дням</h2>
      </div>
      <input v-model="month" type="month" class="field-input field-input--compact" @change="loadMonth" />
    </section>

    <div v-if="store.loading" class="soft-panel">Собираем расписание…</div>
    <div v-else class="calendar-list">
      <article v-for="day in store.calendar" :key="day.date" class="calendar-day">
        <header>
          <strong>{{ formatDay(day.date) }}</strong>
        </header>
        <router-link
          v-for="event in day.events"
          :key="event.id"
          :to="`/event/${event.slug}`"
          class="calendar-day__event"
        >
          <span>{{ event.title }}</span>
          <strong>{{ formatTime(event.starts_at) }}</strong>
        </router-link>
      </article>
    </div>
  </section>
</template>

<script setup>
import { onMounted, ref } from "vue";

import { useEventsStore } from "../stores/events";

const store = useEventsStore();
const now = new Date();
const month = ref(`${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`);

function loadMonth() {
  store.loadCalendar(month.value);
}

function formatDay(value) {
  return new Intl.DateTimeFormat("ru-RU", {
    weekday: "long",
    day: "numeric",
    month: "long",
  }).format(new Date(value));
}

function formatTime(value) {
  return new Intl.DateTimeFormat("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

onMounted(loadMonth);
</script>
