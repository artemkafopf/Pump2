<template>
  <section class="page-stack">
    <section class="hero-panel">
      <span class="hero-panel__eyebrow">Мобильная афиша</span>
      <h1>Вечера с мясом, вином и понятной записью без лишних шагов.</h1>
      <p>
        Выберите событие, проверьте остаток мест, получите QR на оплату и сразу увидите статус своей брони.
      </p>
    </section>

    <section class="section-heading">
      <div>
        <span class="kicker">Ближайшие события</span>
        <h2>Афиша</h2>
      </div>
    </section>

    <div v-if="store.loading" class="soft-panel">Загружаем афишу…</div>
    <div v-else class="card-list">
      <EventCard v-for="event in store.events" :key="event.id" :event="event" />
    </div>
  </section>
</template>

<script setup>
import { onMounted } from "vue";

import EventCard from "../components/EventCard.vue";
import { useEventsStore } from "../stores/events";

const store = useEventsStore();

onMounted(() => {
  store.loadEvents();
});
</script>
