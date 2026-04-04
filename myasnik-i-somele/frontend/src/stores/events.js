import { defineStore } from "pinia";

import { fetchCalendar, fetchPublicEvent, fetchPublicEvents } from "../services/api";

export const useEventsStore = defineStore("events", {
  state: () => ({
    events: [],
    calendar: [],
    loading: false,
    activeEvent: null,
  }),
  actions: {
    async loadEvents() {
      this.loading = true;
      try {
        this.events = await fetchPublicEvents();
      } finally {
        this.loading = false;
      }
    },
    async loadCalendar(month) {
      this.loading = true;
      try {
        this.calendar = await fetchCalendar(month);
      } finally {
        this.loading = false;
      }
    },
    async loadEvent(slug) {
      this.loading = true;
      try {
        this.activeEvent = await fetchPublicEvent(slug);
      } finally {
        this.loading = false;
      }
    },
  },
});
