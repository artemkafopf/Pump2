<template>
  <div class="app-shell">
    <header class="topbar">
      <router-link to="/" class="brand-mark">
        <span class="brand-mark__eyebrow">Гастрономические вечера</span>
        <strong>Мясник и Сомелье</strong>
      </router-link>
      <router-link to="/admin" class="ghost-chip">Админ</router-link>
    </header>

    <main class="page-shell">
      <router-view />
    </main>

    <BottomNav v-if="showBottomNav" />
  </div>
</template>

<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";

import BottomNav from "./components/BottomNav.vue";

const route = useRoute();
const showBottomNav = computed(() => route.meta.showBottomNav !== false);
</script>
