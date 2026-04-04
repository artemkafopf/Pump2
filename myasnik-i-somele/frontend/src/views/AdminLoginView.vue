<template>
  <section class="admin-auth">
    <form class="booking-form" @submit.prevent="submit">
      <h1>Вход администратора</h1>
      <input v-model.trim="email" class="field-input" type="email" placeholder="Email" required />
      <input v-model.trim="password" class="field-input" type="password" placeholder="Пароль" required />
      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
      <button class="primary-button" :disabled="auth.loading">
        {{ auth.loading ? "Входим…" : "Войти" }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";

import { useAuthStore } from "../stores/auth";

const auth = useAuthStore();
const router = useRouter();
const email = ref("admin@mis.local");
const password = ref("Admin123!");
const errorMessage = ref("");

async function submit() {
  errorMessage.value = "";
  try {
    await auth.login({ email: email.value, password: password.value });
    router.push("/admin/dashboard");
  } catch (error) {
    errorMessage.value = error.message;
  }
}
</script>
