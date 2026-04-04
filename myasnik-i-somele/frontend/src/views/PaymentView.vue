<template>
  <section v-if="booking" class="page-stack">
    <section class="hero-panel hero-panel--detail">
      <span class="hero-panel__eyebrow">Оплата</span>
      <h1>{{ booking.event.title }}</h1>
      <p>Статус брони: <strong>{{ statusLabel(booking.status) }}</strong></p>
    </section>

    <section class="soft-panel qr-panel">
      <img v-if="qrCodeDataUrl" :src="qrCodeDataUrl" alt="QR-код для оплаты" class="qr-image" />
      <div class="info-row"><span>Сумма</span><strong>{{ formatMoney(booking.amount_due) }}</strong></div>
      <div class="info-row"><span>Получатель</span><strong>{{ booking.payment?.recipient_name }}</strong></div>
      <div class="info-row"><span>Банк</span><strong>{{ booking.payment?.recipient_bank }}</strong></div>
      <div class="info-row"><span>Телефон</span><strong>{{ booking.payment?.recipient_phone }}</strong></div>
      <div class="info-row"><span>Назначение</span><strong>{{ booking.payment?.payment_purpose }}</strong></div>
    </section>

    <section class="soft-panel">
      <p class="muted-text">
        После перевода администратор вручную подтвердит оплату в админ-панели. Эту страницу можно открыть снова по ссылке.
      </p>
    </section>
  </section>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import QRCode from "qrcode";

import { fetchBooking } from "../services/api";

const route = useRoute();
const booking = ref(null);
const qrCodeDataUrl = ref("");

function formatMoney(value) {
  return new Intl.NumberFormat("ru-RU", {
    style: "currency",
    currency: "RUB",
    maximumFractionDigits: 0,
  }).format(value);
}

function statusLabel(status) {
  return (
    {
      pending_payment: "ожидает оплаты",
      paid: "оплачено",
      cancelled: "отменено",
      sold_out: "мест нет",
    }[status] || status
  );
}

onMounted(async () => {
  booking.value = await fetchBooking(route.params.token);
  if (booking.value?.payment?.qr_payload) {
    qrCodeDataUrl.value = await QRCode.toDataURL(booking.value.payment.qr_payload, {
      width: 320,
      margin: 1,
      color: { dark: "#2d1a16", light: "#f7efe6" },
    });
  }
});
</script>
