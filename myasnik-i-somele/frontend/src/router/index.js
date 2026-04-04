import { createRouter, createWebHistory } from "vue-router";

import AdminDashboardView from "../views/AdminDashboardView.vue";
import AdminLoginView from "../views/AdminLoginView.vue";
import CalendarView from "../views/CalendarView.vue";
import EventView from "../views/EventView.vue";
import HomeView from "../views/HomeView.vue";
import PaymentView from "../views/PaymentView.vue";

const routes = [
  { path: "/", name: "home", component: HomeView, meta: { showBottomNav: true } },
  { path: "/calendar", name: "calendar", component: CalendarView, meta: { showBottomNav: true } },
  { path: "/event/:slug", name: "event", component: EventView, meta: { showBottomNav: true } },
  { path: "/booking/:token/payment", name: "payment", component: PaymentView, meta: { showBottomNav: true } },
  { path: "/admin", name: "admin-login", component: AdminLoginView, meta: { showBottomNav: false } },
  { path: "/admin/dashboard", name: "admin-dashboard", component: AdminDashboardView, meta: { showBottomNav: false } }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
