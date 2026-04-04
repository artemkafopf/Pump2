import { defineStore } from "pinia";

import { adminLogin } from "../services/api";

const TOKEN_KEY = "mis_admin_token";
const NAME_KEY = "mis_admin_name";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    token: localStorage.getItem(TOKEN_KEY) || "",
    adminName: localStorage.getItem(NAME_KEY) || "",
    loading: false,
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.token),
  },
  actions: {
    async login(payload) {
      this.loading = true;
      try {
        const result = await adminLogin(payload);
        this.token = result.access_token;
        this.adminName = result.admin_name;
        localStorage.setItem(TOKEN_KEY, this.token);
        localStorage.setItem(NAME_KEY, this.adminName);
      } finally {
        this.loading = false;
      }
    },
    logout() {
      this.token = "";
      this.adminName = "";
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(NAME_KEY);
    },
  },
});
