import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 15000,
});

export const fetchHealth = async () => (await api.get("/health")).data;
export const fetchMetrics = async () => (await api.get("/api/metrics")).data;
export const fetchModelComparison = async () => (await api.get("/api/models/comparison")).data;
export const fetchCacheStats = async () => (await api.get("/api/cache/stats")).data;
export const fetchRouterHistory = async () => (await api.get("/api/router/history")).data;
export const fetchBudgetStatus = async () => (await api.get("/api/budget/status")).data;
export const fetchAbResults = async () => (await api.get("/api/ab-test/results")).data;

/** Recent rows from Postgres query_logs (same records that power metrics). */
export const fetchQueryHistory = async (limit = 40) =>
  (await api.get("/api/query/history", { params: { limit } })).data;

export const fetchDashboardData = async () => {
  const [health, metrics, modelComparison, cacheStats, routerHistory, budgetStatus, abResults] =
    await Promise.all([
      fetchHealth(),
      fetchMetrics(),
      fetchModelComparison(),
      fetchCacheStats(),
      fetchRouterHistory(),
      fetchBudgetStatus(),
      fetchAbResults(),
    ]);

  return { health, metrics, modelComparison, cacheStats, routerHistory, budgetStatus, abResults };
};

export default api;
