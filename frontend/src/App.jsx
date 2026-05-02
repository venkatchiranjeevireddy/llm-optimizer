import React from "react";
import CostTracker from "./components/CostTracker";
import ModelPerformance from "./components/ModelPerformance";
import CacheMetrics from "./components/CacheMetrics";
import RouterAccuracy from "./components/RouterAccuracy";
import LatencyChart from "./components/LatencyChart";
import ABTestResults from "./components/ABTestResults";
import BudgetTracker from "./components/BudgetTracker";
import PromptLab from "./components/PromptLab";
import TokenCompressionHero from "./components/TokenCompressionHero";
import { fetchDashboardData } from "./api/client";

const NAV_ITEMS = [
  { label: "Try query", href: "#prompt-lab" },
  { label: "Compression", href: "#compression" },
  { label: "Cost", href: "#cost-intel" },
  { label: "Models", href: "#models" },
  { label: "Cache", href: "#cache" },
  { label: "Router", href: "#router" },
  { label: "Latency", href: "#latency" },
  { label: "A/B", href: "#ab" },
  { label: "Budget", href: "#budget" },
];

function App() {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState("");
  const [lastUpdated, setLastUpdated] = React.useState(null);

  const loadDashboard = React.useCallback(async () => {
    try {
      const payload = await fetchDashboardData();
      setData(payload);
      setError("");
      setLastUpdated(new Date());
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to load dashboard data.");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadDashboard();
    const timer = setInterval(loadDashboard, 30000);
    return () => clearInterval(timer);
  }, [loadDashboard]);

  const healthOk = data?.health?.status === "ok";

  return (
    <div className="mesh-page-bg min-h-screen">
      <div className="mx-auto flex max-w-[1820px]">
        <aside className="sidebar-glow sticky top-0 hidden h-screen w-64 border-r border-slate-800/80 bg-slate-950/90 p-6 backdrop-blur-md lg:block">
          <div className="mb-8 flex items-center gap-2">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-violet-600 text-lg font-bold text-white shadow-lg shadow-cyan-900/50">
              ◈
            </span>
            <div>
              <h1 className="font-display text-lg font-bold tracking-tight text-white">LLM Optimizer</h1>
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Cost-aware routing</p>
            </div>
          </div>
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="block rounded-lg px-3 py-2 text-sm text-slate-400 transition hover:bg-slate-800/90 hover:text-white"
              >
                {item.label}
              </a>
            ))}
          </nav>
        </aside>

        <main className="flex-1 p-4 md:p-8">
          <header className="glass-card-glow mb-8 flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="font-display text-2xl font-bold tracking-tight text-white md:text-3xl">
                <span className="title-gradient">Optimization Control Center</span>
              </h2>
              <p className="mt-1 text-sm text-slate-500">Live metrics · auto-refresh every 30s</p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div
                className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${
                  healthOk
                    ? "border-emerald-500/40 bg-emerald-950/40 text-emerald-200"
                    : "border-rose-500/40 bg-rose-950/40 text-rose-200"
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${healthOk ? "bg-emerald-400" : "bg-rose-400"}`} />
                {healthOk ? "Healthy" : "Degraded"}
              </div>
              <span className="font-mono text-xs text-slate-500">
                {lastUpdated ? lastUpdated.toLocaleTimeString() : "—"}
              </span>
            </div>
          </header>

          {loading && (
            <div className="glass-card animate-pulse p-8 text-center text-slate-400">Loading metrics…</div>
          )}
          {error && (
            <div className="glass-card mb-5 border-rose-500/40 bg-rose-950/30 p-5 text-sm text-rose-100">{error}</div>
          )}

          {!loading && data && (
            <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
              <PromptLab onQueryLogged={loadDashboard} />
              <TokenCompressionHero metrics={data.metrics} />
              <div id="cost-intel" className="scroll-mt-24">
                <CostTracker metrics={data.metrics} />
              </div>
              <div id="models" className="scroll-mt-24">
                <ModelPerformance modelComparison={data.modelComparison} />
              </div>
              <div id="cache" className="scroll-mt-24">
                <CacheMetrics cacheStats={data.cacheStats} />
              </div>
              <div id="router" className="scroll-mt-24">
                <RouterAccuracy routerHistory={data.routerHistory} />
              </div>
              <div id="latency" className="scroll-mt-24">
                <LatencyChart metrics={data.metrics} modelComparison={data.modelComparison} />
              </div>
              <div id="ab" className="scroll-mt-24">
                <ABTestResults abResults={data.abResults} />
              </div>
              <div id="budget" className="xl:col-span-2 scroll-mt-24">
                <BudgetTracker budgetStatus={data.budgetStatus} routerEngine={data.metrics?.router_engine} />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
