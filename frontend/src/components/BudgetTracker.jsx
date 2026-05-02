import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function BudgetTracker({ budgetStatus, routerEngine }) {
  const used = Number(budgetStatus?.used_percent || 0);
  const remaining = Number(budgetStatus?.remaining_percent || 0);
  const alert = used >= 80;
  const lowBudget = remaining <= 20;

  const trend = Array.from({ length: 8 }).map((_, idx) => {
    const hour = `${idx * 3}:00`;
    const factor = (idx + 1) / 8;
    return { hour, value: Math.min(100, used * factor) };
  });

  return (
    <section className="glass-card-glow p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">Budget & router</h3>
      <p className="mb-4 text-sm text-slate-400">
        Daily budget from Postgres spend · Routing algorithm is reported below (not implied by “normal”).
      </p>

      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4">
          <p className="text-xs uppercase text-slate-400">Used</p>
          <p className="text-3xl font-bold">{used.toFixed(2)}%</p>
          <div className="mt-3 h-2 rounded-full bg-slate-700">
            <div
              className={`h-2 rounded-full ${alert ? "bg-rose-400" : "bg-emerald-400"}`}
              style={{ width: `${Math.min(100, used)}%` }}
            />
          </div>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4">
          <p className="text-xs uppercase text-slate-400">Remaining</p>
          <p className="text-3xl font-bold text-cyan-300">{remaining.toFixed(2)}%</p>
          <p className="mt-2 text-xs text-slate-400">Budget refreshes at UTC midnight.</p>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4">
          <p className="text-xs uppercase text-slate-400">Router engine</p>
          <p className="text-lg font-semibold leading-snug text-cyan-100">
            {routerEngine?.title || "Loading…"}
          </p>
          <p className="mt-2 text-xs leading-relaxed text-slate-400">{routerEngine?.detail}</p>
          <p className="mt-3 border-t border-slate-700 pt-3 text-xs uppercase text-slate-500">Budget guardrail</p>
          <p className={`text-sm font-medium ${lowBudget ? "text-amber-300" : "text-emerald-300"}`}>
            {lowBudget ? "Low remaining budget (≤20%) — tighten usage" : "Headroom OK (>20% remaining)"}
          </p>
          <p className="mt-1 text-[11px] text-slate-500">
            Retraining uses XGBoost in code when samples exist; runtime routing uses the engine above if pickle loaded.
          </p>
          {alert && <p className="mt-2 text-xs text-rose-300">Alert: Daily budget exceeded 80%.</p>}
        </div>
      </div>

      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="hour" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" domain={[0, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default BudgetTracker;
