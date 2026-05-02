import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ROUTE_LABELS } from "../modelDisplay";

function currency(value) {
  return `$${Number(value || 0).toFixed(6)}`;
}

function CostTracker({ metrics }) {
  const totalSaved = Number(metrics?.cost_saved_usd || 0);
  const actual = Number(metrics?.total_cost_usd || 0);
  const baseline = Number(metrics?.baseline_cost_usd || 0);
  const pctSaved = Number(metrics?.cost_saved_percentage || 0);
  const totals = metrics?.model_cost_totals_usd || {};

  const compareData = [
    { name: "Actual spend (routed)", cost: actual },
    { name: "All-Gemini estimate", cost: baseline },
  ];

  const byModel = ["groq_simple", "groq_mixtral", "gemini_pro"].map((model) => ({
    model,
    label: ROUTE_LABELS[model] || model,
    cost: Number(totals[model] || 0),
  }));

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display text-lg font-semibold tracking-tight text-white">Cost intelligence</h3>
      <p className="mb-4 text-sm text-slate-400">
        Aggregated from PostgreSQL <code className="text-slate-300">query_logs</code> after each API query.
        Baseline = estimated cost if every logged answer used Gemini Flash pricing with the same rough input/output
        sizes.
      </p>

      <div className="mb-5 rounded-xl bg-emerald-500/10 p-4">
        <p className="text-sm text-emerald-300">Estimated savings vs all-Gemini baseline</p>
        <p className="text-4xl font-bold text-emerald-200">{currency(totalSaved)}</p>
        <p className="mt-1 text-xs text-emerald-400/90">
          {pctSaved.toFixed(2)}% of baseline · Negative means routed stack cost more than that Gemini estimate for your
          workload.
        </p>
      </div>

      <div className="mb-5 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={compareData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
            <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v}`} />
            <Tooltip formatter={(v) => currency(v)} />
            <Legend />
            <Bar dataKey="cost" fill="#22d3ee" radius={[6, 6, 0, 0]} name="USD" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mb-5 grid grid-cols-1 gap-3 md:grid-cols-3">
        {byModel.map((item) => (
          <div key={item.model} className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
            <p className="text-xs font-medium leading-snug text-slate-300">{item.label}</p>
            <p className="mt-1 text-[10px] text-slate-600">{item.model}</p>
            <p className="mt-2 text-lg font-semibold">{currency(item.cost)}</p>
            <p className="text-[11px] text-slate-500">Sum of logged calls</p>
          </div>
        ))}
      </div>

      <p className="mt-4 text-center text-xs text-slate-500">
        Token compression stats live in the <a className="text-cyan-400 hover:text-cyan-300" href="#compression">compression strip</a>{" "}
        above · routing keys (<code className="text-slate-400">groq_*</code>) are internal IDs.
      </p>
    </section>
  );
}

export default CostTracker;
