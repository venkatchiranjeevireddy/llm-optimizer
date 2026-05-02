import React from "react";

import { displayModelName } from "../modelDisplay";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function LatencyChart({ metrics, modelComparison }) {
  const avgLatency = Number(metrics?.avg_latency_ms || 0);
  const cacheHitRate = Number(metrics?.cache_hit_rate || 0);
  const hitMs = Number(metrics?.avg_latency_cache_hit_ms ?? 0);
  const missMs = Number(metrics?.avg_latency_cache_miss_ms ?? 0);

  const areaData = [
    { name: "Cache-hit avg", latency: hitMs || (cacheHitRate > 0 ? avgLatency : 0) },
    { name: "Cache-miss avg", latency: missMs || avgLatency },
    { name: "Overall avg", latency: avgLatency },
  ];

  const barData = Object.entries(modelComparison || {}).map(([model, values]) => ({
    model: displayModelName(model),
    latency: Number(values?.avg_latency || 0),
  }));

  const noHits = cacheHitRate <= 0 || (!hitMs && !missMs);

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">Latency</h3>
      <p className="mb-4 text-sm text-slate-400">
        Hit rate {cacheHitRate.toFixed(2)}% · Bars use logged{" "}
        <strong className="text-slate-200">average latency per model</strong> from PostgreSQL (not simulated curves).
      </p>

      <div className="mb-5 h-52">
        {noHits ? (
          <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-700 text-sm text-slate-500">
            No cache hits yet — cache-hit latency will appear once semantic cache serves requests.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={areaData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#94a3b8" }} />
              <Tooltip formatter={(v) => [`${Number(v).toFixed(1)} ms`, "Latency"]} />
              <Area type="monotone" dataKey="latency" stroke="#22d3ee" fill="#22d3ee44" name="Avg latency (ms)" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="model" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} angle={-12} textAnchor="end" height={56} />
            <YAxis stroke="#94a3b8" />
            <Tooltip formatter={(v) => [`${Number(v).toFixed(1)} ms`, "Avg latency"]} />
            <Bar dataKey="latency" fill="#60a5fa" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default LatencyChart;
