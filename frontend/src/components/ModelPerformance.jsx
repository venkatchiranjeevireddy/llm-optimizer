import React from "react";

import { displayModelName } from "../modelDisplay";

function qualityClass(quality) {
  if (quality > 0.7) return "text-emerald-400";
  if (quality > 0.5) return "text-amber-400";
  return "text-rose-400";
}

function ModelPerformance({ modelComparison }) {
  const rows = Object.entries(modelComparison || {}).map(([model, values]) => ({
    model,
    calls: values.count || 0,
    latency: values.avg_latency || 0,
    quality: values.avg_quality || 0,
    avgCost: values.avg_cost || 0,
    totalCost: values.total_cost ?? values.avg_cost * (values.count || 0),
    success: Math.max(0, Math.min(100, (values.avg_quality || 0) * 100)),
  }));

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">Model performance</h3>
      <p className="mb-4 text-sm text-slate-400">
        Per routed tier from Postgres — labels show the real provider/model (Groq Llama vs Gemini Flash).
      </p>

      <div className="max-w-full overflow-x-auto">
        <table className="w-full table-fixed border-collapse text-left text-sm">
          <thead className="border-b border-slate-700 text-slate-400">
            <tr>
              <th className="w-[24%] py-2 pr-2">Model</th>
              <th className="w-[10%] py-2 pr-2">Calls</th>
              <th className="w-[15%] py-2 pr-2">Avg latency</th>
              <th className="w-[12%] py-2 pr-2">Avg quality</th>
              <th className="w-[13%] py-2 pr-2">Avg $/call</th>
              <th className="w-[13%] py-2 pr-2">Total $</th>
              <th className="w-[17%] py-2">Quality→%</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model} className="border-b border-slate-800">
                <td className="py-3 pr-2">
                  <span className="font-medium text-slate-100">{displayModelName(row.model)}</span>
                  <span className="mt-0.5 block truncate text-[11px] text-slate-500" title={row.model}>
                    {row.model}
                  </span>
                </td>
                <td className="py-3 pr-2">{row.calls}</td>
                <td className="py-3 pr-2">{row.latency.toFixed(1)} ms</td>
                <td className={`py-3 pr-2 font-semibold ${qualityClass(row.quality)}`}>{row.quality.toFixed(3)}</td>
                <td className="py-3 pr-2">${Number(row.avgCost).toFixed(6)}</td>
                <td className="py-3 pr-2">${Number(row.totalCost).toFixed(6)}</td>
                <td className="py-3">{row.success.toFixed(1)}%</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-500">
                  No model performance data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default ModelPerformance;
