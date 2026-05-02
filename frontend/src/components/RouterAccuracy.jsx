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

function RouterAccuracy({ routerHistory }) {
  const chartData = (routerHistory || []).map((row, index) => ({
    event: index + 1,
    accuracy: Number(row.accuracy_after || 0) * 100,
    samples: row.training_samples || 0,
    reason: row.trigger_reason || "unknown",
  }));

  const hasData = chartData.length > 0;
  const accValues = chartData.map((d) => d.accuracy);
  const yMin = hasData ? Math.max(0, Math.min(...accValues) - 5) : 0;
  const yMax = hasData ? Math.min(100, Math.max(...accValues) + 5) : 100;

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">Router accuracy</h3>
      <p className="mb-4 text-sm text-slate-400">Logged router retraining events from PostgreSQL.</p>

      <div className="h-64">
        {!hasData ? (
          <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-700 text-sm text-slate-500">
            No retraining events yet — accuracy points appear after scheduled or triggered retrains.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="event" stroke="#94a3b8" />
              <YAxis domain={[yMin, yMax]} stroke="#94a3b8" />
              <Tooltip
                formatter={(value, name) => [`${Number(value).toFixed(2)}%`, name]}
                labelFormatter={(label, payload) => {
                  const row = payload?.[0]?.payload;
                  return `Event ${label} · samples: ${row?.samples || 0} · reason: ${row?.reason || "n/a"}`;
                }}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#34d399" strokeWidth={3} dot={{ r: 4 }} name="Accuracy %" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

export default RouterAccuracy;
