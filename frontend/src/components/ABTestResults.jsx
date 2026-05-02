import React from "react";

function ABTestResults({ abResults }) {
  const rows = [];
  Object.entries(abResults || {}).forEach(([intent, groups]) => {
    const groupA = groups?.A;
    const groupB = groups?.B;
    rows.push({
      intent,
      modelA: groupA?.winner || "router_default",
      modelB: groupB?.winner || "next_model_up",
      scoreA: Number(groupA?.avg_score || 0),
      scoreB: Number(groupB?.avg_score || 0),
      winner: Number(groupA?.avg_score || 0) >= Number(groupB?.avg_score || 0) ? "A" : "B",
      comparisons: (groupA?.comparisons || 0) + (groupB?.comparisons || 0),
    });
  });

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">A/B tests</h3>
      <p className="mb-4 text-sm text-slate-400">Quality comparison by intent (fills when evaluation records A/B buckets).</p>

      <div className="max-w-full overflow-x-auto">
        <table className="w-full table-auto text-left text-sm">
          <thead className="border-b border-slate-700 text-slate-400">
            <tr>
              <th className="py-2 pr-3">Query type</th>
              <th className="py-2 pr-3">Model A</th>
              <th className="py-2 pr-3">Model B</th>
              <th className="py-2 pr-3">Score A</th>
              <th className="py-2 pr-3">Score B</th>
              <th className="py-2">Winner</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.intent} className="border-b border-slate-800">
                <td className="py-3 pr-3 font-medium">{row.intent}</td>
                <td className="py-3 pr-3">{row.modelA}</td>
                <td className="py-3 pr-3">{row.modelB}</td>
                <td className="py-3 pr-3">{row.scoreA.toFixed(3)}</td>
                <td className="py-3 pr-3">{row.scoreB.toFixed(3)}</td>
                <td className={`py-3 font-semibold ${row.winner === "A" ? "text-emerald-400" : "text-cyan-300"}`}>
                  Group {row.winner}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-500">
                  No A/B comparisons available yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default ABTestResults;
