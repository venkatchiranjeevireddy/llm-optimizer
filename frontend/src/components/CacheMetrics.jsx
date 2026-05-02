import React from "react";

/** Redis / FAISS semantic cache — avoids LLM calls when a similar prompt was answered before. */
function CacheMetrics({ cacheStats }) {
  const hitRate = Number(cacheStats?.hit_rate ?? 0);
  const entries = Number(cacheStats?.total_entries ?? 0);
  const cold = entries === 0 && hitRate === 0;

  return (
    <section className="glass-card-glow overflow-hidden p-5">
      <h3 className="font-display mb-1 text-lg font-semibold tracking-tight text-white">Semantic cache</h3>
      <p className="mb-4 text-sm text-slate-400">
        Shows how often similar prompts are answered from Redis/FAISS instead of calling the model again. Different from{" "}
        <code className="text-slate-300">query_logs</code> (full pipeline history).
      </p>

      <div className="mb-5 flex h-36 items-center justify-center rounded-full border border-cyan-500/40 bg-cyan-500/10">
        <div className="text-center">
          <p className="text-xs uppercase text-cyan-300">Hit rate</p>
          <p className="text-4xl font-bold text-cyan-100">{hitRate.toFixed(2)}%</p>
        </div>
      </div>

      {cold && (
        <p className="mb-4 rounded-lg border border-slate-700/80 bg-slate-900/60 px-3 py-2 text-xs text-slate-400">
          No entries yet — repeat or paraphrase a prompt to see similarity hits climb.
        </p>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <p className="text-xs text-slate-400">Entries</p>
          <p className="text-lg font-semibold">{entries}</p>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <p className="text-xs text-slate-400">Avg match quality</p>
          <p className="text-lg font-semibold">{Number(cacheStats?.avg_quality ?? 0).toFixed(3)}</p>
        </div>
        <div className="col-span-2 rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <p className="text-xs text-slate-400">Est. cost avoided (cache)</p>
          <p className="text-lg font-semibold text-emerald-300">
            ${Number(cacheStats?.estimated_cost_saved ?? 0).toFixed(6)}
          </p>
        </div>
      </div>
    </section>
  );
}

export default CacheMetrics;
