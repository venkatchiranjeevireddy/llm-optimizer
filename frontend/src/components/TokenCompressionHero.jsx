import React from "react";

/** Aggregate compression from Postgres query_logs (Latin token sums). */
function TokenCompressionHero({ metrics }) {
  const tokOrig = Number(metrics?.token_original_total ?? 0);
  const tokComp = Number(metrics?.token_compressed_total ?? 0);
  const tokSaved = Number(metrics?.tokens_saved_total ?? 0);
  const pct =
    tokOrig > 0 ? Math.max(0, Math.min(100, ((tokOrig - tokComp) / tokOrig) * 100)) : 0;

  const zeroShrink = pct < 0.01 && tokOrig > 0;

  return (
    <section
      id="compression"
      className="relative overflow-hidden rounded-2xl border border-teal-500/25 bg-gradient-to-br from-slate-900/95 via-slate-900/90 to-cyan-950/40 p-6 shadow-xl shadow-cyan-950/30 scroll-mt-24"
    >
      <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 -left-16 h-56 w-56 rounded-full bg-violet-600/10 blur-3xl" />

      <div className="relative flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-xl">
          <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-teal-400/90">
            Pipeline step 0
          </p>
          <h3 className="font-display text-xl font-bold tracking-tight text-white md:text-2xl">
            Prompt compression
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-slate-400">
            Strips polite filler and near-duplicate sentences before routing. Metrics above sum{" "}
            <strong className="text-slate-300">Latin word tokens</strong> across logged queries (same regex as the backend).
          </p>
          <p className="mt-2 text-sm leading-relaxed text-slate-500">
            Seeing <strong className="text-slate-400">0% shrink</strong> is normal when prompts are already concise — nothing was wasted.
          </p>
        </div>

        <div className="flex w-full max-w-md flex-col gap-3 rounded-xl border border-slate-700/80 bg-slate-950/60 p-4 lg:w-auto lg:min-w-[320px]">
          <div className="flex justify-between text-xs text-slate-500">
            <span title="Share of Latin tokens removed vs original (aggregate)">Token shrink (aggregate)</span>
            <span className={`font-mono ${zeroShrink ? "text-slate-400" : "text-teal-400"}`}>
              {pct.toFixed(1)}% removed
            </span>
          </div>
          <div className="h-4 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-600 to-cyan-400 transition-all duration-500"
              style={{ width: `${pct}%` }}
              title={`${pct.toFixed(1)}% of aggregate Latin tokens removed by compression`}
            />
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="rounded-lg bg-slate-800/80 py-2">
              <p className="text-slate-500">Original Σ tokens</p>
              <p className="font-mono text-lg font-semibold text-white">{tokOrig.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-slate-800/80 py-2">
              <p className="text-slate-500">After trim Σ</p>
              <p className="font-mono text-lg font-semibold text-cyan-200">{tokComp.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-slate-800/80 py-2">
              <p className="text-slate-500">Removed Σ</p>
              <p className="font-mono text-lg font-semibold text-emerald-400">{tokSaved.toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="relative mt-6 rounded-xl border border-amber-500/30 bg-amber-950/25 px-4 py-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-amber-200/90">CJK · 中文 · 日本語 · 한국어</p>
        <p className="mt-1 text-sm leading-relaxed text-amber-100/85">
          Latin tokens can read <strong>0</strong> for East Asian text. Use <strong>Unicode chars</strong> in the playground row
          below after you send a prompt.
        </p>
      </div>
    </section>
  );
}

export default TokenCompressionHero;
