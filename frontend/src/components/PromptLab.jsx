import React from "react";
import api, { fetchQueryHistory } from "../api/client";
import { displayModelName } from "../modelDisplay";

function CompressionStrip({ c }) {
  if (!c) return null;
  const ot = Number(c.original_tokens ?? 0);
  const ct = Number(c.compressed_tokens ?? 0);
  const saved = Number(c.tokens_saved ?? 0);
  const oc = Number(c.original_chars ?? 0);
  const cc = Number(c.compressed_chars ?? 0);
  const cs = Number(c.chars_saved ?? 0);
  const shrinkPct = ot > 0 ? Math.min(100, (saved / ot) * 100) : 0;
  const noop = saved === 0 && cs === 0;

  return (
    <div className="mt-4 rounded-xl border border-teal-500/30 bg-gradient-to-r from-teal-950/50 to-slate-900/80 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wider text-teal-400/90">Compression (this query)</p>
        {noop ? (
          <span className="rounded-full border border-slate-600/80 bg-slate-900/80 px-2 py-0.5 text-[11px] text-slate-400">
            No trim — text was already lean
          </span>
        ) : (
          <span className="font-mono text-[11px] text-teal-400/90">{shrinkPct.toFixed(1)}% Latin tokens removed</span>
        )}
      </div>
      <div className="mb-3 h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-gradient-to-r from-teal-500 to-cyan-400 transition-all"
          style={{ width: `${shrinkPct}%` }}
          title={`${shrinkPct.toFixed(1)}% of Latin tokens removed`}
        />
      </div>
      <div className="grid grid-cols-2 gap-3 font-mono text-xs md:grid-cols-4">
        <div>
          <p className="text-slate-500">Latin tokens (orig → trimmed)</p>
          <p className="text-sm font-semibold text-white">
            {ot} → {ct} <span className="text-emerald-400">(Δ {saved})</span>
          </p>
        </div>
        <div>
          <p className="text-slate-500">Unicode chars</p>
          <p className="text-sm font-semibold text-white">
            {oc} → {cc} <span className="text-cyan-300">(Δ {cs})</span>
          </p>
        </div>
        <div className="col-span-2 text-[11px] leading-relaxed text-slate-400 md:col-span-2">
          {ot === 0 && oc > 0
            ? "No Latin tokens counted — rely on char counts for CJK."
            : "Token counts match the backend regex; chars are full Unicode length."}
        </div>
      </div>
    </div>
  );
}

function PromptLab({ onQueryLogged }) {
  const [prompt, setPrompt] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [lastReply, setLastReply] = React.useState(null);
  const [history, setHistory] = React.useState([]);

  const loadHistory = React.useCallback(async () => {
    try {
      const data = await fetchQueryHistory(40);
      setHistory(data.items || []);
    } catch (e) {
      setHistory([]);
    }
  }, []);

  React.useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  async function submit(e) {
    e.preventDefault();
    const text = prompt.trim();
    if (!text) return;
    setBusy(true);
    setError("");
    try {
      const { data } = await api.post("/api/query", { prompt: text }, { timeout: 120000 });
      setLastReply(data);
      setPrompt("");
      await loadHistory();
      onQueryLogged?.();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : JSON.stringify(detail || err.message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section id="prompt-lab" className="glass-card-glow relative overflow-hidden p-6 xl:col-span-2">
      <div className="pointer-events-none absolute right-0 top-0 h-32 w-32 rounded-full bg-violet-600/10 blur-2xl" />
      <h3 className="font-display relative text-xl font-bold tracking-tight text-white">Playground · history</h3>
      <p className="relative mb-5 text-sm text-slate-400">
        Full pipeline → Postgres <code className="text-cyan-400/90">query_logs</code>. Refresh charts after each send.
      </p>

      <form onSubmit={submit} className="relative mb-6 flex flex-col gap-3 md:flex-row md:items-start">
        <textarea
          className="min-h-[100px] flex-1 resize-y rounded-xl border border-slate-600/90 bg-slate-950/90 px-4 py-3 text-sm text-slate-100 shadow-inner placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
          placeholder="Ask in English or any language — char stats update below…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !prompt.trim()}
          className="rounded-xl bg-gradient-to-r from-cyan-600 to-teal-600 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-cyan-900/40 transition hover:from-cyan-500 hover:to-teal-500 disabled:cursor-not-allowed disabled:from-slate-700 disabled:to-slate-700 disabled:shadow-none"
        >
          {busy ? "Running…" : "Run pipeline"}
        </button>
      </form>

      {error && (
        <div className="mb-4 rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-100">{error}</div>
      )}

      {lastReply && (
        <div className="relative mb-6 rounded-xl border border-slate-700/90 bg-slate-950/60 p-5 shadow-inner">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium text-slate-300">Last response</p>
            {lastReply.cache_hit ? (
              <span
                className="rounded-lg border border-emerald-500/45 bg-emerald-950/55 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-200"
                title="Answer came from Redis/FAISS semantic cache, not a new model call."
              >
                Cache hit
              </span>
            ) : (
              <span
                className="rounded-lg border border-slate-600/70 bg-slate-900/80 px-3 py-1 text-[11px] text-slate-400"
                title="Full pipeline. Answers enter the semantic cache only when quality score is above 0.7 — repeating the same prompt may still be a miss."
              >
                Cache miss
              </span>
            )}
          </div>
          <p className="max-h-52 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
            {lastReply.response}
          </p>
          <CompressionStrip c={lastReply.compression} />
          <dl className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-400 md:grid-cols-4">
            <div>
              <dt className="text-slate-500">Model</dt>
              <dd className="font-medium text-slate-100">{displayModelName(lastReply.model_used)}</dd>
              <dd className="font-mono text-[10px] text-slate-600">{lastReply.model_used}</dd>
            </div>
            <div>
              <dt className="text-slate-500">cost_usd</dt>
              <dd className="font-mono text-slate-200">{lastReply.cost_usd}</dd>
            </div>
            <div>
              <dt className="text-slate-500">latency_ms</dt>
              <dd className="font-mono text-slate-200">{lastReply.latency_ms}</dd>
            </div>
            <div>
              <dt className="text-slate-500">cache_hit</dt>
              <dd className="text-slate-200">{String(lastReply.cache_hit)}</dd>
            </div>
          </dl>
        </div>
      )}

      <div>
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-medium text-slate-300">Recent queries</p>
          <button type="button" onClick={loadHistory} className="text-xs font-medium text-cyan-400 hover:text-cyan-300">
            Refresh
          </button>
        </div>
        <div className="max-h-[340px] overflow-x-auto overflow-y-auto rounded-xl border border-slate-700/80">
          <table className="w-full min-w-[640px] text-left text-xs text-slate-300">
            <thead className="sticky top-0 z-10 bg-slate-900/98 text-[10px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2">When</th>
                <th className="px-3 py-2">Prompt</th>
                <th className="px-3 py-2">Tier</th>
                <th className="px-3 py-2">$</th>
                <th className="px-3 py-2" title="Latin token delta (pipeline regex)">
                  Δ tok EN
                </th>
                <th className="px-3 py-2" title="Unicode length of original prompt — use for CJK">
                  Chars
                </th>
              </tr>
            </thead>
            <tbody>
              {history.map((row) => (
                <tr key={row.id} className="border-t border-slate-800/90 align-top hover:bg-slate-900/50">
                  <td className="whitespace-nowrap px-3 py-2.5 text-slate-500">
                    {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
                  </td>
                  <td className="max-w-[240px] px-3 py-2.5 md:max-w-[320px]">
                    <span className="line-clamp-2" title={row.prompt_preview}>
                      {row.prompt_preview}
                    </span>
                  </td>
                  <td className="min-w-[130px] px-3 py-2.5">
                    <span className="text-slate-200">{displayModelName(row.routed_to)}</span>
                    <span className="mt-0.5 block font-mono text-[10px] text-slate-600">{row.routed_to}</span>
                  </td>
                  <td className="whitespace-nowrap px-3 py-2.5 font-mono">{Number(row.cost_usd).toFixed(6)}</td>
                  <td className="whitespace-nowrap px-3 py-2.5 font-mono text-teal-300">{row.tokens_saved}</td>
                  <td className="whitespace-nowrap px-3 py-2.5 font-mono text-violet-300">
                    {row.prompt_chars != null ? row.prompt_chars : "—"}
                  </td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-8 text-center text-slate-500">
                    No rows yet — run a query above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default PromptLab;
