/**
 * Display names for route keys. `gemini_pro` is the premium tier label in UI/logs (router escalations).
 */
export const ROUTE_LABELS = {
  groq_simple: "Groq · Llama 3.1 8B Instant",
  groq_mixtral: "Groq · Llama 3.3 70B Versatile (medium)",
  gemini_pro: "Google · Gemini 1.5 Flash (premium tier)",
  cache: "Semantic cache hit",
  none: "—",
};

/** Pretty label for tables, charts, history. Keeps raw key in `title` tooltips where useful. */
export function displayModelName(routeKey) {
  if (routeKey == null || routeKey === "") return "—";
  const k = String(routeKey);
  return ROUTE_LABELS[k] || k;
}
