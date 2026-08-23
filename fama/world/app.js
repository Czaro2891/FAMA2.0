/* FAMA 2.0 — World interface.
   Everything rendered here comes from real system state via the API/SSE. */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const state = {
  selected: null,       // task_id
  tasks: [],
  events: [],           // selected task events
  taskState: null,      // /api/tasks/{id} snapshot
  replay: null,         // replay object when viewing a recording
  es: null,             // EventSource
  health: null,
};

// ---------------------------------------------------------------- utils

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function fmtCost(c) { return "$" + (Number(c) || 0).toFixed(4); }
function stBadge(v) { return `<span class="st ${esc(v)}">${esc(v)}</span>`; }
function timeOf(ts) { return String(ts || "").split("T")[1] || ""; }
function pct(x) { return Math.round(100 * (Number(x) || 0)) + "%"; }

async function api(path, opts) {
  const r = await fetch(path, opts ? Object.assign({ headers: { "Content-Type": "application/json" } }, opts) : undefined);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- boot

async function boot() {
  state.health = await api("/api/health");
  renderProviderPill();
  await refreshScenarios();
  await refreshReplays();
  await refreshTasks();
  await refreshMetrics();
  setInterval(refreshTasks, 4000);
  setInterval(refreshMetrics, 6000);
  connectSSE();
}

function renderProviderPill() {
  const p = $("#provider-pill");
  const h = state.health;
  const provs = Object.entries(h.providers || {}).filter(([k]) => !["any_real", "scripted"].includes(k))
    .map(([k, v]) => `${k}:${v ? "✓" : "—"}`).join("  ");
  if (h.providers && h.providers.bridge) {
    p.className = "pill ok"; p.textContent = "LIVE · BRIDGE (Twój lokalny model)";
  } else if (h.providers && h.providers.any_real) {
    p.className = "pill warn"; p.textContent = "KEY ✓ / SIEĆ ✗ · połącz Bridge →";
    p.title = "Klucz API jest, ale endpoint bywa nieosiągalny z tego środowiska. Panel 'Bridge' łączy przez Twoją przeglądarkę.";
  } else {
    p.className = "pill warn"; p.textContent = "NO PROVIDER · " + provs;
    p.title = h.warning || "";
  }
}

async function refreshMetrics() {
  try {
    const m = await api("/api/metrics");
    $("#metrics-pill").textContent =
      `tasks ${m.counters.tasks_total || 0} · ok ${m.counters.tasks_completed || 0} · ` +
      `replans ${m.counters.replans || 0} · ${fmtCost(m.total_cost_usd)}`;
  } catch (e) { /* noop */ }
}

// ---------------------------------------------------------------- SSE

function connectSSE() {
  if (state.es) state.es.close();
  const url = state.selected ? `/api/stream?task_id=${state.selected}` : `/api/stream`;
  state.es = new EventSource(url);
  state.es.onmessage = (m) => {
    try {
      const ev = JSON.parse(m.data);
      if (!state.selected || ev.task_id === state.selected) {
        state.events.push(ev);
        if (state.events.length > 3000) state.events.splice(0, state.events.length - 3000);
        appendTimelineEvent(ev);
        if (["task_finished", "clarification_requested", "approval_required",
             "strategy_selected", "plan_changed", "oracle_run", "verification_failure",
             "task_started", "step_started", "step_done"].includes(ev.type)) {
          refreshSelectedState();
        }
      }
      if (ev.type === "task_started") refreshTasks();
      if (ev.type === "task_finished") { refreshTasks(); refreshMetrics(); }
    } catch (e) { /* noop */ }
  };
}

// ---------------------------------------------------------------- tasks list

async function refreshTasks() {
  try {
    state.tasks = await api("/api/tasks");
    const el = $("#tasks");
    el.innerHTML = state.tasks.map(t => `
      <div class="task-item ${t.id === state.selected ? "sel" : ""}" data-id="${t.id}">
        <div class="t">${esc(t.input.slice(0, 64))}${t.input.length > 64 ? "…" : ""}</div>
        <div style="display:flex;gap:6px;align-items:center;margin-top:3px;">
          ${stBadge(t.result || t.status)}
          ${t.scripted ? '<span class="tag scripted">SCRIPTED</span>' : ""}
          <span class="dim small" style="margin-left:auto">${fmtCost(t.cost_usd)}</span>
        </div>
      </div>`).join("") || '<div class="dim small">brak zadań</div>';
    $$(".task-item", el).forEach(n => n.onclick = () => selectTask(n.dataset.id));
  } catch (e) { /* noop */ }
}

function selectTask(id) {
  state.selected = id;
  state.replay = null;
  state.events = [];
  $("#empty-state").hidden = true;
  $("#task-view").hidden = false;
  connectSSE();
  api(`/api/tasks/${id}/events`).then(evts => {
    state.events = evts;
    renderAll();
  }).catch(() => renderAll());
  refreshSelectedState();
  refreshTasks();
}

async function refreshSelectedState() {
  if (!state.selected || state.replay) return;
  try {
    state.taskState = await api(`/api/tasks/${state.selected}`);
    renderAll();
  } catch (e) { /* task may belong to demo instance */ }
}

// ---------------------------------------------------------------- scenarios & replays

async function refreshScenarios() {
  const data = await api("/api/scenarios");
  $("#scenarios").innerHTML = data.scenarios.map(s => `
    <div class="scenario">
      <b>${esc(s.title)}</b>
      <p>${esc(s.description)}</p>
      <button class="mini" data-run="${s.name}">▶ uruchom (offline)</button>
    </div>`).join("");
  $$("[data-run]").forEach(b => b.onclick = async () => {
    b.disabled = true; b.textContent = "…";
    try {
      const r = await api(`/api/scenarios/${b.dataset.run}/run`, { method: "POST" });
      selectTask(r.task_id);
    } catch (e) { b.textContent = "błąd"; }
    b.disabled = false; b.textContent = "▶ uruchom (offline)";
  });
}

async function refreshReplays() {
  try {
    const data = await api("/api/replays");
    const el = $("#replays");
    if (!data.replays.length) { el.innerHTML = '<span class="dim small">brak — użyj <span class="mono">fama record</span></span>'; return; }
    el.innerHTML = data.replays.map(r => `
      <div class="replay-item" data-replay="${r.name}">
        <span>${esc(r.title)}</span>
        <button class="mini" data-record="${r.name}" title="nagraj ponownie">↺</button>
      </div>`).join("");
    $$(".replay-item", el).forEach(n => n.onclick = () => openReplay(n.dataset.replay));
    $$("[data-record]", el).forEach(b => b.onclick = async (e) => {
      e.stopPropagation(); b.textContent = "…";
      try { await api(`/api/replays/${b.dataset.record}/record`, { method: "POST" }); refreshReplays(); }
      catch (err) { b.textContent = "×"; }
    });
  } catch (e) { /* noop */ }
}

async function openReplay(name) {
  const rep = await api(`/api/replays/${name}`);
  state.replay = rep;
  state.selected = "replay:" + name;
  state.events = rep.events || [];
  state.taskState = rep.final_state || null;
  $("#empty-state").hidden = true;
  $("#task-view").hidden = false;
  renderAll();
}

// ---------------------------------------------------------------- bridge

const bridge = { connected: false, base: "", timer: null, models: [] };

$("#btn-bridge").onclick = async () => {
  const btn = $("#btn-bridge"), status = $("#bridge-status"), tag = $("#bridge-tag");
  if (bridge.connected) {
    bridgeDisconnect();
    return;
  }
  const base = $("#bridge-url").value.trim().replace(/\/+$/, "");
  btn.disabled = true;
  status.textContent = "łączę z " + base + " …";
  try {
    // the BROWSER reaches the user's localhost (the sandbox cannot)
    const r = await fetch(base + "/models", { headers: { "Content-Type": "application/json" } });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    const ids = (data.data || []).map(m => m.id).filter(Boolean).slice(0, 40);
    if (!ids.length) throw new Error("endpoint nie zwrócił modeli");
    await api("/api/bridge/models", { method: "POST", body: JSON.stringify({ models: ids, base_url: base }) });
    bridge.connected = true;
    bridge.base = base;
    bridge.models = ids;
    tag.textContent = "LIVE · " + ids.length + " modeli";
    tag.className = "tag scripted";
    tag.style.color = "var(--ok)"; tag.style.borderColor = "var(--ok)";
    btn.textContent = "Rozłącz bridge";
    status.innerHTML = "✓ Połączono: " + esc(ids.slice(0, 3).join(", ")) +
      (ids.length > 3 ? " +" + (ids.length - 3) : "") +
      " — zadania lecą przez TWÓJ model (koszt $0). Pętla obsługi aktywna.";
    startBridgeLoop();
    state.health = await api("/api/health");
    renderProviderPill();
  } catch (e) {
    const msg = String(e.message || e);
    let hint;
    if (/Failed to fetch|NetworkError|load failed/i.test(msg)) {
      hint = "✗ Przeglądarka zablokowała połączenie (Chrome PNA / CORS) albo nic nie słucha na tym porcie." +
        "<br><b>Naprawa:</b> uruchom <span class='mono'>python examples/bridge_helper.py</span> na swoim komputerze " +
        "(plik w repo FAMA2.0) i użyj URL <span class='mono'>http://localhost:8790/v1</span>. " +
        "Firefox: bez helpera, z <span class='mono'>OLLAMA_ORIGINS=* ollama serve</span>.";
    } else {
      hint = "✗ " + esc(msg) + " — sprawdź, czy model działa; pomocne: <span class='mono'>curl http://localhost:11434/v1/models</span>";
    }
    status.innerHTML = hint;
  }
  btn.disabled = false;
};

function bridgeDisconnect() {
  bridge.connected = false;
  stopBridgeLoop();
  api("/api/bridge/disable", { method: "POST" }).catch(() => {});
  const tag = $("#bridge-tag");
  tag.textContent = "OFF"; tag.style.color = ""; tag.style.borderColor = "";
  $("#btn-bridge").textContent = "Połącz z lokalnym modelem";
  $("#bridge-status").textContent = "Bridge rozłączony.";
  api("/api/health").then(h => { state.health = h; renderProviderPill(); }).catch(() => {});
}

function startBridgeLoop() {
  stopBridgeLoop();
  bridge.timer = setInterval(serveBridge, 1500);
  serveBridge();
}

function stopBridgeLoop() {
  if (bridge.timer) { clearInterval(bridge.timer); bridge.timer = null; }
}

async function serveBridge() {
  if (!bridge.connected || document.hidden) return;
  let pend;
  try {
    pend = await api("/api/bridge/pending");
  } catch (e) { return; }
  for (const p of pend.pending || []) {
    try {
      const r = await fetch(bridge.base + "/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: p.model, messages: p.messages, max_tokens: p.max_tokens,
          temperature: p.temperature, stream: false,
        }),
      });
      if (!r.ok) throw new Error("local HTTP " + r.status);
      const data = await r.json();
      const content = data.choices && data.choices[0] &&
        data.choices[0].message && data.choices[0].message.content || "";
      const usage = data.usage || {};
      await api("/api/bridge/complete", { method: "POST", body: JSON.stringify({
        id: p.id, content,
        tokens_in: usage.prompt_tokens || usage.input_tokens || 0,
        tokens_out: usage.completion_tokens || usage.output_tokens || 0,
      }) });
    } catch (e) {
      await api("/api/bridge/fail", { method: "POST", body: JSON.stringify({
        id: p.id, error: "browser->local call failed: " + e.message }) }).catch(() => {});
      $("#bridge-status").innerHTML = "✗ błąd wywołania lokalnego: " + esc(e.message) +
        " (CORS? OLLAMA_ORIGINS?)";
    }
  }
}

// ---------------------------------------------------------------- new task

$("#btn-submit").onclick = async () => {
  const input = $("#task-input").value.trim();
  const hint = $("#submit-hint");
  if (!input) { hint.textContent = "wpisz opis problemu"; return; }
  hint.textContent = "wysyłanie…";
  try {
    const r = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify({
        input,
        autonomy: $("#task-autonomy").value || null,
        max_cost_usd: parseFloat($("#task-cost").value) || 2,
      }),
    });
    hint.textContent = "";
    $("#task-input").value = "";
    selectTask(r.task_id);
  } catch (e) { hint.textContent = "błąd: " + e.message; }
};

// ---------------------------------------------------------------- tabs

$$("#tabs button").forEach(b => b.onclick = () => {
  $$("#tabs button").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  $$(".tab").forEach(t => (t.hidden = true));
  $("#tab-" + b.dataset.tab).hidden = false;
});

// ---------------------------------------------------------------- rendering

function renderAll() {
  renderHeader();
  renderInteract();
  renderTimeline();
  renderUnderstanding();
  renderStrategies();
  renderTeam();
  renderVerification();
  renderEvidence();
  renderDecisions();
  renderResult();
}

function ts() {
  // task snapshot (works for live tasks and replays)
  return state.taskState;
}

function renderHeader() {
  const s = ts();
  if (!s) return;
  const t = s.task;
  $("#task-title").textContent = t.input;
  const badges = [stBadge(t.status)];
  if (t.result_status) badges.push(stBadge(t.result_status));
  if (s.scripted || state.replay) badges.push('<span class="tag scripted">SCRIPTED/REPLAY — deterministiczne demo, nie model AI</span>');
  if (t.understanding) {
    badges.push(`<span class="tag">${esc(t.understanding.task_type)}</span>`);
    badges.push(`<span class="tag">risk: ${esc(t.understanding.risk_level)}</span>`);
    badges.push(`<span class="tag">${esc(t.understanding.complexity)}</span>`);
    badges.push(`<span class="tag">autonomia: ${esc(t.autonomy_override || (t.understanding && t.understanding.autonomy) || "auto")}</span>`);
  }
  $("#task-badges").innerHTML = badges.join(" ");
  $("#task-cost-box").innerHTML =
    `<div>${fmtCost(t.cost_usd)} · ${((t.tokens && t.tokens.input) || 0) + ((t.tokens && t.tokens.output) || 0)} tok</div>` +
    `<div class="dim small">${t.duration_s || 0}s · planów: ${t.plan_versions || 0} · błędów: ${t.failure_count || 0}</div>`;
}

function renderInteract() {
  const el = $("#interact");
  const s = ts();
  if (!s || state.replay) { el.innerHTML = ""; return; }
  const t = s.task;
  let html = "";
  if (t.status === "awaiting_clarification" && s.clarifications && s.clarifications.length) {
    html += `<div class="gate clar">
      <h3>FAMA pyta zamiast zakładać (niejednoznaczność)</h3>
      ${s.clarifications.map(q => `<p>• ${esc(q)}</p>`).join("")}
      <textarea id="clar-answers" rows="2" placeholder="Odpowiedzi — po jednej na linię"></textarea>
      <button class="primary" id="btn-clar" style="margin-top:8px">Odpowiedz</button>
    </div>`;
  }
  const pending = (s.gates || []).filter(g => g.status === "pending");
  if (t.status === "awaiting_approval" && pending.length) {
    html += `<div class="gate">
      <h3>Wymagana zatwierdzenie człowieka (governance)</h3>
      ${pending.map(g => `
        <p><b>${esc(g.reason)}</b><br><span class="dim small">${esc(g.detail)}</span></p>
        <button class="mini" data-approve="${g.id}">✓ zatwierdź</button>
        <button class="mini danger" data-reject="${g.id}">✗ odrzuć</button>`).join("")}
    </div>`;
  }
  el.innerHTML = html;
  const clarBtn = $("#btn-clar");
  if (clarBtn) clarBtn.onclick = async () => {
    const answers = $("#clar-answers").value.split("\n").map(x => x.trim()).filter(Boolean);
    await api(`/api/tasks/${t.id}/clarify`, { method: "POST", body: JSON.stringify({ answers }) });
    setTimeout(refreshSelectedState, 300);
  };
  $$("[data-approve]", el).forEach(b => b.onclick = () =>
    api(`/api/tasks/${t.id}/approve`, { method: "POST", body: JSON.stringify({ gate_id: b.dataset.approve, approve: true }) }).then(() => setTimeout(refreshSelectedState, 300)));
  $$("[data-reject]", el).forEach(b => b.onclick = () =>
    api(`/api/tasks/${t.id}/approve`, { method: "POST", body: JSON.stringify({ gate_id: b.dataset.reject, approve: false }) }).then(() => setTimeout(refreshSelectedState, 300)));
}

// ---------------------------------------------------------------- timeline

const PHASE_ICON = { understanding: "🧠", strategy: "🗺", planning: "📋", capabilities: "🧩",
  governance: "🛡", execution: "⚙", verification: "🔬", done: "🏁", ingest: "📥" };

function eventRow(ev) {
  const icon = PHASE_ICON[ev.phase] || "·";
  let extra = "";
  const p = ev.payload || {};
  if (ev.type === "strategies_compared" && p.candidates) {
    extra = p.candidates.map(c =>
      `<div>◦ ${esc(c.name)} — utility ${Number(c.utility).toFixed(3)}, est $${c.est_cost_usd}, ` +
      `ver ${pct(c.ver_strength)} <span class="tag twin">PREDICTION</span></div>`).join("");
  } else if (ev.type === "oracle_run") {
    extra = `<div>${esc(p.detail || "")}</div>`;
  } else if (ev.type === "common_mode_risk") {
    extra = (p.findings || []).map(f => `<div>◦ ${esc(f)}</div>`).join("");
  } else if (ev.type === "memory_recalled") {
    extra = `<div>${esc(p.n || "")} podobnych zadań — historia jest punktem odniesienia, nie prawdą</div>`;
  } else if (ev.type === "assumptions_identified" && p.assumptions) {
    extra = p.assumptions.map(a => `<div>◦ ${esc(a.statement)} [${esc(a.status)}] conf ${pct(a.confidence)}</div>`).join("");
  }
  return `<div class="ev ${ev.level}">
    <div class="t">${timeOf(ev.ts)}</div>
    <div class="body"><span class="ty">${icon} ${esc(ev.type)}</span><b>${esc(ev.title)}</b>${extra ? `<div class="extra">${extra}</div>` : ""}</div>
  </div>`;
}

function renderTimeline() {
  const el = $("#tab-timeline");
  el.innerHTML = state.events.map(eventRow).join("") ||
    '<p class="dim">brak zdarzeń</p>';
}

function appendTimelineEvent(ev) {
  const el = $("#tab-timeline");
  if (!el || el.hidden === false) { }
  el.insertAdjacentHTML("beforeend", eventRow(ev));
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- understanding

function renderUnderstanding() {
  const el = $("#tab-understanding");
  const s = ts();
  const u = s && s.understanding;
  if (!u) { el.innerHTML = '<p class="dim">interpretacja niedostępna</p>'; return; }
  const caps = (u.required_capabilities || []).map(c => `
    <tr><td class="mono">${esc(c.capability)}</td><td>${pct(c.min_quality)}</td>
    <td>${pct(c.importance)}</td><td class="dim">${esc(c.why || "")}</td></tr>`).join("");
  el.innerHTML = `
    <div class="cards">
      <div class="card"><h4>Cel</h4><div>${esc(u.goal)}</div></div>
      <div class="card"><h4>Rezultat</h4><div>${esc(u.deliverable)}</div></div>
      <div class="card"><h4>Interpretacja (bezpieczna)</h4><div>${esc(u.interpretation || "—")}</div></div>
      <div class="card"><h4>Pewność rozumienia</h4>
        <div class="bar ${u.confidence < 0.5 ? "warn" : ""}"><i style="width:${pct(u.confidence)}"></i></div>
        <div class="dim small" style="margin-top:4px">${pct(u.confidence)}</div></div>
    </div>
    <h3>Wymagane capabilities</h3>
    <table><tr><th>capability</th><th>min. jakość</th><th>ważność</th><th>uzasadnienie</th></tr>${caps}</table>
    <div class="cards" style="margin-top:12px">
      <div class="card"><h4>Kryteria sukcesu</h4>${(u.success_criteria || []).map(c => `<div>✓ ${esc(c)}</div>`).join("") || "—"}</div>
      <div class="card"><h4>Ograniczenia</h4>${(u.constraints || []).map(c => `<div>• ${esc(c)}</div>`).join("") || "—"}</div>
      <div class="card"><h4>Ryzyka</h4>${(u.risks || []).map(c => `<div>⚠ ${esc(c)}</div>`).join("") || "—"}<div class="dim small" style="margin-top:6px">poziom: ${esc(u.risk_level)}</div></div>
      <div class="card"><h3></h3><h4>Niepewności</h4>${(u.uncertainties || []).map(c => `<div>? ${esc(c)}</div>`).join("") || "—"}
        ${u.ambiguities && u.ambiguities.length ? `<h4>Ambiwalencje</h4>${u.ambiguities.map(c => `<div>≠ ${esc(c)}</div>`).join("")}` : ""}</div>
    </div>`;
}

// ---------------------------------------------------------------- strategies

function renderStrategies() {
  const el = $("#tab-strategies");
  const s = ts();
  const list = s && s.strategies;
  if (!list || !list.length) { el.innerHTML = '<p class="dim">brak kandydatów (zadanie nie doszło do fazy strategii)</p>'; return; }
  const chosenId = s.chosen_strategy && s.chosen_strategy.id;
  el.innerHTML = list.map(x => {
    const st = x.strategy;
    const f = x.factors || {};
    const chosen = st.id === chosenId;
    return `<div class="card ${chosen ? "chosen" : ""}" style="margin-bottom:10px">
      <h4>${chosen ? "★ " : ""}${esc(st.name)} ${st.memory_ref ? `<span class="tag">📊 ${esc(st.memory_ref)}</span>` : ""}</h4>
      <div class="dim small" style="margin-bottom:6px">${esc(st.description)}</div>
      <div class="kv">
        <span class="k">utility</span><span><b>${Number(st.utility).toFixed(4)}</b>
          <span class="dim small">(profil wag: ${esc(x.weight_profile)})</span></span>
        <span class="k">koszt / czas</span><span class="mono">${fmtCost(st.est_cost_usd)} · ${st.est_seconds}s</span>
        <span class="k">szansa sukcesu</span><span>${pct(st.est_success_prob)}</span>
        <span class="k">siła weryfikacji</span><span>${pct(st.verification_strength)}
          — ${(st.verification_bundle || []).map(v => `<span class="tag">${esc(typeof v === "string" ? v : v.value || v)}</span>`).join(" ")}</span>
      </div>
      <div style="display:flex;gap:10px;margin-top:8px;flex-wrap:wrap">
        ${Object.entries(f).filter(([k, v]) => v !== null).map(([k, v]) =>
          `<div style="flex:1;min-width:90px"><div class="dim small">${esc(k)}</div>
           <div class="bar"><i style="width:${pct(typeof v === "number" ? v : 0)}"></i></div></div>`).join("")}
      </div>
      <div class="dim small" style="margin-top:8px">
        <span class="tag twin">TWIN PREDICTION — szacunki z symulacji, nie rezultaty</span>
        ${chosen ? " · wybrana strategia" : ""}
      </div>
    </div>`;
  }).join("");
}

// ---------------------------------------------------------------- team & plan

function renderTeam() {
  const el = $("#tab-team");
  const s = ts();
  if (!s) return;
  const team = (s.team || []).map(m => `
    <div class="card">
      <h4>${esc(m.agent ? m.agent.name : "?")} <span class="dim small">(${esc(m.role)})</span></h4>
      <div class="kv">
        <span class="k">model</span><span class="mono">${esc(m.model || "auto")}</span>
        <span class="k">capabilities</span><span class="small">${m.agent ? (m.agent.capabilities || []).map(c => `${esc(c.id)} (${Number(c.quality).toFixed(2)})`).join(", ") : ""}</span>
        <span class="k">reliability</span><span>${m.agent ? m.agent.reliability : ""}</span>
        <span class="k">trust</span><span>${m.agent ? m.agent.trust : ""}${m.agent && m.agent.probation ? " · PROBATION" : ""}</span>
      </div>
    </div>`).join("");
  const plans = (s.plans || []);
  const planHtml = plans.map(p => `
    <div class="planver">
      <h3>PLAN V${p.version} — ${esc(p.strategy_name)}
        ${p.change_reason && p.version > 1 ? `<span class="tag" style="color:var(--warn)">zmiana: ${esc(p.change_reason)}</span>` : ""}
      </h3>
      <div class="flow">
        ${p.steps.map(st => `
          <div class="step">
            <span class="nm">${esc(st.name)}</span>
            <span class="gl">${esc(st.goal)}</span>
            <span class="ag">${esc(agentName(s, st.agent_id))}${st.model ? " · " + esc(st.model) : ""}</span>
            ${stBadge(st.status)}
            ${st.attempts > 1 ? `<span class="tag">próby: ${st.attempts}</span>` : ""}
          </div>`).join("")}
      </div>
    </div>`).join("");
  const sel = (s.selection_trace || []).filter(t => t.decision).map(t => `
    <tr><td class="mono">${esc(t.capability || "")}</td><td>${esc(t.decision)}</td>
    <td>${esc(t.selected || "—")}</td><td class="dim small">${t.candidates ? t.candidates.length : 0} kandydatów</td></tr>`).join("");
  el.innerHTML = `
    <h3>Zespół (najmniejszy wystarczający)</h3>
    <div class="cards">${team || '<span class="dim">—</span>'}</div>
    ${sel ? `<h3>Dobór agentów (capability matching)</h3><table>
      <tr><th>capability</th><th>decyzja</th><th>wybrany agent</th><th></th></tr>${sel}</table>` : ""}
    ${planHtml || '<p class="dim">brak planu</p>'}`;
}

function agentName(s, id) {
  const m = (s.team || []).find(x => x.agent && x.agent.id === id);
  return m ? m.agent.name : "—";
}

// ---------------------------------------------------------------- verification

function renderVerification() {
  const el = $("#tab-verification");
  const s = ts();
  if (!s) return;
  const runs = (s.oracle_runs || []).map(r => `
    <div class="oracle ${esc(r.verdict)}">
      <div class="head"><b>${esc(r.kind.value || r.kind)}</b>${stBadge(r.verdict)}
        <span class="dim small">siła dowodu ${Number(r.strength || 0).toFixed(2)}</span></div>
      <div class="detail">${esc(r.detail || "")}</div>
      ${r.measurements && r.measurements.score !== undefined
        ? `<div class="bar ${r.verdict === "weak" ? "warn" : ""}" style="margin-top:6px"><i style="width:${pct(r.measurements.score)}"></i></div>` : ""}
    </div>`).join("");
  const asm = (s.assumptions || []).map(a => `
    <tr><td>${esc(a.statement)}</td><td>${stBadge(a.status)}</td>
    <td>${pct(a.confidence)}</td><td>${pct(a.importance)}</td><td class="mono">${esc(a.verification_method)}</td></tr>`).join("");
  const vb = s.verification_budget;
  const cm = s.common_mode;
  const aut = (s.autopsies || []).map(a => `
    <div class="card"><h4>Autopsy ${esc(a.failure_id)}</h4>
      <div>${esc(a.root_cause || "")}</div>
      <div class="dim small" style="margin-top:4px">lesson: ${esc(a.lesson || "")}</div></div>`).join("");
  el.innerHTML = `
    ${vb ? `<div class="cards" style="margin-bottom:12px">
      <div class="card"><h4>Budżet weryfikacji</h4><div class="kv">
        <span class="k">ryzyko</span><span>${esc(vb.risk)}</span>
        <span class="k">max oracles</span><span>${vb.max_oracles}${vb.escalation_level ? ` (+${vb.escalation_level} eskalacja)` : ""}</span>
        <span class="k">niezależność</span><span>${vb.require_independent ? "wymagana" : "—"} </span>
        <span class="k">human sign-off</span><span>${vb.require_human ? "wymagany" : "—"}</span>
      </div></div>
      ${cm ? `<div class="card"><h4>Common-mode risk</h4>
        <div class="bar ${cm.score > 0.3 ? "warn" : ""}"><i style="width:${pct(cm.score)}"></i></div>
        ${(cm.findings || []).map(f => `<div class="small" style="margin-top:4px">⚠ ${esc(f)}</div>`).join("") || '<div class="dim small">brak wspólnych punktów błędu</div>'}</div>` : ""}
    </div>` : ""}
    <h3>Oracle runs</h3>
    ${runs || '<p class="dim">weryfikacja nie uruchomiona</p>'}
    ${asm ? `<h3>Założenia (Assumption Engine)</h3><table>
      <tr><th>założenie</th><th>status</th><th>pewność</th><th>ważność</th><th>weryfikacja</th></tr>${asm}</table>` : ""}
    ${aut ? `<h3>Autopsje błędów</h3><div class="cards">${aut}</div>` : ""}`;
}

// ---------------------------------------------------------------- evidence graph

function renderEvidence() {
  const el = $("#tab-evidence");
  const s = ts();
  const evd = s && s.evidence;
  if (!evd || !evd.nodes) { el.innerHTML = '<p class="dim">brak dowodów</p>'; return; }
  const nodes = evd.nodes, edges = evd.edges || [];
  const byKind = {};
  nodes.forEach(n => (byKind[n.kind] = byKind[n.kind] || []).push(n));
  const order = ["claim", "action", "test", "countertest", "measurement", "artifact", "failure", "approval"];
  const rows = [];
  const pos = {};
  order.forEach(kind => (byKind[kind] || []).forEach((n, i) => {
    rows.push({ kind, n, i, count: (byKind[kind] || []).length });
    pos[n.id] = { x: 30 + (order.indexOf(kind) * 210), y: 40 + i * 52 + 20 };
  }));
  const W = order.length * 220 + 40;
  const H = Math.max(...rows.map(r => pos[r.n.id].y)) + 60;
  const COLOR = { claim: "#4fd1c5", action: "#63b3ed", test: "#48bb78",
                  countertest: "#f56565", measurement: "#b794f4", artifact: "#d7e2ee",
                  failure: "#f56565", approval: "#ed8936", decision: "#d7e2ee", source: "#48bb78" };
  let svg = `<svg width="${W}" height="${H}" style="min-width:100%">`;
  edges.forEach(e => {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) return;
    const bad = e.relation === "refuted_by";
    svg += `<line x1="${a.x + 190}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
      stroke="${bad ? "#f56565" : "#3b5068"}" stroke-width="${bad ? 2 : 1.2}"
      ${bad ? 'stroke-dasharray="5,3"' : ""}/>`;
  });
  rows.forEach(r => {
    const p = pos[r.n.id];
    const c = COLOR[r.kind] || "#888";
    svg += `<g class="evd-node">
      <rect x="${p.x}" y="${p.y - 16}" width="190" height="36" rx="7" fill="#18222f" stroke="${c}"/>
      <text class="claimlabel" x="${p.x + 9}" y="${p.y - 2}">${esc(r.n.label).slice(0, 26)}</text>
      <text class="evtype" x="${p.x + 9}" y="${p.y + 12}">${esc(r.kind)} · ${esc(r.n.result || "")} ${r.n.hash ? "· " + esc(r.n.hash.slice(0, 6)) : ""}</text>
    </g>`;
  });
  svg += "</svg>";
  const final = nodes.filter(n => n.label && n.label.startsWith("FINAL"));
  el.innerHTML = `
    <p class="dim small">Łańcuch dowodów odpowiada na pytanie „dlaczego FAMA uznała ten wynik?”.
      Czerwone przerywane krawędzie = obalenie (refuted_by).</p>
    <div id="evidence-wrap">${svg}</div>
    ${final.length ? `<h3>Werdykt końcowy</h3><div class="cards">${final.map(f =>
      `<div class="card"><h4>${esc(f.result)}</h4><div class="small">${esc((f.payload || {}).summary || "")}</div></div>`).join("")}</div>` : ""}`;
}

// ---------------------------------------------------------------- decisions

function renderDecisions() {
  const el = $("#tab-decisions");
  const s = ts();
  const list = s && s.decisions;
  if (!list || !list.length) { el.innerHTML = '<p class="dim">brak zapisanych decyzji</p>'; return; }
  el.innerHTML = list.map(d => `
    <div class="card" style="margin-bottom:10px">
      <h4>${esc(d.decision)} → <span style="color:var(--acc)">${esc(d.selected)}</span></h4>
      <div class="kv">
        <span class="k">score</span><span>${Number(d.score).toFixed(4)}</span>
        <span class="k">pewność</span><span>${pct(d.confidence)}</span>
        <span class="k">ryzyko</span><span>${esc(d.risk)}</span>
      </div>
      <div class="dim small" style="margin:6px 0">${esc(d.reason)}</div>
      ${d.options && d.options.length ? `<table><tr><th>opcja</th><th>utility</th></tr>
        ${d.options.map(o => `<tr class="${o.name === d.selected ? "" : ""}"><td>${esc(o.name)}</td><td>${Number(o.utility || 0).toFixed(4)}</td></tr>`).join("")}</table>` : ""}
      <div class="dim small" style="margin-top:6px">trace pokazuje wynik procesu decyzyjnego, nie prywatne rozumowanie modelu</div>
    </div>`).join("");
}

// ---------------------------------------------------------------- result

function renderResult() {
  const el = $("#tab-result");
  const s = ts();
  if (!s) return;
  const t = s.task;
  const rs = t.result_status || "(w toku)";
  const art = t.final_artifact || "";
  let preview = "";
  if (art.endsWith(".html") && !state.replay && t.id) {
    preview = `
      <h3>Artefakt — podgląd na żywo</h3>
      <iframe src="/api/tasks/${t.id}/artifact"
        style="width:100%;height:430px;border:1px solid var(--line);border-radius:10px;
               background:#0b0f14"></iframe>
      <p class="small" style="margin-top:6px">
        <a href="/api/tasks/${t.id}/artifact" target="_blank">otwórz artefakt w nowej karcie ↗</a>
      </p>`;
  }
  const blockedHelp = (rs === "blocked" &&
    /model unavailable|no provider/i.test(t.result_summary || "")) ? `
      <div class="gate clar">
        <h3>Jak odpalić LIVE</h3>
        <p><b>Opcja 1 — Bridge (tu, w tym UI):</b> panel „Bridge — Twój lokalny LLM” po lewej →
        uruchom model lokalnie (np. <span class="mono">OLLAMA_ORIGINS=* ollama serve</span>) →
        „Połącz z lokalnym modelem” → wyślij zadanie ponownie.</p>
        <p><b>Opcja 2 — na Twojej maszynie:</b></p>
        <pre class="code">git clone https://github.com/Czaro2891/FAMA2.0 &amp;&amp; cd FAMA2.0
pip install -r requirements.txt
echo 'OPENAI_BASE_URL=http://localhost:11434/v1' &gt; .env
python -m fama serve</pre>
      </div>` : "";
  el.innerHTML = `
    <div class="result-banner ${esc(rs)}">${esc(rs.toUpperCase())}</div>
    <p>${esc(t.result_summary || "")}</p>
    ${blockedHelp}
    ${preview}
    <div class="cards" style="margin-top:12px">
      <div class="card"><h4>Koszt</h4>${fmtCost(t.cost_usd)}</div>
      <div class="card"><h4>Czas</h4>${t.duration_s || 0}s</div>
      <div class="card"><h4>Tokeny</h4>in ${((t.tokens || {}).input) || 0} / out ${((t.tokens || {}).output) || 0}</div>
      <div class="card"><h4>Wersje planu</h4>${t.plan_versions || 0} (replany: ${t.replan_count || 0})</div>
      <div class="card"><h4>Błędy</h4>${t.failure_count || 0}</div>
      <div class="card"><h4>Artefakt</h4><span class="mono">${esc(art || "—")}</span></div>
    </div>`;
}

boot();
