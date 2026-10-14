/* Frontend de la demo: consume la API FastAPI (/health, /model/info, /predict, /predict/batch). */
(() => {
  "use strict";
  const API = (window.API_BASE || "").replace(/\/$/, "");
  const BASE_RATE = 0.117;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ---------- Niveles de las variables categóricas (valor del modelo → etiqueta) ----------
  const LEVELS = {
    job: [["admin.", "Administrativo"], ["blue-collar", "Obrero / operario"], ["entrepreneur", "Emprendedor"],
      ["housemaid", "Empleada doméstica"], ["management", "Gerencial"], ["retired", "Jubilado"],
      ["self-employed", "Independiente"], ["services", "Servicios"], ["student", "Estudiante"],
      ["technician", "Técnico"], ["unemployed", "Desempleado"], ["unknown", "Desconocido"]],
    marital: [["married", "Casado/a"], ["single", "Soltero/a"], ["divorced", "Divorciado/a o viudo/a"]],
    education: [["primary", "Primaria"], ["secondary", "Secundaria"], ["tertiary", "Terciaria / universitaria"], ["unknown", "Desconocida"]],
    yesno: [["no", "No"], ["yes", "Sí"]],
    contact: [["cellular", "Celular"], ["telephone", "Teléfono fijo"], ["unknown", "Desconocido"]],
    month: [["jan", "Enero"], ["feb", "Febrero"], ["mar", "Marzo"], ["apr", "Abril"], ["may", "Mayo"], ["jun", "Junio"],
      ["jul", "Julio"], ["aug", "Agosto"], ["sep", "Septiembre"], ["oct", "Octubre"], ["nov", "Noviembre"], ["dec", "Diciembre"]],
    poutcome: [["unknown", "Sin campaña previa"], ["failure", "Fracaso"], ["other", "Otro"], ["success", "Éxito"]],
  };
  const LABEL = Object.fromEntries(Object.entries(LEVELS).map(([k, v]) => [k, Object.fromEntries(v)]));
  const VALUE_ES = { ...LABEL.job, ...LABEL.education, ...LABEL.contact, ...LABEL.month, ...LABEL.poutcome,
    married: "casado/a", single: "soltero/a", divorced: "divorciado/a", yes: "sí", no: "no" };

  const PRESETS = {
    exito: { age: 64, job: "retired", marital: "married", education: "primary", default: "no", balance: 43, housing: "no",
      loan: "no", contact: "cellular", day: 10, month: "mar", campaign: 1, pdays: 90, previous: 2, poutcome: "success" },
    profesional: { age: 35, job: "management", marital: "single", education: "tertiary", default: "no", balance: 1500,
      housing: "no", loan: "no", contact: "cellular", day: 15, month: "oct", campaign: 1, pdays: -1, previous: 0, poutcome: "unknown" },
    masiva: { age: 45, job: "blue-collar", marital: "married", education: "secondary", default: "no", balance: 300,
      housing: "yes", loan: "yes", contact: "unknown", day: 20, month: "may", campaign: 4, pdays: -1, previous: 0, poutcome: "unknown" },
  };

  // ---------- Formato ----------
  const nf = (d) => new Intl.NumberFormat("es-AR", { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmt = (x, d = 0) => nf(d).format(x);
  const pct = (x, d = 1) => `${fmt(100 * x, d)} %`;
  const money = (x) => (x < 0 ? "−" : "") + fmt(Math.abs(x), 0);

  // ---------- Estado ----------
  const state = { V: 20, C: 1, threshold: 0.05, lastPrediction: null, rows: null, hasY: false, cumP: [], cumY: [], k: 0 };

  // ---------- Tema ----------
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) { /* sin storage */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* sin storage */ }
    if (state.rows) renderCampaign();
  });

  // ---------- Pestañas (con #hash) ----------
  function showTab(name) {
    if (!["cliente", "campana", "modelo"].includes(name)) name = "cliente";
    $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    if (name === "campana" && state.rows) renderCampaign();
  }
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // ---------- API ----------
  async function api(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) {
      let detail = `Error ${r.status}`;
      try { const j = await r.json(); detail = Array.isArray(j.detail) ? j.detail.map((d) => d.msg || d).join(" · ") : (j.detail || detail); }
      catch (e) { /* respuesta no JSON */ }
      throw new Error(String(detail).replace(/^Value error, /, ""));
    }
    return r;
  }
  async function checkHealth() {
    const el = $("#api-status");
    try {
      const j = await (await api("/health")).json();
      el.className = "api-status ok"; el.lastElementChild.textContent = j.model_loaded ? "API en línea" : "API sin modelo";
    } catch (e) {
      el.className = "api-status err"; el.lastElementChild.textContent = "API no disponible";
    }
  }

  // ---------- Supuestos económicos ----------
  function readEconomics() {
    const V = parseFloat($("#v-value").value), C = parseFloat($("#c-value").value);
    state.V = V > 0 ? V : 20; state.C = C > 0 ? C : 1;
    const thr = state.C / state.V;
    $("#rule-text").innerHTML = "";
    const b = document.createElement("b");
    b.textContent = `p > ${fmt(state.C, 2)} / ${fmt(state.V, 2)} = ${pct(thr, 1)}`;
    $("#rule-text").append("Conviene llamar si la probabilidad supera C/V: ", b,
      ". Así lo que se espera ganar (p × V) supera lo que cuesta la llamada (C).");
    if (state.lastPrediction) renderPrediction(state.lastPrediction);
    if (state.rows) renderCampaign();
  }
  ["#v-value", "#c-value"].forEach((s) => $(s).addEventListener("input", readEconomics));

  // ---------- KPIs del modelo (data/summary.json) ----------
  function kpi(value, label, accent) {
    const d = document.createElement("div"); d.className = "kpi" + (accent ? " accent" : "");
    const v = document.createElement("div"); v.className = "v"; v.textContent = value;
    const l = document.createElement("div"); l.className = "l"; l.textContent = label;
    d.append(v, l); return d;
  }
  async function loadSummary() {
    try {
      const s = await (await fetch("data/summary.json")).json();
      const t = s.test.final;
      $("#kpis").replaceChildren(
        kpi(fmt(t.roc_auc, 3), "ROC-AUC en datos de prueba", true),
        kpi(fmt(t.pr_auc, 3), `PR-AUC (al azar ${fmt(s.test.dummy.pr_auc, 3)})`, true),
        kpi(pct(t.gain20, 0), "de las conversiones llamando al 20 % mejor rankeado"),
        kpi(`${fmt(t.lift10, 1)}×`, "más conversiones que al azar en el top 10 %"));
      $("#m-dur").textContent = fmt(s.test.with_duration.roc_auc, 3);
      $("#m-overlap").textContent = fmt(s.overlap_removed);
      $("#m-train").textContent = fmt(s.train_rows);
      $("#m-test").textContent = fmt(s.test_rows);
      const tbl = $("#metrics-table");
      const head = tbl.createTHead().insertRow();
      ["Modelo", "ROC-AUC", "PR-AUC", "Lift top 10 %", "Conversiones en top 20 %", "Brier"].forEach((h) => {
        const th = document.createElement("th"); th.textContent = h; head.append(th); });
      const body = tbl.createTBody();
      ["final", "logreg", "dummy", "with_duration"].forEach((k) => {
        const m = s.test[k]; const tr = body.insertRow(); if (k === "final") tr.className = "hl";
        [m.label, fmt(m.roc_auc, 3), fmt(m.pr_auc, 3), fmt(m.lift10, 2), pct(m.gain20, 0), fmt(m.brier, 3)]
          .forEach((v) => { tr.insertCell().textContent = v; });
      });
      state.threshold = s.threshold;
    } catch (e) { $("#kpis").replaceChildren(); }
  }

  // ---------- Formulario del cliente ----------
  $$("select[data-levels]").forEach((sel) => {
    LEVELS[sel.dataset.levels].forEach(([v, l]) => sel.add(new Option(l, v)));
  });
  function fillForm(data) {
    const f = $("#client-form");
    Object.entries(data).forEach(([k, v]) => { if (f.elements[k]) f.elements[k].value = v; });
  }
  fillForm(PRESETS.profesional);
  $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    $$("[data-preset]").forEach((x) => x.classList.toggle("active", x === b));
    fillForm(PRESETS[b.dataset.preset]);
    $("#client-form").requestSubmit();
  }));

  $("#client-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target, err = $("#form-error"), btn = $("button[type=submit]", f);
    err.textContent = "";
    if (!f.reportValidity()) return;
    const payload = {};
    ["age", "balance", "day", "campaign", "pdays", "previous"].forEach((k) => { payload[k] = parseInt(f.elements[k].value, 10); });
    ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"].forEach((k) => { payload[k] = f.elements[k].value; });
    btn.disabled = true;
    try {
      const res = await (await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })).json();
      state.lastPrediction = res;
      renderPrediction(res);
    } catch (e) { err.textContent = e.message; }
    finally { btn.disabled = false; }
  });

  function describeFactor(f) {
    const v = f.value;
    const map = {
      prev_campaign_success: v ? "tuvo éxito en la campaña anterior" : "sin éxito en campaña anterior",
      contact_known: v ? "canal de contacto registrado" : "canal de contacto desconocido",
      previously_contacted: v ? "fue contactado en campañas anteriores" : "nunca fue contactado antes",
      balance_negative: v ? "saldo negativo" : "saldo no negativo",
      balance_zero: v ? "saldo en cero" : "saldo distinto de cero",
      n_credit_products: `${v} préstamo(s)`,
      balance_log: "saldo (escala logarítmica)",
      day_sin: "día del mes", day_cos: "día del mes",
      campaign_w: `${v} contacto(s) en esta campaña`,
      previous_w: `${v} contacto(s) previo(s)`,
      pdays_clean: v === null ? "nunca contactado antes" : `${v} días desde el contacto previo`,
      age: `${v} años`, age_group: `${v} años`,
    };
    if (f.feature in map) return map[f.feature];
    return VALUE_ES[v] || String(v);
  }

  function renderPrediction(res) {
    $("#result-empty").hidden = true; $("#result-body").hidden = false;
    const p = res.probability, thr = state.C / state.V, call = p >= thr;
    $("#r-prob").textContent = pct(p, 1);
    const dec = $("#r-decision");
    dec.className = "decision " + (call ? "call" : "skip");
    dec.innerHTML = call
      ? '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
      : '<svg viewBox="0 0 24 24"><path d="M12 7v6M12 17h.01"/><circle cx="12" cy="12" r="9"/></svg>';
    dec.append(call ? "Llamar" : "No llamar");
    $("#r-meter").style.width = `${Math.max(1, 100 * p)}%`;
    $("#r-thr").style.left = `${100 * thr}%`;
    const tl = $("#r-thr-label"); tl.style.left = `${Math.min(92, Math.max(6, 100 * thr))}%`; tl.textContent = `umbral ${pct(thr, 1)}`;
    $("#r-decile").textContent = res.decile;
    const ev = p * state.V - state.C;
    $("#r-ev").textContent = (ev >= 0 ? "+" : "−") + fmt(Math.abs(ev), 2);
    $("#r-lift").textContent = `${fmt(p / BASE_RATE, 1)}×`;

    const box = $("#r-factors"); box.replaceChildren();
    const maxAbs = Math.max(0.5, ...res.top_factors.map((f) => Math.abs(f.shap)));
    res.top_factors.forEach((f) => {
      const row = document.createElement("div"); row.className = "factor"; row.tabIndex = 0;
      const name = document.createElement("div"); name.className = "name";
      const b = document.createElement("b"); b.textContent = f.label;
      const s = document.createElement("span"); s.textContent = describeFactor(f);
      name.append(b, s);
      const area = document.createElement("div"); area.className = "bar-area";
      const axis = document.createElement("div"); axis.className = "axis";
      const bar = document.createElement("div"); bar.className = "bar " + (f.shap >= 0 ? "pos" : "neg");
      bar.style.width = `${(50 * Math.abs(f.shap)) / maxAbs}%`;
      area.append(axis, bar);
      const val = document.createElement("div"); val.className = "val";
      val.textContent = (f.shap >= 0 ? "+" : "−") + fmt(Math.abs(f.shap), 2);
      row.append(name, area, val);
      const tip = () => [`${f.label}: ${describeFactor(f)}`, `${f.shap >= 0 ? "Aumenta" : "Reduce"} la probabilidad`, `${val.textContent} (log-odds)`];
      bindTooltip(row, tip);
      box.append(row);
    });
    const leg = document.createElement("div"); leg.className = "factor-legend";
    const l1 = document.createElement("span"); l1.textContent = "◀ reduce la probabilidad";
    const l2 = document.createElement("span"); l2.textContent = "aumenta la probabilidad ▶";
    leg.append(l1, l2); box.append(leg);
    const ups = res.top_factors.filter((f) => f.shap > 0).map((f) => `${f.label.toLowerCase()} (${describeFactor(f)})`);
    const downs = res.top_factors.filter((f) => f.shap < 0).map((f) => `${f.label.toLowerCase()} (${describeFactor(f)})`);
    let txt = `La probabilidad estimada es ${pct(p, 1)}, ${call ? "por encima" : "por debajo"} del umbral de ${pct(thr, 1)}: `
      + (call ? `conviene llamar (se espera ganar ${fmt(p * state.V, 2)} por una llamada que cuesta ${fmt(state.C, 2)}).`
              : `no conviene llamar (se espera ganar ${fmt(p * state.V, 2)} por una llamada que cuesta ${fmt(state.C, 2)}).`);
    if (ups.length) txt += ` Lo que más la sube: ${ups.join(", ")}.`;
    if (downs.length) txt += ` Lo que más la baja: ${downs.join(", ")}.`;
    $("#r-text").textContent = txt;
  }

  // ---------- Tooltip compartido (contenido siempre por textContent) ----------
  const tooltip = $("#tooltip");
  function showTooltip(x, y, head, rows) {
    tooltip.replaceChildren();
    const h = document.createElement("div"); h.className = "t-head"; h.textContent = head; tooltip.append(h);
    rows.forEach(([key, val, label]) => {
      const r = document.createElement("div"); r.className = "t-row";
      if (key) { const k = document.createElement("span"); k.className = "t-key " + key; r.append(k); }
      const b = document.createElement("b"); b.textContent = val;
      const s = document.createElement("span"); s.textContent = label || "";
      r.append(b, s); tooltip.append(r);
    });
    tooltip.hidden = false;
    const w = tooltip.offsetWidth, hgt = tooltip.offsetHeight;
    tooltip.style.left = `${Math.min(innerWidth - w - 8, x + 14)}px`;
    tooltip.style.top = `${Math.max(8, y - hgt - 12)}px`;
  }
  const hideTooltip = () => { tooltip.hidden = true; };
  function bindTooltip(el, linesFn) {
    const show = (x, y) => { const [h, ...rest] = linesFn(); showTooltip(x, y, h, rest.map((t) => ["", t, ""])); };
    el.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
    el.addEventListener("pointerleave", hideTooltip);
    el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); show(r.left + r.width / 2, r.top); });
    el.addEventListener("blur", hideTooltip);
  }

  // ---------- Campaña ----------
  function parseCSV(text) {
    const rows = []; let row = [], field = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c; }
      else if (c === '"') q = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(field); rows.push(row); row = []; field = ""; }
      else field += c;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    const head = rows.shift();
    return rows.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
  }

  async function scoreFile(blob, name) {
    const err = $("#batch-error"); err.textContent = "";
    const btns = [$("#btn-sample"), $(".file-btn")]; btns.forEach((b) => b.setAttribute("disabled", ""));
    try {
      const fd = new FormData(); fd.append("file", blob, name);
      const text = await (await api("/predict/batch", { method: "POST", body: fd })).text();
      const rows = parseCSV(text);
      rows.forEach((r) => { r.probability = parseFloat(r.probability); r.decile = parseInt(r.decile, 10); });
      state.rows = rows;
      state.hasY = rows.length > 0 && "y" in rows[0];
      state.cumP = [0]; state.cumY = [0];
      rows.forEach((r, i) => {
        state.cumP.push(state.cumP[i] + r.probability);
        state.cumY.push(state.cumY[i] + (state.hasY && (r.y === "yes" || r.y === "1") ? 1 : 0));
      });
      $("#campaign-body").hidden = false;
      renderCampaign();
    } catch (e) { err.textContent = e.message; }
    finally { btns.forEach((b) => b.removeAttribute("disabled")); }
  }
  $("#btn-sample").addEventListener("click", async () => {
    try {
      const blob = await (await fetch("data/clientes_ejemplo.csv")).blob();
      scoreFile(blob, "clientes_ejemplo.csv");
    } catch (e) { $("#batch-error").textContent = "No se pudo cargar la muestra de ejemplo."; }
  });
  $("#file-input").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) scoreFile(f, f.name); e.target.value = ""; });

  const pctInput = $("#pct");
  pctInput.addEventListener("input", () => renderCampaign());
  $$("[data-pct]").forEach((b) => b.addEventListener("click", () => {
    if (!state.rows) return;
    if (b.dataset.pct === "thr") {
      const thr = state.C / state.V, k = state.rows.filter((r) => r.probability >= thr).length;
      pctInput.value = Math.max(1, Math.round((100 * k) / state.rows.length));
    } else pctInput.value = b.dataset.pct;
    renderCampaign();
  }));

  function renderCampaign() {
    if (!state.rows || $("#tab-campana").hidden) return;
    const n = state.rows.length, p = parseInt(pctInput.value, 10), k = Math.max(1, Math.ceil((p / 100) * n));
    state.k = k;
    $("#pct-out").textContent = `${p} %`;
    const { V, C } = state, expAll = state.cumP[n], expK = state.cumP[k];
    const profitK = V * expK - C * k, profitAll = V * expAll - C * n;
    const tiles = [
      kpi(`${fmt(k)} de ${fmt(n)}`, "clientes a llamar"),
      kpi(fmt(expK, 1), `conversiones esperadas (${pct(expK / expAll, 0)} del total esperado)`, true),
      kpi(money(profitK), "beneficio esperado", true),
      kpi((profitK - profitAll >= 0 ? "+" : "−") + fmt(Math.abs(profitK - profitAll), 0), "vs llamar a todos"),
    ];
    if (state.hasY) {
      const yAll = state.cumY[n], yK = state.cumY[k];
      tiles.push(kpi(`${fmt(yK)} de ${fmt(yAll)}`, `conversiones reales capturadas (${pct(yAll ? yK / yAll : 0, 0)}) — dato conocido de la muestra`));
    }
    const kp = $("#camp-kpis"); kp.replaceChildren(...tiles);
    kp.style.gridTemplateColumns = `repeat(${tiles.length}, minmax(0, 1fr))`;
    if (innerWidth < 640) kp.style.gridTemplateColumns = "";

    // Series muestreadas cada 1 % para los gráficos.
    const xs = Array.from({ length: 101 }, (_, i) => i);
    const kAt = (x) => Math.round((x / 100) * n);
    const gainExp = xs.map((x) => (100 * state.cumP[kAt(x)]) / expAll);
    const series = [{ key: "", label: "Modelo (esperado)", values: gainExp }];
    if (state.hasY && state.cumY[n] > 0) series.push({ key: "real", label: "Modelo (real en la muestra)", values: xs.map((x) => (100 * state.cumY[kAt(x)]) / state.cumY[n]) });
    series.push({ key: "ref", label: "Al azar", values: xs.slice() });
    renderLegend(series);
    lineChart($("#chart-gain"), { xs, series, yMin: 0, yMax: 100, sel: p, yFmt: (v) => `${fmt(v, 0)} %`,
      head: (x) => `Llamando al ${x} % (${fmt(kAt(x))} clientes)`, valFmt: (v) => `${fmt(v, 1)} %` });
    const profit = xs.map((x) => V * state.cumP[kAt(x)] - C * kAt(x));
    const pMin = Math.min(0, ...profit), pMax = Math.max(...profit, 1);
    lineChart($("#chart-profit"), { xs, series: [{ key: "", label: "Beneficio esperado", values: profit }], yMin: pMin, yMax: pMax,
      sel: p, zero: true, yFmt: (v) => money(v), head: (x) => `Llamando al ${x} % (${fmt(kAt(x))} clientes)`, valFmt: (v) => money(v) });

    renderTable(k);
  }

  function renderLegend(series) {
    const lg = $("#legend-gain"); lg.replaceChildren();
    series.forEach((s) => {
      const sp = document.createElement("span"); const i = document.createElement("i"); if (s.key) i.className = s.key;
      sp.append(i, document.createTextNode(s.label)); lg.append(sp);
    });
  }

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (parent) parent.append(el);
    return el;
  }

  function lineChart(host, o) {
    const W = host.clientWidth || 500, H = host.clientHeight || 260, m = { l: 62, r: 14, t: 10, b: 30 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const X = (x) => m.l + (x / 100) * iw;
    const Y = (v) => m.t + ih - ((v - o.yMin) / (o.yMax - o.yMin || 1)) * ih;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.series.map((s) => s.label).join(", ") });
    // Grilla y ejes (recesivos).
    for (let i = 0; i <= 4; i++) {
      const v = o.yMin + ((o.yMax - o.yMin) * i) / 4, y = Y(v);
      svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, svg);
      svgEl("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", class: "axis-label" }, svg).textContent = o.yFmt(v);
    }
    [0, 25, 50, 75, 100].forEach((x) => {
      svgEl("text", { x: X(x), y: H - 8, "text-anchor": "middle", class: "axis-label" }, svg).textContent = `${x} %`;
    });
    if (o.zero && o.yMin < 0) svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    // Series.
    const css = getComputedStyle(document.documentElement);
    const color = { "": css.getPropertyValue("--pos"), real: css.getPropertyValue("--s2"), ref: css.getPropertyValue("--chart-ref") };
    o.series.forEach((s) => {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${X(o.xs[i]).toFixed(1)},${Y(v).toFixed(1)}`).join("");
      const path = svgEl("path", { d, class: "line" + (s.key === "ref" ? " ref" : "") }, svg);
      path.style.stroke = color[s.key];
    });
    // Selección actual.
    svgEl("line", { x1: X(o.sel), x2: X(o.sel), y1: m.t, y2: m.t + ih, class: "sel-line" }, svg);
    const main = o.series[0], selV = main.values[o.sel];
    const mk = svgEl("circle", { cx: X(o.sel), cy: Y(selV), r: 5, class: "marker" }, svg); mk.style.fill = color[""];
    const lbl = svgEl("text", { x: X(o.sel) + (o.sel > 80 ? -8 : 8), y: Y(selV) - 9, "text-anchor": o.sel > 80 ? "end" : "start", class: "direct" }, svg);
    lbl.textContent = o.valFmt(selV);
    // Capa de hover: cruz vertical + tooltip con todas las series.
    const hover = svgEl("line", { x1: 0, x2: 0, y1: m.t, y2: m.t + ih, class: "hover-line", visibility: "hidden" }, svg);
    const hit = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    const at = (clientX) => {
      const r = svg.getBoundingClientRect(), sx = ((clientX - r.left) / r.width) * W;
      return Math.max(0, Math.min(100, Math.round(((sx - m.l) / iw) * 100)));
    };
    const show = (x, cx, cy) => {
      hover.setAttribute("x1", X(x)); hover.setAttribute("x2", X(x)); hover.setAttribute("visibility", "visible");
      showTooltip(cx, cy, o.head(x), o.series.map((s) => [s.key || "base", o.valFmt(s.values[x]), s.label]));
    };
    hit.addEventListener("pointermove", (e) => show(at(e.clientX), e.clientX, e.clientY));
    hit.addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); hideTooltip(); });
    hit.addEventListener("focus", () => { const r = svg.getBoundingClientRect(); show(o.sel, r.left + (X(o.sel) / W) * r.width, r.top + 20); });
    hit.addEventListener("blur", () => { hover.setAttribute("visibility", "hidden"); hideTooltip(); });
    hit.addEventListener("click", (e) => { pctInput.value = Math.max(1, at(e.clientX)); renderCampaign(); });
    host.replaceChildren(svg);
  }

  const SHOW_COLS = [["job", "Ocupación", (v) => LABEL.job[v] || v], ["age", "Edad"], ["balance", "Saldo"],
    ["contact", "Canal", (v) => LABEL.contact[v] || v], ["month", "Mes", (v) => LABEL.month[v] || v],
    ["poutcome", "Campaña previa", (v) => LABEL.poutcome[v] || v]];

  function renderTable(k) {
    const tbl = $("#rank-table"); tbl.replaceChildren();
    const head = tbl.createTHead().insertRow();
    const cols = ["#", "Probabilidad", "Decil", "Recomendación", ...SHOW_COLS.map((c) => c[1])];
    if (state.hasY) cols.push("¿Convirtió? (real)");
    cols.forEach((c) => { const th = document.createElement("th"); th.textContent = c; head.append(th); });
    const body = tbl.createTBody(), shown = Math.min(k, 100), thr = state.C / state.V;
    state.rows.slice(0, shown).forEach((r, i) => {
      const tr = body.insertRow();
      tr.insertCell().textContent = i + 1;
      const pc = tr.insertCell(); const bar = document.createElement("span"); bar.className = "pbar";
      const fill = document.createElement("i"); fill.style.width = `${Math.min(100, 100 * r.probability)}%`; bar.append(fill);
      pc.append(bar, document.createTextNode(pct(r.probability, 1)));
      tr.insertCell().textContent = r.decile;
      const rc = tr.insertCell(); const tg = document.createElement("span");
      const call = r.probability >= thr; tg.className = "tag " + (call ? "call" : "skip"); tg.textContent = call ? "✓ Llamar" : "! Bajo umbral";
      rc.append(tg);
      SHOW_COLS.forEach(([key, , f]) => { tr.insertCell().textContent = f ? f(r[key]) : r[key]; });
      if (state.hasY) {
        const yc = tr.insertCell();
        if (r.y === "yes" || r.y === "1") { const t = document.createElement("span"); t.className = "tag yes"; t.textContent = "Sí"; yc.append(t); }
        else yc.textContent = "No";
      }
    });
    $("#table-title").textContent = `Clientes a llamar (${fmt(k)})`;
    $("#table-note").textContent = k > shown ? `Se muestran los primeros ${shown}; la descarga incluye los ${fmt(k)}.` : "";
  }

  $("#btn-download").addEventListener("click", () => {
    if (!state.rows) return;
    const rows = state.rows.slice(0, state.k), cols = Object.keys(rows[0]);
    const esc = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    a.download = "clientes_a_llamar.csv"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  window.addEventListener("resize", () => { if (state.rows) renderCampaign(); });

  // ---------- Inicio ----------
  $("#docs-link").href = `${API || "."}/docs`;
  readEconomics();
  showTab(location.hash.slice(1));
  checkHealth();
  loadSummary();
  // ?demo: carga un ejemplo en cada pestaña (útil para mostrar la app en una presentación).
  if (new URLSearchParams(location.search).has("demo")) {
    $("[data-preset=\"exito\"]").click();
    $("#btn-sample").click();
  }
})();
