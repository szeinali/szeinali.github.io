// Manifest-driven project log renderer. No external dependencies.
// Each entry in entries.json lists "visuals"; each visual has a "type" handled below.
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const COLORS = { trained: "--trained", random_policy: "--random", classical_controller: "--classical" };
  const LABELS = { trained: "Learned policy", random_policy: "Random actions", classical_controller: "Classical controller (Jᵀ PD)" };
  const col = (k) => css(COLORS[k] || "--accent");
  const cache = {};
  const load = (src) => (cache[src] ||= fetch(src).then((r) => { if (!r.ok) throw new Error(src); return r.json(); }));
  const h = (tag, attrs = {}, html) => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (html != null) e.innerHTML = html; return e; };
  const s = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const pct = (v) => `${(100 * v).toFixed(0)}%`;
  const tip = h("div", { class: "tip" }); document.body.appendChild(tip);

  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => span / x <= n) || mag * 10;
    const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10)); return t;
  }

  // ---------- generic line chart with optional bands, reference lines, hover ----------
  function lineChart(el, o) {
    const W = 760, H = 300, m = { l: 52, r: 16, t: 10, b: 38 };
    const all = o.series.flatMap((d) => [...d.points, ...(d.band || []).flatMap((b) => [[b[0], b[1]], [b[0], b[2]]])]);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).concat((o.hlines || []).map((r) => r.y));
    let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = o.yRange || [Math.min(...ys), Math.max(...ys)];
    if (!o.yRange) { const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad; }
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r), Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": o.aria || "" });
    niceTicks(y0, y1).forEach((t) => { svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }));
      const tx = s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }); tx.textContent = (o.yFmt || String)(t); svg.append(tx); });
    niceTicks(x0, x1, 6).forEach((t) => { const tx = s("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }); tx.textContent = (o.xFmt || String)(t); svg.append(tx); });
    const xl = s("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }); xl.textContent = o.xLabel; svg.append(xl);
    const yl = s("text", { x: 12, y: (m.t + H - m.b) / 2, transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle" }); yl.textContent = o.yLabel; svg.append(yl);
    (o.hlines || []).forEach((r) => { svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: r.color, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
      const t = s("text", { x: W - m.r - 4, y: Y(r.y) - 5, "text-anchor": "end", style: `fill:${r.color}` }); t.textContent = r.label; svg.append(t); });
    o.series.forEach((d) => {
      if (d.band) svg.append(s("path", { d: "M" + d.band.map((b) => `${X(b[0])},${Y(b[2])}`).join("L") + "L" + d.band.slice().reverse().map((b) => `${X(b[0])},${Y(b[1])}`).join("L") + "Z", fill: d.color, opacity: 0.14 }));
      svg.append(s("path", { d: "M" + d.points.map((p) => `${X(p[0])},${Y(p[1])}`).join("L"), fill: "none", stroke: d.color, "stroke-width": d.width || 2, opacity: d.opacity || 1, "stroke-linejoin": "round" }));
    });
    // hover
    const guide = s("line", { y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-width": 1, opacity: 0 }); svg.append(guide);
    const hov = o.series.filter((d) => d.hover !== false);
    svg.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect(), xv = x0 + (((ev.clientX - r.left) * W) / r.width - m.l) / (W - m.l - m.r) * (x1 - x0);
      if (xv < x0 || xv > x1) { tip.style.opacity = 0; guide.setAttribute("opacity", 0); return; }
      const rows = hov.map((d) => { const p = d.points.reduce((a, b) => (Math.abs(b[0] - xv) < Math.abs(a[0] - xv) ? b : a)); return [d, p]; });
      guide.setAttribute("x1", X(rows[0][1][0])); guide.setAttribute("x2", X(rows[0][1][0])); guide.setAttribute("opacity", 0.5);
      tip.innerHTML = `<b>${(o.xFmt || String)(rows[0][1][0])} ${o.xUnit || ""}</b><br>` + rows.map(([d, p]) => `${d.name}: ${(o.tipFmt || o.yFmt || String)(p[1])}`).join("<br>");
      tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY - 10 + "px"; tip.style.opacity = 1;
    });
    svg.addEventListener("mouseleave", () => { tip.style.opacity = 0; guide.setAttribute("opacity", 0); });
    el.append(svg);
    if (o.legend) el.append(h("div", { class: "legend" }, o.legend.map(([n, c]) => `<span style="--c:${c}">${n}</span>`).join("")));
  }

  const meanSeeds = (runs, key) => runs[0].log.map((_, i) => {
    const v = runs.map((r) => r.log[i][key]); const mu = v.reduce((a, b) => a + b) / v.length;
    return [runs[0].log[i].env_steps / 1e6, mu, Math.min(...v), Math.max(...v)];
  });

  // ---------- renderers ----------
  const R = {
    async kpis(el, v) {
      const m = await load(v.src), last = m.runs.map((r) => r.log[r.log.length - 1]);
      const avg = (k) => last.reduce((a, b) => a + b[k], 0) / last.length;
      const sps = m.runs.reduce((a, r) => a + r.steps_per_sec, 0) / m.runs.length;
      const rf = m.reference;
      const items = [
        [pct(avg("success_rate")), `Learned policy success (≤${m.success_threshold_cm} cm), mean of ${m.runs.length} seeds`],
        [`${avg("mean_final_dist_cm").toFixed(1)} cm`, `Learned: mean final distance (random: ${rf.random_policy.mean_final_dist_cm.toFixed(1)} cm)`],
        ...(rf.classical_controller ? [[pct(rf.classical_controller.success_rate), "Classical controller success: the bar to beat"]] : []),
        [`${(sps / 1e6).toFixed(2)}M`, v.throughput_label || "Env steps / sec"],
      ];
      el.append(h("div", { class: "kpis" }, items.map(([a, b]) => `<div class="kpi"><div class="v">${a}</div><div class="l">${b}</div></div>`).join("")));
    },
    async learning_curve(el, v) {
      const m = await load(v.src), key = v.metric, isRate = key === "success_rate";
      const fmt = isRate ? pct : (x) => (+x).toFixed(key.includes("cm") ? 1 : 2);
      const mu = meanSeeds(m.runs, key);
      const series = m.runs.map((r) => ({ name: `seed ${r.seed}`, color: col("trained"), width: 1, opacity: 0.35, hover: false, points: r.log.map((e) => [e.env_steps / 1e6, e[key]]) }));
      series.push({ name: "Learned (mean)", color: col("trained"), width: 2.5, points: mu.map((p) => [p[0], p[1]]), band: mu.map((p) => [p[0], p[2], p[3]]) });
      const refs = ["classical_controller", "random_policy"].filter((k) => m.reference[k]);
      const hl = refs.map((k) => ({ y: m.reference[k][key], color: col(k), label: LABELS[k] }));
      lineChart(el, { series, hlines: hl, xLabel: "environment steps (millions)", xUnit: "M steps", yLabel: v.y_label || key, yFmt: fmt,
        yRange: isRate ? [0, 1] : null, xFmt: (x) => x, aria: v.title,
        legend: [["Learned: mean, min–max band over seeds", col("trained")], ...refs.map((k) => [LABELS[k], col(k)])] });
    },
    async dist_over_time(el, v) {
      const d = await load(v.src), t = d.time_s;
      const ks = ["random_policy", "trained", "classical_controller"].filter((k) => d.dist_over_time_cm[k]);
      const series = ks.map((k) => { const q = d.dist_over_time_cm[k]; return { name: LABELS[k], color: col(k), points: t.map((x, i) => [x, q.mean[i]]), band: t.map((x, i) => [x, q.p10[i], q.p90[i]]) }; });
      lineChart(el, { series, xLabel: "time in episode (s)", xUnit: "s", yLabel: "distance to target (cm)", yFmt: (x) => (+x).toFixed(1), xFmt: (x) => x,
        hlines: [{ y: 2, color: css("--good"), label: "success threshold" }], legend: ks.map((k) => [LABELS[k] + " (mean, 10–90th pct)", col(k)]) });
    },
    async histogram(el, v) {
      const d = await load(v.src), b = d.bins_cm, ks = Object.keys(d.final_dist_hist);
      const W = 760, H = 260, m = { l: 44, r: 16, t: 10, b: 38 };
      const tot = d.final_dist_hist[ks[0]].reduce((a, c) => a + c, 0);
      const ymax = Math.max(...ks.flatMap((k) => d.final_dist_hist[k])) / tot;
      const X = (x) => m.l + (x / b[b.length - 1]) * (W - m.l - m.r), Y = (y) => H - m.b - (y / (ymax * 1.08)) * (H - m.t - m.b);
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
      niceTicks(0, ymax * 1.08, 4).forEach((t) => { svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" })); const tx = s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }); tx.textContent = pct(t); svg.append(tx); });
      b.forEach((x, i) => { if (i % 5 === 0) { const tx = s("text", { x: X(x), y: H - m.b + 16, "text-anchor": "middle" }); tx.textContent = x + (i === b.length - 1 ? "+" : ""); svg.append(tx); } });
      const xl = s("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }); xl.textContent = "final distance to target (cm)"; svg.append(xl);
      svg.append(s("line", { x1: X(2), x2: X(2), y1: m.t, y2: H - m.b, stroke: css("--good"), "stroke-dasharray": "4 3" }));
      const bw = (X(b[1]) - X(b[0])) / ks.length;
      ks.forEach((k, j) => d.final_dist_hist[k].forEach((c, i) => {
        const r = s("rect", { x: X(b[i]) + j * bw, y: Y(c / tot), width: Math.max(bw - 1, 1), height: Y(0) - Y(c / tot), fill: col(k), rx: 1 });
        r.addEventListener("mousemove", (ev) => { tip.innerHTML = `${LABELS[k]}<br>${b[i]}–${b[i + 1]} cm: <b>${pct(c / tot)}</b> (${c})`; tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY - 10 + "px"; tip.style.opacity = 1; });
        r.addEventListener("mouseleave", () => (tip.style.opacity = 0)); svg.append(r);
      }));
      el.append(svg, h("div", { class: "legend" }, ks.map((k) => `<span style="--c:${col(k)}">${LABELS[k]}</span>`).join("") + `<span style="--c:${css("--good")}">2 cm success line</span>`));
    },
    async arm_replay(el, v) {
      const d = await load(v.src), stages = Object.keys(d.stages).map(Number).sort((a, b) => a - b);
      let stage = stages[stages.length - 1], t = 0, playing = true;
      const T = d.stages[stage].q[0].length, reach = d.links[0] + d.links[1];
      const ctr = h("div", { class: "controls" });
      ctr.append(h("span", { class: "muted" }, "Training iteration:"));
      const btns = stages.map((st) => { const b = h("button", {}, st === 0 ? "0 (init)" : String(st)); b.onclick = () => { stage = st; t = 0; btns.forEach((x) => x.classList.toggle("on", x === b)); upd(); }; return b; });
      btns[btns.length - 1].classList.add("on"); ctr.append(...btns);
      const play = h("button", {}, "Pause"); play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; }; ctr.append(play);
      const slider = h("input", { type: "range", min: 0, max: T - 1, value: 0, "aria-label": "time step" }); slider.oninput = () => { t = +slider.value; playing = false; play.textContent = "Play"; }; ctr.append(slider);
      const clock = h("span", { class: "muted small" }); ctr.append(clock);
      const grid = h("div", { class: "replay" }); const cells = d.targets.map((_, i) => { const c = h("div"); const cv = h("canvas", { width: 260, height: 260 }); const lab = h("div", { class: "cell" }); c.append(cv, lab); grid.append(c); return { cv, lab }; });
      el.append(ctr, grid);
      function upd() { cells.forEach((c, i) => { const fd = d.stages[stage].final_dist_cm[i]; c.lab.innerHTML = `final: <b style="color:${fd <= 2 ? css("--good") : "inherit"}">${fd.toFixed(1)} cm</b>`; }); }
      function fk(q) { const [a, b] = q, [l1, l2] = d.links; const e = [l1 * Math.cos(a), l1 * Math.sin(a)]; return [e, [e[0] + l2 * Math.cos(a + b), e[1] + l2 * Math.sin(a + b)]]; }
      function draw() {
        const ink = css("--ink"), mut = css("--muted"), tr = col("trained"), good = css("--good");
        cells.forEach(({ cv }, i) => {
          const g = cv.getContext("2d"), S = cv.width, sc = (S / 2 - 12) / reach, P = (p) => [S / 2 + p[0] * sc, S / 2 - p[1] * sc];
          g.clearRect(0, 0, S, S);
          g.strokeStyle = css("--line"); g.lineWidth = 1; g.beginPath(); g.arc(S / 2, S / 2, reach * sc, 0, 7); g.stroke();
          const tg = P(d.targets[i]); g.fillStyle = good; g.globalAlpha = 0.18; g.beginPath(); g.arc(tg[0], tg[1], 0.02 * sc, 0, 7); g.fill(); g.globalAlpha = 1;
          g.beginPath(); g.arc(tg[0], tg[1], 4, 0, 7); g.fill();
          const qs = d.stages[stage].q[i];
          g.strokeStyle = tr; g.globalAlpha = 0.5; g.lineWidth = 1.5; g.beginPath();
          for (let k = 0; k <= t; k++) { const p = P(fk(qs[k])[1]); k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); } g.stroke(); g.globalAlpha = 1;
          const [e, tip_] = fk(qs[t]).map(P); g.strokeStyle = ink; g.lineWidth = 5; g.lineCap = "round";
          g.beginPath(); g.moveTo(S / 2, S / 2); g.lineTo(e[0], e[1]); g.lineTo(tip_[0], tip_[1]); g.stroke();
          g.fillStyle = mut; g.beginPath(); g.arc(S / 2, S / 2, 5, 0, 7); g.fill(); g.fillStyle = tr; g.beginPath(); g.arc(tip_[0], tip_[1], 4.5, 0, 7); g.fill();
        });
        slider.value = t; clock.textContent = `t = ${(t * d.dt + d.dt).toFixed(2)} s`;
      }
      upd(); let last = 0;
      (function loop(ts) { if (playing && ts - last > 1000 * d.dt * 1.5) { t = (t + 1) % T; if (t === 0) last = ts + 700; else last = ts; } draw(); requestAnimationFrame(loop); })(0);
    },
  };

  async function main() {
    const root = document.getElementById("log"); if (!root) return;
    const man = await load(root.dataset.manifest);
    for (const e of man.entries.slice().sort((a, b) => (a.date < b.date ? 1 : -1))) {
      const sec = h("section", { class: "entry", id: e.id });
      sec.append(h("div", { class: "entry-head" }, `<h2 style="margin:0">${e.title}</h2><span class="date">${e.date}</span>`));
      if (e.summary) sec.append(h("p", {}, e.summary));
      if (e.takeaways) sec.append(h("ul", { class: "takeaways" }, e.takeaways.map((t) => `<li>${t}</li>`).join("")));
      for (const v of e.visuals || []) {
        const card = h("div", { class: "card" });
        if (v.title) card.append(h("div", { class: "viz-title" }, v.title));
        if (v.caption) card.append(h("p", { class: "viz-cap" }, v.caption));
        const body = h("div"); card.append(body);
        if (v.why) card.append(h("p", { class: "viz-why" }, `<b>What it means:</b> ${v.why}`));
        sec.append(card);
        try { await R[v.type](body, v); } catch (err) { body.textContent = `Could not render ${v.type}: ${err.message}`; }
      }
      root.append(sec);
    }
    if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
  }
  main();
})();
