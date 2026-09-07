(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const RS_SOURCE = "Fred6725/rs-log";
  const COMMIT_URL = "https://api.github.com/repos/Fred6725/rs-log/commits?path=output/rs_stocks.csv&per_page=1";
  const CSV_URL = (sha) => `https://raw.githubusercontent.com/Fred6725/rs-log/${sha}/output/rs_stocks.csv`;

  let dashboard = null;
  let reference = {
    status: "loading",
    sourceDate: null,
    ratings: new Map(),
    error: null,
  };
  let refreshQueued = false;

  function nyDate(isoTimestamp) {
    const value = new Date(isoTimestamp);
    if (Number.isNaN(value.getTime())) return null;
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(value);
    const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
    return values.year && values.month && values.day
      ? `${values.year}-${values.month}-${values.day}`
      : null;
  }

  function parseCsvLine(line) {
    const cells = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (char === '"') {
        if (quoted && line[index + 1] === '"') {
          value += '"';
          index += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        cells.push(value);
        value = "";
      } else {
        value += char;
      }
    }
    cells.push(value);
    return cells;
  }

  function percentile(value) {
    const parsed = Number(String(value ?? "").trim());
    return Number.isFinite(parsed) && parsed >= 0 && parsed <= 99 ? Math.round(parsed) : null;
  }

  function parseRatings(csvText) {
    const lines = csvText.split(/\r?\n/).filter((line) => line.trim());
    if (!lines.length) throw new Error("RS CSV is empty");
    const headers = parseCsvLine(lines[0]).map((value) => value.trim());
    const index = Object.fromEntries(headers.map((name, position) => [name, position]));
    for (const required of ["Ticker", "Percentile", "1M_RS_Percentile", "3M_RS_Percentile", "6M_RS_Percentile"]) {
      if (!(required in index)) throw new Error(`RS CSV missing ${required}`);
    }

    const ratings = new Map();
    for (const line of lines.slice(1)) {
      const cells = parseCsvLine(line);
      const code = String(cells[index.Ticker] ?? "").trim().toUpperCase();
      if (!code) continue;
      ratings.set(code, {
        current: percentile(cells[index.Percentile]),
        m1: percentile(cells[index["1M_RS_Percentile"]]),
        m3: percentile(cells[index["3M_RS_Percentile"]]),
        m6: percentile(cells[index["6M_RS_Percentile"]]),
      });
    }
    if (!ratings.size) throw new Error("RS CSV contains no ticker rows");
    return ratings;
  }

  function currentPoolDate() {
    if (!dashboard) return null;
    const period = app.querySelector('[data-action="period"][aria-pressed="true"]')?.dataset.value
      || dashboard.default_period
      || "WEEKEND";
    return period === "MIDWEEK"
      ? dashboard.meta?.midweek_snapshot_date || null
      : dashboard.meta?.complete_snapshot_date || null;
  }

  function ratingFor(code) {
    return reference.ratings.get(String(code ?? "").trim().toUpperCase()) || null;
  }

  function stateFor(poolDate) {
    if (reference.status === "loading") return "loading";
    if (reference.status !== "ready") return "unavailable";
    if (reference.sourceDate && poolDate && reference.sourceDate < poolDate) return "stale";
    return "ready";
  }

  function tooltip(code) {
    const poolDate = currentPoolDate();
    const state = stateFor(poolDate);
    const rating = ratingFor(code);
    if (state === "loading") {
      return `RS N/A\nSource: ${RS_SOURCE}\nLoading public reference…`;
    }
    if (state === "unavailable") {
      return `RS N/A\nSource: ${RS_SOURCE}\nReference unavailable. Dashboard data is unaffected.`;
    }
    if (!rating || rating.current === null) {
      return [
        "RS N/A",
        `Source: ${RS_SOURCE}`,
        `Updated (ET): ${reference.sourceDate || "N/A"}`,
        "Ticker is not present in the current public RS dataset.",
      ].join("\n");
    }
    return [
      `${state === "stale" ? "RS stale" : "RS Percentile"}: ${rating.current}`,
      `1M ago: ${rating.m1 ?? "N/A"}`,
      `3M ago: ${rating.m3 ?? "N/A"}`,
      `6M ago: ${rating.m6 ?? "N/A"}`,
      `Updated (ET): ${reference.sourceDate || "N/A"}`,
      `Pool snapshot: ${poolDate || "N/A"}`,
      `Source: ${RS_SOURCE}`,
      "Reference only; never used by Pool, Gate, Top3 or default ordering.",
    ].join("\n");
  }

  function applyTooltip(node, copy) {
    if (node.dataset.rsEnhanced === "true") {
      node.dataset.rsTooltip = copy;
      node.removeAttribute("title");
      node.setAttribute("aria-label", `${node.textContent.trim() || "RS"}. Tap for RS reference details.`);
    } else if (node.getAttribute("title") !== copy) {
      node.setAttribute("title", copy);
    }
  }

  function setReferenceNode(node, code) {
    const poolDate = currentPoolDate();
    const state = stateFor(poolDate);
    const rating = ratingFor(code);
    const current = rating?.current ?? null;
    const display = current === null ? "N/A" : String(current);
    if (node.textContent !== display) node.textContent = display;
    node.classList.toggle("rs-stale", current !== null && state === "stale");
    applyTooltip(node, tooltip(code));
  }

  function updateTable() {
    app.querySelectorAll("[data-table-shell]").forEach((shell) => {
      const headers = [...shell.querySelectorAll("thead th")];
      const rsIndex = headers.findIndex((header) => {
        const label = header.querySelector(".table-header-label")?.textContent || header.textContent;
        return String(label).trim() === "RS";
      });
      if (rsIndex < 0) return;
      shell.querySelectorAll("tbody tr[data-code]").forEach((row) => {
        const cell = row.children[rsIndex];
        if (!cell) return;
        let target = cell.querySelector("span");
        if (!target) {
          target = document.createElement("span");
          cell.replaceChildren(target);
        }
        setReferenceNode(target, row.dataset.code);
      });
    });
  }

  function updateSelected() {
    const selected = app.querySelector(".selected-strip:not(.empty)");
    if (!selected) return;
    const code = selected.querySelector(".selected-code")?.textContent?.trim();
    if (!code) return;
    const cell = [...selected.querySelectorAll(".selected-cell")].find(
      (item) => item.querySelector(".selected-key")?.textContent?.trim() === "RS Reference",
    );
    const value = cell?.querySelector(".selected-value");
    if (!value) return;

    const rating = ratingFor(code);
    const poolDate = currentPoolDate();
    const state = stateFor(poolDate);
    if (!rating || rating.current === null || state === "loading" || state === "unavailable") {
      if (value.textContent !== "N/A") value.textContent = "N/A";
    } else {
      const html = `${rating.current} <small>1M ${rating.m1 ?? "N/A"} · 3M ${rating.m3 ?? "N/A"} · 6M ${rating.m6 ?? "N/A"}${state === "stale" ? ` · stale ${reference.sourceDate || ""}` : ""}</small>`;
      if (value.innerHTML !== html) value.innerHTML = html;
    }
    applyTooltip(value, tooltip(code));
  }

  function refresh() {
    updateTable();
    updateSelected();
  }

  function scheduleRefresh() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      refresh();
    });
  }

  async function loadReference() {
    try {
      const [dashboardResponse, commitResponse] = await Promise.all([
        fetch("./data/dashboard.json", { cache: "no-store" }),
        fetch(COMMIT_URL, { cache: "no-store" }),
      ]);
      if (!dashboardResponse.ok) throw new Error(`Dashboard HTTP ${dashboardResponse.status}`);
      if (!commitResponse.ok) throw new Error(`RS metadata HTTP ${commitResponse.status}`);
      dashboard = await dashboardResponse.json();
      const commits = await commitResponse.json();
      const latest = Array.isArray(commits) ? commits[0] : null;
      const sha = latest?.sha;
      const timestamp = latest?.commit?.committer?.date;
      if (!sha || !timestamp) throw new Error("RS metadata is incomplete");

      const csvResponse = await fetch(CSV_URL(sha), { cache: "no-store" });
      if (!csvResponse.ok) throw new Error(`RS CSV HTTP ${csvResponse.status}`);
      reference = {
        status: "ready",
        sourceDate: nyDate(timestamp),
        ratings: parseRatings(await csvResponse.text()),
        error: null,
      };
    } catch (error) {
      reference = {
        status: "error",
        sourceDate: null,
        ratings: new Map(),
        error: String(error?.message || error),
      };
    }
    scheduleRefresh();
  }

  const observer = new MutationObserver(scheduleRefresh);
  observer.observe(app, { childList: true, subtree: true });
  scheduleRefresh();
  loadReference();
})();
