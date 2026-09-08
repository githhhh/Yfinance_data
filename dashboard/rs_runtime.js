(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const RS_SOURCE = "Fred6725 / rs-log";
  const RS_SOURCE_URL = "https://github.com/Fred6725/rs-log";
  const COMMIT_URL = "https://api.github.com/repos/Fred6725/rs-log/commits?path=output/rs_stocks.csv&per_page=1";
  const CSV_URL = (sha) => `https://raw.githubusercontent.com/Fred6725/rs-log/${sha}/output/rs_stocks.csv`;

  // Contract: Reference only; never used by Pool, Gate, Top3 or default ordering.
  let dashboard = null;
  let dashboardPromise = null;
  let reference = {
    status: "loading",
    sourceDate: null,
    publishedDate: null,
    ratings: new Map(),
    error: null,
    checkedAt: null,
  };
  let refreshQueued = false;
  let popover = null;
  let popoverAnchor = null;
  let loading = false;

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

  function parseDateKey(value) {
    const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;
    const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function dateKey(date) {
    return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
  }

  function shiftDate(value, days) {
    const date = parseDateKey(value);
    if (!date) return null;
    date.setUTCDate(date.getUTCDate() + days);
    return dateKey(date);
  }

  function nthWeekday(year, month, weekday, occurrence) {
    const first = new Date(Date.UTC(year, month - 1, 1));
    const offset = (weekday - first.getUTCDay() + 7) % 7;
    first.setUTCDate(1 + offset + (occurrence - 1) * 7);
    return dateKey(first);
  }

  function lastWeekday(year, month, weekday) {
    const last = new Date(Date.UTC(year, month, 0));
    const offset = (last.getUTCDay() - weekday + 7) % 7;
    last.setUTCDate(last.getUTCDate() - offset);
    return dateKey(last);
  }

  function easterSunday(year) {
    const a = year % 19;
    const b = Math.floor(year / 100);
    const c = year % 100;
    const d = Math.floor(b / 4);
    const e = b % 4;
    const f = Math.floor((b + 8) / 25);
    const g = Math.floor((b - f + 1) / 3);
    const h = (19 * a + b - d - g + 15) % 30;
    const i = Math.floor(c / 4);
    const k = c % 4;
    const l = (32 + 2 * e + 2 * i - h - k) % 7;
    const m = Math.floor((a + 11 * h + 22 * l) / 451);
    const month = Math.floor((h + l - 7 * m + 114) / 31);
    const day = ((h + l - 7 * m + 114) % 31) + 1;
    return dateKey(new Date(Date.UTC(year, month - 1, day)));
  }

  function observedFixedHoliday(year, month, day) {
    const key = dateKey(new Date(Date.UTC(year, month - 1, day)));
    const weekday = parseDateKey(key)?.getUTCDay();
    if (weekday === 6) return shiftDate(key, -1);
    if (weekday === 0) return shiftDate(key, 1);
    return key;
  }

  function marketHolidayDates(year) {
    const holidays = new Set();
    const newYear = dateKey(new Date(Date.UTC(year, 0, 1)));
    const newYearWeekday = parseDateKey(newYear)?.getUTCDay();
    if (newYearWeekday === 0) holidays.add(shiftDate(newYear, 1));
    else if (newYearWeekday !== 6) holidays.add(newYear);

    holidays.add(nthWeekday(year, 1, 1, 3)); // Martin Luther King Jr. Day
    holidays.add(nthWeekday(year, 2, 1, 3)); // Presidents' Day
    holidays.add(shiftDate(easterSunday(year), -2)); // Good Friday
    holidays.add(lastWeekday(year, 5, 1)); // Memorial Day
    holidays.add(observedFixedHoliday(year, 6, 19)); // Juneteenth
    holidays.add(observedFixedHoliday(year, 7, 4)); // Independence Day
    holidays.add(nthWeekday(year, 9, 1, 1)); // Labor Day
    holidays.add(nthWeekday(year, 11, 4, 4)); // Thanksgiving
    holidays.add(observedFixedHoliday(year, 12, 25)); // Christmas
    return holidays;
  }

  function isMarketSessionDate(value) {
    const date = parseDateKey(value);
    if (!date) return false;
    const weekday = date.getUTCDay();
    if (weekday === 0 || weekday === 6) return false;
    return !marketHolidayDates(date.getUTCFullYear()).has(value);
  }

  function marketSessionDateForCommit(isoTimestamp) {
    // rs-log may still run on a weekend/NYSE holiday. Its prices then represent
    // the most recent completed US session, not the GitHub publication date.
    let candidate = nyDate(isoTimestamp);
    for (let attempts = 0; candidate && attempts < 10; attempts += 1) {
      if (isMarketSessionDate(candidate)) return candidate;
      candidate = shiftDate(candidate, -1);
    }
    return null;
  }

  function clockTime(value) {
    if (!(value instanceof Date) || Number.isNaN(value.getTime())) return "N/A";
    return new Intl.DateTimeFormat(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    }).format(value);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
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

  function dateState(poolDate) {
    if (!reference.sourceDate || !poolDate) return "current";
    if (reference.sourceDate < poolDate) return "stale";
    if (reference.sourceDate > poolDate) return "newer";
    return "current";
  }

  function stateFor(poolDate = currentPoolDate()) {
    if (reference.status === "loading") return "loading";
    if (reference.status === "refreshing") return "refreshing";
    if (reference.status === "error" && !reference.ratings.size) return "unavailable";
    if (reference.status === "error" && reference.ratings.size) return "refresh_failed";
    return dateState(poolDate);
  }

  function stateCopy(state) {
    return {
      loading: { icon: "···", label: "Loading reference", tone: "loading" },
      refreshing: { icon: "↻", label: "Refreshing", tone: "loading" },
      current: { icon: "ⓘ", label: "Current", tone: "current" },
      stale: { icon: "!", label: "Older than Pool", tone: "stale" },
      newer: { icon: "•", label: "Newer than Pool", tone: "newer" },
      unavailable: { icon: "○", label: "Unavailable", tone: "unavailable" },
      refresh_failed: { icon: "!", label: "Refresh failed", tone: "stale" },
    }[state] || { icon: "ⓘ", label: "Reference", tone: "current" };
  }

  function ensureStyles() {
    if (document.getElementById("rs-reference-styles")) return;
    const style = document.createElement("style");
    style.id = "rs-reference-styles";
    style.textContent = `
      th[data-sort-field="rs_percentile"] { position: relative; }
      th[data-sort-field="rs_percentile"] > button:not(.rs-info-button) { padding-right: 24px !important; }
      .rs-info-button {
        position: absolute;
        right: 7px;
        top: 50%;
        transform: translateY(-50%);
        box-sizing: border-box;
        width: 18px;
        height: 18px;
        display: grid;
        place-items: center;
        padding: 0;
        border: 1px solid #465365;
        border-radius: 50%;
        background: #11171e;
        color: #9ca8b7;
        font: 800 10px/1 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        cursor: help;
        z-index: 3;
      }
      .rs-info-button[data-state="current"] { color: #70e8d6; border-color: rgb(31 205 180 / 45%); }
      .rs-info-button[data-state="newer"] { color: #60a5fa; border-color: rgb(96 165 250 / 48%); }
      .rs-info-button[data-state="stale"],
      .rs-info-button[data-state="refresh_failed"] { color: #ffd21f; border-color: rgb(255 210 31 / 52%); }
      .rs-info-button[data-state="unavailable"] { color: #9ca8b7; border-color: #465365; }
      .rs-info-button[data-state="loading"],
      .rs-info-button[data-state="refreshing"] { color: #9ca8b7; border-color: #465365; }
      [data-rs-enhanced="true"] { pointer-events: none !important; }
      .rs-reference-popover {
        position: fixed;
        width: min(330px, calc(100vw - 20px));
        padding: 13px;
        border: 1px solid rgb(148 163 184 / 24%);
        border-radius: 9px;
        background: #0b1320;
        color: #e2e8f0;
        box-shadow: 0 12px 32px rgb(0 0 0 / 44%);
        z-index: 1000000;
        font: 11px/1.45 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .rs-popover-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
      .rs-popover-title { color: #fff; font-size: 13px; font-weight: 800; }
      .rs-popover-subtitle { margin-top: 2px; color: #8fa0b4; font-size: 10px; }
      .rs-state {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        min-height: 23px;
        margin-top: 11px;
        padding: 0 8px;
        border: 1px solid #35404d;
        border-radius: 99px;
        color: #cbd5e1;
        background: rgb(255 255 255 / 2%);
        font-size: 10px;
        font-weight: 750;
      }
      .rs-state[data-tone="current"] { color: #70e8d6; border-color: rgb(31 205 180 / 38%); }
      .rs-state[data-tone="newer"] { color: #93c5fd; border-color: rgb(96 165 250 / 38%); }
      .rs-state[data-tone="stale"] { color: #fde68a; border-color: rgb(255 210 31 / 42%); }
      .rs-meta-grid {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 5px 12px;
        margin-top: 11px;
        padding-top: 10px;
        border-top: 1px solid rgb(148 163 184 / 14%);
      }
      .rs-meta-grid span { color: #8391a3; }
      .rs-meta-grid b { color: #dbe3ec; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; }
      .rs-source-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 10px;
      }
      .rs-source-link { color: #70e8d6; text-decoration: none; font-weight: 700; }
      .rs-source-link:hover, .rs-source-link:focus-visible { text-decoration: underline; }
      .rs-disclaimer { margin-top: 3px; color: #7f8da0; font-size: 10px; }
      .rs-refresh-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-top: 11px;
        padding-top: 9px;
        border-top: 1px solid rgb(148 163 184 / 14%);
      }
      .rs-checked { color: #758397; font-size: 10px; }
      .rs-refresh-button {
        min-height: 29px;
        padding: 0 10px;
        border: 1px solid #465365;
        border-radius: 6px;
        background: #151b23;
        color: #d5dde7;
        font-family: inherit;
        font-size: 10px;
        font-weight: 700;
        line-height: 1;
        cursor: pointer;
      }
      .rs-refresh-button:disabled { opacity: .55; cursor: wait; }
      .rs-reference-popover a,
      .rs-reference-popover button { -webkit-tap-highlight-color: transparent; }
      @media (width <= 760px) {
        .rs-reference-popover {
          left: 10px !important;
          right: 10px;
          bottom: max(10px, env(safe-area-inset-bottom));
          top: auto !important;
          width: auto;
          max-width: none;
          border-radius: 11px;
          padding: 14px;
        }
        .rs-info-button { width: 19px; height: 19px; right: 5px; }
      }
    `;
    document.head.appendChild(style);
  }

  function plainRsNode(node) {
    if (!node) return node;
    if (node.dataset.rsEnhanced === "true") {
      const replacement = node.cloneNode(true);
      node.replaceWith(replacement);
      node = replacement;
    }
    node.removeAttribute("title");
    node.removeAttribute("data-rs-enhanced");
    node.removeAttribute("data-rs-tooltip");
    node.removeAttribute("tabindex");
    node.removeAttribute("role");
    node.style.removeProperty("cursor");
    node.classList.remove("rs-stale");
    return node;
  }

  function displayValue(code) {
    const state = stateFor();
    const rating = ratingFor(code);
    if (state === "loading") return "—";
    if (state === "unavailable") return "N/A";
    return rating?.current === null || rating?.current === undefined ? "N/A" : String(rating.current);
  }

  function updateTable() {
    app.querySelectorAll("[data-table-shell]").forEach((shell) => {
      const headers = [...shell.querySelectorAll("thead th")];
      const rsIndex = headers.findIndex((header) => header.dataset.sortField === "rs_percentile"
        || String(header.querySelector(".table-header-label")?.textContent || header.textContent).trim() === "RS");
      if (rsIndex < 0) return;
      shell.querySelectorAll("tbody tr[data-code]").forEach((row) => {
        const cell = row.children[rsIndex];
        if (!cell) return;
        let target = plainRsNode(cell.querySelector("span"));
        if (!target) {
          target = document.createElement("span");
          cell.replaceChildren(target);
        }
        const display = displayValue(row.dataset.code);
        if (target.textContent !== display) target.textContent = display;
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
    const value = plainRsNode(cell?.querySelector(".selected-value"));
    if (!value) return;

    const state = stateFor();
    const rating = ratingFor(code);
    if (state === "loading") {
      if (value.textContent !== "—") value.textContent = "—";
      return;
    }
    if (state === "unavailable" || !rating || rating.current === null) {
      if (value.textContent !== "N/A") value.textContent = "N/A";
      return;
    }
    const suffix = [
      `1M ${rating.m1 ?? "N/A"}`,
      `3M ${rating.m3 ?? "N/A"}`,
      `6M ${rating.m6 ?? "N/A"}`,
    ].join(" · ");
    const html = `${rating.current} <small>${suffix}</small>`;
    if (value.innerHTML !== html) value.innerHTML = html;
  }

  function infoHeader() {
    const header = app.querySelector('th[data-sort-field="rs_percentile"]');
    if (!header) return null;
    let button = header.querySelector("[data-rs-info]");
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "rs-info-button";
      button.dataset.rsInfo = "true";
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (popover && popoverAnchor === button) {
          closePopover();
        } else {
          openPopover(button);
        }
      });
      button.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closePopover();
      });
      header.appendChild(button);
    }
    const state = stateFor();
    const copy = stateCopy(state);
    button.dataset.state = state;
    if (button.textContent !== "i") button.textContent = "i";
    button.setAttribute("aria-label", `RS reference: ${copy.label}. Open details.`);
    button.title = `RS reference · ${copy.label}`;
    return button;
  }

  function positionPopover() {
    if (!popover || !popoverAnchor || window.innerWidth <= 760) return;
    const anchor = popoverAnchor.getBoundingClientRect();
    const tip = popover.getBoundingClientRect();
    const padding = 8;
    const left = Math.min(
      Math.max(padding, anchor.right - tip.width),
      Math.max(padding, window.innerWidth - tip.width - padding),
    );
    const below = anchor.bottom + 7;
    const top = below + tip.height <= window.innerHeight - padding
      ? below
      : Math.max(padding, anchor.top - tip.height - 7);
    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
  }

  function closePopover() {
    popover?.remove();
    popover = null;
    popoverAnchor = null;
  }

  function popoverHtml() {
    const poolDate = currentPoolDate();
    const state = stateFor(poolDate);
    const copy = stateCopy(state);
    const checked = reference.checkedAt ? `Checked ${clockTime(reference.checkedAt)}` : "Not checked yet";
    let note = "";
    if (state === "stale") note = "RS market session is older than the selected Pool snapshot.";
    if (state === "newer") note = "RS market session is newer than the selected Pool snapshot.";
    if (state === "unavailable") note = "Public RS reference is unavailable. Pool review is unaffected.";
    if (state === "refresh_failed") note = "Refresh failed. Showing the last successfully loaded RS data.";
    if (state === "loading") note = "Loading public RS reference…";
    if (state === "refreshing") note = "Refreshing public RS reference…";

    return `
      <div class="rs-popover-head">
        <div>
          <div class="rs-popover-title">Relative Strength</div>
          <div class="rs-popover-subtitle">IBD-style percentile · 0–99</div>
        </div>
      </div>
      <div class="rs-state" data-tone="${copy.tone}"><span>${copy.icon}</span><span>${copy.label}</span></div>
      ${note ? `<div class="rs-disclaimer">${note}</div>` : ""}
      <div class="rs-meta-grid">
        <span>RS market session</span><b>${escapeHtml(reference.sourceDate || "N/A")}</b>
        <span>Published (ET)</span><b>${escapeHtml(reference.publishedDate || "N/A")}</b>
        <span>Pool snapshot</span><b>${escapeHtml(poolDate || "N/A")}</b>
      </div>
      <div class="rs-source-row">
        <a class="rs-source-link" href="${RS_SOURCE_URL}" target="_blank" rel="noopener noreferrer">${RS_SOURCE} ↗</a>
      </div>
      <div class="rs-disclaimer">Public reference · not official IBD RS</div>
      <div class="rs-refresh-row">
        <span class="rs-checked">${escapeHtml(checked)}</span>
        <button class="rs-refresh-button" type="button" data-rs-refresh ${loading ? "disabled" : ""}>
          ${state === "unavailable" || state === "refresh_failed" ? "↻ Retry" : "↻ Refresh"}
        </button>
      </div>
    `;
  }

  function renderPopover() {
    if (!popover) return;
    popover.innerHTML = popoverHtml();
    popover.querySelector("[data-rs-refresh]")?.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await loadReference({ preserve: reference.ratings.size > 0 });
    });
    popover.querySelector(".rs-source-link")?.addEventListener("click", (event) => {
      event.stopPropagation();
    });
    requestAnimationFrame(positionPopover);
  }

  function openPopover(anchor) {
    closePopover();
    popoverAnchor = anchor;
    popover = document.createElement("section");
    popover.className = "rs-reference-popover";
    popover.setAttribute("role", "dialog");
    popover.setAttribute("aria-label", "Relative Strength reference details");
    document.body.appendChild(popover);
    renderPopover();
  }

  function updateOpenPopover() {
    if (popover) renderPopover();
  }

  function refresh() {
    ensureStyles();
    infoHeader();
    updateTable();
    updateSelected();
    updateOpenPopover();
  }

  function scheduleRefresh() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      refresh();
    });
  }

  async function ensureDashboard() {
    if (dashboard) return dashboard;
    if (!dashboardPromise) {
      dashboardPromise = fetch("./data/dashboard.json", { cache: "no-store" })
        .then((response) => {
          if (!response.ok) throw new Error(`Dashboard HTTP ${response.status}`);
          return response.json();
        })
        .then((payload) => {
          dashboard = payload;
          return payload;
        })
        .finally(() => {
          dashboardPromise = null;
        });
    }
    return dashboardPromise;
  }

  async function loadReference({ preserve = false } = {}) {
    if (loading) return;
    loading = true;
    const previous = reference;
    reference = preserve
      ? { ...previous, status: "refreshing", error: null }
      : {
        status: "loading",
        sourceDate: null,
        publishedDate: null,
        ratings: new Map(),
        error: null,
        checkedAt: previous.checkedAt,
      };
    scheduleRefresh();

    try {
      const [, commitResponse] = await Promise.all([
        ensureDashboard(),
        fetch(COMMIT_URL, { cache: "no-store" }),
      ]);
      if (!commitResponse.ok) throw new Error(`RS metadata HTTP ${commitResponse.status}`);
      const commits = await commitResponse.json();
      const latest = Array.isArray(commits) ? commits[0] : null;
      const sha = latest?.sha;
      const timestamp = latest?.commit?.committer?.date;
      if (!sha || !timestamp) throw new Error("RS metadata is incomplete");

      const publishedDate = nyDate(timestamp);
      const sourceDate = marketSessionDateForCommit(timestamp);
      if (!publishedDate || !sourceDate) throw new Error("RS publication date is invalid");

      const csvResponse = await fetch(CSV_URL(sha), { cache: "force-cache" });
      if (!csvResponse.ok) throw new Error(`RS CSV HTTP ${csvResponse.status}`);
      reference = {
        status: "ready",
        sourceDate,
        publishedDate,
        ratings: parseRatings(await csvResponse.text()),
        error: null,
        checkedAt: new Date(),
      };
    } catch (error) {
      reference = preserve
        ? { ...previous, status: "error", error: String(error?.message || error), checkedAt: new Date() }
        : {
          status: "error",
          sourceDate: null,
          publishedDate: null,
          ratings: new Map(),
          error: String(error?.message || error),
          checkedAt: new Date(),
        };
    } finally {
      loading = false;
      scheduleRefresh();
    }
  }

  document.addEventListener("pointerdown", (event) => {
    if (!popover) return;
    if (event.target.closest?.(".rs-reference-popover") || event.target.closest?.("[data-rs-info]")) return;
    closePopover();
  });
  window.addEventListener("resize", () => {
    if (popover && window.innerWidth > 760) positionPopover();
  });
  window.addEventListener("scroll", () => {
    if (popover && window.innerWidth > 760) closePopover();
  }, true);

  const observer = new MutationObserver(scheduleRefresh);
  observer.observe(app, { childList: true, subtree: true });

  ensureStyles();
  scheduleRefresh();
  loadReference();
})();