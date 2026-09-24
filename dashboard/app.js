(() => {
  "use strict";

  const app = document.getElementById("app");
  const WATCH_STAGE = "NEAR_BREAKOUT";
  const ENTRY_STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"];
  const REVIEW_STATE_ORDER = [WATCH_STAGE, ...ENTRY_STATUS_ORDER];
  const CHANGE_ORDER = ["BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES"];
  const ORIGIN_ORDER = ["NEW", "CARRY", "RECONFIRMED"];
  const ROUTE_LABELS = {
    All: "All",
    ceiling: "Ceiling",
    ceiling_pullback: "Ceiling Pullback",
    ma10_pullback: "MA10 Pullback",
    ma10_touch_confirm: "MA10 Touch",
    pivot: "Pivot",
    three_weeks_tight: "Three Weeks Tight",
  };

  let data = null;
  let state = null;

  function esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function bool(value) {
    if (value === true || value === 1) return true;
    const normalized = String(value ?? "").trim().toLowerCase();
    return ["true", "1", "1.0", "yes", "y", "t"].includes(normalized);
  }

  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function fmt(value, kind = "number") {
    const parsed = num(value);
    if (parsed === null) return "n/a";
    if (kind === "pct") return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(2)}%`;
    if (kind === "pct1") return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(1)}%`;
    if (kind === "x") return `${parsed.toFixed(2)}×`;
    if (kind === "x1") return `${parsed.toFixed(1)}×`;
    if (kind === "int") return Number.isInteger(parsed) ? String(parsed) : parsed.toFixed(1);
    return parsed.toFixed(2);
  }

  function text(value, fallback = "n/a") {
    if (value === null || value === undefined) return fallback;
    const out = String(value).trim();
    return out && !["nan", "none", "<na>"].includes(out.toLowerCase()) ? out : fallback;
  }

  function dateText(value) {
    const out = text(value, "");
    return /^\d{4}-\d{2}-\d{2}/.test(out) ? out.slice(0, 10) : out;
  }

  function statusLabel(status) {
    return text(status, "N/A").replaceAll("_", " ");
  }

  function statusColor(status) {
    return data?.ui?.status_meta?.[status]?.color || "#f4f5f7";
  }

  function routeLabel(route) {
    return ROUTE_LABELS[route] || text(route, "N/A").replaceAll("_", " ");
  }

  function qualityClass(value) {
    const quality = text(value, "").toLowerCase();
    if (quality.includes("powerful")) return "quality-powerful";
    if (quality.includes("strong")) return "quality-strong";
    if (quality.includes("constructive")) return "quality-constructive";
    if (quality.includes("marginal")) return "quality-marginal";
    if (quality.includes("weak")) return "quality-weak";
    return "";
  }

  function defaultPeriodContext(period) {
    return {
      scope: period === "MIDWEEK" && data.meta.midweek_baseline_available ? "CHANGES" : "ALL_SIGNALS",
      change: "ALL",
      origin: "ALL",
      status: "ALL",
      route: "All",
      distanceMin: null,
      distanceMax: null,
      entryVolumeMin: null,
      weeklyVolumeMin: null,
      filtersExpanded: false,
    };
  }

  function initialState() {
    const period = data.default_period === "MIDWEEK" && data.meta.midweek_available ? "MIDWEEK" : "WEEKEND";
    const context = defaultPeriodContext(period);
    return {
      period,
      ...context,
      selected: { WEEKEND: null, MIDWEEK: null },
      periodContexts: { [period]: { ...context } },
    };
  }

  function periodContextSnapshot() {
    return {
      scope: state.scope,
      change: state.change,
      origin: state.origin,
      status: state.status,
      route: state.route,
      distanceMin: state.distanceMin,
      distanceMax: state.distanceMax,
      entryVolumeMin: state.entryVolumeMin,
      weeklyVolumeMin: state.weeklyVolumeMin,
      filtersExpanded: state.filtersExpanded,
    };
  }

  function savePeriodContext(period = state.period) {
    state.periodContexts[period] = periodContextSnapshot();
  }

  function restorePeriodContext(period) {
    const context = state.periodContexts[period] || defaultPeriodContext(period);
    state.period = period;
    Object.assign(state, context);
    if (period !== "MIDWEEK" || !data.meta.midweek_baseline_available) {
      state.scope = "ALL_SIGNALS";
      state.change = "ALL";
      state.origin = "ALL";
    }
  }

  function rowsForPeriod(period = state.period) {
    return period === "MIDWEEK" ? data.views.midweek.rows : data.views.weekend.rows;
  }

  function isSignalActive(row) {
    if (Object.hasOwn(row, "review_watch_active")) return bool(row.review_watch_active);
    return bool(row.signal);
  }

  function isNearBreakout(row) {
    return !isSignalActive(row) && bool(row.bf_watch_active);
  }

  function isActive(row) {
    return isSignalActive(row) || isNearBreakout(row);
  }

  function displayStatus(row) {
    return isNearBreakout(row) ? WATCH_STAGE : row.ibd_entry_status;
  }

  function reviewSetup(row) {
    return isNearBreakout(row) ? text(row.bf_watch_type, "pivot") : row.ibd_candidate_rule;
  }

  function nearBreakoutTarget(row) {
    const genericTarget = num(row.bf_watch_trigger_price);
    if (genericTarget !== null) return genericTarget;
    const sResistance = num(row.bf_watch_s_resistance);
    return sResistance !== null ? sResistance : num(row.bf_watch_m_resistance);
  }

  function nearBreakoutDistance(row) {
    const genericDistance = num(row.bf_watch_distance_pct);
    if (genericDistance !== null) return genericDistance;
    const sResistance = num(row.bf_watch_s_resistance);
    return sResistance !== null
      ? num(row.bf_watch_s_distance_pct)
      : num(row.bf_watch_m_distance_pct);
  }

  function watchTargetSource(row) {
    const watchType = reviewSetup(row);
    if (watchType === "ceiling_pullback") return "Recovery High";
    if (watchType === "three_weeks_tight") return "TWK High";
    if (watchType === "ma10_pullback") return "Pending High";
    if (num(row.bf_watch_trigger_price) !== null) return "Pivot Resistance";
    return num(row.bf_watch_s_resistance) !== null ? "S Resistance" : "M Resistance";
  }

  function reviewDistance(row) {
    return isNearBreakout(row) ? nearBreakoutDistance(row) : num(row.current_vs_ibd_candidate_pct);
  }

  function reviewReferencePrice(row) {
    return isNearBreakout(row) ? nearBreakoutTarget(row) : num(row.ibd_candidate_price);
  }

  function displayChange(row) {
    return isNearBreakout(row) ? "" : text(row.review_change_label, "");
  }

  function currentHasComparison() {
    return state.period === "MIDWEEK" && data.meta.midweek_baseline_available;
  }

  function expectedMarketDate() {
    return state.period === "MIDWEEK"
      ? data.meta.midweek_snapshot_date || null
      : data.meta.complete_snapshot_date || null;
  }

  function rsTitle(row) {
    const meta = data?.meta?.rs_reference || {};
    const source = meta.source || "Fred6725/rs-log";
    const poolDate = expectedMarketDate();
    const sourceDate = meta.market_date || null;
    const current = num(row?.rs_percentile);

    if (current === null) {
      if (!meta.available) {
        return `RS N/A\nSource: ${source}\nPublic reference unavailable.`;
      }
      if (!poolDate || sourceDate !== poolDate) {
        return `RS N/A\nSource: ${source}\nRS market date: ${sourceDate || "N/A"}\nPool market date: ${poolDate || "N/A"}\nExact trading-date match required.`;
      }
      return `RS N/A\nSource: ${source}\nTicker is not present in the current public RS dataset.`;
    }

    return [
      `RS Percentile: ${current}`,
      `1M ago: ${num(row.rs_1m_percentile) ?? "N/A"}`,
      `3M ago: ${num(row.rs_3m_percentile) ?? "N/A"}`,
      `6M ago: ${num(row.rs_6m_percentile) ?? "N/A"}`,
      `Market date: ${sourceDate || "N/A"}`,
      `Source: ${source}`,
      "IBD-style reference; not official IBD RS and not a ranking gate.",
    ].join("\n");
  }

  function filterRows(rows, exclude = "") {
    let result = rows.filter(isActive);
    const comparison = currentHasComparison();

    if (comparison && state.scope === "CHANGES") {
      result = result.filter((row) => (
        text(row.review_change_group, "UNCHANGED") !== "UNCHANGED" || isNearBreakout(row)
      ));
    }
    if (comparison && exclude !== "change" && state.change !== "ALL") {
      result = result.filter((row) => row.review_change_group === state.change);
    }
    if (comparison && exclude !== "origin" && state.origin !== "ALL") {
      result = result.filter((row) => row.review_signal_origin === state.origin);
    }
    if (exclude !== "status" && state.status !== "ALL") {
      result = result.filter((row) => displayStatus(row) === state.status);
    }
    if (exclude !== "advanced") {
      if (state.route !== "All") {
        result = result.filter((row) => reviewSetup(row) === state.route);
      }
      if (state.distanceMin !== null) {
        result = result.filter((row) => {
          const value = reviewDistance(row);
          return value !== null && value >= state.distanceMin;
        });
      }
      if (state.distanceMax !== null) {
        result = result.filter((row) => {
          const value = reviewDistance(row);
          return value !== null && value <= state.distanceMax;
        });
      }
      if (state.entryVolumeMin !== null) {
        result = result.filter((row) => {
          if (isNearBreakout(row)) return true;
          const value = num(row.ibd_entry_volume_ratio);
          return value !== null && value >= state.entryVolumeMin;
        });
      }
      if (state.weeklyVolumeMin !== null) {
        result = result.filter((row) => {
          const value = num(row.volume_ratio);
          return value !== null && value >= state.weeklyVolumeMin;
        });
      }
    }
    return result;
  }

  function filterCounts(rows) {
    const statusBase = filterRows(rows, "status");
    const changeBase = filterRows(rows, "change");
    const originBase = filterRows(rows, "origin");
    return {
      status: Object.fromEntries(REVIEW_STATE_ORDER.map((key) => [
        key,
        statusBase.filter((row) => displayStatus(row) === key).length,
      ])),
      change: Object.fromEntries(CHANGE_ORDER.map((key) => [
        key,
        changeBase.filter((row) => row.review_change_group === key).length,
      ])),
      origin: Object.fromEntries(ORIGIN_ORDER.map((key) => [
        key,
        originBase.filter((row) => row.review_signal_origin === key).length,
      ])),
    };
  }

  function sortRows(rows) {
    const result = [...rows];
    if (currentHasComparison() && state.scope === "CHANGES") {
      result.sort((a, b) => {
        const ap = isNearBreakout(a) ? 5 : num(a.review_priority) ?? 9999;
        const bp = isNearBreakout(b) ? 5 : num(b.review_priority) ?? 9999;
        if (ap !== bp) return ap - bp;
        const as = REVIEW_STATE_ORDER.indexOf(displayStatus(a));
        const bs = REVIEW_STATE_ORDER.indexOf(displayStatus(b));
        if (as !== bs) return as - bs;
        return String(a.code).localeCompare(String(b.code));
      });
      return { rows: result, label: "Review Priority" };
    }
    result.sort((a, b) => String(a.code).localeCompare(String(b.code), undefined, {
      numeric: true,
      sensitivity: "base",
    }));
    return { rows: result, label: "Code" };
  }

  function advancedCount() {
    return [
      state.route !== "All",
      state.distanceMin !== null || state.distanceMax !== null,
      state.entryVolumeMin !== null,
      state.weeklyVolumeMin !== null,
    ].filter(Boolean).length;
  }

  function quickCount() {
    return [state.change !== "ALL", state.origin !== "ALL"].filter(Boolean).length;
  }

  function resetAdvanced() {
    state.route = "All";
    state.distanceMin = null;
    state.distanceMax = null;
    state.entryVolumeMin = null;
    state.weeklyVolumeMin = null;
  }

  function resetPeriodState(period) {
    if (period === state.period) return;
    savePeriodContext();
    restorePeriodContext(period);
  }

  function freshness(snapshot) {
    if (!snapshot) return { status: "UNKNOWN", label: "Unknown", age: null };
    const snap = new Date(`${snapshot}T12:00:00Z`);
    if (Number.isNaN(snap.getTime())) return { status: "UNKNOWN", label: "Unknown", age: null };
    const now = new Date();
    const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
    const day = Date.UTC(snap.getUTCFullYear(), snap.getUTCMonth(), snap.getUTCDate());
    const age = Math.max(0, Math.floor((today - day) / 86400000));
    if (age <= 3) return { status: "FRESH", label: "Fresh", age };
    if (age <= 6) return { status: "AGING", label: "Aging", age };
    return { status: "STALE", label: "Stale", age };
  }

  function headerHtml(rows) {
    const isMidweek = state.period === "MIDWEEK";
    const snapshot = isMidweek ? data.meta.midweek_snapshot_date : data.meta.complete_snapshot_date;
    const signalCount = rows.filter(isSignalActive).length;
    const nearCount = rows.filter(isNearBreakout).length;
    const fresh = freshness(snapshot);
    const badge = isMidweek
      ? `<span class="data-badge loaded">Data Loaded</span>`
      : `<span class="data-badge ${fresh.status.toLowerCase()}">Data ${esc(fresh.label)}</span>`;
    const snapshotText = isMidweek
      ? `Snapshot <b>${esc(snapshot || "N/A")}</b> · Midweek · baseline <b>${esc(data.meta.midweek_baseline_available ? data.meta.complete_snapshot_date : "unavailable")}</b>`
      : `Snapshot <b>${esc(snapshot || "N/A")}</b>${fresh.age === null ? "" : ` · ${fresh.age}d old`}`;
    return `
      <header class="dashboard-header">
        <div>
          <div class="dashboard-title-row"><div class="dashboard-title">Breakout Pool</div>${badge}</div>
          <div class="dashboard-snapshot">${snapshotText} · <b>${rows.length}</b> Total Pool · <b>${signalCount}</b> Active Signals · <b>${nearCount}</b> Near Breakout</div>
        </div>
      </header>`;
  }

  function queueHtml(rows, counts) {
    const comparison = currentHasComparison();
    const activeTotal = rows.filter(isActive).length;
    const changeTotal = comparison
      ? rows.filter(isActive).filter((row) => row.review_change_group !== "UNCHANGED" || isNearBreakout(row)).length
      : 0;
    const midweekDisabled = !data.meta.midweek_available;
    const scope = comparison
      ? `<div><div class="control-group-label">Scope</div><div class="segmented">
           <button data-action="scope" data-value="CHANGES" aria-pressed="${state.scope === "CHANGES"}" title="Changed signals plus current Near Breakout candidates">Review Now · ${changeTotal}</button>
           <button data-action="scope" data-value="ALL_SIGNALS" aria-pressed="${state.scope === "ALL_SIGNALS"}">All Review · ${activeTotal}</button>
         </div></div>`
      : `<div><div class="control-group-label">Scope</div><div class="scope-static">All Review · ${activeTotal}</div></div>`;

    return `
      <section class="review-section">
        <div class="queue-heading">
          <div><h2>Review Queue</h2></div>
          <div><div class="control-group-label">Period</div><div class="segmented">
            <button data-action="period" data-value="MIDWEEK" aria-pressed="${state.period === "MIDWEEK"}" ${midweekDisabled ? 'disabled title="Midweek snapshot unavailable"' : ""}>Midweek Review</button>
            <button data-action="period" data-value="WEEKEND" aria-pressed="${state.period === "WEEKEND"}">Weekend Pool</button>
          </div></div>
          ${scope}
        </div>
        ${contextHtml(counts)}
        ${statusCardsHtml(counts.status)}
      </section>`;
  }

  function contextHtml(counts) {
    if (state.period !== "MIDWEEK") {
      if (!data.meta.midweek_available) return "";
      return `<div class="context-panel"><div class="context-note"><strong>Weekend Baseline</strong><span>Complete weekly pool</span><span>Midweek comparison is not applied in this view.</span></div></div>`;
    }
    if (!data.meta.midweek_baseline_available) {
      return `<div class="context-panel"><div class="context-note"><strong>Midweek Snapshot</strong><span>No valid complete-week baseline</span><span>Change and Origin comparison is unavailable.</span></div></div>`;
    }
    const changeButtons = CHANGE_ORDER.map((key) => quickButton(key, counts.change[key], "change")).join("");
    const originButtons = ORIGIN_ORDER.map((key) => quickButton(key, counts.origin[key], "origin")).join("");
    return `
      <div class="context-panel">
        <div class="quick-groups">
          <div><div class="eyebrow">What Changed</div><div class="quick-grid">${changeButtons}</div></div>
          <div><div class="eyebrow">Signal Source</div><div class="quick-grid">${originButtons}</div></div>
          <div>${quickCount() ? `<button class="small-button" data-action="clear-quick">Clear</button>` : ""}</div>
        </div>
      </div>`;
  }

  function quickButton(key, count, field) {
    const meta = data.ui.flow_meta[key] || {};
    const selected = state[field] === key;
    return `<button class="quick-button" style="--quick-color:${esc(meta.color || "#1fcdb4")}" data-action="quick" data-field="${field}" data-value="${key}" aria-pressed="${selected}" title="${esc(meta.tooltip || "")}">
      <span class="symbol">${esc(meta.symbol || "•")}</span><span>${esc(meta.label || key)}</span><span class="count">${count ?? 0}</span>
    </button>`;
  }

  function statusCardHtml(key, count) {
    const meta = data.ui.status_meta[key] || {};
    return `<button class="status-card" style="--tone:${esc(meta.color || "#9ca8b7")}" data-action="status" data-value="${key}" aria-pressed="${state.status === key}" title="${esc(meta.tooltip || "")}">
      <span class="status-orb"></span><span><span class="status-label">${esc(meta.label || key)}</span><span class="status-subtitle">${esc(meta.subtitle || "")}</span></span><span class="status-count">${count ?? 0}</span>
    </button>`;
  }

  function statusCardsHtml(counts) {
    const watch = statusCardHtml(WATCH_STAGE, counts[WATCH_STAGE]);
    const entries = ENTRY_STATUS_ORDER.map((key) => statusCardHtml(key, counts[key])).join("");
    return `<div class="review-stage-status">
      <div class="review-flow-group">
        <div class="review-flow-label">Watch Stage <span>pre-signal · current candidates</span></div>
        <div class="status-grid review-watch-grid">${watch}</div>
      </div>
      <div class="review-flow-group">
        <div class="review-flow-label">Entry Status <span>active signals</span></div>
        <div class="status-grid review-entry-grid">${entries}</div>
      </div>
    </div>`;
  }

  function bounds(rows, field, floor, ceiling) {
    const values = rows.map((row) => (
      field === "review_distance_pct" ? reviewDistance(row) : num(row[field])
    )).filter((value) => value !== null);
    if (!values.length) return null;
    const low = Math.min(floor, Math.floor(Math.min(...values) * 10) / 10);
    const high = Math.max(ceiling, Math.ceil(Math.max(...values) * 10) / 10);
    return [Number(low.toFixed(1)), Number(high.toFixed(1))];
  }

  function filtersHtml(rows) {
    const active = advancedCount();
    if (!state.filtersExpanded) {
      return `<section class="filters-wrap"><div class="filters-head"><button class="filter-toggle" data-action="toggle-filters">More Filters · ${active ? `${active} active` : "None"}</button>${active ? `<button class="reset-button" data-action="reset-filters">Reset</button>` : ""}</div></section>`;
    }
    const distance = bounds(rows, "review_distance_pct", -5, 5);
    const entry = bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio", 0, 1);
    const weekly = bounds(rows, "volume_ratio", 0, 1);
    const dMin = state.distanceMin ?? distance?.[0] ?? -5;
    const dMax = state.distanceMax ?? distance?.[1] ?? 5;
    const entryValue = state.entryVolumeMin ?? entry?.[0] ?? 0;
    const weeklyValue = state.weeklyVolumeMin ?? weekly?.[0] ?? 0;
    return `
      <section class="filters-wrap">
        <div class="filters-head"><button class="filter-toggle expanded" data-action="toggle-filters">More Filters · ${active ? `${active} active` : "None"}</button>${active ? `<button class="reset-button" data-action="reset-filters">Reset</button>` : ""}</div>
        <div class="filter-controls">
          <div class="filter-field"><div class="eyebrow">Setup</div><label>Setup type</label><select data-control="route">${data.ui.setup_options.map((value) => `<option value="${esc(value)}" ${state.route === value ? "selected" : ""}>${esc(routeLabel(value))}</option>`).join("")}</select></div>
          ${distance ? `<div class="filter-field"><div class="eyebrow">Price Position</div><label>Vs Reference · Min</label><input data-control="distance-min" data-dynamic-bounds="true" type="range" min="${distance[0]}" max="${distance[1]}" step="0.1" value="${dMin}"><div class="range-values"><small>${fmt(dMin, "pct1")}</small><small>${state.distanceMin === null ? "Any" : "Active"}</small></div><small class="filter-helper">Watch → trigger · Signal → buy point</small></div>
          <div class="filter-field"><div class="eyebrow">Price Position</div><label>Vs Reference · Max</label><input data-control="distance-max" data-dynamic-bounds="true" type="range" min="${distance[0]}" max="${distance[1]}" step="0.1" value="${dMax}"><div class="range-values"><small>${fmt(dMax, "pct1")}</small><small>${state.distanceMax === null ? "Any" : "Active"}</small></div></div>` : ""}
          ${entry ? `<div class="filter-field"><div class="eyebrow">Volume</div><label>${state.entryVolumeMin === null ? "Entry Volume ≥ Any" : `Entry Volume ≥ ${fmt(entryValue, "x1")}`}</label><input data-control="entry-volume" data-dynamic-bounds="true" type="range" min="${entry[0]}" max="${entry[1]}" step="0.1" value="${entryValue}"><div class="range-values"><small>${fmt(entryValue, "x1")}</small><small>${state.entryVolumeMin === null ? "Any" : "Active"}</small></div><small class="filter-helper">Signal stage only · Watch candidates stay visible as N/A</small></div>` : ""}
          ${weekly ? `<div class="filter-field"><div class="eyebrow">Volume</div><label>${state.weeklyVolumeMin === null ? "Weekly Volume ≥ Any" : `Weekly Volume ≥ ${fmt(weeklyValue, "x1")}`}</label><input data-control="weekly-volume" data-dynamic-bounds="true" type="range" min="${weekly[0]}" max="${weekly[1]}" step="0.1" value="${weeklyValue}"><div class="range-values"><small>${fmt(weeklyValue, "x1")}</small><small>${state.weeklyVolumeMin === null ? "Any" : "Active"}</small></div></div>` : ""}
        </div>
      </section>`;
  }

  function resultsHtml(rows, sortedLabel) {
    const selectedCode = state.selected[state.period];
    const selectedRow = rows.find((row) => String(row.code) === String(selectedCode)) || null;
    return `
      <section>
        <div class="results-toolbar"><div class="results-summary">${rows.length} results · Sorted by ${esc(sortedLabel)}</div><button class="copy-button" data-action="copy-codes">Copy ${rows.length} Codes</button><div></div></div>
        ${selectedHtml(selectedRow)}
        ${tableHtml(rows)}
      </section>`;
  }

  function selectedHtml(row) {
    if (!row) return `<div class="selected-strip empty"><span>${filterRows(rowsForPeriod()).length ? "Select a row · Use ↑↓ to review" : "No matching records found with current filter criteria."}</span></div>`;
    const near = isNearBreakout(row);
    const referenceKey = near ? "Watch Trigger" : "Buy Point";
    const entryDate = dateText(row.ibd_entry_date);
    const buyPointDate = dateText(row.buy_point_date);
    const referenceContext = near ? watchTargetSource(row) : entryDate || buyPointDate;
    const contextLabel = near ? "Source" : entryDate ? "Entry" : "Buy Point Date";
    const referenceNote = `${referenceKey} ${fmt(reviewReferencePrice(row))}${referenceContext ? ` · ${contextLabel} ${esc(referenceContext)}` : ""}`;
    const baseDepth = num(row.base_depth_pct);
    const baseDuration = num(row.base_duration_weeks);
    const baseValue = baseDepth === null && baseDuration === null
      ? "—"
      : `${baseDepth === null ? "—" : fmt(baseDepth, "pct1")} · ${baseDuration === null ? "—" : `${fmt(baseDuration, "int")}w`}`;
    const pullbackDepth = num(row.pullback_pct);
    const pullbackDuration = num(row.pullback_duration_weeks);
    const pullbackDry = row.pullback_v_is_dry === null || row.pullback_v_is_dry === undefined
      ? null : bool(row.pullback_v_is_dry);
    const pullbackVisible = pullbackDepth !== null || pullbackDuration !== null || pullbackDry === true;
    const pullbackValue = `${pullbackDepth === null ? "—" : fmt(pullbackDepth, "pct1")} · ${pullbackDuration === null ? "—" : `${fmt(pullbackDuration, "int")}w`}`;
    const currentRs = num(row.rs_percentile);
    const rsValue = currentRs === null
      ? "N/A"
      : `${currentRs} <small>1M ${num(row.rs_1m_percentile) ?? "N/A"} · 3M ${num(row.rs_3m_percentile) ?? "N/A"} · 6M ${num(row.rs_6m_percentile) ?? "N/A"}</small>`;
    return `<div class="selected-strip ${pullbackVisible ? "has-pullback" : ""}" aria-label="Selected overview for ${esc(row.code)}">
      <div class="selected-cell selected-identity"><div class="selected-key">Selected Overview</div><div class="selected-value selected-code">${esc(row.code)}</div><div class="selected-industry" title="${esc(text(row.industry))}">${esc(text(row.industry, "Industry N/A"))}</div><div class="selected-reference">${referenceNote}</div></div>
      <div class="selected-cell"><div class="selected-key">EPS YoY</div><div class="selected-value">${fmt(row.eps_yoy_growth, "pct1")}</div></div>
      <div class="selected-cell"><div class="selected-key">Base</div><div class="selected-value">${baseValue}</div><div class="selected-hint">Depth · Duration</div></div>
      <div class="selected-cell"><div class="selected-key">To 52W High</div><div class="selected-value">${fmt(row.dist_to_52w_high_pct, "pct1")}</div></div>
      ${pullbackVisible ? `<div class="selected-cell selected-pullback"><div class="selected-key">Pullback</div><div class="selected-value">${pullbackValue}</div><div class="selected-hint">Depth · Duration${pullbackDry === null ? "" : ` · Dry ${pullbackDry ? "Yes" : "No"}`}</div></div>` : ""}
      <div class="selected-cell selected-rs"><div class="selected-key">RS Reference</div><div class="selected-value" title="${esc(rsTitle(row))}">${rsValue}</div></div>
    </div>`;
  }

  function renderSelection(currentRows, { focusTable = false, scrollCode = null } = {}) {
    const selectedCode = state.selected[state.period];
    const selectedRow = currentRows.find((row) => String(row.code) === String(selectedCode)) || null;
    const strip = app.querySelector(".selected-strip");
    if (strip) {
      const template = document.createElement("template");
      template.innerHTML = selectedHtml(selectedRow).trim();
      const replacement = template.content.firstElementChild;
      if (replacement) strip.replaceWith(replacement);
    }

    app.querySelectorAll("tbody tr[data-code]").forEach((row) => {
      row.classList.toggle("selected", String(row.dataset.code) === String(selectedCode));
    });

    const shell = app.querySelector("[data-table-shell]");
    if (shell && scrollCode) {
      const scrollLeft = shell.scrollLeft;
      const target = shell.querySelector(`tr[data-code="${CSS.escape(String(scrollCode))}"]`);
      target?.scrollIntoView({ block: "nearest", inline: "nearest" });
      shell.scrollLeft = scrollLeft;
    }
    if (focusTable) shell?.focus({ preventScroll: true });
  }

  function tableHtml(rows) {
    if (!rows.length) return `<div class="table-shell"><div class="no-results">No matching records.</div></div>`;
    const comparison = currentHasComparison();
    const columns = [
      ["code", "Code"],
      ...(comparison ? [["review_change_label", "Change"]] : []),
      ["ibd_entry_status", "Stage / Status"],
      ["ibd_candidate_rule", "Setup"],
      ["current_vs_ibd_candidate_pct", "Vs Reference"],
      ["ibd_breakout_quality", "Breakout Price Quality"],
      ["latest_close", "Latest"],
      ["ibd_entry_vol_or_reject", "Entry / Reason"],
      ["volume_ratio", "Weekly Vol"],
      ["rs_percentile", "RS"],
    ];
    const selected = state.selected[state.period];
    return `<div class="table-shell" tabindex="0" data-table-shell><table class="review-table"><thead><tr>${columns.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr data-code="${esc(row.code)}" class="${String(row.code) === String(selected) ? "selected" : ""}">${columns.map(([field]) => `<td class="${field === "code" ? "code-cell" : ""}">${cellHtml(row, field)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function cellHtml(row, field) {
    const value = row[field];
    if (field === "review_change_label") {
      return isNearBreakout(row)
        ? `<span class="change-badge" title="Current watch-stage candidate; signal transition labels do not apply yet.">—</span>`
        : `<span class="change-badge">${esc(displayChange(row) || "n/a")}</span>`;
    }
    if (field === "ibd_entry_status") {
      const status = displayStatus(row);
      return `<span class="status-text" style="color:${statusColor(status)}">${esc(statusLabel(status))}</span>`;
    }
    if (field === "ibd_candidate_rule") return esc(routeLabel(reviewSetup(row)));
    if (field === "current_vs_ibd_candidate_pct") return esc(fmt(reviewDistance(row), "pct"));
    if (field === "ibd_breakout_quality") return `<span class="quality-text ${qualityClass(value)}">${esc(text(value))}</span>`;
    if (field === "latest_close") return esc(fmt(value));
    if (field === "volume_ratio") return esc(fmt(value, "x"));
    if (field === "rs_percentile") {
      const current = num(value);
      return `<span title="${esc(rsTitle(row))}">${current === null ? "N/A" : esc(String(current))}</span>`;
    }
    if (field === "ibd_entry_vol_or_reject") return isNearBreakout(row) ? "Pre-signal" : esc(text(value).replace(/x$/, "×"));
    return esc(text(value));
  }

  function warningsHtml() {
    const warnings = data.meta.warnings || [];
    if (!warnings.length || state.period !== "MIDWEEK") return "";
    return `<div class="warning-stack">${warnings.map((warning) => `<div class="warning">⚠ ${esc(warning)}</div>`).join("")}</div>`;
  }

  function footerHtml() {
    return `<div class="footer-note">Static snapshot · source: Yfinance_data authoritative BreakoutFollow pool · RS reference: Fred6725/rs-log</div>`;
  }

  function render() {
    if (!data || !state) return;

    const sourceRows = rowsForPeriod();
    const counts = filterCounts(sourceRows);
    const filtered = filterRows(sourceRows);
    const sorted = sortRows(filtered);
    const selectedCode = state.selected[state.period];
    if (selectedCode && !sorted.rows.some((row) => String(row.code) === String(selectedCode))) {
      state.selected[state.period] = null;
    }
    app.innerHTML = `${headerHtml(sourceRows)}${warningsHtml()}${queueHtml(sourceRows, counts)}${filtersHtml(sourceRows)}${resultsHtml(sorted.rows, sorted.label)}${footerHtml()}`;
    bindEvents(sorted.rows);
  }

  function bindEvents(currentRows = []) {
    app.querySelectorAll("[data-action]").forEach((element) => {
      element.addEventListener("click", async () => {
        const action = element.dataset.action;
        if (action === "period") {
          if (!element.disabled) resetPeriodState(element.dataset.value);
          render();
        } else if (action === "scope") {
          state.scope = element.dataset.value;
          render();
        } else if (action === "quick") {
          const field = element.dataset.field;
          const value = element.dataset.value;
          state[field] = state[field] === value ? "ALL" : value;
          render();
        } else if (action === "clear-quick") {
          state.change = "ALL";
          state.origin = "ALL";
          render();
        } else if (action === "status") {
          const value = element.dataset.value;
          state.status = state.status === value ? "ALL" : value;
          render();
        } else if (action === "toggle-filters") {
          state.filtersExpanded = !state.filtersExpanded;
          render();
        } else if (action === "reset-filters") {
          resetAdvanced();
          render();
        } else if (action === "copy-codes") {
          const visible = [...app.querySelectorAll("[data-table-shell] tbody tr[data-code]")]
            .map((row) => row.dataset.code)
            .filter(Boolean);
          await copyCodes(visible.length ? visible : currentRows.map((row) => row.code), element);
        }
      });
    });

    const route = app.querySelector('[data-control="route"]');
    if (route) route.addEventListener("change", () => { state.route = route.value; render(); });

    bindRange("distance-min", (value, element) => {
      const lower = Number(element.min);
      state.distanceMin = Math.abs(value - lower) < 1e-9 ? null : value;
      if (state.distanceMax !== null && state.distanceMin !== null && state.distanceMin > state.distanceMax) state.distanceMax = state.distanceMin;
    });
    bindRange("distance-max", (value, element) => {
      const upper = Number(element.max);
      state.distanceMax = Math.abs(value - upper) < 1e-9 ? null : value;
      if (state.distanceMin !== null && state.distanceMax !== null && state.distanceMax < state.distanceMin) state.distanceMin = state.distanceMax;
    });
    bindRange("entry-volume", (value, element) => { state.entryVolumeMin = Math.abs(value - Number(element.min)) < 1e-9 ? null : value; });
    bindRange("weekly-volume", (value, element) => { state.weeklyVolumeMin = Math.abs(value - Number(element.min)) < 1e-9 ? null : value; });

    app.querySelectorAll("tbody tr[data-code]").forEach((row) => {
      row.addEventListener("click", () => {
        state.selected[state.period] = row.dataset.code;
        renderSelection(currentRows, { focusTable: true });
      });
    });

    const reviewShell = app.querySelector("[data-table-shell]");
    if (reviewShell) reviewShell.addEventListener("keydown", (event) => handleArrow(event, currentRows, state.period));
  }

  function bindRange(name, update) {
    const element = app.querySelector(`[data-control="${name}"]`);
    if (!element) return;
    element.addEventListener("change", () => {
      update(Number(element.value), element);
      render();
    });
  }

  function handleArrow(event, rows, key) {
    if (!["ArrowDown", "ArrowUp"].includes(event.key) || !rows.length) return;
    event.preventDefault();
    const current = state.selected[key];
    let index = rows.findIndex((row) => String(row.code) === String(current));
    if (index < 0) index = event.key === "ArrowDown" ? -1 : rows.length;
    index += event.key === "ArrowDown" ? 1 : -1;
    index = Math.max(0, Math.min(rows.length - 1, index));
    state.selected[key] = String(rows[index].code);
    renderSelection(rows, { focusTable: true, scrollCode: rows[index].code });
  }

  async function copyCodes(codes, button) {
    const clean = codes.map((code) => String(code).trim()).filter(Boolean);
    const payload = clean.join(", ");
    let success = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(payload);
        success = true;
      }
    } catch (_) {
      success = false;
    }
    if (!success) {
      const area = document.createElement("textarea");
      area.value = payload;
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.select();
      try { success = document.execCommand("copy"); } catch (_) { success = false; }
      area.remove();
    }
    const original = button.textContent;
    button.classList.add(success ? "success" : "fail");
    button.textContent = success ? `✓ Copied ${clean.length}` : "Copy failed";
    setTimeout(() => {
      button.classList.remove("success", "fail");
      button.textContent = original;
    }, 1600);
  }

  async function boot() {
    try {
      const response = await fetch("./data/dashboard.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = await response.json();
      if (!data?.views?.weekend?.rows) throw new Error("Dashboard payload is incomplete");
      state = initialState();
      render();
    } catch (error) {
      app.innerHTML = `<section class="error-card"><div class="boot-mark"></div><div><strong>Dashboard data unavailable</strong><span>${esc(error?.message || error)}</span></div></section>`;
    }
  }

  boot();
})();
