(() => {
  "use strict";

  const app = document.getElementById("app");
  const STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"];
  const CHANGE_ORDER = ["BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES"];
  const ORIGIN_ORDER = ["NEW", "CARRY", "RECONFIRMED"];
  const ROUTE_LABELS = {
    All: "All",
    ceiling: "Ceiling",
    ceiling_pullback: "Ceiling Pullback",
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
    const text = String(value ?? "").trim().toLowerCase();
    return ["true", "1", "1.0", "yes", "y", "t"].includes(text);
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

  function initialState() {
    const period = data.default_period === "MIDWEEK" && data.meta.midweek_available ? "MIDWEEK" : "WEEKEND";
    return {
      globalMode: "IBD",
      period,
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
      selected: { WEEKEND: null, MIDWEEK: null, C_RANK: null },
      detailOpen: false,
      cRankTopN: "ALL",
    };
  }

  function rowsForPeriod(period = state.period) {
    return period === "MIDWEEK" ? data.views.midweek.rows : data.views.weekend.rows;
  }

  function isActive(row) {
    if (Object.hasOwn(row, "review_watch_active")) return bool(row.review_watch_active);
    return bool(row.signal);
  }

  function currentHasComparison() {
    return state.period === "MIDWEEK" && data.meta.midweek_baseline_available;
  }

  function filterRows(rows, exclude = "") {
    let result = rows.filter(isActive);
    const comparison = currentHasComparison();

    if (comparison && state.scope === "CHANGES") {
      result = result.filter((row) => text(row.review_change_group, "UNCHANGED") !== "UNCHANGED");
    }
    if (comparison && exclude !== "change" && state.change !== "ALL") {
      result = result.filter((row) => row.review_change_group === state.change);
    }
    if (comparison && exclude !== "origin" && state.origin !== "ALL") {
      result = result.filter((row) => row.review_signal_origin === state.origin);
    }
    if (exclude !== "status" && state.status !== "ALL") {
      result = result.filter((row) => row.ibd_entry_status === state.status);
    }
    if (exclude !== "advanced") {
      if (state.route !== "All") {
        result = result.filter((row) => row.ibd_candidate_rule === state.route);
      }
      if (state.distanceMin !== null) {
        result = result.filter((row) => {
          const value = num(row.current_vs_ibd_candidate_pct);
          return value !== null && value >= state.distanceMin;
        });
      }
      if (state.distanceMax !== null) {
        result = result.filter((row) => {
          const value = num(row.current_vs_ibd_candidate_pct);
          return value !== null && value <= state.distanceMax;
        });
      }
      if (state.entryVolumeMin !== null) {
        result = result.filter((row) => {
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
      status: Object.fromEntries(STATUS_ORDER.map((key) => [key, statusBase.filter((row) => row.ibd_entry_status === key).length])),
      change: Object.fromEntries(CHANGE_ORDER.map((key) => [key, changeBase.filter((row) => row.review_change_group === key).length])),
      origin: Object.fromEntries(ORIGIN_ORDER.map((key) => [key, originBase.filter((row) => row.review_signal_origin === key).length])),
    };
  }

  function sortRows(rows) {
    const result = [...rows];
    if (currentHasComparison() && state.scope === "CHANGES") {
      result.sort((a, b) => {
        const ap = num(a.review_priority) ?? 9999;
        const bp = num(b.review_priority) ?? 9999;
        if (ap !== bp) return ap - bp;
        const as = STATUS_ORDER.indexOf(a.ibd_entry_status);
        const bs = STATUS_ORDER.indexOf(b.ibd_entry_status);
        if (as !== bs) return as - bs;
        return String(a.code).localeCompare(String(b.code));
      });
      return { rows: result, label: "Review Priority" };
    }
    result.sort((a, b) => {
      const ar = num(a.rank_C_continuous) ?? 999999;
      const br = num(b.rank_C_continuous) ?? 999999;
      return ar !== br ? ar - br : String(a.code).localeCompare(String(b.code));
    });
    return { rows: result, label: "C Rank" };
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
    state.period = period;
    state.scope = period === "MIDWEEK" && data.meta.midweek_baseline_available ? "CHANGES" : "ALL_SIGNALS";
    state.change = "ALL";
    state.origin = "ALL";
    state.status = "ALL";
    state.filtersExpanded = false;
    state.detailOpen = false;
    resetAdvanced();
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
    const isMidweek = state.globalMode === "IBD" && state.period === "MIDWEEK";
    const snapshot = isMidweek ? data.meta.midweek_snapshot_date : data.meta.complete_snapshot_date;
    const active = rows.filter(isActive).length;
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
          <div class="dashboard-snapshot">${snapshotText} · <b>${rows.length}</b> Total Pool · <b>${active}</b> Active Signals</div>
        </div>
        <div class="segmented" aria-label="Dashboard mode">
          <button data-action="global-mode" data-value="IBD" aria-pressed="${state.globalMode === "IBD"}">IBD Review</button>
          <button data-action="global-mode" data-value="C_RANK" aria-pressed="${state.globalMode === "C_RANK"}">C Rank Reference</button>
        </div>
      </header>`;
  }

  function queueHtml(rows, counts) {
    const comparison = currentHasComparison();
    const activeTotal = rows.filter(isActive).length;
    const changeTotal = comparison ? rows.filter(isActive).filter((row) => row.review_change_group !== "UNCHANGED").length : 0;
    const midweekDisabled = !data.meta.midweek_available;
    const scope = comparison
      ? `<div><div class="control-group-label">Scope</div><div class="segmented">
           <button data-action="scope" data-value="CHANGES" aria-pressed="${state.scope === "CHANGES"}">Changes · ${changeTotal}</button>
           <button data-action="scope" data-value="ALL_SIGNALS" aria-pressed="${state.scope === "ALL_SIGNALS"}">All Signals · ${activeTotal}</button>
         </div></div>`
      : `<div><div class="control-group-label">Scope</div><div class="scope-static">All Signals · ${activeTotal}</div></div>`;

    return `
      <section class="review-section">
        <div class="queue-heading">
          <div><h2>Review Queue</h2></div>
          <div><div class="control-group-label">Period</div><div class="segmented">
            <button data-action="period" data-value="MIDWEEK" aria-pressed="${state.period === "MIDWEEK"}" ${midweekDisabled ? "disabled title=\"Midweek snapshot unavailable\"" : ""}>Midweek Review</button>
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

  function statusCardsHtml(counts) {
    return `<div class="status-grid">${STATUS_ORDER.map((key) => {
      const meta = data.ui.status_meta[key] || {};
      return `<button class="status-card" style="--tone:${esc(meta.color || "#9ca8b7")}" data-action="status" data-value="${key}" aria-pressed="${state.status === key}" title="${esc(meta.tooltip || "")}">
        <span class="status-orb"></span><span><span class="status-label">${esc(meta.label || key)}</span><span class="status-subtitle">${esc(meta.subtitle || "")}</span></span><span class="status-count">${counts[key] ?? 0}</span>
      </button>`;
    }).join("")}</div>`;
  }

  function bounds(rows, field, floor, ceiling) {
    const values = rows.map((row) => num(row[field])).filter((value) => value !== null);
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
    const distance = bounds(rows, "current_vs_ibd_candidate_pct", -5, 5);
    const entry = bounds(rows, "ibd_entry_volume_ratio", 0, 1);
    const weekly = bounds(rows, "volume_ratio", 0, 1);
    const dMin = state.distanceMin ?? distance?.[0] ?? -5;
    const dMax = state.distanceMax ?? distance?.[1] ?? 5;
    const entryValue = state.entryVolumeMin ?? entry?.[0] ?? 0;
    const weeklyValue = state.weeklyVolumeMin ?? weekly?.[0] ?? 0;
    return `
      <section class="filters-wrap">
        <div class="filters-head"><button class="filter-toggle expanded" data-action="toggle-filters">More Filters · ${active ? `${active} active` : "None"}</button>${active ? `<button class="reset-button" data-action="reset-filters">Reset</button>` : ""}</div>
        <div class="filter-controls">
          <div class="filter-field"><div class="eyebrow">Setup</div><label>Signal setup</label><select data-control="route">${data.ui.setup_options.map((value) => `<option value="${esc(value)}" ${state.route === value ? "selected" : ""}>${esc(routeLabel(value))}</option>`).join("")}</select></div>
          ${distance ? `<div class="filter-field"><div class="eyebrow">Price Position</div><label>Vs Buy Point · Min</label><input data-control="distance-min" type="range" min="${distance[0]}" max="${distance[1]}" step="0.1" value="${dMin}"><div class="range-values"><small>${fmt(dMin, "pct1")}</small><small>${state.distanceMin === null ? "Any" : "Active"}</small></div></div>
          <div class="filter-field"><div class="eyebrow">Price Position</div><label>Vs Buy Point · Max</label><input data-control="distance-max" type="range" min="${distance[0]}" max="${distance[1]}" step="0.1" value="${dMax}"><div class="range-values"><small>${fmt(dMax, "pct1")}</small><small>${state.distanceMax === null ? "Any" : "Active"}</small></div></div>` : ""}
          ${entry ? `<div class="filter-field"><div class="eyebrow">Volume</div><label>${state.entryVolumeMin === null ? "Entry Volume ≥ Any" : `Entry Volume ≥ ${fmt(entryValue, "x1")}`}</label><input data-control="entry-volume" type="range" min="${entry[0]}" max="${entry[1]}" step="0.1" value="${entryValue}"><div class="range-values"><small>${fmt(entryValue, "x1")}</small><small>${state.entryVolumeMin === null ? "Any" : "Active"}</small></div></div>` : ""}
          ${weekly ? `<div class="filter-field"><div class="eyebrow">Volume</div><label>${state.weeklyVolumeMin === null ? "Weekly Volume ≥ Any" : `Weekly Volume ≥ ${fmt(weeklyValue, "x1")}`}</label><input data-control="weekly-volume" type="range" min="${weekly[0]}" max="${weekly[1]}" step="0.1" value="${weeklyValue}"><div class="range-values"><small>${fmt(weeklyValue, "x1")}</small><small>${state.weeklyVolumeMin === null ? "Any" : "Active"}</small></div></div>` : ""}
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

  function selectedHtml(row, modeKey = state.period) {
    if (!row) return `<div class="selected-strip empty"><span>${filterRows(rowsForPeriod()).length ? "Select a row · Use ↑↓ to review" : "No matching records found with current filter criteria."}</span></div>`;
    const code = esc(row.code);
    const status = row.ibd_entry_status;
    const baseline = text(row.review_baseline_entry_status, "");
    const volReason = text(row.ibd_entry_vol_or_reject, "n/a").replace(/x$/, "×");
    const transition = baseline
      ? `${esc(statusLabel(baseline))} → <span style="color:${statusColor(status)}">${esc(statusLabel(status))}</span>`
      : `<span style="color:${statusColor(status)}">${esc(statusLabel(status))}</span>`;
    const change = text(row.review_change_label, "");
    return `<div class="selected-strip">
      <div class="selected-cell"><div class="selected-key">Selected</div><div class="selected-value selected-code">${code}</div>${change ? `<div class="selected-change">${esc(change)}</div>` : ""}<button class="detail-toggle" data-action="detail">${state.detailOpen ? "Hide details ▴" : "Details ▾"}</button></div>
      <div class="selected-cell"><div class="selected-key">Buy Point</div><div class="selected-value">${fmt(row.ibd_candidate_price)} <small>(${esc(routeLabel(row.ibd_candidate_rule))})</small></div></div>
      <div class="selected-cell"><div class="selected-key">Vs Buy Point</div><div class="selected-value">${fmt(row.current_vs_ibd_candidate_pct, "pct")} <small>(Close: ${fmt(row.latest_close)})</small></div></div>
      <div class="selected-cell"><div class="selected-key">Entry Status</div><div class="selected-value">${transition} <small>(${esc(volReason)})</small></div></div>
      <div class="selected-cell"><div class="selected-key">C Rank & Continuous</div><div class="selected-value">#${fmt(row.rank_C_continuous, "int")} <small>(${fmt(row.C_continuous)})</small></div></div>
      ${state.detailOpen ? detailHtml(row) : ""}
    </div>`;
  }

  function detailHtml(row) {
    const valid = bool(row.ibd_entry_valid);
    const reject = valid ? "" : `<div class="detail-reject">Unconfirmed · ${esc(text(row.ibd_entry_reject_reason, "Volume not confirmed"))}</div>`;
    const pullbackVisible = num(row.pullback_pct) !== null || num(row.pullback_pct_off_peak) !== null;
    return `<div class="detail-panel">
      <div class="detail-section"><div class="detail-title">1. Daily Entry</div><div class="detail-grid">
        ${detailItem("Trigger", fmt(row.ibd_trigger_price))}${detailItem("Entry Date", text(row.ibd_entry_date))}${detailItem("Daily Entry Vol", fmt(row.ibd_entry_volume_ratio, "x"))}${detailItem("Close Position", fmt(row.ibd_entry_close_position))}${detailItem("Range Ratio", fmt(row.ibd_entry_breakout_range_ratio, "x"))}${detailItem("Price Quality", text(row.ibd_breakout_quality))}
      </div>${reject}</div>
      ${pullbackVisible ? `<div class="detail-section"><div class="detail-title">2. Pullback</div><div class="detail-grid">${detailItem("Pullback Depth", fmt(row.pullback_pct, "pct1"))}${detailItem("Off Peak", fmt(row.pullback_pct_off_peak, "pct1"))}${detailItem("Duration", num(row.pullback_duration_weeks) === null ? "n/a" : `${fmt(row.pullback_duration_weeks, "int")}w`)}${detailItem("Volume Dry", row.pullback_v_is_dry === null ? "n/a" : bool(row.pullback_v_is_dry) ? "Yes" : "No")}</div></div>` : `<div class="detail-section"><div class="detail-title">2. Pullback</div><div class="detail-grid">${detailItem("Evidence", "n/a")}</div></div>`}
      <div class="detail-section"><div class="detail-title">3. CANSLIM / Base</div><div class="detail-grid">${detailItem("EPS YoY", fmt(row.eps_yoy_growth, "pct1"))}${detailItem("To 52W High", fmt(row.dist_to_52w_high_pct, "pct1"))}${detailItem("52W High", fmt(row.price_52_week_high))}${detailItem("Base Depth", fmt(row.base_depth_pct, "pct1"))}${detailItem("Base Duration", num(row.base_duration_weeks) === null ? "n/a" : `${fmt(row.base_duration_weeks, "int")}w`)}${detailItem("Industry", text(row.industry))}</div></div>
    </div>`;
  }

  function detailItem(label, value) {
    return `<div class="detail-item"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
  }

  function tableHtml(rows) {
    if (!rows.length) return `<div class="table-shell"><div class="no-results">No matching records.</div></div>`;
    const comparison = currentHasComparison();
    const columns = [
      ["code", "Code"],
      ...(comparison ? [["review_change_label", "Change"]] : []),
      ["ibd_entry_status", "Status"],
      ["ibd_candidate_rule", "Setup"],
      ["current_vs_ibd_candidate_pct", "Vs Buy Point"],
      ["ibd_breakout_quality", "Breakout Price Quality"],
      ["latest_close", "Latest"],
      ["ibd_entry_vol_or_reject", "Entry / Reason"],
      ["volume_ratio", "Weekly Vol"],
      ["rank_C_continuous", "C Rank"],
    ];
    const selected = state.selected[state.period];
    return `<div class="table-shell" tabindex="0" data-table-shell><table class="review-table"><thead><tr>${columns.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr data-code="${esc(row.code)}" class="${String(row.code) === String(selected) ? "selected" : ""}">${columns.map(([field]) => `<td class="${field === "code" ? "code-cell" : ""}">${cellHtml(row, field)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function cellHtml(row, field) {
    const value = row[field];
    if (field === "review_change_label") return `<span class="change-badge">${esc(text(value))}</span>`;
    if (field === "ibd_entry_status") return `<span class="status-text" style="color:${statusColor(value)}">${esc(statusLabel(value))}</span>`;
    if (field === "ibd_candidate_rule") return esc(routeLabel(value));
    if (field === "current_vs_ibd_candidate_pct") return esc(fmt(value, "pct"));
    if (field === "ibd_breakout_quality") return `<span class="quality-text ${qualityClass(value)}">${esc(text(value))}</span>`;
    if (field === "latest_close") return esc(fmt(value));
    if (field === "volume_ratio") return esc(fmt(value, "x"));
    if (field === "rank_C_continuous") return esc(fmt(value, "int"));
    if (field === "ibd_entry_vol_or_reject") return esc(text(value).replace(/x$/, "×"));
    return esc(text(value));
  }

  function warningsHtml() {
    const warnings = data.meta.warnings || [];
    if (!warnings.length || state.globalMode !== "IBD" || state.period !== "MIDWEEK") return "";
    return `<div class="warning-stack">${warnings.map((warning) => `<div class="warning">⚠ ${esc(warning)}</div>`).join("")}</div>`;
  }

  function cRankHtml() {
    const all = data.views.c_rank.rows.filter(isActive).sort((a, b) => (num(a.rank_C_continuous) ?? 999999) - (num(b.rank_C_continuous) ?? 999999));
    const limit = state.cRankTopN === "ALL" ? all.length : Number(state.cRankTopN);
    const rows = all.slice(0, limit);
    const selectedCode = state.selected.C_RANK;
    const selected = rows.find((row) => String(row.code) === String(selectedCode)) || null;
    return `${headerHtml(data.views.weekend.rows)}
      <section>
        <div class="reference-header"><div><h2>C Rank Reference View</h2><p>Active Signals · C Rank · Best First · reference only</p></div><select class="topn-select" data-control="top-n"><option value="ALL" ${state.cRankTopN === "ALL" ? "selected" : ""}>All rows</option><option value="10" ${state.cRankTopN === "10" ? "selected" : ""}>Top 10</option><option value="25" ${state.cRankTopN === "25" ? "selected" : ""}>Top 25</option><option value="50" ${state.cRankTopN === "50" ? "selected" : ""}>Top 50</option></select><button class="copy-button" data-action="copy-c-rank">Copy ${rows.length} Codes</button></div>
        <div class="reference-rule">Fixed mode: evaluates Active Signals only; C Rank best first; Top N slice only. This is a horizontal quality reference and does not replace IBD entry status.</div>
        <div class="results-toolbar"><div class="results-summary">Showing: ${rows.length} of ${all.length} Active Signals · Reference Only</div><div></div><div></div></div>
        ${selectedHtml(selected, "C_RANK")}
        ${cRankTableHtml(rows)}
      </section>${footerHtml()}`;
  }

  function cRankTableHtml(rows) {
    if (!rows.length) return `<div class="table-shell"><div class="no-results">No active signals.</div></div>`;
    const columns = [["code", "Code"], ["rank_C_continuous", "C Rank"], ["C_continuous", "Continuous C"], ["ibd_entry_status", "Status"], ["current_vs_ibd_candidate_pct", "Vs Buy Point"], ["ibd_candidate_rule", "Setup"], ["volume_ratio", "Weekly Vol"], ["latest_close", "Latest"]];
    return `<div class="table-shell" tabindex="0" data-c-rank-table><table class="review-table"><thead><tr>${columns.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr data-code="${esc(row.code)}" class="${String(row.code) === String(state.selected.C_RANK) ? "selected" : ""}">${columns.map(([field]) => `<td class="${field === "code" ? "code-cell" : ""}">${field === "C_continuous" ? esc(fmt(row[field])) : cellHtml(row, field)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function footerHtml() {
    return `<div class="footer-note">Static snapshot · source: Yfinance_data authoritative BreakoutFollow pool</div>`;
  }

  function render() {
    if (!data || !state) return;
    if (state.globalMode === "C_RANK") {
      app.innerHTML = cRankHtml();
      bindEvents();
      return;
    }

    const sourceRows = rowsForPeriod();
    const counts = filterCounts(sourceRows);
    const filtered = filterRows(sourceRows);
    const sorted = sortRows(filtered);
    const selectedCode = state.selected[state.period];
    if (selectedCode && !sorted.rows.some((row) => String(row.code) === String(selectedCode))) {
      state.selected[state.period] = null;
      state.detailOpen = false;
    }
    app.innerHTML = `${headerHtml(sourceRows)}${warningsHtml()}${queueHtml(sourceRows, counts)}${filtersHtml(sourceRows)}${resultsHtml(sorted.rows, sorted.label)}${footerHtml()}`;
    bindEvents(sorted.rows);
  }

  function bindEvents(currentRows = []) {
    app.querySelectorAll("[data-action]").forEach((element) => {
      element.addEventListener("click", async () => {
        const action = element.dataset.action;
        if (action === "global-mode") {
          state.globalMode = element.dataset.value;
          state.detailOpen = false;
          render();
        } else if (action === "period") {
          if (!element.disabled) resetPeriodState(element.dataset.value);
          render();
        } else if (action === "scope") {
          state.scope = element.dataset.value;
          state.change = "ALL";
          state.origin = "ALL";
          state.status = "ALL";
          state.detailOpen = false;
          render();
        } else if (action === "quick") {
          const field = element.dataset.field;
          const value = element.dataset.value;
          state[field] = state[field] === value ? "ALL" : value;
          state.detailOpen = false;
          render();
        } else if (action === "clear-quick") {
          state.change = "ALL";
          state.origin = "ALL";
          render();
        } else if (action === "status") {
          const value = element.dataset.value;
          state.status = state.status === value ? "ALL" : value;
          state.detailOpen = false;
          render();
        } else if (action === "toggle-filters") {
          state.filtersExpanded = !state.filtersExpanded;
          render();
        } else if (action === "reset-filters") {
          resetAdvanced();
          render();
        } else if (action === "detail") {
          state.detailOpen = !state.detailOpen;
          render();
        } else if (action === "copy-codes") {
          await copyCodes(currentRows.map((row) => row.code), element);
        } else if (action === "copy-c-rank") {
          const rows = data.views.c_rank.rows.filter(isActive).sort((a, b) => (num(a.rank_C_continuous) ?? 999999) - (num(b.rank_C_continuous) ?? 999999));
          const limit = state.cRankTopN === "ALL" ? rows.length : Number(state.cRankTopN);
          await copyCodes(rows.slice(0, limit).map((row) => row.code), element);
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

    const topN = app.querySelector('[data-control="top-n"]');
    if (topN) topN.addEventListener("change", () => { state.cRankTopN = topN.value; state.detailOpen = false; render(); });

    app.querySelectorAll("tbody tr[data-code]").forEach((row) => {
      row.addEventListener("click", () => {
        const key = state.globalMode === "C_RANK" ? "C_RANK" : state.period;
        state.selected[key] = row.dataset.code;
        state.detailOpen = false;
        render();
      });
    });

    const reviewShell = app.querySelector("[data-table-shell]");
    if (reviewShell) reviewShell.addEventListener("keydown", (event) => handleArrow(event, currentRows, state.period));
    const cRankShell = app.querySelector("[data-c-rank-table]");
    if (cRankShell) {
      const rows = data.views.c_rank.rows.filter(isActive).sort((a, b) => (num(a.rank_C_continuous) ?? 999999) - (num(b.rank_C_continuous) ?? 999999));
      const limit = state.cRankTopN === "ALL" ? rows.length : Number(state.cRankTopN);
      cRankShell.addEventListener("keydown", (event) => handleArrow(event, rows.slice(0, limit), "C_RANK"));
    }
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
    state.detailOpen = false;
    render();
    requestAnimationFrame(() => {
      const target = app.querySelector(`tr[data-code="${CSS.escape(String(rows[index].code))}"]`);
      target?.scrollIntoView({ block: "nearest", inline: "nearest" });
      const shell = key === "C_RANK" ? app.querySelector("[data-c-rank-table]") : app.querySelector("[data-table-shell]");
      shell?.focus({ preventScroll: true });
    });
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
