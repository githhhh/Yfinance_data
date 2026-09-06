(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const sortState = {
    review: { field: null, direction: "asc" },
    cRank: { field: null, direction: "asc" },
  };

  const STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW TRIGGER", "EXTENDED"];
  const QUALITY_ORDER = ["POWERFUL", "STRONG", "CONSTRUCTIVE", "MARGINAL", "WEAK"];
  let qualityTooltip = null;
  let qualityTooltipPinned = false;
  let qualityTooltipAnchor = null;
  let refreshQueued = false;

  function normalizeText(value) {
    return String(value ?? "").trim();
  }

  function numericValue(value) {
    const cleaned = normalizeText(value)
      .replaceAll(",", "")
      .replaceAll("#", "")
      .replaceAll("+", "")
      .replaceAll("%", "")
      .replaceAll("×", "")
      .replace(/x$/i, "")
      .trim();
    if (!cleaned || cleaned.toLowerCase() === "n/a") return null;
    const parsed = Number(cleaned);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function ordinalValue(field, value) {
    const upper = normalizeText(value).replaceAll("_", " ").toUpperCase();
    if (field === "ibd_entry_status") {
      const index = STATUS_ORDER.indexOf(upper);
      return index < 0 ? STATUS_ORDER.length : index;
    }
    if (field === "ibd_breakout_quality") {
      const index = QUALITY_ORDER.findIndex((item) => upper.includes(item));
      return index < 0 ? QUALITY_ORDER.length : index;
    }
    return null;
  }

  function compareValues(field, left, right, direction) {
    const ordinalLeft = ordinalValue(field, left);
    const ordinalRight = ordinalValue(field, right);
    let result = 0;

    if (ordinalLeft !== null && ordinalRight !== null) {
      result = ordinalLeft - ordinalRight;
    } else {
      const numericLeft = numericValue(left);
      const numericRight = numericValue(right);
      if (numericLeft !== null && numericRight !== null) {
        result = numericLeft - numericRight;
      } else if (numericLeft !== null) {
        result = -1;
      } else if (numericRight !== null) {
        result = 1;
      } else {
        result = normalizeText(left).localeCompare(normalizeText(right), undefined, {
          numeric: true,
          sensitivity: "base",
        });
      }
    }
    return direction === "desc" ? -result : result;
  }

  function tableKind(shell) {
    return shell.hasAttribute("data-c-rank-table") ? "cRank" : "review";
  }

  function columnField(label, kind) {
    const normalized = normalizeText(label).replace(/[▲▼▾]/g, "").trim();
    const common = {
      Code: "code",
      Status: "ibd_entry_status",
      Setup: "ibd_candidate_rule",
      "Vs Buy Point": "current_vs_ibd_candidate_pct",
      Latest: "latest_close",
      "Weekly Vol": "volume_ratio",
      "C Rank": "rank_C_continuous",
    };
    if (kind === "review") {
      return {
        ...common,
        Change: "review_change_label",
        "Breakout Price Quality": "ibd_breakout_quality",
        "Entry / Reason": "ibd_entry_vol_or_reject",
      }[normalized] || null;
    }
    return {
      ...common,
      "Continuous C": "C_continuous",
    }[normalized] || null;
  }

  function sortTable(shell, field, direction) {
    const headers = [...shell.querySelectorAll("thead th")];
    const index = headers.findIndex((header) => header.dataset.sortField === field);
    const body = shell.querySelector("tbody");
    if (index < 0 || !body) return;

    const rows = [...body.querySelectorAll("tr[data-code]")];
    rows.sort((a, b) => {
      const left = a.children[index]?.textContent || "";
      const right = b.children[index]?.textContent || "";
      const compared = compareValues(field, left, right, direction);
      if (compared !== 0) return compared;
      return normalizeText(a.dataset.code).localeCompare(normalizeText(b.dataset.code));
    });
    rows.forEach((row) => body.appendChild(row));
  }

  function updateSortIndicators(shell) {
    const kind = tableKind(shell);
    const active = sortState[kind];
    shell.querySelectorAll("thead th[data-sort-field]").forEach((header) => {
      const icon = header.querySelector(".table-sort-icon");
      const isActive = active.field === header.dataset.sortField;
      header.setAttribute("aria-sort", isActive ? (active.direction === "asc" ? "ascending" : "descending") : "none");
      if (icon) icon.textContent = isActive ? (active.direction === "asc" ? "▲" : "▼") : "";
    });
  }

  function updateSummary(kind) {
    const active = sortState[kind];
    if (!active.field) return;
    const label = document.querySelector(
      `${kind === "cRank" ? "[data-c-rank-table]" : "[data-table-shell]"} thead th[data-sort-field="${active.field}"] .table-header-label`,
    )?.textContent;
    const summary = app.querySelector(".results-summary");
    if (summary && label) {
      const count = summary.textContent.match(/^\d+\s+(results|of)/i)?.[0];
      summary.textContent = `${count ? `${count} · ` : ""}Sorted by ${label} ${active.direction === "asc" ? "↑" : "↓"}`;
    }
  }

  function onHeaderSort(event) {
    const button = event.currentTarget;
    if (event.target.closest("[data-quality-info]")) return;
    const shell = button.closest(".table-shell");
    if (!shell) return;
    const kind = tableKind(shell);
    const field = button.closest("th")?.dataset.sortField;
    if (!field) return;

    const active = sortState[kind];
    if (active.field === field) {
      active.direction = active.direction === "asc" ? "desc" : "asc";
    } else {
      active.field = field;
      active.direction = "asc";
    }
    sortTable(shell, active.field, active.direction);
    updateSortIndicators(shell);
    updateSummary(kind);
  }

  function qualityTooltipHtml() {
    const rows = [
      ["Powerful", "High close + full clearance", "#22c55e", "38px"],
      ["Strong", "One strong, one solid", "rgba(34,197,94,.78)", "32px"],
      ["Constructive", "Mixed but valid", "rgba(74,222,128,.58)", "26px"],
      ["Marginal", "Valid, little edge", "rgba(134,239,172,.38)", "20px"],
      ["Weak", "Low close", "rgba(187,247,208,.22)", "14px"],
    ];
    return `
      <div style="display:flex;justify-content:space-between;gap:18px;align-items:baseline;margin-bottom:10px;">
        <strong style="font-size:12px;color:#fff;">Breakout Price Quality</strong>
        <span style="font-size:10px;color:#94a3b8;">strong → weak</span>
      </div>
      <div style="display:grid;gap:5px;">
        ${rows.map(([label, note, color, width]) => `
          <div style="display:grid;grid-template-columns:44px 86px minmax(0,1fr);gap:9px;align-items:center;">
            <span style="display:block;width:${width};height:11px;margin:auto;background:${color};clip-path:polygon(0 0,100% 0,50% 100%);"></span>
            <b style="font-size:11px;color:#d9fbe4;">${label}</b>
            <span style="font-size:11px;color:#94a3b8;white-space:nowrap;">${note}</span>
          </div>`).join("")}
      </div>
      <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(148,163,184,.16);font-size:10px;color:#94a3b8;line-height:1.45;">
        <div>Price only: Close Position + Trigger Clearance.</div>
        <div>Volume is separate.</div>
      </div>`;
  }

  function positionQualityTooltip() {
    if (!qualityTooltip || !qualityTooltipAnchor) return;
    const anchor = qualityTooltipAnchor.getBoundingClientRect();
    const tip = qualityTooltip.getBoundingClientRect();
    const padding = 8;
    const left = Math.min(Math.max(padding, anchor.left), Math.max(padding, window.innerWidth - tip.width - padding));
    const below = anchor.bottom + 6;
    const top = below + tip.height <= window.innerHeight - padding
      ? below
      : Math.max(padding, anchor.top - tip.height - 6);
    qualityTooltip.style.left = `${Math.round(left)}px`;
    qualityTooltip.style.top = `${Math.round(top)}px`;
  }

  function hideQualityTooltip(force = false) {
    if (qualityTooltipPinned && !force) return;
    qualityTooltip?.remove();
    qualityTooltip = null;
    qualityTooltipAnchor = null;
    if (force) qualityTooltipPinned = false;
  }

  function showQualityTooltip(anchor, pinned = false) {
    hideQualityTooltip(true);
    qualityTooltipPinned = pinned;
    qualityTooltipAnchor = anchor;
    const tooltip = document.createElement("div");
    tooltip.setAttribute("role", "tooltip");
    tooltip.style.cssText = "position:fixed;width:min(338px,calc(100vw - 16px));padding:12px 13px;border-radius:7px;background:#0b1329;color:#e2e8f0;box-shadow:0 8px 24px rgba(0,0,0,.45);border:1px solid rgba(148,163,184,.22);z-index:999999;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;pointer-events:none;";
    tooltip.innerHTML = qualityTooltipHtml();
    document.body.appendChild(tooltip);
    qualityTooltip = tooltip;
    positionQualityTooltip();
  }

  function decorateTable(shell) {
    const kind = tableKind(shell);
    const headers = [...shell.querySelectorAll("thead th")];
    headers.forEach((header) => {
      if (header.dataset.sortEnhanced === "true") return;
      const originalLabel = normalizeText(header.textContent);
      const field = columnField(originalLabel, kind);
      if (!field) return;

      header.dataset.sortEnhanced = "true";
      header.dataset.sortField = field;
      header.setAttribute("aria-sort", "none");
      header.textContent = "";

      const button = document.createElement("button");
      button.type = "button";
      button.style.cssText = "width:100%;height:38px;display:flex;align-items:center;gap:5px;padding:0;border:0;background:transparent;color:inherit;font:inherit;font-weight:inherit;letter-spacing:inherit;text-transform:inherit;text-align:left;cursor:pointer;white-space:nowrap;";
      button.setAttribute("aria-label", `Sort by ${originalLabel}`);

      const label = document.createElement("span");
      label.className = "table-header-label";
      label.textContent = originalLabel;
      const icon = document.createElement("span");
      icon.className = "table-sort-icon";
      icon.style.cssText = "display:inline-block;min-width:10px;color:#94a3b8;font-size:8px;line-height:1;";
      button.append(label, icon);

      if (field === "ibd_breakout_quality") {
        const info = document.createElement("span");
        info.dataset.qualityInfo = "true";
        info.textContent = "▾";
        info.setAttribute("role", "button");
        info.setAttribute("tabindex", "0");
        info.setAttribute("aria-label", "Explain Breakout Price Quality strength");
        info.style.cssText = "display:inline-grid;place-items:center;width:17px;height:17px;margin-left:2px;border:1px solid #475569;border-radius:4px;color:#86efac;font-size:10px;cursor:help;";
        const openPinned = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (qualityTooltipPinned && qualityTooltipAnchor === button) hideQualityTooltip(true);
          else showQualityTooltip(button, true);
        };
        info.addEventListener("click", openPinned);
        info.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") openPinned(event);
          if (event.key === "Escape") hideQualityTooltip(true);
        });
        button.appendChild(info);
        button.addEventListener("mouseenter", () => {
          if (!qualityTooltipPinned) showQualityTooltip(button, false);
        });
        button.addEventListener("mouseleave", () => hideQualityTooltip(false));
        button.addEventListener("focus", () => {
          if (!qualityTooltipPinned) showQualityTooltip(button, false);
        });
        button.addEventListener("blur", () => hideQualityTooltip(false));
      }

      button.addEventListener("click", onHeaderSort);
      header.appendChild(button);
    });

    const active = sortState[kind];
    if (active.field) {
      sortTable(shell, active.field, active.direction);
      updateSummary(kind);
    }
    updateSortIndicators(shell);
  }

  function enhanceTables() {
    app.querySelectorAll("[data-table-shell], [data-c-rank-table]").forEach(decorateTable);
  }

  function scheduleEnhance() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      enhanceTables();
    });
  }

  app.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    const shell = event.target.closest?.("[data-table-shell], [data-c-rank-table]");
    if (!shell) return;
    const kind = tableKind(shell);
    if (!sortState[kind].field) return;

    const rows = [...shell.querySelectorAll("tbody tr[data-code]")];
    if (!rows.length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    let index = rows.findIndex((row) => row.classList.contains("selected"));
    if (index < 0) index = event.key === "ArrowDown" ? -1 : rows.length;
    index += event.key === "ArrowDown" ? 1 : -1;
    index = Math.max(0, Math.min(rows.length - 1, index));
    rows[index].click();
    requestAnimationFrame(() => {
      const refreshed = [...shell.querySelectorAll("tbody tr[data-code]")].find((row) => row.dataset.code === rows[index].dataset.code);
      refreshed?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  }, true);

  document.addEventListener("pointerdown", (event) => {
    if (qualityTooltipPinned && !event.target.closest?.("[data-quality-info]")) hideQualityTooltip(true);
  });
  window.addEventListener("resize", positionQualityTooltip);
  window.addEventListener("scroll", () => hideQualityTooltip(true), true);

  const observer = new MutationObserver(scheduleEnhance);
  observer.observe(app, { childList: true, subtree: true });
  scheduleEnhance();
})();

(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  let dashboardData = null;
  let refreshQueued = false;

  function bool(value) {
    if (value === true || value === 1) return true;
    return ["true", "1", "1.0", "yes", "y", "t"].includes(String(value ?? "").trim().toLowerCase());
  }

  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function isActive(row) {
    if (Object.hasOwn(row, "review_watch_active")) return bool(row.review_watch_active);
    return bool(row.signal);
  }

  function pressedValue(action) {
    return app.querySelector(`[data-action="${action}"][aria-pressed="true"]`)?.dataset.value || null;
  }

  function quickValue(field) {
    return app.querySelector(`[data-action="quick"][data-field="${field}"][aria-pressed="true"]`)?.dataset.value || "ALL";
  }

  function contextRows() {
    if (!dashboardData) return [];
    const period = pressedValue("period") || "WEEKEND";
    const comparison = period === "MIDWEEK" && Boolean(dashboardData.meta?.midweek_baseline_available);
    const source = period === "MIDWEEK" ? dashboardData.views?.midweek?.rows : dashboardData.views?.weekend?.rows;
    let rows = Array.isArray(source) ? source.filter(isActive) : [];

    if (comparison && pressedValue("scope") === "CHANGES") {
      rows = rows.filter((row) => String(row.review_change_group || "UNCHANGED") !== "UNCHANGED");
    }

    const change = quickValue("change");
    if (comparison && change !== "ALL") rows = rows.filter((row) => row.review_change_group === change);

    const origin = quickValue("origin");
    if (comparison && origin !== "ALL") rows = rows.filter((row) => row.review_signal_origin === origin);

    const status = pressedValue("status");
    if (status) rows = rows.filter((row) => row.ibd_entry_status === status);

    const route = app.querySelector('[data-control="route"]')?.value || "All";
    if (route !== "All") rows = rows.filter((row) => row.ibd_candidate_rule === route);

    return rows;
  }

  function bounds(rows, field) {
    const values = rows.map((row) => num(row[field])).filter((value) => value !== null);
    if (!values.length) return null;
    const low = Math.floor(Math.min(...values) * 10) / 10;
    const high = Math.ceil(Math.max(...values) * 10) / 10;
    return [Number(low.toFixed(1)), Number(high.toFixed(1))];
  }

  function format(value, kind) {
    const parsed = num(value);
    if (parsed === null) return "n/a";
    if (kind === "pct") return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(1)}%`;
    return `${parsed.toFixed(1)}×`;
  }

  function labelText(name, inactive, value, kind) {
    if (name === "distance-min") return `Vs Buy Point · Min · ${inactive ? "Full range" : format(value, kind)}`;
    if (name === "distance-max") return `Vs Buy Point · Max · ${inactive ? "Full range" : format(value, kind)}`;
    if (name === "entry-volume") return inactive ? "Entry Volume · Min · Full range" : `Entry Volume ≥ ${format(value, kind)}`;
    return inactive ? "Weekly Volume · Min · Full range" : `Weekly Volume ≥ ${format(value, kind)}`;
  }

  function enhanceRange(name, rangeBounds, edge, kind) {
    const input = app.querySelector(`[data-control="${name}"]`);
    if (!input || !rangeBounds || input.dataset.dynamicBounds === "true") return;

    const field = input.closest(".filter-field");
    const label = field?.querySelector("label");
    const valueLabels = field ? [...field.querySelectorAll(".range-values small")] : [];
    const appStatus = valueLabels[1]?.textContent?.trim() || "";
    const inactive = appStatus === "Any" || label?.textContent?.includes("Any");
    const [low, high] = rangeBounds;
    const previous = Number(input.value);

    input.dataset.dynamicBounds = "true";
    input.min = String(low);
    input.max = String(high);
    input.step = "0.1";

    let nextValue = inactive ? (edge === "max" ? high : low) : Math.min(high, Math.max(low, previous));
    if (!Number.isFinite(nextValue)) nextValue = edge === "max" ? high : low;
    input.value = String(nextValue);

    if (low === high) input.disabled = true;
    if (valueLabels[0]) valueLabels[0].textContent = format(low, kind);
    if (valueLabels[1]) valueLabels[1].textContent = format(high, kind);
    if (label) label.textContent = labelText(name, inactive, nextValue, kind);

    input.addEventListener("input", () => {
      if (label) label.textContent = labelText(name, false, Number(input.value), kind);
    });

    if (!inactive && Math.abs(previous - nextValue) > 1e-9) {
      queueMicrotask(() => input.dispatchEvent(new Event("change", { bubbles: true })));
    }
  }

  function enhanceRanges() {
    if (!dashboardData) return;
    const rows = contextRows();
    if (!rows.length) return;
    enhanceRange("distance-min", bounds(rows, "current_vs_ibd_candidate_pct"), "min", "pct");
    enhanceRange("distance-max", bounds(rows, "current_vs_ibd_candidate_pct"), "max", "pct");
    enhanceRange("entry-volume", bounds(rows, "ibd_entry_volume_ratio"), "min", "x");
    enhanceRange("weekly-volume", bounds(rows, "volume_ratio"), "min", "x");
  }

  function scheduleEnhance() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      enhanceRanges();
    });
  }

  const observer = new MutationObserver(scheduleEnhance);
  observer.observe(app, { childList: true, subtree: true });

  fetch("./data/dashboard.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      dashboardData = payload;
      scheduleEnhance();
    })
    .catch(() => {
      dashboardData = null;
    });
})();
