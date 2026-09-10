(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const sortState = { field: null, direction: "asc", label: null };
  const STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW TRIGGER", "EXTENDED"];
  const QUALITY_ORDER = ["POWERFUL", "STRONG", "CONSTRUCTIVE", "MARGINAL", "WEAK"];
  let rsBackdrop = null;

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
    if (!cleaned || cleaned.toLowerCase() === "n/a" || cleaned === "—") return null;
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
    if (ordinalLeft !== null && ordinalRight !== null) {
      const result = ordinalLeft - ordinalRight;
      return direction === "desc" ? -result : result;
    }

    const numericLeft = numericValue(left);
    const numericRight = numericValue(right);
    if (field === "rs_percentile") {
      if (numericLeft === null && numericRight !== null) return 1;
      if (numericLeft !== null && numericRight === null) return -1;
    }

    let result = 0;
    if (numericLeft !== null && numericRight !== null) result = numericLeft - numericRight;
    else if (numericLeft !== null) result = -1;
    else if (numericRight !== null) result = 1;
    else {
      result = normalizeText(left).localeCompare(normalizeText(right), undefined, {
        numeric: true,
        sensitivity: "base",
      });
    }
    return direction === "desc" ? -result : result;
  }

  function columnField(label) {
    const normalized = normalizeText(label).replace(/[▲▼▾]/g, "").trim();
    return {
      Code: "code",
      Change: "review_change_label",
      Status: "ibd_entry_status",
      Setup: "ibd_candidate_rule",
      "Vs Buy Point": "current_vs_ibd_candidate_pct",
      "Breakout Price Quality": "ibd_breakout_quality",
      Latest: "latest_close",
      "Entry / Reason": "ibd_entry_vol_or_reject",
      "Weekly Vol": "volume_ratio",
      RS: "rs_percentile",
    }[normalized] || null;
  }

  function headerField(header) {
    return header?.dataset.sortField
      || columnField(header?.querySelector(".table-header-label")?.textContent || header?.textContent);
  }

  function headerLabel(header) {
    return normalizeText(header?.querySelector(".table-header-label")?.textContent || header?.textContent)
      .replace(/[▲▼▾]/g, "")
      .trim();
  }

  function rememberManualSort(header) {
    if (!header) return;
    const field = headerField(header);
    const ariaSort = header.getAttribute("aria-sort");
    if (!field || !["ascending", "descending"].includes(ariaSort)) return;
    sortState.field = field;
    sortState.direction = ariaSort === "descending" ? "desc" : "asc";
    sortState.label = headerLabel(header);
  }

  function applyRememberedSort() {
    if (!sortState.field) return;
    const shell = app.querySelector("[data-table-shell]");
    if (!shell) return;

    const headers = [...shell.querySelectorAll("thead th")];
    const index = headers.findIndex((header) => headerField(header) === sortState.field);
    const body = shell.querySelector("tbody");
    if (index < 0 || !body) return;

    const rows = [...body.querySelectorAll("tr[data-code]")];
    const sortedRows = [...rows].sort((leftRow, rightRow) => {
      const left = leftRow.children[index]?.textContent || "";
      const right = rightRow.children[index]?.textContent || "";
      const compared = compareValues(sortState.field, left, right, sortState.direction);
      if (compared !== 0) return compared;
      return normalizeText(leftRow.dataset.code).localeCompare(normalizeText(rightRow.dataset.code));
    });
    const orderChanged = sortedRows.some((row, position) => row !== rows[position]);
    if (orderChanged) sortedRows.forEach((row) => body.appendChild(row));

    const label = sortState.label || headerLabel(headers[index]);
    const summary = app.querySelector(".results-summary");
    const nextSummary = `${rows.length} results · Sorted by ${label} ${sortState.direction === "asc" ? "↑" : "↓"}`;
    if (summary && summary.textContent !== nextSummary) summary.textContent = nextSummary;
  }

  // app.js emits initial range markup with the same marker the legacy range
  // enhancer uses as its "already enhanced" flag. Clear that marker exactly
  // once per newly rendered input so the existing authoritative context-bound
  // enhancement still runs; keep a separate bootstrap flag to avoid duplicate
  // listeners during partial Selected Detail updates.
  function prepareRangeInputs() {
    app.querySelectorAll('input[data-dynamic-bounds="true"]').forEach((input) => {
      if (input.dataset.rangeBootstrap === "true") return;
      input.removeAttribute("data-dynamic-bounds");
      input.dataset.rangeBootstrap = "true";
    });
  }

  // table_enhancements owns the visible sort controls. Capture its committed
  // state after the button handler runs, then synchronously re-apply that order
  // from MutationObserver callbacks before the next paint whenever app.js has
  // to rebuild the table for filters/period changes.
  app.addEventListener("click", (event) => {
    if (event.target.closest?.("[data-rs-info], [data-quality-info]")) return;
    const button = event.target.closest?.("thead th > button");
    if (!button) return;
    rememberManualSort(button.closest("th"));
  });

  // The legacy table enhancer also has a manual-sort ArrowUp/ArrowDown handler.
  // Intercept one level earlier (document capture) so keyboard review preserves
  // the current horizontal scroll position just like pointer row selection.
  document.addEventListener("keydown", (event) => {
    if (!sortState.field || !["ArrowDown", "ArrowUp"].includes(event.key)) return;
    const shell = event.target.closest?.("[data-table-shell]");
    if (!shell) return;
    const rows = [...shell.querySelectorAll("tbody tr[data-code]")];
    if (!rows.length) return;

    event.preventDefault();
    event.stopImmediatePropagation();
    let index = rows.findIndex((row) => row.classList.contains("selected"));
    if (index < 0) index = event.key === "ArrowDown" ? -1 : rows.length;
    index += event.key === "ArrowDown" ? 1 : -1;
    index = Math.max(0, Math.min(rows.length - 1, index));

    const code = rows[index].dataset.code;
    const scrollLeft = shell.scrollLeft;
    rows[index].click();
    requestAnimationFrame(() => {
      const currentShell = app.querySelector("[data-table-shell]");
      const target = currentShell?.querySelector(`tr[data-code="${CSS.escape(String(code))}"]`);
      target?.scrollIntoView({ block: "nearest", inline: "nearest" });
      if (currentShell) {
        currentShell.scrollLeft = scrollLeft;
        currentShell.focus({ preventScroll: true });
      }
    });
  }, true);

  function syncRsBackdrop() {
    const popover = document.querySelector(".rs-reference-popover");
    if (popover && !rsBackdrop) {
      rsBackdrop = document.createElement("div");
      rsBackdrop.className = "rs-popover-backdrop";
      rsBackdrop.setAttribute("aria-hidden", "true");
      rsBackdrop.addEventListener("pointerdown", (event) => {
        // Keep the backdrop alive through the pointer sequence so the following
        // click still targets it instead of a filter/sort control underneath.
        event.stopPropagation();
      });
      rsBackdrop.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        document.querySelector("[data-rs-info]")?.click();
      });
      document.body.insertBefore(rsBackdrop, popover);
    } else if (!popover && rsBackdrop) {
      rsBackdrop.remove();
      rsBackdrop = null;
    }
  }

  const appObserver = new MutationObserver(() => {
    prepareRangeInputs();
    applyRememberedSort();
  });
  appObserver.observe(app, { childList: true, subtree: true });

  const bodyObserver = new MutationObserver(syncRsBackdrop);
  bodyObserver.observe(document.body, { childList: true });
  prepareRangeInputs();
  syncRsBackdrop();
})();
