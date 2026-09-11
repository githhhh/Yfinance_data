(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  const sortState = { field: null, direction: "asc", label: null };
  const STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW TRIGGER", "EXTENDED"];
  const QUALITY_ORDER = ["POWERFUL", "STRONG", "CONSTRUCTIVE", "MARGINAL", "WEAK"];
  let rsBackdrop = null;
  let pendingTableViewport = null;
  let pendingReviewAnchor = null;

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

  function ensureInteractionStyles() {
    if (document.getElementById("interaction-runtime-styles")) return;
    const style = document.createElement("style");
    style.id = "interaction-runtime-styles";
    style.textContent = `
      /* Horizontal table gestures stay contained, but vertical gestures must
         chain back to the page at the table's top/bottom instead of trapping
         the user inside the 58vh review surface on mobile. */
      .table-shell {
        overscroll-behavior-x: none !important;
        overscroll-behavior-y: auto !important;
      }
      .review-table th:first-child::after,
      .review-table td:first-child::after {
        content: "";
        position: absolute;
        top: 0;
        right: -5px;
        width: 5px;
        height: 100%;
        pointer-events: none;
        background: linear-gradient(to right, rgb(0 0 0 / 34%), transparent);
      }
      /* The visible Quality glyph stays compact while the pseudo-element gives
         it a ~44px touch target. It remains the event target, so its existing
         stopPropagation prevents accidental column sorting. */
      [data-quality-info] {
        position: relative !important;
        width: 32px !important;
        height: 32px !important;
        flex: 0 0 32px !important;
        margin-left: 4px !important;
        touch-action: manipulation;
        z-index: 2;
      }
      [data-quality-info]::after {
        content: "";
        position: absolute;
        inset: -6px;
      }
      .rs-popover-backdrop {
        background: rgb(0 0 0 / 28%) !important;
      }
      .rs-runtime-close {
        width: 30px;
        height: 30px;
        display: grid;
        place-items: center;
        flex: 0 0 30px;
        padding: 0;
        border: 1px solid #465365;
        border-radius: 7px;
        background: #151b23;
        color: #b7c1ce;
        font: 700 17px/1 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        cursor: pointer;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
      }
    `;
    document.head.appendChild(style);
  }

  function prepareRangeInputs() {
    app.querySelectorAll('input[data-dynamic-bounds="true"]').forEach((input) => {
      if (input.dataset.rangeBootstrap === "true") return;
      input.removeAttribute("data-dynamic-bounds");
      input.dataset.rangeBootstrap = "true";
    });
  }

  function captureTableViewport() {
    const shell = app.querySelector("[data-table-shell]");
    if (!shell) return;
    pendingTableViewport = {
      shell,
      scrollLeft: shell.scrollLeft,
      scrollTop: shell.scrollTop,
    };
  }

  function restoreTableViewport() {
    if (!pendingTableViewport) return;
    const snapshot = pendingTableViewport;
    const shell = app.querySelector("[data-table-shell]");
    pendingTableViewport = null;
    if (!shell || shell === snapshot.shell) return;
    shell.scrollLeft = snapshot.scrollLeft;
    shell.scrollTop = snapshot.scrollTop;
  }

  function captureReviewAnchor(event) {
    const row = event.target.closest?.("tbody tr[data-code]");
    const detail = event.target.closest?.('[data-action="detail"]');
    const anchor = row || (detail ? app.querySelector("tbody tr.selected[data-code]") : null);
    if (!anchor) return;
    pendingReviewAnchor = {
      code: anchor.dataset.code,
      top: anchor.getBoundingClientRect().top,
    };
  }

  function restoreReviewAnchor() {
    if (!pendingReviewAnchor) return;
    const snapshot = pendingReviewAnchor;
    pendingReviewAnchor = null;
    const row = app.querySelector(`tbody tr[data-code="${CSS.escape(String(snapshot.code))}"]`);
    if (!row) return;
    const delta = row.getBoundingClientRect().top - snapshot.top;
    if (Number.isFinite(delta) && Math.abs(delta) > 0.5) window.scrollBy(0, delta);
  }

  app.addEventListener("click", (event) => {
    if (event.target.closest?.("[data-rs-info], [data-quality-info]")) return;
    const button = event.target.closest?.("thead th > button");
    if (!button) return;
    rememberManualSort(button.closest("th"));
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest?.(
      '[data-action="period"], [data-action="scope"], [data-action="quick"], '
      + '[data-action="clear-quick"], [data-action="status"], '
      + '[data-action="toggle-filters"], [data-action="reset-filters"]',
    )) captureTableViewport();
    captureReviewAnchor(event);
  }, true);
  document.addEventListener("change", (event) => {
    if (event.target.matches?.('[data-control="route"], input[type="range"][data-control]')) {
      captureTableViewport();
    }
  }, true);

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

  function closeRsPopover() {
    document.querySelector("[data-rs-info]")?.click();
  }

  function ensureRsCloseButton(popover) {
    const head = popover?.querySelector(".rs-popover-head");
    if (!head || head.querySelector("[data-rs-runtime-close]")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rs-runtime-close";
    button.dataset.rsRuntimeClose = "true";
    button.setAttribute("aria-label", "Close Relative Strength details");
    button.textContent = "×";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeRsPopover();
    });
    head.appendChild(button);
  }

  function syncRsBackdrop() {
    const popover = document.querySelector(".rs-reference-popover");
    if (popover) {
      popover.setAttribute("aria-modal", "true");
      ensureRsCloseButton(popover);
      if (!rsBackdrop) {
        rsBackdrop = document.createElement("div");
        rsBackdrop.className = "rs-popover-backdrop";
        rsBackdrop.setAttribute("aria-hidden", "true");
        rsBackdrop.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
        });
        rsBackdrop.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          closeRsPopover();
        });
        document.body.insertBefore(rsBackdrop, popover);
      }
    } else if (rsBackdrop) {
      rsBackdrop.remove();
      rsBackdrop = null;
    }
  }

  const appObserver = new MutationObserver(() => {
    prepareRangeInputs();
    applyRememberedSort();
    restoreTableViewport();
    restoreReviewAnchor();
  });
  appObserver.observe(app, { childList: true, subtree: true });

  const bodyObserver = new MutationObserver(syncRsBackdrop);
  bodyObserver.observe(document.body, { childList: true, subtree: true });
  ensureInteractionStyles();
  prepareRangeInputs();
  syncRsBackdrop();
})();
