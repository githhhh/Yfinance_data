(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  let dashboardData = null;
  let refreshQueued = false;
  let rsSortDirection = null;

  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function activePeriod() {
    return app.querySelector('[data-action="period"][aria-pressed="true"]')?.dataset.value || "WEEKEND";
  }

  function currentRows() {
    if (!dashboardData) return [];
    return activePeriod() === "MIDWEEK"
      ? dashboardData.views?.midweek?.rows || []
      : dashboardData.views?.weekend?.rows || [];
  }

  function rowByCode(code) {
    const target = String(code || "").trim();
    return currentRows().find((row) => String(row.code) === target) || null;
  }

  function expectedMarketDate() {
    if (!dashboardData) return null;
    return activePeriod() === "MIDWEEK"
      ? dashboardData.meta?.midweek_snapshot_date || null
      : dashboardData.meta?.complete_snapshot_date || null;
  }

  function hideLegacyModeSwitch() {
    const modeSwitch = app.querySelector('.dashboard-header .segmented[aria-label="Dashboard mode"]');
    if (modeSwitch) modeSwitch.style.display = "none";
  }

  function replaceLegacySummary() {
    const summary = app.querySelector(".results-summary");
    if (!summary || rsSortDirection) return;
    if (summary.textContent.includes("Sorted by C Rank")) {
      summary.textContent = summary.textContent.replace("Sorted by C Rank", "Sorted by Code");
    }
  }

  function rsTitle(row) {
    const meta = dashboardData?.meta?.rs_reference || {};
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
    const lines = [
      `RS Percentile: ${current}`,
      `1M ago: ${num(row.rs_1m_percentile) ?? "N/A"}`,
      `3M ago: ${num(row.rs_3m_percentile) ?? "N/A"}`,
      `6M ago: ${num(row.rs_6m_percentile) ?? "N/A"}`,
      `Market date: ${sourceDate || "N/A"}`,
      `Source: ${source}`,
      "IBD-style reference; not official IBD RS and not a ranking gate.",
    ];
    return lines.join("\n");
  }

  function rsHeader(shell) {
    return [...shell.querySelectorAll("thead th")].find((header) => {
      const label = header.querySelector(".table-header-label")?.textContent?.trim();
      return label === "C Rank" || label === "RS" || header.dataset.sortField === "rs_percentile";
    }) || null;
  }

  function replaceReviewColumn() {
    const shell = app.querySelector("[data-table-shell]");
    if (!shell || !dashboardData) return;
    const header = rsHeader(shell);
    if (!header || !header.querySelector(".table-header-label")) return;

    const headers = [...shell.querySelectorAll("thead th")];
    const index = headers.indexOf(header);
    if (index < 0) return;

    const label = header.querySelector(".table-header-label");
    const button = header.querySelector("button");
    label.textContent = "RS";
    header.dataset.sortField = "rs_percentile";
    if (button) button.setAttribute("aria-label", "Sort by RS");

    shell.querySelectorAll("tbody tr[data-code]").forEach((tr) => {
      const row = rowByCode(tr.dataset.code);
      const cell = tr.children[index];
      if (!cell) return;
      const value = num(row?.rs_percentile);
      cell.textContent = value === null ? "N/A" : String(value);
      cell.title = rsTitle(row);
      cell.dataset.rsValue = value === null ? "" : String(value);
    });

    if (button && button.dataset.rsSortBound !== "true") {
      button.dataset.rsSortBound = "true";
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        rsSortDirection = rsSortDirection === "desc" ? "asc" : "desc";
        sortByRs(shell, rsSortDirection);
        updateRsSortUi(shell);
      }, true);
    }

    if (rsSortDirection) {
      sortByRs(shell, rsSortDirection);
      updateRsSortUi(shell);
    }
  }

  function sortByRs(shell, direction) {
    const header = rsHeader(shell);
    const headers = [...shell.querySelectorAll("thead th")];
    const index = headers.indexOf(header);
    const body = shell.querySelector("tbody");
    if (index < 0 || !body) return;
    const rows = [...body.querySelectorAll("tr[data-code]")];
    rows.sort((a, b) => {
      const av = num(a.children[index]?.dataset.rsValue);
      const bv = num(b.children[index]?.dataset.rsValue);
      if (av === null && bv === null) return String(a.dataset.code).localeCompare(String(b.dataset.code));
      if (av === null) return 1;
      if (bv === null) return -1;
      const diff = av - bv;
      if (diff !== 0) return direction === "asc" ? diff : -diff;
      return String(a.dataset.code).localeCompare(String(b.dataset.code));
    });
    rows.forEach((row) => body.appendChild(row));
  }

  function updateRsSortUi(shell) {
    shell.querySelectorAll('thead th[data-sort-field]').forEach((header) => {
      const isRs = header.dataset.sortField === "rs_percentile";
      const icon = header.querySelector(".table-sort-icon");
      if (!isRs) {
        header.setAttribute("aria-sort", "none");
        if (icon) icon.textContent = "";
        return;
      }
      header.setAttribute("aria-sort", rsSortDirection === "asc" ? "ascending" : "descending");
      if (icon) icon.textContent = rsSortDirection === "asc" ? "▲" : "▼";
    });
    const summary = app.querySelector(".results-summary");
    if (summary) {
      const count = summary.textContent.match(/^\d+\s+results/i)?.[0];
      summary.textContent = `${count || "Results"} · Sorted by RS ${rsSortDirection === "asc" ? "↑" : "↓"}`;
    }
  }

  function replaceSelectedReference() {
    const selectedCode = app.querySelector(".selected-code")?.textContent?.trim();
    if (!selectedCode || !dashboardData) return;
    const row = rowByCode(selectedCode);
    app.querySelectorAll(".selected-cell").forEach((cell) => {
      const key = cell.querySelector(".selected-key");
      if (!key || !key.textContent.includes("C Rank")) return;
      const value = cell.querySelector(".selected-value");
      const current = num(row?.rs_percentile);
      key.textContent = "RS Reference";
      if (value) {
        value.innerHTML = current === null
          ? "N/A"
          : `${current} <small>1M ${num(row.rs_1m_percentile) ?? "N/A"} · 3M ${num(row.rs_3m_percentile) ?? "N/A"} · 6M ${num(row.rs_6m_percentile) ?? "N/A"}</small>`;
        value.title = rsTitle(row);
      }
    });
  }

  async function copyVisibleCodes(button) {
    const shell = app.querySelector("[data-table-shell]");
    const codes = shell ? [...shell.querySelectorAll("tbody tr[data-code]")].map((row) => row.dataset.code).filter(Boolean) : [];
    if (!codes.length) return;
    const payload = codes.join(", ");
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
    button.textContent = success ? `✓ Copied ${codes.length}` : "Copy failed";
    setTimeout(() => { button.textContent = original; }, 1600);
  }

  function enhance() {
    hideLegacyModeSwitch();
    replaceLegacySummary();
    replaceReviewColumn();
    replaceSelectedReference();
  }

  function scheduleEnhance() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      enhance();
    });
  }

  app.addEventListener("click", (event) => {
    const copy = event.target.closest?.('[data-action="copy-codes"]');
    if (copy) {
      event.preventDefault();
      event.stopImmediatePropagation();
      copyVisibleCodes(copy);
      return;
    }
    const headerButton = event.target.closest?.("[data-table-shell] thead th button");
    if (headerButton && headerButton.closest("th")?.dataset.sortField !== "rs_percentile") {
      rsSortDirection = null;
    }
  }, true);

  app.addEventListener("keydown", (event) => {
    if (!rsSortDirection || !["ArrowDown", "ArrowUp"].includes(event.key)) return;
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
    rows[index].click();
  }, true);

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
      scheduleEnhance();
    });

  scheduleEnhance();
})();
