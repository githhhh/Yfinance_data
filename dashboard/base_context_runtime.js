(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  let payload = null;
  let scheduled = false;

  function injectStyles() {
    if (document.getElementById("base-context-runtime-styles")) return;
    const style = document.createElement("style");
    style.id = "base-context-runtime-styles";
    style.textContent = `
      .detail-grid.base-context-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    `;
    document.head.appendChild(style);
  }

  function activePeriod() {
    const selected = app.querySelector('[data-action="period"][aria-pressed="true"]');
    return selected?.dataset.value === "MIDWEEK" ? "midweek" : "weekend";
  }

  function selectedCode() {
    return app.querySelector(".selected-code")?.textContent?.trim() || null;
  }

  function currentRow() {
    if (!payload) return null;
    const code = selectedCode();
    if (!code) return null;
    const rows = payload?.views?.[activePeriod()]?.rows;
    if (!Array.isArray(rows)) return null;
    return rows.find((row) => String(row?.code) === code) || null;
  }

  function displayNumber(value) {
    if (value === null || value === undefined || value === "") return "n/a";
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(2) : "n/a";
  }

  function displayText(value) {
    if (value === null || value === undefined) return "n/a";
    const text = String(value).trim();
    return text && !["nan", "none", "<na>"].includes(text.toLowerCase()) ? text : "n/a";
  }

  function findBaseGrid() {
    const sections = [...app.querySelectorAll(".detail-section")];
    const section = sections.find((node) =>
      node.querySelector(".detail-title")?.textContent?.trim() === "3. CANSLIM / Base"
    );
    return section?.querySelector(".detail-grid") || null;
  }

  function findItem(grid, label) {
    return [...grid.querySelectorAll(":scope > .detail-item")].find(
      (item) => item.querySelector("span")?.textContent?.trim() === label
    ) || null;
  }

  function upsertItem(grid, key, label, value) {
    let item = grid.querySelector(`[data-base-context="${key}"]`);
    if (!item) {
      item = document.createElement("div");
      item.className = "detail-item";
      item.dataset.baseContext = key;
      const labelNode = document.createElement("span");
      const valueNode = document.createElement("b");
      labelNode.textContent = label;
      item.append(labelNode, valueNode);
      grid.appendChild(item);
    }
    const valueNode = item.querySelector("b");
    if (valueNode && valueNode.textContent !== value) valueNode.textContent = value;
    return item;
  }

  function enhanceSelectedDetail() {
    scheduled = false;
    const row = currentRow();
    const grid = findBaseGrid();
    if (!row || !grid) return;

    grid.classList.add("base-context-grid");
    upsertItem(grid, "base-ceiling", "Base Ceiling", displayNumber(row.ceiling));
    upsertItem(grid, "base-ceiling-date", "Base Ceiling Date", displayText(row.ceiling_date));
    upsertItem(grid, "buy-point-date", "Buy Point Date", displayText(row.review_buy_point_date));

    // Keep the nine-field desktop grid semantically grouped:
    // fundamentals -> base geometry -> date provenance / industry.
    const industry = findItem(grid, "Industry");
    if (industry && grid.lastElementChild !== industry) grid.appendChild(industry);
  }

  function scheduleEnhancement() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(enhanceSelectedDetail);
  }

  injectStyles();

  const observer = new MutationObserver(scheduleEnhancement);
  observer.observe(app, { childList: true, subtree: true });

  fetch("./data/dashboard.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      payload = data;
      scheduleEnhancement();
    })
    .catch(() => {
      // Optional display context only; the authoritative dashboard remains usable.
    });
})();
