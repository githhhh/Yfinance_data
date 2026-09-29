(() => {
  "use strict";

  const app = document.getElementById("app");
  if (!app) return;

  let rsBackdrop = null;
  let pendingTableViewport = null;

  function ensureInteractionStyles() {
    if (document.getElementById("interaction-runtime-styles")) return;
    const style = document.createElement("style");
    style.id = "interaction-runtime-styles";
    style.textContent = `
      /* Keep horizontal gestures contained. On mobile, suppress vertical
         rubber-band overscroll so the sticky review header stays visually
         fixed while the result body scrolls. */
      .table-shell {
        overscroll-behavior-x: none !important;
        overscroll-behavior-y: auto !important;
      }
      @media (max-width: 760px) {
        .table-shell {
          overscroll-behavior-y: none !important;
        }
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

  document.addEventListener("click", (event) => {
    if (event.target.closest?.(
      '[data-action="period"], [data-action="scope"], [data-action="quick"], '
      + '[data-action="clear-quick"], [data-action="status"], '
      + '[data-action="toggle-filters"], [data-action="reset-filters"]',
    )) captureTableViewport();
  }, true);

  document.addEventListener("change", (event) => {
    if (event.target.matches?.('[data-control="route"], input[type="range"][data-control]')) {
      captureTableViewport();
    }
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
    restoreTableViewport();
  });
  appObserver.observe(app, { childList: true, subtree: true });

  const bodyObserver = new MutationObserver(syncRsBackdrop);
  bodyObserver.observe(document.body, { childList: true, subtree: true });

  ensureInteractionStyles();
  prepareRangeInputs();
  syncRsBackdrop();
})();
