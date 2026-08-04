from __future__ import annotations


# st.components.v1.html runs in a same-origin sandboxed iframe. This bridge keeps
# the filter buttons native to Streamlit while giving their independent info
# triggers one shared hover, focus, and touch tooltip in the parent document.
FLOW_TOOLTIP_BRIDGE_HTML = r"""
<script>
(() => {
    const parentWindow = window.parent;
    const parentDocument = window.parent.document;
    const controllerKey = "__breakoutPoolFlowTooltipController";
    const previousController = parentWindow[controllerKey];
    if (previousController && typeof previousController.destroy === "function") {
        previousController.destroy();
    }

    const tooltip = parentDocument.createElement("div");
    tooltip.id = "breakout-pool-flow-tooltip";
    tooltip.className = "flow-tooltip-surface";
    tooltip.setAttribute("role", "tooltip");
    tooltip.hidden = true;
    parentDocument.body.appendChild(tooltip);

    let activeCard = null;
    let pinned = false;
    let describedElements = [];
    const hoverDelayMs = 275;
    let hoverTimer = null;
    let pendingCard = null;

    const cancelHoverTimer = () => {
        if (hoverTimer !== null) {
            parentWindow.clearTimeout(hoverTimer);
            hoverTimer = null;
        }
        pendingCard = null;
    };

    const cardFor = target => target instanceof parentWindow.Element
        ? target.closest('div[class*="st-key-flow_card_"]')
        : null;

    const ibdDetailsFor = target => target instanceof parentWindow.Element
        ? target.closest(".st-key-ibd_selected_row .code-detail")
        : null;

    const restoreDescriptions = () => {
        for (const [element, priorValue] of describedElements) {
            if (!element.isConnected) continue;
            if (priorValue === null) {
                element.removeAttribute("aria-describedby");
            } else {
                element.setAttribute("aria-describedby", priorValue);
            }
        }
        describedElements = [];
    };

    const describeCard = card => {
        restoreDescriptions();
        const targets = card.querySelectorAll("button[kind], .flow-info-trigger");
        describedElements = Array.from(targets, element => [
            element,
            element.getAttribute("aria-describedby"),
        ]);
        for (const [element] of describedElements) {
            element.setAttribute("aria-describedby", tooltip.id);
        }
    };

    const positionTooltip = card => {
        const anchor = card.querySelector(".flow-info-trigger") || card;
        const anchorBox = anchor.getBoundingClientRect();
        const viewportPadding = 12;
        tooltip.style.left = `${viewportPadding}px`;
        tooltip.style.top = `${viewportPadding}px`;
        tooltip.style.visibility = "hidden";
        const tooltipBox = tooltip.getBoundingClientRect();
        const maxLeft = Math.max(
            viewportPadding,
            parentWindow.innerWidth - tooltipBox.width - viewportPadding,
        );
        const left = Math.min(Math.max(anchorBox.right - tooltipBox.width, viewportPadding), maxLeft);
        let top = anchorBox.bottom + 8;
        if (top + tooltipBox.height > parentWindow.innerHeight - viewportPadding) {
            top = Math.max(viewportPadding, anchorBox.top - tooltipBox.height - 8);
        }
        tooltip.style.left = `${Math.round(left)}px`;
        tooltip.style.top = `${Math.round(top)}px`;
        tooltip.style.visibility = "visible";
    };

    const hideTooltip = () => {
        cancelHoverTimer();
        if (activeCard) {
            const trigger = activeCard.querySelector(".flow-info-trigger");
            if (trigger) trigger.setAttribute("aria-expanded", "false");
        }
        restoreDescriptions();
        tooltip.hidden = true;
        tooltip.textContent = "";
        activeCard = null;
        pinned = false;
    };

    const showTooltip = (card, shouldPin = false) => {
        const trigger = card && card.querySelector(".flow-info-trigger");
        const title = trigger && trigger.dataset.flowTooltipTitle;
        const content = trigger && trigger.dataset.flowTooltip;
        if (!title || !content) return;
        if (activeCard && activeCard !== card) hideTooltip();
        activeCard = card;
        pinned = shouldPin;
        tooltip.textContent = "";
        const titleElement = parentDocument.createElement("strong");
        titleElement.textContent = title;
        titleElement.style.display = "block";
        const body = parentDocument.createElement("span");
        body.textContent = content;
        tooltip.append(titleElement, body);
        tooltip.hidden = false;
        trigger.setAttribute("aria-expanded", shouldPin ? "true" : "false");
        describeCard(card);
        positionTooltip(card);
    };

    const scheduleTooltip = card => {
        if (pendingCard === card) return;
        cancelHoverTimer();
        pendingCard = card;
        hoverTimer = parentWindow.setTimeout(() => {
            hoverTimer = null;
            pendingCard = null;
            showTooltip(card, false);
        }, hoverDelayMs);
    };

    const onPointerOver = event => {
        const ibdDetails = ibdDetailsFor(event.target);
        if (ibdDetails) delete ibdDetails.dataset.escapeDismissed;
        const card = cardFor(event.target);
        if (!card || (pinned && activeCard !== card)) return;
        if (event.relatedTarget instanceof parentWindow.Node && card.contains(event.relatedTarget)) return;
        if (pinned && activeCard === card) return;
        scheduleTooltip(card);
    };

    const onPointerOut = event => {
        const card = cardFor(event.target);
        const nextTarget = event.relatedTarget;
        if (
            nextTarget instanceof parentWindow.Node
            && card
            && card.contains(nextTarget)
        ) return;
        if (card && pendingCard === card) cancelHoverTimer();
        if (!activeCard || pinned) return;
        if (nextTarget instanceof parentWindow.Node && tooltip.contains(nextTarget)) return;
        if (activeCard.contains(parentDocument.activeElement)) return;
        hideTooltip();
    };

    const onFocusIn = event => {
        const card = cardFor(event.target);
        if (!card || (pinned && activeCard !== card)) return;
        cancelHoverTimer();
        showTooltip(card, pinned && activeCard === card);
    };

    const onFocusOut = () => {
        parentWindow.setTimeout(() => {
            if (!activeCard || pinned || activeCard.contains(parentDocument.activeElement)) return;
            hideTooltip();
        }, 0);
    };

    const onClick = event => {
        const ibdDetails = ibdDetailsFor(event.target);
        if (ibdDetails) delete ibdDetails.dataset.escapeDismissed;
        const trigger = event.target instanceof parentWindow.Element
            ? event.target.closest(".flow-info-trigger")
            : null;
        if (trigger) {
            event.preventDefault();
            event.stopPropagation();
            cancelHoverTimer();
            const card = cardFor(trigger);
            if (!card) return;
            if (activeCard === card && pinned) {
                hideTooltip();
            } else {
                showTooltip(card, true);
            }
            return;
        }
        if (activeCard && !tooltip.contains(event.target)) hideTooltip();
        else cancelHoverTimer();
    };

    const onKeyDown = event => {
        const targetDetails = ibdDetailsFor(event.target);
        if (targetDetails && event.key !== "Escape") {
            delete targetDetails.dataset.escapeDismissed;
        }
        if (event.key === "Escape") {
            const details = targetDetails
                || parentDocument.querySelector(".st-key-ibd_selected_row .code-detail[open]");
            if (details) {
                event.preventDefault();
                details.open = false;
                details.dataset.escapeDismissed = "true";
                const trigger = details.querySelector(".code-hover-trigger");
                if (trigger) trigger.focus({preventScroll: true});
                return;
            }
        }
        if (event.key === "Escape" && (activeCard || hoverTimer !== null)) {
            event.preventDefault();
            if (activeCard) hideTooltip();
            else cancelHoverTimer();
        }
    };

    const onScroll = () => {
        if (activeCard) hideTooltip();
        else cancelHoverTimer();
    };

    const onResize = () => {
        if (activeCard) positionTooltip(activeCard);
    };

    const syncFiltersDisclosure = () => {
        const filterHeader = parentDocument.querySelector(".st-key-filters_header");
        const marker = filterHeader && filterHeader.querySelector(".filters-state-marker");
        const button = filterHeader && filterHeader.querySelector("button");
        if (button && marker) {
            button.setAttribute("aria-expanded", marker.dataset.expanded);
        }
    };

    const filterObserver = new parentWindow.MutationObserver(syncFiltersDisclosure);
    filterObserver.observe(parentDocument.body, {childList: true, subtree: true});
    parentWindow.requestAnimationFrame(syncFiltersDisclosure);

    parentDocument.addEventListener("pointerover", onPointerOver);
    parentDocument.addEventListener("pointerout", onPointerOut);
    parentDocument.addEventListener("focusin", onFocusIn);
    parentDocument.addEventListener("focusout", onFocusOut);
    parentDocument.addEventListener("click", onClick);
    parentDocument.addEventListener("keydown", onKeyDown);
    parentDocument.addEventListener("scroll", onScroll, true);
    parentWindow.addEventListener("resize", onResize);

    parentWindow[controllerKey] = {
        destroy: () => {
            cancelHoverTimer();
            filterObserver.disconnect();
            parentDocument.removeEventListener("pointerover", onPointerOver);
            parentDocument.removeEventListener("pointerout", onPointerOut);
            parentDocument.removeEventListener("focusin", onFocusIn);
            parentDocument.removeEventListener("focusout", onFocusOut);
            parentDocument.removeEventListener("click", onClick);
            parentDocument.removeEventListener("keydown", onKeyDown);
            parentDocument.removeEventListener("scroll", onScroll, true);
            parentWindow.removeEventListener("resize", onResize);
            restoreDescriptions();
            tooltip.remove();
        },
    };
})();
</script>
"""
