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

    const cardFor = target => target instanceof parentWindow.Element
        ? target.closest('div[class*="st-key-flow_card_"]')
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
        const content = trigger && trigger.dataset.flowTooltip;
        if (!content) return;
        if (activeCard && activeCard !== card) hideTooltip();
        activeCard = card;
        pinned = shouldPin;
        tooltip.textContent = content;
        tooltip.hidden = false;
        trigger.setAttribute("aria-expanded", shouldPin ? "true" : "false");
        describeCard(card);
        positionTooltip(card);
    };

    const onPointerOver = event => {
        const card = cardFor(event.target);
        if (!card || (pinned && activeCard !== card)) return;
        showTooltip(card, pinned && activeCard === card);
    };

    const onPointerOut = event => {
        if (!activeCard || pinned) return;
        const nextTarget = event.relatedTarget;
        if (
            nextTarget instanceof parentWindow.Node
            && (activeCard.contains(nextTarget) || tooltip.contains(nextTarget))
        ) return;
        if (activeCard.contains(parentDocument.activeElement)) return;
        hideTooltip();
    };

    const onFocusIn = event => {
        const card = cardFor(event.target);
        if (!card || (pinned && activeCard !== card)) return;
        showTooltip(card, pinned && activeCard === card);
    };

    const onFocusOut = () => {
        parentWindow.setTimeout(() => {
            if (!activeCard || pinned || activeCard.contains(parentDocument.activeElement)) return;
            hideTooltip();
        }, 0);
    };

    const onClick = event => {
        const trigger = event.target instanceof parentWindow.Element
            ? event.target.closest(".flow-info-trigger")
            : null;
        if (trigger) {
            event.preventDefault();
            event.stopPropagation();
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
    };

    const onKeyDown = event => {
        if (event.key === "Escape" && activeCard) {
            event.preventDefault();
            hideTooltip();
        }
    };

    const onScroll = () => {
        if (activeCard) hideTooltip();
    };

    const onResize = () => {
        if (activeCard) positionTooltip(activeCard);
    };

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
