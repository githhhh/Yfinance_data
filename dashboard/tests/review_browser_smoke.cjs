"use strict";
// Real Chrome smoke with synthetic A/B data over the generated static site.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const fs = require("node:fs");
const { spawn } = require("node:child_process");
const path = require("node:path");
const net = require("node:net");

const port = 8765;
const root = path.resolve("_site");
const payload = JSON.parse(fs.readFileSync(path.join(root, "data/dashboard.json"), "utf8"));
const original = payload.views.midweek.rows[0] || payload.views.weekend.rows[0] || {};
function row(code, rest) {
  return { ...original, code, signal: true, review_watch_active: true,
    review_membership: "IN", bf_watch_active: false,
    ibd_entry_status: "ACTIONABLE", ibd_candidate_rule: "ceiling",
    ibd_candidate_price: 170, latest_close: 175,
    current_vs_ibd_candidate_pct: 2.94, base_duration_weeks: 22,
    base_depth_pct: -22, ceiling: 160,
    review_a_active: true, review_a_price: 170, review_a_pct: 2.94,
    review_a_zone: "ZONE", review_a_as_of: "2026-10-14",
    review_a_has_value_change: false, review_b_has_value_change: false,
    review_b_active: false, review_has_value_change: false,
    review_weekend_buy_point_change: "UNCHANGED", review_new_buy_point: false,
    review_new_buy_point_zone: null, review_b_event: "",
    review_change_label: "", review_buy_point_change_label: "",
    review_buy_point_priority: 9, ...rest };
}
const bOnly = row("BONLY", {
  review_b_active: true, review_b_price: 185, review_b_status: "UNCONFIRMED",
  review_b_rule: "pivot", review_b_close: 180, review_b_pct: -2.70,
  review_b_zone: "BELOW", review_b_event: "NEW BP",
  review_b_date: "2026-10-14", review_b_entry_reason: "Volume not confirmed",
  review_b_details: {
    base_duration_weeks: 11, base_depth_pct: -12, ceiling: 181,
    ceiling_date: "2026-10-02", breakout_date: "2026-10-12",
    ibd_entry_close_position: 0.8, ibd_entry_breakout_range_ratio: 0.15,
    eps_yoy_growth: 40, dist_to_52w_high_pct: -1.3, volume_ratio: 1.5,
  },
  review_new_buy_point: true, review_new_buy_point_zone: "BELOW_BUY_POINT",
  review_has_value_change: true, review_b_has_value_change: true,
});
const returned = row("RETURNED", {
  review_selected_price_event: "BP CHANGED",
  review_a_active: true, review_b_active: false,
  review_a_has_value_change: false, review_b_has_value_change: false,
  review_has_value_change: true, review_a_event: "", review_a_week_event: "",
  review_weekend_buy_point_change: "UNCHANGED",
});
const aOnly = row("AWEEK", {
  ibd_entry_status: "UNCONFIRMED",
  review_membership: "OUT", review_a_has_value_change: true,
  review_has_value_change: true, review_a_event: "BELOW → ZONE",
  review_a_week_event: "EXT → ZONE",
  review_change_label: "BELOW → ZONE · OUT",
  review_buy_point_change_label: "BELOW → ZONE · OUT",
  review_weekend_buy_point_change: "ENTERED_BUY_ZONE",
  review_buy_point_priority: 0,
});
const belowOut = row("BELOWOUT", {
  ibd_entry_status: "BELOW_TRIGGER",
  review_membership: "OUT", review_a_active: true,
  review_a_has_value_change: true, review_has_value_change: true,
  review_a_event: "ZONE → BELOW",
  review_weekend_buy_point_change: "BELOW_BUY_POINT",
  review_buy_point_change_label: "ZONE → BELOW",
  review_buy_point_priority: 1,
});
const dual = row("DUAL", {
  review_membership: "OUT", review_a_active: true,
  review_a_event: "ZONE → EXT", review_a_has_value_change: true,
  review_b_active: true, review_b_status: "ACTIONABLE", review_b_entry_valid: true,
  review_b_display_event: "UNCONFIRMED → ACTIONABLE", review_b_has_value_change: true,
  review_b_price: 171, review_b_pct: 2.34, review_b_close: 175,
  review_has_value_change: true, review_weekend_buy_point_change: "BECAME_EXTENDED",
});
const poolOnly = row("POOLONLY", {
  ibd_entry_status: "EXTENDED", review_a_active: true,
  review_membership: "OUT", review_membership_event: "IN → OUT",
  review_a_event: "", review_a_week_event: "ZONE → EXT",
  review_a_has_value_change: true, review_has_value_change: true,
  review_weekend_buy_point_change: "BECAME_EXTENDED",
  review_b_active: true, review_b_entry_valid: true, review_b_status: "ACTIONABLE",
  review_b_display_event: "CURRENT", review_b_has_value_change: false,
});
const near = row("NEAR", {
  signal: false, review_watch_active: false,
  bf_watch_active: true, review_a_active: false, review_b_active: false,
  bf_watch_type: "pivot", bf_watch_trigger_price: 180, ibd_entry_status: null,
});
payload.default_period = "MIDWEEK";
payload.meta.midweek_available = true;
payload.meta.midweek_baseline_available = true;
payload.views.midweek.rows = [aOnly, belowOut, bOnly, dual, poolOnly, returned, near];

const server = spawn("python", ["-m", "http.server", String(port), "--directory", root], { stdio: "ignore" });
async function ready() {
  for (let i = 0; i < 80; i++) {
    try {
      const c = net.connect(port, "127.0.0.1");
      await new Promise((resolve, reject) => { c.once("connect", resolve); c.once("error", reject); });
      c.destroy(); return;
    } catch (_) { await new Promise(r => setTimeout(r, 100)); }
  }
  throw new Error("Local static server failed to start");
}
(async () => {
  await ready();
  const executablePath = ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]
    .find(p => fs.existsSync(p));
  const browser = await chromium.launch(executablePath ? { executablePath, headless: true, args: ["--no-sandbox"] } : { channel: "chrome", headless: true });
  try {
    for (const viewport of [
      { width: 1680, height: 960 }, { width: 1360, height: 900 },
      { width: 1024, height: 900 }, { width: 760, height: 844 },
      { width: 430, height: 844 }, { width: 390, height: 844 },
      { width: 350, height: 844 }, { width: 320, height: 844 },
    ]) {
      const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
      page.on("pageerror", error => { throw error; });
      await page.route("**/data/dashboard.json", route => route.fulfill({
        status: 200, contentType: "application/json", body: JSON.stringify(payload),
      }));
      await page.goto("http://127.0.0.1:8765/", { waitUntil: "domcontentloaded" });
      await page.locator(".review-section").waitFor();
      assert.equal(await page.locator(".review-table tbody tr[data-code]").count(), 6, "Changes scope excludes Near");
      assert(await page.locator('[data-action="status"][data-value="NEAR_BREAKOUT"]').isDisabled());

      for (const outCode of ["AWEEK", "BELOWOUT"]) {
        const outRow = page.locator(`.review-table tbody tr[data-code="${outCode}"]`);
        assert((await outRow.innerText()).includes("OUT"), "OUT must be visible without expanding");
        const statusCell = outRow.locator('td[data-field="ibd_entry_status"]');
        const statusBox = await statusCell.boundingBox();
        const statusRect = await statusCell.locator(".status-stack .status-text").boundingBox();
        const outRect = await statusCell.locator(".status-stack .pool-out-indicator").boundingBox();
        assert(statusBox && statusRect && outRect, "Status and OUT must be visible");
        assert(outRect.y >= statusRect.y + statusRect.height - 1,
          `${outCode} OUT must appear on the second line (${viewport.width}px)`);
        assert(Math.abs((outRect.x + outRect.width / 2) - (statusRect.x + statusRect.width / 2)) <= 2,
          `${outCode} OUT must be centered below status (${viewport.width}px)`);
        assert(statusRect.x >= statusBox.x - 1 &&
          statusRect.x + statusRect.width <= statusBox.x + statusBox.width + 1,
          `${outCode} status must not be horizontally clipped (${viewport.width}px)`);
        assert(outRect.y + outRect.height <= statusBox.y + statusBox.height + 1,
          `${outCode} OUT must fit inside the existing row height (${viewport.width}px)`);
      }
      const returnedRow = page.locator('.review-table tbody tr[data-code="RETURNED"]');
      assert((await returnedRow.locator('td[data-field="review_change_label"]').innerText()).includes("TO WEEKEND"),
        "A return from B to the immutable Weekend price is a visible one-day event");
      if (viewport.width <= 760) {
        assert.equal((await returnedRow.locator(".mobile-change-hint").innerText()).trim(), "To Weekend");
      }
      const poolRow = page.locator('.review-table tbody tr[data-code="POOLONLY"]');
      const poolChange = poolRow.locator('td[data-field="review_change_label"]');
      assert((await poolChange.innerText()).includes("POOL OUT"),
        "Review Now reveals membership-only event, not older Weekend history");
      assert(!(await poolChange.innerText()).includes("ZONE → EXT"));
      if (viewport.width <= 760) {
        assert.equal((await poolRow.locator(".mobile-change-hint").innerText()).trim(), "Pool OUT",
          "Mobile Ticker hints and expanded Change must share event provenance");
        await poolRow.click();
        const poolDetail = page.locator(".mobile-detail-row .mobile-event-breakdown");
        assert((await poolDetail.innerText()).includes("POOL OUT"));
      }
      await page.locator('[data-action="status"][data-value="ACTIONABLE"]').click();
      assert.equal(await page.locator('.review-table tbody tr[data-code="POOLONLY"]').count(), 0,
        "Unchanged Midweek ACTIONABLE cannot pull changed Weekend EXTENDED into Review Now");
      await page.locator('[data-action="status"][data-value="ACTIONABLE"]').click();
      const dualRow = page.locator('.review-table tbody tr[data-code="DUAL"]');
      if (viewport.width > 760) {
        const changeCell = dualRow.locator('td[data-field="review_change_label"]');
        const changeBounds = await changeCell.boundingBox();
        const dualLines = changeCell.locator(".change-line");
        assert.equal(await dualLines.count(), 2, "Two sources produce exactly two rows");
        assert(changeBounds, "Dual-event Change cell exists");
        for (let i = 0; i < 2; i++) {
          const line = await dualLines.nth(i).boundingBox();
          assert(line && line.y >= changeBounds.y - 1 &&
            line.y + line.height <= changeBounds.y + changeBounds.height + 1,
            "Dual Change line must stay within its compact row");
          assert(line.x + line.width <= changeBounds.x + changeBounds.width + 1,
            "Dual Change line must not spill into Stage or neighboring cells");
        }
        const rowBounds = await dualRow.boundingBox();
        assert(rowBounds && rowBounds.height <= 72,
          `Dual-event + OUT may grow only enough to show all three facts (${rowBounds?.height}px)`);
        const status = dualRow.locator('td[data-field="ibd_entry_status"]');
        const statusBounds = await status.boundingBox();
        const out = await status.locator(".pool-out-indicator").boundingBox();
        assert(statusBounds && out && out.y + out.height <= statusBounds.y + statusBounds.height + 1,
          "Dual-BP + OUT stack must fit the compact status cell");
      } else {
        const active = dualRow.locator(".bp-status-line.is-active .status-text");
        assert.equal(await active.count(), 1, "Mobile dual-BP row marks one active status");
        assert(await active.isVisible(), "Mobile dual-BP status is visible");
      }
      if (viewport.width > 760) {
        const empty = page.locator(".selected-strip.empty");
        const emptyBox = await empty.boundingBox();
        assert(emptyBox && emptyBox.height <= 44,
          `Empty Selected Overview must be compact (${emptyBox?.height}px)`);
      }
      const bRow = page.locator('.review-table tbody tr[data-code="BONLY"]');
      if (viewport.width > 760) {
        const mainChange = bRow.locator('td[data-field="review_change_label"]');
        assert((await mainChange.locator(".change-origin").innerText()) === "Midweek");
        assert((await mainChange.locator(".change-event").innerText()) === "NEW");
        assert(await mainChange.locator(".change-basis").isHidden(),
          "Historical comparison belongs to hover or details, not the table");
        const rect = await bRow.boundingBox();
        assert(rect && rect.height <= 48, `Single-event row must stay compact at ${viewport.width}px (was ${rect?.height}px)`);
      }
      assert((await bRow.innerText()).includes("UNCONFIRMED"), "B-only event uses B status");
      assert((await bRow.innerText()).includes("-2.70%"), "B-only event uses B reference");
      await bRow.click();
      if (viewport.width <= 760) {
        const detail = page.locator(".mobile-inline-review");
        await detail.waitFor();
        const mobileText = await detail.innerText();
        assert(mobileText.toUpperCase().includes("MIDWEEK BUY POINT"), "Mobile B label");
        assert(!mobileText.includes("4W"), "Missing B pullback must not borrow A history");
        assert((await detail.innerText()).includes("11W"), "Mobile B base, not A");
      } else {
        const selected = page.locator(".selected-strip");
        assert((await selected.innerText()).includes("Midweek Buy Point 185"), "Desktop B label");
        assert((await selected.innerText()).includes("11W"), "Desktop B base, not A");
      }

      await page.locator('[data-action="scope"][data-value="ALL_SIGNALS"]').click();
      assert.equal(await page.locator(".review-table tbody tr[data-code]").count(), 7, "All Review includes Near");
      // Scope changes retain selection; clicking the same mobile row again
      // would intentionally toggle the selection off.
      const selected = await page.locator(".selected-strip").innerText();
      assert(selected.includes("Weekend Buy Point 170"), "All Review restores A primary");

      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
      assert(!overflow, "No body-level horizontal overflow at " + viewport.width);
      await page.evaluate(() => {
        window.BFIndustryRS = {
          status: () => "ready", industryForCode: () => "Software",
          industryRS: () => 92, stockRS: code => ({ BONLY: 99, AWEEK: 96, BELOWOUT: 93, NEAR: 90 })[code],
        };
      });
      await page.locator('[data-action="toggle-view"]').click();
      const toggle = page.locator(".industry-group-toggle").first();
      await toggle.waitFor();
      await toggle.click();
      assert.equal(await toggle.getAttribute("aria-expanded"), "true");
      const industryPulse = page.locator(".industry-group-pulse").first();
      assert((await industryPulse.innerText()).includes("Actionable"), "Industry shows verified actionable concentration");
      assert((await industryPulse.innerText()).includes("Changed"), "Industry shows active review concentration");
      assert(await page.locator('.industry-stock-table tbody tr[data-code]').count() >= 2);
      // At <=760px all inactive BP status lines are CSS-hidden. The Industry
      // row must tag exactly one active source, otherwise dual-BP status
      // becomes entirely invisible while the rest of the row is present.
      const dualIndustry = page.locator('.industry-stock-table tbody tr[data-code="BONLY"]');
      const dualStatus = dualIndustry.locator(".bp-status-line.is-active .status-text");
      assert.equal(await dualStatus.count(), 1, "Industry dual-BP row must designate one visible status");
      assert(await dualStatus.isVisible(), `Industry dual-BP status is visible at ${viewport.width}px`);
      for (const outCode of ["AWEEK", "BELOWOUT"]) {
        const industryOut = page.locator(`.industry-stock-table tbody tr[data-code="${outCode}"]`);
        const cell = industryOut.locator("td:nth-child(3)");
        const cellRect = await cell.boundingBox();
        const labelRect = await cell.locator(".status-stack .status-text").boundingBox();
        const outRect = await cell.locator(".status-stack .pool-out-indicator").boundingBox();
        assert(cellRect && labelRect && outRect, "Industry status and OUT must be visible");
        assert(outRect.y >= labelRect.y + labelRect.height - 1,
          `${outCode} Industry OUT must appear below the status (${viewport.width}px)`);
        assert(labelRect.x >= cellRect.x - 1 &&
          labelRect.x + labelRect.width <= cellRect.x + cellRect.width + 1,
          `${outCode} Industry status must not clip (${viewport.width}px)`);
        assert(outRect.y + outRect.height <= cellRect.y + cellRect.height + 1,
          `${outCode} Industry OUT must fit within row height (${viewport.width}px)`);
      }
      const after = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
      assert(!after, "Industry view overflow at " + viewport.width);

      // Re-sort groups so a CLOSED higher-RS industry precedes the expanded
      // group. Arrow navigation must skip its DOM-mounted hidden stock rows.
      await page.evaluate(() => {
        window.BFIndustryRS.industryForCode = code => code === "NEAR" ? "Energy" : "Software";
        window.BFIndustryRS.industryRS = industry => industry === "Energy" ? 99 : 92;
        document.getElementById("app").dispatchEvent(new Event("bf-rs-updated"));
      });
      // The earlier view can have expanded a stale/unclassified group
      // before the synthetic RS source was installed. Expand the actual
      // newly resolved Software group after the reference refresh.
      const softwareToggle = page.locator('.industry-group-toggle[data-industry="Software"]');
      assert.equal(await softwareToggle.count(), 1, "Refreshed Industry groups have their real labels");
      if ((await softwareToggle.getAttribute("aria-expanded")) !== "true") await softwareToggle.click();
      const industryShell = page.locator("[data-industry-list]");
      assert(await industryShell.locator('.industry-group-body:not([hidden]) tbody tr[data-code]').count() > 0,
        "Keyboard test requires visible stock rows in the expanded group");
      assert(await industryShell.locator('.industry-group-body[hidden] tbody tr[data-code]').count() > 0,
        "Keyboard test requires an additional collapsed industry group");
      await industryShell.focus();
      await page.keyboard.press("ArrowUp");
      const lastVisible = page.locator(".industry-stock-table tbody tr.selected");
      assert.equal(await lastVisible.count(), 1, "ArrowUp selects one visible industry ticker");
      assert(await lastVisible.isVisible(), "Keyboard may not select a collapsed group row");
      const originalCode = await lastVisible.getAttribute("data-code");
      await page.keyboard.press("ArrowDown");
      assert.equal(await page.locator(".industry-stock-table tbody tr.selected").getAttribute("data-code"), originalCode,
        "ArrowDown at the last visible ticker cannot deselect the row");
      await page.locator('[data-action="quick"][data-field="change"][data-value="ENTERED_BUY_ZONE"]').click();
      assert((await page.locator(".industry-group-pulse").first().innerText()).includes("1 Vs Weekend"),
        "What Changed industry concentration must match the cumulative Weekend quick group");
      const quickA = page.locator('.industry-stock-table tbody tr[data-code="AWEEK"] td').first();
      assert((await quickA.count()) === 1);
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2)),
        "Weekend quick group cannot cause Industry overflow");
      await page.close();
    }
  } finally { await browser.close(); }
  console.log("1680/1360/1024 desktop + 320/350/390/430/760px browser status-layout smoke passed");
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => server.kill());
