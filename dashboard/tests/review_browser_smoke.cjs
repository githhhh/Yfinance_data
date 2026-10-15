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
const aOnly = row("AWEEK", {
  review_membership: "OUT", review_a_has_value_change: true,
  review_has_value_change: true, review_a_event: "BELOW → ZONE",
  review_change_label: "BELOW → ZONE · OUT",
  review_buy_point_change_label: "BELOW → ZONE · OUT",
  review_weekend_buy_point_change: "ENTERED_BUY_ZONE",
  review_buy_point_priority: 0,
});
const near = row("NEAR", {
  signal: false, review_watch_active: false,
  bf_watch_active: true, review_a_active: false, review_b_active: false,
  bf_watch_type: "pivot", bf_watch_trigger_price: 180, ibd_entry_status: null,
});
payload.default_period = "MIDWEEK";
payload.meta.midweek_available = true;
payload.meta.midweek_baseline_available = true;
payload.views.midweek.rows = [aOnly, bOnly, near];

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
    for (const viewport of [{ width: 1360, height: 900 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
      page.on("pageerror", error => { throw error; });
      await page.route("**/data/dashboard.json", route => route.fulfill({
        status: 200, contentType: "application/json", body: JSON.stringify(payload),
      }));
      await page.goto("http://127.0.0.1:8765/", { waitUntil: "domcontentloaded" });
      await page.locator(".review-section").waitFor();
      assert.equal(await page.locator(".review-table tbody tr[data-code]").count(), 2, "Review Now excludes Near");
      assert(await page.locator('[data-action="status"][data-value="NEAR_BREAKOUT"]').isDisabled());

      const outRow = page.locator('.review-table tbody tr[data-code="AWEEK"]');
      assert((await outRow.innerText()).includes("OUT"), "OUT must be visible without expanding");
      const bRow = page.locator('.review-table tbody tr[data-code="BONLY"]');
      assert((await bRow.innerText()).includes("UNCONFIRMED"), "B-only event uses B status");
      assert((await bRow.innerText()).includes("-2.70%"), "B-only event uses B reference");
      await bRow.click();
      if (viewport.width <= 760) {
        const detail = page.locator(".mobile-inline-review");
        await detail.waitFor();
        const mobileText = await detail.innerText();
        assert(mobileText.toUpperCase().includes("MIDWEEK BP"), "Mobile B label");
        assert(!mobileText.includes("4W"), "Missing B pullback must not borrow A history");
        assert((await detail.innerText()).includes("11W"), "Mobile B base, not A");
      } else {
        const selected = page.locator(".selected-strip");
        assert((await selected.innerText()).includes("Midweek BP 185"), "Desktop B label");
        assert((await selected.innerText()).includes("11W"), "Desktop B base, not A");
      }

      await page.locator('[data-action="scope"][data-value="ALL_SIGNALS"]').click();
      assert.equal(await page.locator(".review-table tbody tr[data-code]").count(), 3, "All Review includes Near");
      // Scope changes retain selection; clicking the same mobile row again
      // would intentionally toggle the selection off.
      const selected = await page.locator(".selected-strip").innerText();
      assert(selected.includes("Weekend BP 170"), "All Review restores A primary");

      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
      assert(!overflow, "No body-level horizontal overflow at " + viewport.width);
      await page.evaluate(() => {
        window.BFIndustryRS = {
          status: () => "ready", industryForCode: () => "Software",
          industryRS: () => 92, stockRS: code => ({ BONLY: 99, AWEEK: 96, NEAR: 90 })[code],
        };
      });
      await page.locator('[data-action="toggle-view"]').click();
      const toggle = page.locator(".industry-group-toggle").first();
      await toggle.waitFor();
      await toggle.click();
      assert.equal(await toggle.getAttribute("aria-expanded"), "true");
      assert(await page.locator('.industry-stock-table tbody tr[data-code]').count() >= 2);
      const after = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
      assert(!after, "Industry view overflow at " + viewport.width);
      await page.close();
    }
  } finally { await browser.close(); }
  console.log("Desktop + 390px mobile browser smoke passed");
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => server.kill());
