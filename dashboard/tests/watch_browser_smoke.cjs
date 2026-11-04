"use strict";
// Run against real generated HTML in Chrome, with isolated adversarial payloads.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { chromium } = require("playwright");

const port = 8766;
const site = path.resolve("_site");
const server = spawn("python", ["-m", "http.server", String(port), "--directory", site], {
  stdio: "ignore",
});
const probe = async () => {
  for (let i = 0; i < 80; i++) {
    try {
      const socket = net.createConnection(port, "127.0.0.1");
      await new Promise((resolve, reject) => {
        socket.once("connect", resolve);
        socket.once("error", reject);
      });
      socket.destroy();
      return;
    } catch {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  throw new Error("Watch browser smoke local server never started");
};
const escaped = "<img src=x onerror=window.__xss=1>";
const baseRow = (code, reason, overrides = {}) => ({
  code, snapshot_date: "2026-10-30", original_ceiling: 100,
  current_ceiling: 110, last_exit_reason: reason,
  phase: "OBSERVING", price_state: "BELOW_ORIGINAL",
  vs_original_pct: -4.21, latest_close: 95.79,
  pending_status: "UNVERIFIED", pending_high: "",
  events: [{ id: "POOL_EXIT:2026-10-26", type: "POOL_EXIT", week: "2026-10-26", reason }],
  ...overrides,
});
const complete = [
  baseRow("TESTX", "NO_CEILING", { events: [{type:"POOL_EXIT", week:"2026-10-26", reason:escaped}], latest_close:escaped }),
  baseRow("OTHER", "MA"),
  baseRow("CEIL", "CEILING", { phase: "RETURNED", price_state: "ABOVE_ORIGINAL", vs_original_pct: 2.34 }),
];

(async () => {
  await probe();
  const executablePath = ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"].find(fs.existsSync);
  const browser = await chromium.launch(executablePath ? {
    executablePath, headless: true, args: ["--no-sandbox"],
  } : { channel: "chrome", headless: true });
  try {
    for (const width of [1360, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 844 } });
      const errors = [];
      page.on("pageerror", error => errors.push(String(error)));
      await page.route("**/data/pool_exit_watch.json", route => route.fulfill({
        status: 200, contentType: "application/json",
        body: JSON.stringify({
          available: true, complete, midweek: [baseRow("OLD", "MA")],
          dates: { complete: "2026-10-30", midweek: "2026-10-28" },
          midweek_fallback: true,
        }),
      }));
      await page.goto("http://127.0.0.1:8766/pool-exit-watch.html", { waitUntil: "networkidle" });
      const body = page.locator("body");
      await page.locator("#rows tr").first().waitFor();
      assert.equal(await page.locator("#rows tr").count(), 3, "all complete observations visible");
      assert((await page.locator("#stats").innerText()).includes("NO_CEILING"), "dedicated NO_CEILING counter");
      assert.equal(await page.evaluate(() => window.__xss), undefined, "malicious text did not execute");
      await page.locator("#rows details").first().evaluate(node => node.open = true);
      assert((await page.locator("#rows tr").first().innerText()).includes(escaped), "malicious source escaped but readable");
      assert.equal(await page.locator("#rows tr img").count(), 0, "malicious markup must never materialize");
      await page.locator("#reason").selectOption("NO_CEILING");
      assert.equal(await page.locator("#rows tr").count(), 1, "NO_CEILING filter");
      assert((await page.locator("#rows tr").first().innerText()).includes("TESTX"));

      await page.locator("#reason").selectOption("ALL");
      await page.locator("#period").selectOption("midweek");
      assert((await page.locator("#asof").innerText()).includes("Midweek unavailable/stale"),
        "stale Midweek must be labeled as Complete fallback");
      assert.equal(await page.locator("#rows tr").count(), 3, "fallback preserves Complete observations");
      assert(!(await body.innerText()).includes("OLD"), "stale Midweek code excluded");
      assert.equal(errors.length, 0, "no browser page errors");
      const dims = await page.evaluate(() => ({ pageWidth: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert(dims.pageWidth <= dims.viewport + 1,
        "Watch table causes page-wide overflow at " + width + "px: " + JSON.stringify(dims));
      await page.close();
    }
    process.stdout.write("Watch browser smoke: desktop, narrow mobile, XSS, filters, fallback OK\n");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => server.kill());
