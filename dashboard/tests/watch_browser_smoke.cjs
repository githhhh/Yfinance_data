"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const {spawn} = require("node:child_process");
const {chromium} = require("playwright");

const port = 8766;
const site = path.resolve("_site");
const server = spawn("python", ["-m", "http.server", String(port), "--directory", site], {stdio:"ignore"});
const malicious = "<img src=x onerror=window.__xss=1>";
const snap = "2026-11-06";
const event = (type, week, extra={}) => ({id:type+":"+week, type, week, ...extra});
const row = (code, overrides={}) => ({
  code, snapshot_date:snap, original_ceiling:100, current_ceiling:110,
  breakout_date:"2026-09-07", absence_week:"2026-10-26",
  current_in_pool:false, last_exit_reason:"CEILING", phase:"OBSERVING",
  price_state:"BELOW_ORIGINAL", vs_original_pct:-3, latest_close:97,
  pending_high:"", pending_status:"NONE",
  events:[event("POOL_EXIT", "2026-10-26", {reason:"CEILING"})],
  ...overrides
});
const rows = [
  row("CONFIRM", {
    pending_status:"CONFIRMED", pending_high:112, latest_close:114,
    price_state:"ABOVE_ORIGINAL", vs_original_pct:14,
    events:[event("CEILING_LOST", "2026-10-26"),event("ZONE_TOUCH","2026-10-26"),
      event("PENDING_CONFIRMED","2026-11-02",{pending_high:112})]
  }),
  row("RECLAIM", {
    pending_status:"PENDING", pending_high:118, latest_close:102,
    price_state:"ABOVE_ORIGINAL", vs_original_pct:2,
    events:[event("CEILING_LOST","2026-10-26"),event("CEILING_RECLAIMED","2026-11-02")]
  }),
  row("DUAL", {
    pending_status:"CONFIRMED", pending_high:120, latest_close:122,
    price_state:"ABOVE_ORIGINAL", vs_original_pct:22,
    events:[event("CEILING_LOST","2026-10-26"),event("PENDING_CONFIRMED","2026-11-02"),
      event("CEILING_RECLAIMED","2026-11-02")]
  }),
  row("BLOCKED", {
    pending_status:"PENDING_BLOCKED", pending_high:130,
    events:[event("CEILING_LOST","2026-11-02"),
      event("PENDING_BLOCKED","2026-11-02",{historical_confirmation:"2026-09-28"})]
  }),
  row("RETURNED", {
    phase:"RETURNED", current_in_pool:true, pending_status:"PENDING",
    price_state:"ABOVE_ORIGINAL", vs_original_pct:1, latest_close:101,
    pending_high:124,
    events:[event("POOL_EXIT","2026-10-26"),event("POOL_RETURN","2026-11-02")]
  }),
  row("TESTX", {
    last_exit_reason:"NO_CEILING", pending_status:"NONE",
    events:[event("POOL_EXIT","2026-10-26",{reason:malicious})]
  }),
  row("GAP", {
    last_exit_reason:"UNVERIFIED", pending_status:"UNVERIFIED",
    events:[event("CEILING_LOST","2026-11-02",{censored_by_gap:true})]
  })
];
const waitForServer = async () => {
  for(let i=0;i<80;i++){
    try {
      const socket=net.createConnection(port,"127.0.0.1");
      await new Promise((resolve,reject)=>{socket.once("connect",resolve);socket.once("error",reject)});
      socket.destroy();return;
    }catch{await new Promise(resolve=>setTimeout(resolve,100))}
  }
  throw Error("Watch local site did not start");
};
(async () => {
  await waitForServer();
  const executablePath=["/usr/bin/google-chrome","/usr/bin/chromium","/usr/bin/chromium-browser"].find(fs.existsSync);
  const browser=await chromium.launch(executablePath?{executablePath,headless:true,args:["--no-sandbox"]}:{channel:"chrome",headless:true});
  try {
    for(const width of [1680,1360,1024,760,430,390,350,320]){
      const page=await browser.newPage({viewport:{width,height:844}});
      const errors=[];page.on("pageerror",error=>errors.push(String(error)));
      await page.route("**/data/pool_exit_watch.json",route=>route.fulfill({
        status:200,contentType:"application/json",
        body:JSON.stringify({available:true,complete:rows,midweek:[row("OLD")],
          dates:{complete:snap,midweek:"2026-10-28"},midweek_fallback:true})
      }));
      await page.goto("http://127.0.0.1:8766/pool-exit-watch.html",{waitUntil:"networkidle"});
      assert.equal(await page.locator("#rows tr[data-code]").count(),7);
      assert.equal(await page.locator("#confirmed-count").innerText(),"2");
      assert.equal(await page.locator("#reclaimed-count").innerText(),"2");
      assert((await page.locator("#rows tr").first().innerText()).includes("CONFIRM"),
        "Verified event comes before ordinary Pending");
      const confirm=page.locator('#rows tr[data-code="CONFIRM"]');
      assert((await confirm.locator("td").nth(3).textContent()).includes("1W"),
        "Lost → Confirm uses verified calendar weeks, even when mobile hides secondary columns");
      assert(!(await page.locator('#rows tr[data-code="BLOCKED"]').innerText()).includes("Pending Confirmed"),
        "Historical blocked confirmation is not a new event");
      assert((await page.locator('#rows tr[data-code="RETURNED"]').innerText()).includes("Returned"),
        "Returned remains a lifecycle fact");
      const before=await confirm.boundingBox();
      await confirm.click();
      const after=await confirm.boundingBox();
      assert(before&&after&&Math.abs(before.height-after.height)<=1,
        "Selecting a row must not inflate the source table row");
      const details=width<=760?page.locator(".watch-detail-row"):page.locator("#selected");
      assert((await details.innerText()).includes("Lost → Confirm: 1 calendar week"));
      assert((await details.innerText()).includes("Original Ceiling"));
      assert((await details.innerText()).includes("Breakout · 2026-09-07"));
      await page.locator('#rows tr[data-code="RECLAIM"]').click();
      const reclaimDetails=width<=760?page.locator(".watch-detail-row"):page.locator("#selected");
      assert((await reclaimDetails.innerText()).includes("Since observed Lost: 1 calendar week"),
        "Unconfirmed observation shows age since the verified Lost, not time to an invented confirmation");
      await page.locator('[data-event-filter="PENDING_CONFIRMED"]').click();
      assert.equal(await page.locator("#rows tr[data-code]").count(),2);
      assert.equal(await page.locator("#confirmed").getAttribute("aria-pressed"),"true");
      await page.locator('[data-event-filter="CEILING_RECLAIMED"]').click();
      assert.equal(await page.locator("#rows tr[data-code]").count(),2);
      assert.equal(await page.locator("#reclaimed").getAttribute("aria-pressed"),"true");
      await page.locator("#clear").click();
      await page.locator("#reason").selectOption("NO_CEILING");
      assert.equal(await page.locator("#rows tr[data-code]").count(),1);
      await page.locator('#rows tr[data-code="TESTX"]').click();
      const text=await (width<=760?page.locator(".watch-detail-row"):page.locator("#selected")).innerText();
      assert(text.includes(malicious),"HTML payload is escaped and readable in event timeline");
      assert.equal(await page.locator("img").count(),0,"Untrusted Watch event cannot materialize markup");
      assert.equal(await page.evaluate(()=>window.__xss),undefined);
      await page.locator("#reason").selectOption("ALL");
      await page.locator("#list-shell").focus();
      await page.keyboard.press("ArrowDown");
      assert.equal(await page.locator("#rows tr.is-selected").count(),1,"ArrowDown selects a visible ticker");
      await page.locator("#period").selectOption("midweek");
      assert((await page.locator("#asof").innerText()).includes("Midweek unavailable/stale"),
        "Stale Midweek is marked as Complete fallback");
      assert.equal(await page.locator("#rows tr[data-code]").count(),7);
      assert.equal(await page.locator('#rows tr[data-code="OLD"]').count(),0);
      const visibleColumns=await page.locator(".watch-table thead th").evaluateAll(nodes=>nodes.filter(n=>getComputedStyle(n).display!=="none").length);
      assert.equal(visibleColumns,width>=521?5:width>=360?4:3,
        "Watch must allocate widths before removing columns at "+width+"px");
      const dims=await page.evaluate(()=>{
        const shell=document.getElementById("list-shell");
        const table=document.querySelector(".watch-table");
        return {body:document.documentElement.scrollWidth,viewport:innerWidth,
          shellWidth:shell.clientWidth,shellScroll:shell.scrollWidth,
          tableWidth:table.getBoundingClientRect().width,
          offenders:Array.from(document.querySelectorAll("body *")).filter(node=>{
            const b=node.getBoundingClientRect();return b.width>0&&b.right>innerWidth+2;
          }).slice(0,8).map(node=>node.tagName.toLowerCase()+"."+node.className)};
      });
      assert(dims.body<=dims.viewport+1,"Watch page-wide overflow at "+width+"px: "+JSON.stringify(dims));
      assert(dims.shellScroll<=dims.shellWidth+1,"Watch table scrolls sideways at "+width+"px: "+JSON.stringify(dims));
      assert.equal(errors.length,0,"Browser errors: "+errors.join(","));
      await page.close();
    }
    console.log("Watch event-first 1680/1360/1024/760/430/390/350/320 browser smoke passed");
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1}).finally(()=>server.kill());
