"use strict";
// Execute the real dashboard logic against fixed A/B contracts without a DOM
// framework. This catches mixed-source rendering/filter regressions at CI time.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const fixture = {
  default_period: "MIDWEEK",
  meta: {
    midweek_baseline_available: true, midweek_available: true,
    midweek_snapshot_date: "2026-10-14", complete_snapshot_date: "2026-10-09",
    rs_reference: {},
  },
  views: { weekend: { rows: [] }, midweek: { rows: [] } },
};
const script = fs.readFileSync("dashboard/app.js", "utf8");
const marker = "  boot();\n})();";
assert(script.includes(marker));
const hooked = script.replace(marker, `
  data = __fixture;
  state = initialState();
  globalThis.__reviewAudit = {
    getState: () => state, displayStatus, displayChange, reviewUsesB,
    reviewReferencePrice, reviewDistance, selectedHtml, mobileDetailHtml,
    filterRows, filterCounts, tableHtml, industryHtml, headerHtml,
  };
})();`);
const scope = {
  __fixture: fixture,
  document: { getElementById: () => ({ addEventListener() {} }) },
  window: {
    matchMedia: () => ({ matches: true }),
    BFIndustryRS: {
      status: () => "ready",
      industryForCode: () => "Software",
      industryRS: () => 92,
      stockRS: (code) => ({ ABC: 99, DEF: 95 })[code] ?? null,
    },
  },
  console,
};
vm.runInNewContext(hooked, scope, { filename: "app.js" });
const app = scope.__reviewAudit;
const state = app.getState();

function entry(code) {
  return {
    code, signal: true, review_watch_active: true, industry: "Software",
    ibd_candidate_rule: "ceiling", ibd_candidate_price: 170,
    ibd_entry_status: "ACTIONABLE", ibd_entry_valid: true,
    latest_close: 175, current_vs_ibd_candidate_pct: 2.94,
    review_a_active: true, review_a_price: 170, review_a_zone: "ZONE",
    review_a_as_of: "2026-10-14", review_a_path: "EXT → ZONE",
    review_membership: "IN",
    review_b_active: true, review_b_price: 185, review_b_close: 180,
    review_b_pct: -2.70, review_b_zone: "BELOW", review_b_rule: "pivot",
    review_b_date: "2026-10-14", review_b_status: "UNCONFIRMED",
    review_b_event: "NEW BP", review_b_quality: "Constructive",
    review_b_entry_volume: null, review_b_entry_reason: "Volume not confirmed",
    review_b_details: {
      base_depth_pct: -12, base_duration_weeks: 11,
      ceiling: 181, ceiling_date: "2026-10-02",
      breakout_date: "2026-10-12", ibd_entry_close_position: 0.8,
      ibd_entry_breakout_range_ratio: 0.15, pullback_pct: -3,
      pullback_duration_weeks: 2, eps_yoy_growth: 40,
      dist_to_52w_high_pct: -1.3, volume_ratio: 1.5,
    },
    ibd_entry_close_position: 0.35, base_depth_pct: -22,
    base_duration_weeks: 22, ceiling: 160, ceiling_date: "2026-08-01",
    breakout_date: "2026-08-03", pullback_pct: -10,
    pullback_duration_weeks: 3, eps_yoy_growth: 25,
    dist_to_52w_high_pct: -5, volume_ratio: 0.7,
    ibd_breakout_quality: "Marginal",
    review_new_buy_point: true, review_new_buy_point_zone: "BELOW_BUY_POINT",
    review_has_value_change: true, review_a_has_value_change: false,
    review_b_has_value_change: true, review_weekend_buy_point_change: "UNCHANGED",
    review_buy_point_change_label: "", review_change_label: "",
  };
}
const bOnly = entry("ABC");
const aChanged = entry("DEF");
aChanged.review_a_has_value_change = true;
aChanged.review_a_event = "BELOW → ZONE";
aChanged.review_buy_point_change_label = "BELOW → ZONE";
aChanged.review_weekend_buy_point_change = "ENTERED_BUY_ZONE";
fixture.views.midweek.rows = [bOnly, aChanged];

assert.equal(app.reviewUsesB(bOnly), true, "B-only delta must select B facts");
assert.equal(app.reviewReferencePrice(bOnly), 185);
assert.equal(app.reviewDistance(bOnly), -2.7);
assert.equal(app.displayStatus(bOnly), "UNCONFIRMED");
assert.equal(app.displayChange(bOnly), "B · NEW BP");
assert.equal(app.reviewUsesB(aChanged), false, "A delta keeps fixed A reference");
assert.equal(app.reviewReferencePrice(aChanged), 170);
assert.equal(app.displayStatus(aChanged), "ACTIONABLE");
assert(app.displayChange(aChanged).includes("B · NEW BP"));

const bDetail = app.selectedHtml(bOnly);
assert(bDetail.includes("Midweek BP 185"));
assert(bDetail.includes("Snapshot 2026-10-14"));
assert(bDetail.includes("181"), "B ceiling, not A ceiling");
assert(bDetail.includes("11W"), "B base duration");
assert(!bDetail.includes("22W"), "A base must not leak into B context");
assert(bDetail.includes("↳ Weekend BP 170"));
const mobile = app.mobileDetailHtml(bOnly);
assert(mobile.includes("Midweek BP"));
assert(mobile.includes("pos 0.80"), "B geometry");
assert(mobile.includes("181"), "B base facts");
assert(!mobile.includes("pos 0.35"), "A geometry must not leak");

const industry = app.industryHtml([bOnly]);
assert(industry.includes("UNCONFIRMED"));
assert(industry.includes("-2.70%"));
const stock = app.tableHtml([bOnly]);
assert(stock.includes("B · NEW BP"));
assert(stock.includes("UNCONFIRMED"));

state.newBuyPoint = "BELOW_BUY_POINT";
assert.equal(app.reviewUsesB(aChanged), true, "B quick filter selects B facts");
assert.equal(app.reviewReferencePrice(aChanged), 185);
let counts = app.filterCounts([bOnly, aChanged]);
assert.equal(counts.change.ENTERED_BUY_ZONE, 1, "A quick counts stay independent");
assert.equal(counts.newBuyPoint.BELOW_BUY_POINT, 2, "B quick counts stay independent");
state.newBuyPoint = "ALL";
state.change = "ENTERED_BUY_ZONE";
counts = app.filterCounts([bOnly, aChanged]);
assert.equal(counts.newBuyPoint.BELOW_BUY_POINT, 2, "switching A quick group must not hide B counts");

state.scope = "ALL_SIGNALS";
assert.equal(app.reviewUsesB(bOnly), false, "All Review keeps the A primary");
assert.equal(app.reviewReferencePrice(bOnly), 170);
assert.equal(app.displayStatus(bOnly), "ACTIONABLE");

console.log("BF Review A/B interaction contract passed");
