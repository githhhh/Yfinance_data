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
  ui: { status_meta: {} },
};
const script = fs.readFileSync("dashboard/app.js", "utf8");
const marker = "  boot();\n})();";
assert(script.includes(marker));
const hooked = script.replace(marker, `
  data = __fixture;
  state = initialState();
  globalThis.__reviewAudit = {
    getState: () => state, displayStatus, displayChange, reviewUsesB,
    reviewReferencePrice, reviewDistance, selectedHtml, mobileDetailHtml, changeLines, statusCardsHtml,
    filterRows, filterCounts, tableHtml, industryHtml, headerHtml, bounds, sortRows,
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
    review_b_entry_valid: false, review_b_verified: true,
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
assert.equal(app.displayChange(bOnly), "NEW BP");
assert.equal(app.reviewUsesB(aChanged), false, "A delta keeps fixed A reference");
assert.equal(app.reviewReferencePrice(aChanged), 170);
assert.equal(app.displayStatus(aChanged), "ACTIONABLE");
assert(app.displayChange(aChanged).includes("BELOW → ZONE"));

const bDetail = app.selectedHtml(bOnly);
assert(bDetail.includes("Midweek Buy Point 185"));
assert(bDetail.includes("Snapshot 2026-10-14"));
assert(bDetail.includes("181"), "B ceiling, not A ceiling");
assert(bDetail.includes("11W"), "B base duration");
assert(!bDetail.includes("22W"), "A base must not leak into B context");
assert(bDetail.includes("↳ Weekend Buy Point 170"));
assert(bDetail.includes("ACTIONABLE · ZONE"), "Alternate Weekend Buy Point has its own status");
const unconfirmedWithVolume = { ...bOnly, review_b_entry_volume: 2.5,
  review_b_entry_reason: "resolver_rejected", review_b_entry_valid: false };
assert(app.mobileDetailHtml(unconfirmedWithVolume).includes('title="resolver_rejected"'),
  "B rejected resolver reason must remain recoverable in mobile details");
assert(app.tableHtml([unconfirmedWithVolume]).includes('title="resolver_rejected"'),
  "B rejection code remains in tooltip");
assert(app.tableHtml([unconfirmedWithVolume]).includes("Entry Rejected"),
  "B rejection receives concise visible wording");
const mobile = app.mobileDetailHtml(bOnly);
assert(mobile.includes("Midweek Buy Point"));
assert(mobile.includes("pos 0.80"), "B geometry");
assert(mobile.includes("181"), "B base facts");
assert(!mobile.includes("pos 0.35"), "A geometry must not leak");

const industry = app.industryHtml([bOnly]);
assert(industry.includes("UNCONFIRMED"));
assert(industry.includes("-2.70%"));
const out = entry("OUTTICK");
out.review_b_active = false;
out.review_membership = "OUT";
assert(app.tableHtml([out]).includes("pool-out-indicator"), "Stock OUT visible");
assert(app.industryHtml([out]).includes("pool-out-indicator"), "Industry OUT visible");
assert(app.tableHtml([out]).includes('<span class="status-stack">'), "Stock OUT below status");
assert(app.industryHtml([out]).includes('<span class="status-stack">'), "Industry OUT below status");
const stock = app.tableHtml([bOnly]);
assert(stock.includes('>NEW</span>'), "Short NEW event in main table");
assert(stock.includes("UNCONFIRMED"));

state.newBuyPoint = "BELOW_BUY_POINT";
assert.equal(app.reviewUsesB(aChanged), true, "B quick filter selects B facts");
assert.equal(app.reviewReferencePrice(aChanged), 185);
state.weeklyVolumeMin = 1.3;
assert.equal(app.filterRows([bOnly, aChanged]).length, 2,
  "B weekly volume (1.5×) is used in B quick filter, not weekend 0.7×");
assert(app.tableHtml([bOnly]).includes("1.50×"), "B Weekly Vol cell must use 1.5×");
state.weeklyVolumeMin = null;
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
state.weeklyVolumeMin = 1.3;
assert.equal(app.filterRows([bOnly, aChanged]).length, 0,
  "All Review weekly-volume filter must use weekend A 0.7×");
state.weeklyVolumeMin = null;
assert.equal(app.displayStatus(bOnly), "ACTIONABLE");

state.change = "ALL"; state.scope = "CHANGES"; state.status = "ACTIONABLE";
const talo = {...entry("TALO"),ibd_entry_status:"EXTENDED",
  current_vs_ibd_candidate_pct:6.15,review_b_status:"ACTIONABLE",
  review_b_pct:0.73,review_b_zone:"ZONE",review_b_verified:true,
  review_b_entry_valid:true,review_new_buy_point_zone:"BUY_ZONE"};
assert.equal(app.filterRows([talo]).length,1,"TALO qualifies through its Midweek Buy Point");
assert.equal(app.reviewUsesB(talo),true);
assert.equal(app.reviewReferencePrice(talo),185);

state.status = "ALL"; state.newBuyPoint = "BUY_ZONE";
const arqt = {...entry("ARQT"),review_a_active:false,
  review_b_status:"ACTIONABLE",review_b_pct:4.36,review_b_zone:"ZONE",
  review_b_verified:true,review_b_display_event:"CURRENT",
  review_new_buy_point_zone:"BUY_ZONE",review_b_has_value_change:false,
  review_has_value_change:false};
assert.equal(app.filterRows([arqt]).length,1,"Stable Midweek BP is in New Buy Points snapshot");
assert.equal(app.filterCounts([arqt]).newBuyPoint.BUY_ZONE,1);

// Industry aggregates must reflect the *selected* buy point, not the
// other source's hidden status or event.
state.newBuyPoint = "ALL"; state.status = "UNCONFIRMED";
const mixed = {...entry("MIXED"),ibd_entry_status:"UNCONFIRMED",
  review_b_status:"ACTIONABLE",review_b_verified:true,
  review_a_has_value_change:true,review_b_has_value_change:false};
assert(app.industryHtml([mixed]).includes("0 Actionable"),
  "UNCONFIRMED filter cannot count a hidden B ACTIONABLE as a selected opportunity");
state.status = "ALL"; state.newBuyPoint = "BUY_ZONE";
mixed.review_new_buy_point_zone = "BUY_ZONE";
assert(app.industryHtml([mixed]).includes("0 Changed"),
  "New Buy Points context cannot count unrelated A changes");

// An unverified structural identity is not a tradable ACTIONABLE B even
// when its raw price/volume status happens to be ACTIONABLE.
state.status = "ALL"; state.newBuyPoint = "ALL"; state.scope = "ALL_SIGNALS";
const ambiguous = {...entry("SN"), review_a_active:false,
  review_b_verified:false, review_b_status:"ACTIONABLE",
  review_b_display_event:"UNVERIFIED",review_new_buy_point:false,
  review_b_identity_state:"AMBIGUOUS"};
assert.equal(app.displayStatus(ambiguous),"UNVERIFIED");
assert(app.tableHtml([ambiguous]).includes("UNVERIFIED"));
state.status = "ACTIONABLE";
assert.equal(app.filterRows([ambiguous]).length,0,
  "Ambiguous Midweek Buy Point must not pass ACTIONABLE screening");
state.status = "ALL";
const dualUnverified = {...ambiguous, review_a_active:true,ibd_entry_status:"UNCONFIRMED"};
assert(app.tableHtml([dualUnverified]).includes("Weekend Buy Point"));
assert(app.industryHtml([dualUnverified]).includes("Midweek Buy Point"));
assert(app.industryHtml([dualUnverified]).includes("UNVERIFIED"),
  "Industry and Stock use consistent buy-point source disclosure");

const updatedPoint = {...entry("UPDATED"),review_b_display_event:"BP UPDATED",review_b_verified:true};
assert(app.tableHtml([updatedPoint]).includes("BUY POINT UPDATED"),
  "Raw BP UPDATED token must not leak to table");
const changedPoint = {...updatedPoint,review_b_display_event:"BP CHANGED"};
assert(app.tableHtml([changedPoint]).includes("BUY POINT CHANGED"),
  "Raw BP CHANGED token must not leak to table");

// A-only What Changed scope cannot be reprioritized by an unrelated B
// ACTIONABLE. Both visible Weekend signals are EXTENDED, so ticker order
// is the tie-breaker instead of the hidden B status.
state.scope = "CHANGES"; state.change = "BECAME_EXTENDED";
state.newBuyPoint = "ALL"; state.status = "ALL";
const firstA = {...entry("ALPHA"),review_a_event:"ZONE → EXT",
  review_weekend_buy_point_change:"BECAME_EXTENDED",
  review_b_active:false,review_b_verified:false};
const hiddenB = {...entry("ZETA"),review_a_event:"ZONE → EXT",
  review_weekend_buy_point_change:"BECAME_EXTENDED",
  review_b_status:"ACTIONABLE",review_b_verified:true,
  review_b_display_event:"NEW BP"};
assert.equal(app.sortRows([hiddenB, firstA]).rows[0].code, "ALPHA",
  "Review Priority must respect selected Weekend Buy Point context");
state.change = "ALL";

// Entry confirmation cannot be inferred from the price zone or the status
// string when resolver validity disagrees.
state.status = "ALL"; state.change = "ALL"; state.newBuyPoint = "ALL";
const rejected = {...entry("REJECT"),review_a_active:false,
  review_b_verified:true,review_b_status:"ACTIONABLE",
  review_b_entry_valid:false,review_b_entry_reason:"resolver_rejected"};
assert.equal(app.displayStatus(rejected),"UNCONFIRMED",
  "Invalid resolver must downgrade the displayed ACTIONABLE B stage");
state.status = "ACTIONABLE";
assert.equal(app.filterRows([rejected]).length,0,
  "Invalid resolver must not appear as ACTIONABLE in filters");
state.status = "ALL";

// Review Now is the latest increment, but Weekend quick filters keep the
// cumulative Vs Weekend position, so a stable below-BP ticker remains
// findable in What Changed even when it is not a new daily event.
state.status = "ALL"; state.scope = "CHANGES";
state.change = "ALL"; state.newBuyPoint = "ALL";
const oldBelow = {...entry("STABLE_A"),review_a_active:true,
  review_b_active:false,review_has_value_change:false,
  review_a_has_value_change:false,review_a_event:"",
  review_a_week_event:"ZONE → BELOW",review_weekend_buy_point_change:"BELOW_BUY_POINT",
  ibd_entry_status:"BELOW_TRIGGER"};
assert.equal(app.filterRows([oldBelow]).length,0,
  "Stable cumulative Weekend difference is not a daily Review Now event");
assert.equal(app.filterCounts([oldBelow]).change.BELOW_BUY_POINT,1,
  "Weekend cumulative quick-filter count must include stable difference");
state.change = "BELOW_BUY_POINT";
assert.equal(app.filterRows([oldBelow]).length,1,
  "Explicit What Changed selection reveals cumulative Weekend difference");
state.change = "ALL";

// Visual compression must preserve structural origin and comparison evidence
// without making the main table a verbose audit log.
state.change="ALL"; state.newBuyPoint="ALL"; state.status="ALL";
const onlyB={...entry("COMPACT"),review_a_active:false,
  review_b_display_event:"NEW BP",review_b_display_basis:"PREVIOUS_SNAPSHOT"};
const compact=app.changeLines(onlyB);
assert(compact.includes(">Midweek</span>"));
assert(compact.includes(">NEW</span>"));
assert(!compact.includes(">Midweek Buy Point</span>"));
assert(compact.includes('title="Midweek Buy Point · NEW BUY POINT · Vs Previous Snapshot"'));
const stable={...onlyB,review_b_display_event:"CURRENT"};
assert(app.changeLines(stable).includes("Still Unconfirmed"));
assert(!app.changeLines(stable).includes("NO CHANGE"));
const uncertain={...onlyB,review_b_display_event:"UNVERIFIED"};
assert(app.changeLines(uncertain).includes(">UNVERIFIED</span>"));
const both={...entry("DUAL"),review_a_event:"ZONE → EXT",
  review_a_has_value_change:true,review_b_has_value_change:true,
  review_b_display_event:"NEW BP"};
assert.equal((app.changeLines(both).match(/class="change-line"/g)||[]).length,2,
  "Only two meaningful source events create two visual lines");

// In Review Now, don't visually append a stale Weekend cumulative
// transition to a newly changed Midweek Buy Point. Explicit Weekend quick
// grouping still exposes that cumulative transition.
state.status = "ALL"; state.scope = "CHANGES";
state.change = "ALL"; state.newBuyPoint = "ALL";
const focusRow = {...entry("FOCUS"), review_a_active:true,
  review_b_active:true, review_a_event:"",
  review_a_week_event:"ZONE → EXT", review_a_has_value_change:false,
  review_b_has_value_change:true, review_b_display_event:"NEW BP",
  review_b_verified:true,review_b_entry_valid:true,
  review_has_value_change:true,review_weekend_buy_point_change:"BECAME_EXTENDED"};
const focusHtml=app.changeLines(focusRow);
assert.equal((focusHtml.match(/class="change-line"/g)||[]).length,1,
  "Review Now should only show the changed buy-point source");
assert(focusHtml.includes(">Midweek</span>"));
assert(!focusHtml.includes(">Weekend</span>"));
state.change = "BECAME_EXTENDED";
const historicalHtml = app.changeLines(focusRow);
assert(historicalHtml.includes(">Weekend</span>"),
  "What Changed still shows the fixed Weekend cumulative change");
assert(!historicalHtml.includes(">Midweek</span>"),
  "Weekend quick filter must not show unrelated Midweek event");
state.change = "ALL";state.status = "ACTIONABLE";
focusRow.ibd_entry_status = "EXTENDED";
const selectedHtml = app.changeLines(focusRow);
assert(!selectedHtml.includes(">Weekend</span>"),
  "The ACTIONABLE-selected B reference must not display an A EXTENDED event");
state.status = "ALL";

// An unchanged row describes its current status; it never uses an
// ambiguous dash. Snapshot-only and absent data must not imply continuity.
state.scope = "ALL_SIGNALS"; state.change = "ALL"; state.newBuyPoint = "ALL"; state.status = "ALL";
const stillNear = {...entry("WATCH"),review_watch_active:false,bf_watch_active:true};
assert(app.changeLines(stillNear).includes("Still Near Breakout"));
const stillA = {...entry("STILL_A"),review_a_active:true,review_b_active:false,
  review_a_event:"",review_a_week_event:"",ibd_entry_status:"EXTENDED"};
assert(app.changeLines(stillA).includes("Still Extended"));
const noData = {...stillA,review_data_available:false};
assert(app.changeLines(noData).includes("No Data"));
const noPrior = {...entry("NO_PRIOR"),review_a_active:false,review_b_display_event:"SNAPSHOT"};
assert(app.changeLines(noPrior).includes("No Prior"));
const joined = {...stillA,review_membership_event:"IN → OUT",review_membership:"OUT"};
assert(app.changeLines(joined).includes("POOL OUT"),
  "Membership-only change must not masquerade as an unchanged status");
// The older cumulative Weekend event is not the cause of today's membership
// change in Review Now. Outside Review Now the cumulative event remains useful.
state.scope = "CHANGES";
const membershipOnly = {...stillA, review_a_event:"", review_a_week_event:"ZONE → EXT",
  review_a_has_value_change:true, review_has_value_change:true,
  review_membership_event:"IN → OUT", review_membership:"OUT"};
assert.equal(app.filterRows([membershipOnly]).length, 1);
assert(app.changeLines(membershipOnly).includes("POOL OUT"),
  "Review Now must show today's membership transition");
assert(app.tableHtml([membershipOnly]).includes("Pool OUT</small>"),
  "Mobile Ticker hint must use the same Pool OUT event as Change");
assert(!app.changeLines(membershipOnly).includes("ZONE → EXT"),
  "Review Now must not misrepresent a past Weekend cumulative change as today's event");
const membershipRank = {...membershipOnly,code:"A_MEMBER"};
const realPriceRank = {...membershipOnly,code:"Z_PRICE",review_a_event:"ZONE → EXT",
  review_membership_event:""};
assert.equal(app.sortRows([membershipRank,realPriceRank]).rows[0].code,"Z_PRICE",
  "Review priority must not rank historical EXT as a new price event on membership-only rows");
const membershipAndPrice = {...membershipOnly, review_a_event:"BELOW → ZONE"};
assert(app.changeLines(membershipAndPrice).includes("BELOW → ZONE"),
  "A genuine new price transition takes priority over membership");
state.scope = "ALL_SIGNALS";
assert(app.changeLines(membershipOnly).includes("ZONE → EXT"),
  "All Review keeps the useful cumulative Weekend comparison");
// Review Now status and advanced filters must apply to *changed* references.
// A changed Weekend EXTENDED + stable Midweek ACTIONABLE must not produce
// a misleading "Still Actionable" row or ACTIONABLE badge count.
const changedAStableB = {...entry("A_CHANGED"), ibd_entry_status:"EXTENDED",
  review_a_active:true, review_a_has_value_change:true, review_a_event:"ZONE → EXT",
  review_b_active:true, review_b_verified:true, review_b_entry_valid:true,
  review_b_status:"ACTIONABLE", review_b_display_event:"CURRENT",
  review_b_has_value_change:false, review_has_value_change:true};
state.scope = "CHANGES"; state.status = "ACTIONABLE";
assert.equal(app.filterRows([changedAStableB]).length, 0,
  "Review Now ACTIONABLE excludes a stable B when only A EXTENDED changed");
assert.equal(app.filterCounts([changedAStableB]).status.ACTIONABLE, 0,
  "Review Now status badges count changed-source opportunities only");
state.status = "EXTENDED";
assert.equal(app.filterRows([changedAStableB]).length, 1,
  "Review Now EXTENDED includes the changed Weekend source");
assert(app.changeLines(changedAStableB).includes("ZONE → EXT"),
  "Change must expose the source responsible for the filtered row");
state.status = "ALL";
changedAStableB.ibd_entry_volume_ratio = 0.8;
changedAStableB.review_b_entry_volume = 2.5;
state.entryVolumeMin = 1.5;
assert.equal(app.filterRows([changedAStableB]).length, 0,
  "Review Now volume filter must not qualify using an unchanged B");
state.entryVolumeMin = null;
state.scope = "ALL_SIGNALS"; state.status = "ACTIONABLE";
assert.equal(app.filterRows([changedAStableB]).length, 1,
  "All Review retains discoverability of stable B ACTIONABLE");
state.scope = "CHANGES"; state.newBuyPoint = "BUY_ZONE";
changedAStableB.review_new_buy_point = true;
changedAStableB.review_new_buy_point_zone = "BUY_ZONE";
assert.equal(app.filterRows([changedAStableB]).length, 1,
  "Explicit Midweek snapshot quick group is not restricted to daily changes");
state.newBuyPoint = "ALL"; state.status = "ALL";
const dualChanged = {...changedAStableB,review_b_display_event:"NEW BP",
  review_b_has_value_change:true};
assert.equal(app.filterRows([dualChanged]).length, 1);
state.status = "ACTIONABLE";
assert.equal(app.filterRows([dualChanged]).length, 1,
  "Changed Midweek ACTIONABLE remains in Review Now");
assert(app.changeLines(dualChanged).includes(">NEW</span>"));
assert(!app.changeLines(dualChanged).includes(">Weekend</span>"),
  "Review Now ACTIONABLE displays only the changed matching B source");
assert(app.tableHtml([dualChanged]).includes("New Point</small>"),
  "Ticker hint uses the filtered B event, not the unrelated A transition");
state.status = "ALL";
state.scope = "ALL_SIGNALS";
const entryReason = {...entry("REASON"),ibd_entry_vol_or_reject:"daily_volume_not_confirmed"};
const entryHtml = app.tableHtml([entryReason]);
assert(entryHtml.includes(">Vol Unconfirmed</span>"));
assert(entryHtml.includes('title="daily_volume_not_confirmed"'));
assert(!entryHtml.includes(">daily_volume_not_confirmed<"));
const noSelection = app.selectedHtml(null);
assert(noSelection.includes("selected-strip empty"));
assert(app.statusCardsHtml({NEAR_BREAKOUT:0,ACTIONABLE:1,UNCONFIRMED:1,
  BELOW_TRIGGER:0,EXTENDED:0,UNVERIFIED:3}).includes("Unverified · 3"),
  "Unverified identities must be discoverable without a sixth large card");
assert.equal(app.filterCounts([{...entry("AMBIG"),review_a_active:false,
  review_b_active:true,review_b_verified:false,review_b_status:"ACTIONABLE"}]).status.UNVERIFIED,1,
  "Unverified status tally must use the same source filter predicate");
state.scope = "CHANGES";

// The range controls must include B even when A is the primary display.
// A=+6.15%, B=+0.73%; entry and weekly volumes can differ.
state.status = "ALL"; state.change = "ALL"; state.newBuyPoint = "ALL";
const rangeCase = {...entry("RANGE"), current_vs_ibd_candidate_pct:6.15,
  review_b_pct:-11.8,ibd_entry_volume_ratio:0.6,
  review_b_entry_volume:3.4,volume_ratio:0.8,
  review_b_details:{volume_ratio:4.2}};
const distances = app.bounds([rangeCase],"review_distance_pct",-5,5);
const entryVolumes = app.bounds([rangeCase],"ibd_entry_volume_ratio",0,1);
const weeklyVolumes = app.bounds([rangeCase],"volume_ratio",0,1);
assert(distances[0] <= -11.8 && distances[1] >= 6.15);
assert(entryVolumes[1] >= 3.4, "B entry volume must be selectable");
assert(weeklyVolumes[1] >= 4.2, "B weekly volume must be selectable");

console.log("BF Review buy-point-centric interaction contract passed");
