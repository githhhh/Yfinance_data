"""Industry RS dual-view contracts with executable parsing and ordering fixtures."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


DASH = Path(__file__).resolve().parents[1]
APP = (DASH / "app.js").read_text(encoding="utf-8")
RS = (DASH / "rs_runtime.js").read_text(encoding="utf-8")
CSS = (DASH / "styles.css").read_text(encoding="utf-8")
TABLE = (DASH / "table_enhancements.js").read_text(encoding="utf-8")


def test_reference_uses_fred6725_same_commit_without_changing_pool() -> None:
    assert "Fred6725/rs-log" in RS
    assert "output/rs_stocks.csv" in RS
    assert "output/rs_industries.csv" in RS
    assert "INDUSTRY_CSV_URL(sha)" in RS
    assert "CSV_URL(sha)" in RS
    assert "parseIndustries(await industryResponse.text())" in RS
    assert 'app.dispatchEvent(new CustomEvent("bf-rs-updated"))' in RS
    assert "status: \"error\"" in RS
    assert "industry: index.Industry" in RS
    assert '["industry_rs", "IND RS"]' in APP
    assert 'if (field === "industry_rs")' in APP
    # No new authoritative data fields / publications in this feature.
    assert "window.BFIndustryRS" in RS


def test_mobile_views_share_filtering_but_isolate_sort_and_copy() -> None:
    assert 'view: "STOCK"' in APP
    assert "tableHtml(rows).replace(" in APP
    assert "industryHtml(rows).replace(" in APP
    assert "switchReviewView(currentRows)" in APP
    assert "toggleIndustryCard(button)" in APP
    assert 'app.addEventListener("bf-rs-updated", refreshIndustryReference);' in APP
    assert "industryStockRows(currentRows).map((row) => row.code)" in APP
    assert "industrySelected: { WEEKEND: null, MIDWEEK: null }" in APP
    assert "expandedIndustries: new Set()" in APP
    assert "state.viewScroll[state.view]" in APP
    assert 'data-action="toggle-industry"' in APP
    assert 'data-action="toggle-view"' in APP
    assert 'class="industry-stock-table"' in APP
    assert 'grid-template-columns: 16% 28% 10% 26% 20%;' in CSS
    assert "IND RS" not in TABLE  # No header sorter for industry RS.
    assert 'app.querySelectorAll("[data-table-shell]").forEach(decorateTable)' in TABLE
    assert '[data-industry-list]' in APP


NODE_TEST = r"""
const fs = require("node:fs");
const source = fs.readFileSync(process.argv[1], "utf8");
const mode = process.argv[2];
function part(start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first + start.length);
  if (first < 0 || last < 0) throw Error("missing source section " + start);
  return source.slice(first, last);
}
function eq(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw Error(JSON.stringify(actual) + " != " + JSON.stringify(expected));
  }
}
if (mode === "parse") {
  const helpers = part("  function parseCsvLine(", "  function parseRatings(")
    + part("  function industryKey(", "  function industryForCode(");
  const parse = new Function("csv", helpers + "\nreturn parseIndustries(csv);");
  const csv = 'Rank,Industry,Sector,Relative Strength,Percentile,1M_RS_Percentile,3M_RS_Percentile,6M_RS_Percentile,Tickers\n'
    + '1,"Semiconductors",Technology,116,97,88,99,95,"SMTC,NVDA,MPWR"\n'
    + '2,"Software—Infrastructure",Technology,89,76,62,50,32,"CRWD,PANW"\n'
    + '3,"Oil, Gas",Energy,84,70,68,62,55,"OXY,COP"';
  const result = parse(csv);
  eq(result.ratings.get("semiconductors").rs, 97);
  eq(result.ratings.get("software - infrastructure").rs, 76);
  eq(result.members.get("NVDA"), "Semiconductors");
  eq(result.ratings.get("oil, gas").name, "Oil, Gas");
} else if (mode === "groups") {
  const methods = part("  function industryGroups(", "  function industryHtml(");
  const fixture = new Function("window", "rows", methods + "\nreturn { groups: industryGroups(rows), all: industryStockRows(rows) };");
  const rs = { SMTC:98, MPWR:98, NVDA:91, CRWD:88, PANW:86, OXY:null };
  const inds = { SMTC:"Semiconductors", MPWR:"Semiconductors", NVDA:"Semiconductors",
    CRWD:"Software", PANW:"Software", OXY:"Unclassified" };
  const ranks = { Semiconductors:97, Software:76, Unclassified:null };
  const input = ["OXY","NVDA","PANW","CRWD","SMTC","MPWR"].map(code => ({code}));
  const result = fixture({ BFIndustryRS: {
    stockRS: code => rs[code], industryForCode: code => inds[code], industryRS: name => ranks[name]
  } }, input);
  eq(result.groups.map(group => group.name), ["Semiconductors","Software","Unclassified"]);
  eq(result.groups[0].rows.map(row => row.code), ["MPWR","SMTC","NVDA"]);
  eq(result.all.map(row => row.code), ["MPWR","SMTC","NVDA","CRWD","PANW","OXY"]);
} else if (mode === "interaction") {
  const toggle = part("  function toggleIndustryCard(", "  function refreshIndustryReference(");
  const state = { expandedIndustries: new Set() };
  const preview = { hidden: false }, detail = { hidden: true };
  const card = { querySelector: selector => selector === ".industry-top3" ? preview : detail };
  const button = {
    dataset: { industry: "Semiconductors" }, expanded: "false",
    getAttribute() { return this.expanded; },
    setAttribute(_, v) { this.expanded = v; },
    closest: () => card
  };
  const click = new Function("state", "button", toggle + "\nreturn toggleIndustryCard(button);");
  click(state, button);
  eq([button.expanded, preview.hidden, detail.hidden, state.expandedIndustries.has("Semiconductors")],
     ["true", true, false, true]);
  click(state, button);
  eq([button.expanded, preview.hidden, detail.hidden, state.expandedIndustries.has("Semiconductors")],
     ["false", false, true, false]);
  // Both DOM subtrees keep object identity across expand / collapse.
  eq(card.querySelector(".industry-top3") === preview, true);
  eq(card.querySelector(".industry-group-body") === detail, true);

  const change = part("  function switchReviewView(", "  function toggleIndustryCard(");
  const stock = {hidden:false, scrollTop:12};
  const industry = {hidden:true, scrollTop:0};
  const label = {textContent:"Stock"};
  const viewButton = {querySelector: () => label, setAttribute(){}, title:""};
  const selectedStrip = {replaceWith(x){this.replacement=x;}};
  const section = {
    dataset: {view:"STOCK"},
    querySelector(s) {
      return {".table-shell":stock,"[data-industry-list]":industry,
              '[data-action="toggle-view"]':viewButton,".selected-strip":selectedStrip}[s];
    }
  };
  const app = {innerHTML:"ORIGINAL",querySelector:()=>section};
  const doc = {createElement:()=>({set innerHTML(x) { this.content={firstElementChild:x}; }})};
  const s = {view:"INDUSTRY",period:"MIDWEEK",
    selected:{MIDWEEK:null},industrySelected:{MIDWEEK:null},
    viewScroll:{INDUSTRY:35,STOCK:12}};
  const switcher = new Function("state","app","document","selectedHtml","rows",
    change + "\nreturn switchReviewView(rows);");
  switcher(s,app,doc,() => "<div></div>",[]);
  eq([stock.hidden,industry.hidden,industry.scrollTop,label.textContent,section.dataset.view],
     [true,false,35,"Industry","INDUSTRY"]);
  eq(app.innerHTML, "ORIGINAL");
}
"""


@pytest.mark.parametrize("filename,mode", [
    ("rs_runtime.js", "parse"),
    ("app.js", "groups"),
    ("app.js", "interaction"),
])
def test_executable_reference_fixtures(filename: str, mode: str) -> None:
    if not shutil.which("node"):
        pytest.skip("Node.js required for executable dashboard JavaScript tests")
    result = subprocess.run(
        ["node", "-e", NODE_TEST, str(DASH / filename), mode],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

def test_desktop_toolbar_remains_single_row_and_mobile_min_width_override_works() -> None:
    assert ".results-toolbar { grid-template-columns: minmax(0,1fr) auto auto 34px 34px; gap: 6px; }" in CSS
    assert ".results-toolbar, .results-toolbar:not(.nothing)" not in CSS
    assert "grid-template-columns: minmax(0,1fr) 45px 80px 32px 32px;" in CSS
    assert ".results-section > .industry-list { order: 3; }" in CSS
    for action in ('toggle-view', 'copy-codes', 'toggle-filters'):
        assert f'data-action="{action}"' in APP


def test_failed_industry_refresh_clears_stale_industry_ranks() -> None:
    assert 'status: "error", ratings: new Map(),' in RS
    assert 'members: new Map(),' in RS
    assert "Do not mix new stock RS with an older industry's RS" in RS
    assert 'Industry RS unavailable' in APP


def test_industry_keyboard_navigation() -> None:
    assert 'tabindex="0" aria-label="Industry review results"' in APP
    assert 'industryShell.addEventListener("keydown", (event) => {' in APP
    assert 'visibleRows.findIndex' in APP
    assert "industryShell.scrollTop += rowRect.top - shellRect.top;" in APP
    assert "scrollIntoView" not in APP



def test_view_switch_and_industry_accordion_do_not_rebuild_dashboard() -> None:
    switch = APP.split("function switchReviewView(", 1)[1].split("function toggleIndustryCard(", 1)[0]
    toggle = APP.split("function toggleIndustryCard(", 1)[1].split("function refreshIndustryReference(", 1)[0]
    assert "app.innerHTML" not in switch + toggle
    assert "render();" not in switch + toggle
    assert "window.scrollTo" not in switch + toggle
    assert "stock.hidden = state.view !== \"STOCK\"" in switch
    assert "industry.hidden = state.view !== \"INDUSTRY\"" in switch
    assert "preview.hidden = expanded;" in toggle
    assert "detail.hidden = !expanded;" in toggle
    assert "industry-group-metrics" in APP
    assert 'Results</div>' in APP
    assert 'results · Sorted by' not in APP
    assert '.industry-group-body[hidden]' in CSS
    assert '.industry-top3[hidden]' in CSS


def test_rs_interaction_budget_and_vs_reference_centering() -> None:
    assert 'grid-template-columns: 16% 28% 10% 26% 20%;' in CSS
    assert 'width: 24px; flex: 0 0 24px;' in CSS
    assert 'th[data-field="current_vs_ibd_candidate_pct"] .table-sort-button' in CSS
    assert 'justify-content: center !important;' in CSS
    assert 'setTextIfChanged(summary, `${count} Results`)' in TABLE


def test_stock_only_reset_keeps_toolbar_controls_in_place() -> None:
    # Stock's sortable table retains its Reset state while Industry never offers
    # sorting; the summary grows into that grid slot so controls stay anchored.
    assert '.results-section[data-view="INDUSTRY"] .results-order-slot { display: none; }' in CSS
    assert '.results-section[data-view="INDUSTRY"] .results-summary { grid-column: 1 / span 2; }' in CSS
    for selector, col in (("view-switch", 3), ("copy-button", 4), ("mobile-filter-button", 5)):
        assert f'.results-section[data-view="INDUSTRY"] .{selector} {{ grid-column: {col}; }}' in CSS
    assert "state.view = state.view === \"STOCK\" ? \"INDUSTRY\" : \"STOCK\";" in APP
    assert "switchReviewView(currentRows)" in APP
    assert "const sortStates = new Map();" in TABLE
    assert "syncDefaultSortButton(shell)" in TABLE


def test_industry_chevron_stays_on_title_row_and_count_is_grammatical() -> None:
    assert 'class="industry-chevron" viewBox="0 0 24 24" width="18" height="18"' in APP
    assert 'stroke-linecap="round" stroke-linejoin="round"' in APP
    assert "group.rows.length === 1 ? \"stock\" : \"stocks\"" in APP
    assert ".industry-group-toggle {" in CSS
    assert "display: flex; align-items: flex-start; width: 100%;" in CSS
    assert "flex: 0 0 18px; width: 18px; height: 18px;" in CSS
    assert "transform-origin: 50% 50%;" in CSS
    assert '.industry-group-toggle[aria-expanded="true"] .industry-chevron { transform: rotate(90deg); }' in CSS


def test_industry_rs_is_neutral_and_mobile_status_header_and_value_share_center() -> None:
    assert ".industry-group-rs strong { color: #B8D5E2;" in CSS
    assert 'grid-template-columns: 16% 28% 10% 26% 20%;' in CSS
    assert 'th[data-field="ibd_entry_status"] .table-header-control' in CSS
    assert 'th[data-field="ibd_entry_status"] .table-sort-button' in CSS
    assert 'td[data-field="ibd_entry_status"]' in CSS
    assert "justify-content: center !important;" in CSS
    assert "font-size: clamp(8px, 2.35vw, 10px);" in CSS
    assert "letter-spacing: -0.2px;" in CSS
