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
    assert 'state.view === "INDUSTRY" ? industryHtml(rows) : tableHtml(rows)' in APP
    assert "industryStockRows(currentRows).map((row) => row.code)" in APP
    assert "industrySelected: { WEEKEND: null, MIDWEEK: null }" in APP
    assert "expandedIndustries: new Set()" in APP
    assert "state.viewScroll[state.view]" in APP
    assert 'data-action="toggle-industry"' in APP
    assert 'data-action="toggle-view"' in APP
    assert 'class="industry-stock-table"' in APP
    assert 'grid-template-columns: 16% 11% 15% 34% 24%;' in CSS
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
}
"""


@pytest.mark.parametrize("filename,mode", [
    ("rs_runtime.js", "parse"),
    ("app.js", "groups"),
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
