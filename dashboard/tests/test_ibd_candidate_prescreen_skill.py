import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / ".agents"
    / "skills"
    / "ibd-candidate-prescreen"
    / "SKILL.md"
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _output_contract() -> str:
    return _skill_text().split("## 输出格式 (Output Format)", maxsplit=1)[1]


def test_skill_frontmatter_has_valid_required_fields():
    text = _skill_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)

    assert match is not None
    fields = dict(line.split(": ", maxsplit=1) for line in match.group(1).splitlines())
    assert set(fields) == {"name", "description"}
    assert fields["name"] == "ibd-candidate-prescreen"
    assert re.fullmatch(r"[a-z0-9-]+", fields["name"])
    assert 0 < len(fields["description"]) <= 1024
    assert "<" not in fields["description"]
    assert ">" not in fields["description"]


def test_prescreen_output_is_chinese_and_decision_first():
    output = _output_contract()

    assert "正文最多 20 行" in output
    assert "Breakout Quality 使用 Dashboard 原始名称" in output
    assert "判断只写自然语言结论" in output
    assert output.count("**突破日：** [Breakout Quality]") == 2
    assert "日线突破突出但结构、基本面或关键数据证据不完整" in output
    assert "只保留原始业务数据，不输出检查项数量或通过/失败统计" in output
    assert "原著上下文未加载时必须说明" in output

    headings = ("## 结论", "## 优先复核", "## 值得留意", "## 暂不优先")
    for heading in headings:
        assert heading in output
    assert [output.index(heading) for heading in headings] == sorted(
        output.index(heading) for heading in headings
    )
    assert output.index("**背景**") < output.index("**优先复核**")

    for label in ("突破日", "优势", "顾虑", "判断"):
        assert label in output

    for internal_term in (
        "Critical",
        "Major",
        "Minor",
        "PASS",
        "FAIL",
        "UNKNOWN",
        "Checklist",
        "Top Picks",
        "Rejected Candidates",
        "Manual Review Queue",
    ):
        assert internal_term not in output
    assert "3/3" not in output


def test_weekly_volume_is_only_a_positive_bonus_in_user_output():
    skill = _skill_text()

    assert "Minor 中 #10 仅执行下述正向加分语义" in skill
    assert "周线量能达到 `1.3x` 时，作为“优势”中的加分项展示" in skill
    assert "低于 `1.3x` 或缺失时直接省略" in skill
    assert "不得作为拒绝、降级或风险理由" in skill
    assert "合并到可选背景句" in skill
    assert "原著上下文未加载时也必须输出" in skill


def test_internal_screening_rules_remain_unchanged():
    skill = _skill_text()

    unchanged_rules = (
        "`ibd_entry_status == 'ACTIONABLE'`",
        "单一板块不超过 2 只",
        "某板块占比 > 50%",
        "距 Candidate Price ≤ 5.0%",
        "Entry Volume Ratio ≥ 1.5x",
        "Close Position ≥ 0.65",
        "`pullback_v_is_dry == True`",
        "距 52 周高点 > -5.0%",
        "EPS YoY 增长 ≥ 25%",
        "近 10 周上涨周成交量 > 下跌周成交量",
        "当周 Volume Ratio ≥ 1.3x",
        "`trigger_pos <= 0`（即 `range_ratio >= pos`），且 `pos >= 0.80`",
        "`pos >= 0.80` 且 `range_ratio >= 0.50`，但 `trigger_pos > 0`",
        "`pos < 0.65`（上影线 $> 35\\%$）",
        "若 $range\\_ratio \\le 0$ ($Close \\le Trigger$)",
        "初始 Base 突破 (`ibd_candidate_rule == 'ceiling'`) → 强制使用 `base_depth_pct` / `base_duration_weeks`",
        "回踩确认 (`ceiling_pullback`, `ma10_touch_confirm`) → 强制使用 `pullback_pct` / `pullback_duration_weeks`",
        "Pivot / Three-Weeks-Tight → 仅当 `pullback_count > 0` 时评估 `pullback_pct` / `pullback_duration_weeks`",
    )
    for rule in unchanged_rules:
        assert rule in skill


def test_signal_field_applicability_prevents_false_missing_data_risks():
    skill = _skill_text()

    expected_guidance = (
        "`ceiling` 首次突破只评估 `base_depth_pct` / `base_duration_weeks`",
        "不得读取或报告 `pullback_v_is_dry`、`pullback_pct`、`pullback_duration_weeks` 缺失",
        "仅当信号存在实际回撤阶段时才评估",
        "`pullback_duration_weeks` 必须来自上游正式导出",
    )
    for guidance in expected_guidance:
        assert guidance in skill
