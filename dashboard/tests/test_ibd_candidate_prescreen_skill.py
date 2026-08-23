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


def test_main_skill_declares_skill_industry_eps_known_as_formal_baseline():
    skill = _skill_text()

    assert "正式生产基线固定为：**`skill_industry_eps_known`**" in skill
    assert "含 EPS 补源后的 ACTIONABLE 原始质量排序 + EPS 已知准入 + Industry 覆盖 Top3" in skill
    assert "它是当前 `main` 的主 skill 基线" in skill


def test_research_variants_cannot_replace_or_pollute_main_baseline():
    skill = _skill_text()

    assert "clean_eps_pass_no_dry_no_geom_caution" not in skill
    assert "signal_shadow_top3" not in skill
    assert "任何研究变体、审计层" in skill
    assert "RD/qlib 研究结果" in skill
    assert "不得进入生产推荐顺序；生产推荐只按 `skill_industry_eps_known` 的确定性规则执行" in skill


def test_cross_model_determinism_contract_has_fixed_tie_breakers():
    skill = _skill_text()

    expected_rules = (
        "不同模型必须输出相同的“优先复核”代码集合和顺序",
        "deterministic artifact",
        "dashboard.skill_industry_eps_known",
        "不得使用模型偏好、图表观感、历史收益、候选名称熟悉度、行业热度或外部记忆作为隐含排序键",
        "完全并列时只用 `code` 大写字典序，再用 CSV 原始行序打破平局",
        "Industry 覆盖选择必须在原始排序冻结后顺序扫描",
        "EPS 数值大小、EPS 是否高于 25%、EPS 缺失状态均不得进入原始排序",
        "若两个模型输出不同，必须按上述排序键逐项回放，差异方视为执行错误",
    )
    for rule in expected_rules:
        assert rule in skill


def test_existing_core_baseline_rules_remain_present():
    skill = _skill_text()

    expected_rules = (
        "Review Universe 仅包含 `signal == True` 且 `ibd_candidate_rule` 非空的行",
        "完整原始质量排序”只对其中 `ibd_entry_status == ACTIONABLE` 的候选编号",
        "Fresh Demand Alpha > Constructive Pullback > Standard Breakout > Incomplete Evidence",
        "更完整的证据确认项数量",
        "优先复核不超过 3 只，且每个已知 Industry 最多 1 只",
        "只用 `current_vs_ibd_candidate_pct`；缺失 UNKNOWN；`0%～5%` PASS",
        "只用 `ibd_entry_volume_ratio`；缺失 UNKNOWN；有效值 `>=1.5` PASS",
        "`pullback_v_is_dry == True` PASS，`False` FAIL",
        "`dist_to_52w_high_pct > -5.0` 为 PASS",
        "`volume_ratio >= 1.3` 时记一个二元正向加分",
    )
    for rule in expected_rules:
        assert rule in skill
