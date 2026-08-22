from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "ibd-candidate-prescreen"
    / "SKILL.md"
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_defines_eps_blind_and_eps_enriched_historical_replay_modes():
    skill = _skill_text()

    required_guidance = (
        "历史 EPS 对照模式",
        "无 EPS 子模式",
        "关闭 `eps_pit.lookup.get_signal_eps`",
        "不得因 EPS 全部缺失而清空技术基础推荐",
        "含 EPS 子模式",
        "先假设补充 EPS 正确",
        "两份报告必须使用同一套基础候选范围、字段解析、Critical、阶段路由、Geometry 与排序框架",
    )
    for guidance in required_guidance:
        assert guidance in skill


def test_skill_forbids_coupling_replay_observations_to_specific_pool_outcomes():
    skill = _skill_text()

    required_guards = (
        "不得把 32 周样本中的 ticker、日期、收益率、中位数、命中率或个别数值范围写成新门槛",
        "只能把对照结果沉淀为证据簇推理顺序、风险提示、报告审计和缺失信息路由",
        "EPS 数值大小不得作为连续排序键",
    )
    for guard in required_guards:
        assert guard in skill


def test_skill_requires_weekly_signal_oracle_for_historical_iteration():
    skill = _skill_text()

    required_guidance = (
        "按周 Signal Oracle 评估",
        "每周所有 `signal == True` 行",
        "winner / loser 必须只在同一 `snapshot_date` 内排序",
        "不得把不同周的 ACTIONABLE 或推荐样本先合并后再计算大赢家/大输家命中率",
        "记录评估 run log",
        "收益窗口、winner/loser 定义、variant 定义、评分函数、发现的问题和修正记录",
    )
    for guidance in required_guidance:
        assert guidance in skill


def test_skill_records_weekly_iteration_rule_learnings_without_overfitting():
    skill = _skill_text()

    required_guidance = (
        "EPS 已知优先于 EPS 缺失，但 `EPS >=25` 不得升级为优先复核硬门槛",
        "`pullback_not_dry` 与 `geometry_caution_not_failure` 默认作为风险披露或同分压制",
        "不得仅因二者存在就把 Critical 通过的候选硬排除",
        "优先选择在周内 winner 命中、bottom loser 暴露、stop 暴露和周内收益之间综合更稳的通用规则",
    )
    for guidance in required_guidance:
        assert guidance in skill


def test_skill_defines_signal_shadow_layer_without_expanding_official_recommendations():
    skill = _skill_text()

    required_guidance = (
        "Signal Shadow Top3",
        "每周所有 `signal == True` 行",
        "不要求 `ibd_entry_status == ACTIONABLE`",
        "不得扩大 0～3 只优先复核和 0～2 只值得留意的正式容量",
        "不得写成当前买点确认或正式推荐",
    )
    for guidance in required_guidance:
        assert guidance in skill
