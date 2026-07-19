# Agent 项目入口

本仓库用于美股筛选、突破候选池维护和本地 IBD 复盘，技术栈为 Python、Pandas、Streamlit。

## 目录

- `us/`：筛选结果与候选池 CSV
- `results_pkl/`：行情缓存
- `dashboard/`：复盘仪表盘及测试
- `doc/`：经确认的正式文档
- `*_screener.py`：股票筛选入口

## 环境

仅使用 Conda `quant_env`，不得创建项目内虚拟环境。

```bash
conda activate quant_env
python dashboard/run_app.py --csv us/breakout_follow_pool.csv
python dashboard/self_check.py --csv us/breakout_follow_pool.csv
python -m pytest dashboard/tests -q
```

## 强制规则

- 先检查 Git 状态，只修改任务范围；未经授权不得改动 `us/`、`results_pkl/` 数据。
- 修改仪表盘后运行相关测试和 `dashboard/self_check.py`。
- 过程笔记、计划、草稿、截图等临时材料放在项目外；禁止创建 `docs/superpowers/`、`dashboard/artifacts/`。
- 正式设计文档必须事先讨论并获得用户批准。
- 新增依赖须先说明；安装、测试和运行统一使用 `quant_env`。
