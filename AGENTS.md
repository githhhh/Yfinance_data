# Agent 项目入口

本仓库用于美股筛选、突破候选池维护和 IBD 复盘，技术栈以 Python、Pandas 和纯静态 Web 为主。

## 目录

- `us/`：筛选结果与候选池 CSV
- `results_pkl/`：行情缓存
- `dashboard/`：静态复盘页面、共享投影逻辑及测试
- `doc/`：经确认的正式文档
- `*_screener.py`：股票筛选入口

## 环境

仅使用 Conda `quant_env`，不得创建项目内虚拟环境。

```bash
conda activate quant_env
python dashboard/self_check.py --csv us/breakout_follow_pool.csv --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
```

## 强制规则

- 先检查 Git 状态，只修改任务范围；未经授权不得改动 `us/`、`results_pkl/` 数据。
- 修改仪表板后运行相关测试、`dashboard/self_check.py` 和静态构建。
- Dashboard 不得重新引入 Streamlit、服务端运行时或第二套交易规则；状态、Midweek projection、Breakout Quality 等业务口径继续由 Python 权威层生成，浏览器只做展示、筛选、排序与交互。
- 静态站不得包含账户、持仓、API Key 或其它私有交易数据。
- 过程笔记、计划、草稿、截图等临时材料放在项目外；禁止创建 `docs/superpowers/`、`dashboard/artifacts/`。
- 正式设计文档必须事先讨论并获得用户批准。
- 新增依赖须先说明；安装、测试和运行统一使用 `quant_env`。
