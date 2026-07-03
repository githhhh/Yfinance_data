# BreakoutFollow Pool 数据结构与计算白皮书 (已统一归口迁移)

> [!IMPORTANT]
> **唯一真实信源 (Single Source of Truth, SSOT) 架构说明**：
> 
> 为彻底解决跨仓库维护白皮书引发的**版本割裂与协议漂移**问题，从本版本起，BreakoutFollow 策略池的全字段数据规范、物理计算方法与业务架构指南已统一归口迁移至**策略主研发库**：
> 
> 📍 **官方唯一信源路径**：  
> **`quant_trade/strategy/doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`**
> 
> ### 为什么归口至 quant_trade？
> 1. **代码与协议同源**：所有底层字段（`POOL_COLUMNS`）、时间周期进位算子（如 `base_duration_weeks`）及 K 线穿透力物理公式，均由 `quant_trade` 计算引擎负责在运行时生成。
> 2. **高内聚低耦合**：保证后续任何字段演进或规则调整，开发者都能在同一个 Git Commit 中同步修改底层算子代码与白皮书契约，杜绝跨库漏改。
> 
> *请前往策略主仓库查阅最新全字段白皮书，本库不再独立维护副本。*
