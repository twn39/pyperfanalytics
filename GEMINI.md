# PyPerfAnalytics 开发指南

本项目旨在将 R 语言的 `PerformanceAnalytics` 包迁移至 Python 生态，确保算法逻辑与原始 R 实现完全对齐。

## 核心原则

1. **算法正确性与一致性**：在逻辑和数学算法正确的前提下，所有计算结果必须与 `PerformanceAnalytics` (R) 保持一致（绝对误差应小于 `1e-6`）。**如果发现 R 版本存在逻辑缺陷或数学错误（例如特殊的硬编码、多频率下未正确年化 Rf 等）导致结果不一致，必须优先保证代码与数学逻辑的正确性**，并在实现或测试中明确提出该差异和原因，坚决不照搬错误的逻辑。
2. **技术栈限制**：仅允许使用 `pandas`, `numpy`, `statsmodels`, `scipy` 作为核心计算库。
3. **源码参考与批判性思维**：`third_party/PerformanceAnalytics/R/` 下的 R 源码是主要的参考标准。在实现前必须仔细阅读对应的 `.R` 文件，但不能盲从，需结合严格的数学推导来验证其内部实现。
4. **验证驱动**：每实现一个核心算法，必须编写验证脚本，通过 `Rscript` 获取基准值并进行对比。如果由于 R 语言的 Bug 导致静态比对失败，应在 Python 测试中通过动态计算正确预期值的方式来覆盖，并明确标注差异原因。

## 技术规范

### 数据处理
- **索引**：统一使用 `pd.DatetimeIndex`。
- **输入类型**：函数应同时支持 `pd.Series` 和 `pd.DataFrame`。
- **频率（Scale）**：年化计算时，必须准确识别或允许手动指定 `scale`：
  - 日频 (Daily)：252
  - 周频 (Weekly)：52
  - 月频 (Monthly)：12
  - 季频 (Quarterly)：4
  - 年频 (Yearly)：1

### 命名约定
- Python 函数使用 `snake_case`（例如 `return_annualized`），但在文档中注明对应的 R 函数名（例如 `Return.annualized`）。

### 常用算法逻辑对齐
- **收益率计算**：
  - `discrete`: `prices.pct_change()`
  - `log`: `np.log(prices).diff()`
- **年化收益 (Geometric)**：`(1 + R).prod() ** (scale / n) - 1`
- **年化标准差**：`R.std() * sqrt(scale)`
- **夏普比率**：`mean(R - Rf) / Risk(R)`。注意年化时，分子是年化超额收益，分母是年化风险指标。

## 项目结构

```text
pyperfanalytics/
├── data/                 # 测试数据目录
├── scripts/              # 脚本目录
├── src/
│   └── pyperfanalytics/  # 核心逻辑
│       ├── returns.py    # 收益率相关计算
│       ├── risk.py       # 风险指标计算
│       ├── drawdowns.py  # 回撤相关
│       ├── tables.py     # 统计报表生成
│       └── utils.py      # 工具函数
├── tests/                # 单元测试与一致性验证
└── third_party/          # 源码参考项目目录 (原始 R 源码)
```

**注意**：
- `data/` 与 `scripts/` 属于项目资产，**已提交至 git 仓库**。
- `third_party/` 与 `GEMINI.md` 仅在本地开发时作为参考，**不加入 `.gitignore`，也不提交到 git**。

## 开发流程

1. **Research**: 阅读 `third_party/PerformanceAnalytics/R/` 中的目标函数源码。
2. **Implementation**: 在 `pyperfanalytics/` 中实现 Python 版本。
3. **Validation**: 
   - 使用 `Rscript -e 'library(PerformanceAnalytics); ...'` 生成基准数据。
   - 编写 Python 脚本对比结果。
4. **Integration**: 更新 `__init__.py` 暴露接口。

## 已发现的 R 库 Bug 与处理方案

### `TotalRisk` (PerformanceAnalytics) 处理 `Rf` 向量时的 Bug
- **问题描述**：当输入 `Rf` 为向量且 `Ra/Rb` 为 `xts` 对象时，R 的 `TotalRisk` 函数内部会将其转换为矩阵。这会导致内部调用的 `Return.excess` 因为数据对齐失败而返回 `NA` 或错误结果。
- **影响**：导致 `TotalRisk` 的计算结果（例如 `0.2097`）与其组成的 `SystematicRisk` 和 `SpecificRisk` 不符合 $TotalRisk = \sqrt{SystematicRisk^2 + SpecificRisk^2}$ 的数学关系。
- **Python 处理**：Python 实现保持逻辑正确性，坚持使用 $\sqrt{SystematicRisk^2 + SpecificRisk^2}$。测试基准通过在 R 中手动绕过该 Bug（使用标量 `Rf` 或手动平方和开方）生成。

### `SystematicRisk` 的标准差计算
- **逻辑对齐**：R 的 `SystematicRisk` 在计算 Benchmark 的标准差时，使用的是 **超额收益 (Excess Returns)** 的标准差，而非原始收益。Python 实现已同步此逻辑。

### `CDaR.alpha` 年度化硬编码 Bug
- **问题描述**：R 的 `CDaR.alpha` 源码中在计算期望年化收益时，直接硬编码了 `(1+mean(Rm))^12-1`，强制假定输入为月度数据。
- **Python 处理**：Python 实现使用了根据数据推断出的 `scale` (默认日频 252)，避免了此硬编码 Bug。在基准测试生成脚本中通过手动使用 `scale=252` 计算 `CDaR.alpha` 以验证 Python 逻辑的正确性。

## 工具链
- 包管理：使用 `uv`。
- 运行测试：`uv run pytest`。
- 辅助资源：`third_party/PerformanceAnalytics/` (R 源码), `third_party/PortfolioAnalytics/` (参考实现)。
