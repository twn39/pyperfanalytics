# Charts API 文档

该模块包含了 `charts` 相关的核心功能函数。

## `chart_acf`
Create an Autocorrelation Function (ACF) chart.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `maxlag` (`int | None`, 默认值: `None`): 最大滞后阶数
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_acf

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_acf(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_acf_plus`
Create a chart with both ACF and PACF subplots.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `maxlag` (`int | None`, 默认值: `None`): 最大滞后阶数
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_acf_plus

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_acf_plus(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_bar_returns`
Create a bar chart of period returns.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `title` (`str`, 默认值: `'Returns'`): 详细说明请参考源码
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_bar_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_bar_returns(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_bar_var`
Plot periodic returns as a bar chart with interactive risk metric overlays.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `width` (`int`, 默认值: `0`): 详细说明请参考源码
- `gap` (`int`, 默认值: `12`): 详细说明请参考源码
- `methods` (`str | list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `show_symmetric` (`bool`, 默认值: `False`): 详细说明请参考源码
- `show_horizontal` (`bool`, 默认值: `False`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_bar_var

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_bar_var(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_boxplot`
Create a horizontal box and whiskers plot to compare return distributions.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `sort_by` (`str | None`, 默认值: `None`): 详细说明请参考源码
- `sort_ascending` (`bool`, 默认值: `True`): 详细说明请参考源码
- `main` (`str`, 默认值: `'Return Distribution Comparison'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_boxplot

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_boxplot(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_capture_ratios`
Scatter plot of Upside Capture versus Downside Capture against a benchmark.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `main` (`str`, 默认值: `'Capture Ratio'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_capture_ratios

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 生成图表
fig = chart_capture_ratios(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_component_returns`
Plots the contribution of each asset to the portfolio return.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `weights` (`list[float] | np.ndarray | pandas.Series | None`, 默认值: `None`): 详细说明请参考源码
- `main` (`str`, 默认值: `'Component Returns Contribution'`): 图表主标题
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_component_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_component_returns(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_correlation`
Visualization of a Correlation Matrix with distributions and scatter plots.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `main` (`str`, 默认值: `'Correlation Matrix'`): 图表主标题
- `method` (`str`, 默认值: `'pearson'`): 计算方法
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_correlation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_correlation(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_cum_returns`
Create a cumulative returns chart using Plotly.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `wealth_index` (`bool`, 默认值: `False`): 详细说明请参考源码
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `begin` (`str`, 默认值: `'axis'`): 详细说明请参考源码
- `title` (`str`, 默认值: `'Cumulative Returns'`): 详细说明请参考源码
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_cum_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_cum_returns(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_drawdown`
Create a professional drawdown (underwater) chart showing losses from peak.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `title` (`str`, 默认值: `'Drawdown (Underwater)'`): 详细说明请参考源码
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_drawdown

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_drawdown(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_ecdf`
Create an aesthetically enhanced ECDF chart overlaid with a normal CDF.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `main` (`str`, 默认值: `'Empirical Cumulative Distribution Function'`): 图表主标题
- `xlab` (`str`, 默认值: `'Returns'`): 详细说明请参考源码
- `ylab` (`str`, 默认值: `'Cumulative Probability'`): 详细说明请参考源码
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_ecdf

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_ecdf(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_events`
Plots a time series with event dates aligned.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `dates` (`list`): 详细说明请参考源码
- `prior` (`int`, 默认值: `12`): 详细说明请参考源码
- `post` (`int`, 默认值: `12`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_events

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_events(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_histogram`
Create a histogram of returns with optional curve fits and risk markers.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `breaks` (`int`, 默认值: `30`): 详细说明请参考源码
- `methods` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_histogram

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_histogram(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_qqplot`
Create a Quantile-Quantile plot with a normal reference and confidence bands.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_qqplot

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_qqplot(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_relative_performance`
Plot the ratio of cumulative performance between two assets over time.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `main` (`str`, 默认值: `'Relative Performance'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_relative_performance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 生成图表
fig = chart_relative_performance(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_risk_return_scatter`
Create a risk-return scatter plot with Sharpe ratio indifference lines.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `add_sharpe` (`list[float] | None`, 默认值: `None`): 详细说明请参考源码
- `main` (`str`, 默认值: `'Annualized Return and Risk'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_risk_return_scatter

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 生成图表
fig = chart_risk_return_scatter(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_rolling_correlation`
Chart of rolling correlation between two sets of assets.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `width` (`int`, 默认值: `12`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_rolling_correlation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 生成图表
fig = chart_rolling_correlation(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_rolling_performance`
Wrapper to chart any performance metric over a rolling window.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `width` (`int`, 默认值: `12`): 详细说明请参考源码
- `FUN` (`str`, 默认值: `'return_annualized'`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_rolling_performance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_rolling_performance(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_rolling_regression`
Chart of rolling regression performance metrics (Alpha, Beta, or R-Squared).

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `width` (`int`, 默认值: `12`): 详细说明请参考源码
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `attribute` (`str`, 默认值: `'Beta'`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_rolling_regression

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 生成图表
fig = chart_rolling_regression(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_scatter`
Create a scatter plot with optional marginal distributions and regression line.

### 输入参数
- `x` (`pandas.Series | pandas.DataFrame`): 详细说明请参考源码
- `y` (`pandas.Series | pandas.DataFrame`): 详细说明请参考源码
- `main` (`str`, 默认值: `'Scatter Plot'`): 图表主标题
- `xlab` (`str | None`, 默认值: `None`): 详细说明请参考源码
- `ylab` (`str | None`, 默认值: `None`): 详细说明请参考源码
- `marginal` (`str`, 默认值: `'rug'`): 详细说明请参考源码
- `add_regression` (`bool`, 默认值: `True`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_scatter

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_scatter()
# fig.show()  # 取消注释以显示图表
```

---

## `chart_snail_trail`
Create a snail trail chart showing rolling risk-return evolution.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `width` (`int`, 默认值: `12`): 详细说明请参考源码
- `stepsize` (`int`, 默认值: `12`): 详细说明请参考源码
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `main` (`str`, 默认值: `'Annualized Return and Risk Trail'`): 图表主标题
- `add_sharpe` (`list[float] | None`, 默认值: `None`): 详细说明请参考源码
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_snail_trail

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 生成图表
fig = chart_snail_trail(R)
# fig.show()  # 取消注释以显示图表
```

---

## `chart_stacked_bar`
Create a stacked bar plot.

### 输入参数
- `w` (`pandas.Series | pandas.DataFrame`): 详细说明请参考源码
- `main` (`str`, 默认值: `'Stacked Bar Chart'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_stacked_bar

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_stacked_bar()
# fig.show()  # 取消注释以显示图表
```

---

## `chart_var_sensitivity`
Create a chart of VaR and ES estimates across a range of confidence levels.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `methods` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import chart_var_sensitivity

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = chart_var_sensitivity(R)
# fig.show()  # 取消注释以显示图表
```

---

## `charts_performance_summary`
Create a combined dashboard containing Cumulative Returns, Period Returns, and Drawdown.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `wealth_index` (`bool`, 默认值: `False`): 详细说明请参考源码
- `main` (`str | None`, 默认值: `None`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import charts_performance_summary

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 生成图表
fig = charts_performance_summary(R)
# fig.show()  # 取消注释以显示图表
```

---

## `charts_rolling_regression`
Dashboard with Rolling Alpha, Rolling Beta, and Rolling R-Squared charts.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `width` (`int`, 默认值: `12`): 详细说明请参考源码
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `main` (`str`, 默认值: `'Rolling Regression Summary'`): 图表主标题
- `colorset` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `**kwargs`: 其他可选参数

### 返回参数
- `<class 'plotly.graph_objs._figure.Figure'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.charts import charts_rolling_regression

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 生成图表
fig = charts_rolling_regression(R)
# fig.show()  # 取消注释以显示图表
```

---
