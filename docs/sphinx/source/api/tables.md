# Tables API 文档

该模块包含了 `tables` 相关的核心功能函数。

## `table_annualized_returns`
Annualized Returns Summary: Statistics and Stylized Facts

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `digits` (`int`, 默认值: `6`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_annualized_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_annualized_returns(R, scale=None, Rf=Rf, geometric=True, digits=6)
print(result)
```

---

## `table_autocorrelation`
Table for calculating the first six (default) autocorrelation coefficients and significance.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数
- `max_lag` (`int`, 默认值: `6`): 最大滞后阶数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_autocorrelation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_autocorrelation(R, digits=4, max_lag=6)
print(result)
```

---

## `table_calendar_returns`
Monthly and Calendar Year Return Table.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `digits` (`int`, 默认值: `1`): 保留小数位数
- `as_perc` (`bool`, 默认值: `True`): 详细说明请参考源码
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_calendar_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_calendar_returns(R, digits=1, as_perc=True, geometric=True)
print(result)
```

---

## `table_capm`
Single Factor Asset-Pricing Model (CAPM) Summary Table.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_capm

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_capm(R, Rb=Rb, scale=None, Rf=Rf, digits=4)
print(result)
```

---

## `table_capture_ratios`
Up and Down Market Capture Ratio Table.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_capture_ratios

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_capture_ratios(R, Rb=Rb, digits=4)
print(result)
```

---

## `table_correlation`
Calculate correlations and significance of multicolumn data.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数
- `conf_level` (`float`, 默认值: `0.95`): 详细说明请参考源码

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_correlation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_correlation(R, Rb=Rb, digits=4, conf_level=0.95)
print(result)
```

---

## `table_distributions`
Distributions Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_distributions

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_distributions(R, scale=None, digits=4)
print(result)
```

---

## `table_downside_risk`
Downside Risk Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `MAR` (`float`, 默认值: `0.008333333333333333`): 最低可接受回报率 (Minimum Acceptable Return)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_downside_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_downside_risk(R, scale=None, Rf=Rf, MAR=0.008333333333333333, p=0.95, digits=4)
print(result)
```

---

## `table_downside_risk_ratio`
Downside Risk Summary: Ratios and Metrics.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0`): 最低可接受回报率 (Minimum Acceptable Return)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_downside_risk_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_downside_risk_ratio(R, MAR=0, scale=None, digits=4)
print(result)
```

---

## `table_drawdowns`
Worst Drawdowns Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `top` (`int`, 默认值: `5`): 显示的最高/最大数量
- `digits` (`int`, 默认值: `4`): 保留小数位数
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_drawdowns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_drawdowns(R, top=5, digits=4, geometric=True)
print(result)
```

---

## `table_drawdowns_ratio`
Drawdowns Summary: Statistics and ratios.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_drawdowns_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_drawdowns_ratio(R, Rf=Rf, scale=None, digits=4)
print(result)
```

---

## `table_higher_moments`
Higher Moments Summary: Statistics and Stylized Facts (Co-Moments).

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_higher_moments

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_higher_moments(R, Rb=Rb, digits=4)
print(result)
```

---

## `table_information_ratio`
Information Ratio Summary: Statistics and Stylized Facts.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_information_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_information_ratio(R, Rb=Rb, scale=None, digits=4)
print(result)
```

---

## `table_monthly_returns`
Returns Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `ci` (`float`, 默认值: `0.95`): 置信区间 (Confidence Interval)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_monthly_returns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_monthly_returns(R, ci=0.95, digits=4)
print(result)
```

---

## `table_prob_outperformance`
Outperformance Report of Asset vs Benchmark.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `period_lengths` (`list[int] | None`, 默认值: `None`): 详细说明请参考源码

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_prob_outperformance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_prob_outperformance(R, Rb=Rb, period_lengths=None)
print(result)
```

---

## `table_prob_sharpe_ratio`
Summary table for Probabilistic Sharpe Ratio across different thresholds.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `refSR` (`float | list | np.ndarray`, 默认值: `0.0`): 详细说明请参考源码
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_prob_sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_prob_sharpe_ratio(R, refSR=0.0, Rf=Rf, digits=4)
print(result)
```

---

## `table_prob_skewness_kurtosis`
Summary table for univariate skewness and kurtosis methods.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_prob_skewness_kurtosis

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_prob_skewness_kurtosis(R, digits=4)
print(result)
```

---

## `table_rolling_periods`
Rolling Periods Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `periods` (`list[int] | None`, 默认值: `None`): 详细说明请参考源码
- `funcs` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `funcs_names` (`list[str] | None`, 默认值: `None`): 详细说明请参考源码
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_rolling_periods

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_rolling_periods(R, periods=None, funcs=None, funcs_names=None, digits=4)
print(result)
```

---

## `table_specific_risk`
Specific risk Summary: Statistics and Stylized Facts

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_specific_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = table_specific_risk(R, Rb=Rb, Rf=Rf, digits=4)
print(result)
```

---

## `table_stats`
Returns Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `ci` (`float`, 默认值: `0.95`): 置信区间 (Confidence Interval)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_stats

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_stats(R, ci=0.95, digits=4)
print(result)
```

---

## `table_up_down_ratios`
Up and Down Market Ratios Table.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_up_down_ratios

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = table_up_down_ratios(R, Rb=Rb, digits=4)
print(result)
```

---

## `table_variability`
Variability Summary: Statistics and Stylized Facts.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `digits` (`int`, 默认值: `4`): 保留小数位数

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.tables import table_variability

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = table_variability(R, scale=None, geometric=True, digits=4)
print(result)
```

---
