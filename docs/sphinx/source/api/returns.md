# Returns API 文档

该模块包含了 `returns` 相关的核心功能函数。

## `active_premium`
Calculate Active Premium or Active Return.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import active_premium

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = active_premium(R, Rb=Rb, scale=None, geometric=True)
print(result)
```

---

## `adjusted_sharpe_ratio`
Calculate Adjusted Sharpe Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import adjusted_sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = adjusted_sharpe_ratio(R, Rf=Rf, scale=None)
print(result)
```

---

## `annualized_excess_return`
Annualized excess return.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import annualized_excess_return

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = annualized_excess_return(R, Rb=Rb, scale=None, geometric=True)
print(result)
```

---

## `appraisal_ratio`
Calculate Appraisal Ratio.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import appraisal_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = appraisal_ratio(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `bernardo_ledoit_ratio`
Calculate Bernardo and Ledoit Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import bernardo_ledoit_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = bernardo_ledoit_ratio(R)
print(result)
```

---

## `burke_ratio`
Calculate Burke Ratio or Modified Burke Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `modified` (`bool`, 默认值: `False`): 详细说明请参考源码
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import burke_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = burke_ratio(R, Rf=Rf, modified=False, scale=None)
print(result)
```

---

## `calmar_ratio`
Calculate Calmar Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import calmar_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = calmar_ratio(R, scale=None, geometric=True)
print(result)
```

---

## `capm_alpha`
Calculate CAPM alpha of returns against a benchmark.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import capm_alpha

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = capm_alpha(R, Rb=Rb, Rf=Rf)
print(result)
```

---

## `d_ratio`
Calculate D Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import d_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = d_ratio(R)
print(result)
```

---

## `down_capture`
Calculate Down Capture Ratio.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import down_capture

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = down_capture(R, Rb=Rb)
print(result)
```

---

## `downside_deviation`
Calculate downside deviation or potential.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `method` (`str`, 默认值: `'full'`): 计算方法
- `potential` (`bool`, 默认值: `False`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import downside_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = downside_deviation(R, MAR=0.0, method='full', potential=False)
print(result)
```

---

## `downside_frequency`
Calculate Downside Frequency.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import downside_frequency

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = downside_frequency(R, MAR=0.0)
print(result)
```

---

## `downside_potential`
Calculate downside potential.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import downside_potential

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = downside_potential(R, MAR=0.0)
print(result)
```

---

## `downside_sharpe_ratio`
Downside Sharpe Ratio = mean(R - Rf) / (sqrt(2) * SemiSD(R))

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import downside_sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = downside_sharpe_ratio(R, Rf=Rf)
print(result)
```

---

## `gain_deviation`
Standard deviation of the positive returns.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import gain_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = gain_deviation(R)
print(result)
```

---

## `hurst_index`
Calculate the Hurst Index (Simplified Rescaled Range analysis).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import hurst_index

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = hurst_index(R)
print(result)
```

---

## `information_ratio`
Calculate the Information Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import information_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = information_ratio(R, Rb=Rb, scale=None, geometric=True)
print(result)
```

---

## `jensen_alpha`
Calculate Jensen's Alpha.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import jensen_alpha

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = jensen_alpha(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `kappa`
Calculate Kappa.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `l` (`int`, 默认值: `2`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import kappa

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = kappa(R, MAR=0.0, l=2)
print(result)
```

---

## `kelly_ratio`
Calculate Kelly criterion ratio (leverage or bet size) for a strategy.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `method` (`str`, 默认值: `'half'`): 计算方法

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import kelly_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = kelly_ratio(R, Rf=Rf, method='half')
print(result)
```

---

## `loss_deviation`
### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import loss_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = loss_deviation(R)
print(result)
```

---

## `m2_sortino`
Calculate M squared for Sortino.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import m2_sortino

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = m2_sortino(R, Rb=Rb, MAR=0.0, scale=None)
print(result)
```

---

## `m_squared`
Calculate M squared.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import m_squared

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = m_squared(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `m_squared_excess`
Calculate M squared excess.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `method` (`str`, 默认值: `'geometric'`): 计算方法
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import m_squared_excess

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = m_squared_excess(R, Rb=Rb, Rf=Rf, method='geometric', scale=None)
print(result)
```

---

## `market_timing`
Estimate Market Timing models (Treynor-Mazuy or Henriksson-Merton).

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `method` (`str`, 默认值: `'TM'`): 计算方法

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import market_timing

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = market_timing(R, Rb=Rb, Rf=Rf, method='TM')
print(result)
```

---

## `martin_ratio`
Calculate Martin ratio of the return distribution.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import martin_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = martin_ratio(R, Rf=Rf, scale=None, geometric=True)
print(result)
```

---

## `mean_absolute_deviation`
Calculate Mean Absolute Deviation (MAD).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import mean_absolute_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = mean_absolute_deviation(R)
print(result)
```

---

## `modigliani`
Calculate Modigliani-Modigliani (M-squared) measure.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import modigliani

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = modigliani(R, Rb=Rb, Rf=Rf)
print(result)
```

---

## `net_selectivity`
Net selectivity = Selectivity - diversification

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import net_selectivity

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = net_selectivity(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `omega_excess_return`
Omega excess return = Rp - 3 * SigmaD * SigmaDM

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import omega_excess_return

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = omega_excess_return(R, Rb=Rb, MAR=0.0, scale=None)
print(result)
```

---

## `omega_ratio`
Calculate Omega Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `L` (`float`, 默认值: `0.0`): 详细说明请参考源码
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `method` (`str`, 默认值: `'simple'`): 计算方法

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import omega_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = omega_ratio(R, L=0.0, Rf=Rf, method='simple')
print(result)
```

---

## `omega_sharpe_ratio`
Omega-Sharpe Ratio = (UpsidePotential - DownsidePotential) / DownsidePotential

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import omega_sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = omega_sharpe_ratio(R, MAR=0.0)
print(result)
```

---

## `pain_ratio`
Calculate Pain ratio of the return distribution.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import pain_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = pain_ratio(R, Rf=Rf, scale=None, geometric=True)
print(result)
```

---

## `prob_sharpe_ratio`
Calculate Probabilistic Sharpe Ratio (PSR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `refSR` (`float | pandas.Series | pandas.DataFrame`): 详细说明请参考源码
- `Rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `ignore_skewness` (`bool`, 默认值: `False`): 详细说明请参考源码
- `ignore_kurtosis` (`bool`, 默认值: `True`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import prob_sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = prob_sharpe_ratio(R, refSR=None, Rf=Rf, ignore_skewness=False, ignore_kurtosis=True)
print(result)
```

---

## `prospect_ratio`
Calculate Prospect Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import prospect_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = prospect_ratio(R, MAR=0.0)
print(result)
```

---

## `rachev_ratio`
Calculate Rachev Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `alpha` (`float`, 默认值: `0.1`): 详细说明请参考源码
- `beta` (`float`, 默认值: `0.1`): 详细说明请参考源码
- `rf` (`float`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import rachev_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = rachev_ratio(R, alpha=0.1, beta=0.1, rf=0.0)
print(result)
```

---

## `return_annualized`
Calculate annualized return.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_annualized

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = return_annualized(R, scale=None, geometric=True)
print(result)
```

---

## `return_calculate`
Calculate returns from a price stream.

### 输入参数
- `prices` (`pandas.Series | pandas.DataFrame`): 资产价格时间序列数据
- `method` (`str`, 默认值: `'discrete'`): 计算方法

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_calculate

# 构造示例价格数据
dates = pd.date_range("2020-01-01", periods=10, freq="D")
prices = pd.Series([100, 102, 101, 105, 104, 106, 108, 107, 110, 112], index=dates)

# 调用函数
result = return_calculate(prices=prices, method='discrete')
print(result)
```

---

## `return_clean`
Clean extreme observations in a time series to provide robust risk estimates.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `method` (`str`, 默认值: `'boudt'`): 计算方法
- `alpha` (`float`, 默认值: `0.01`): 详细说明请参考源码
- `trim` (`float`, 默认值: `0.001`): 详细说明请参考源码

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_clean

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = return_clean(R, method='boudt', alpha=0.01, trim=0.001)
print(result)
```

---

## `return_cumulative`
Calculate a compounded (geometric) or simple cumulative return.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_cumulative

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = return_cumulative(R, geometric=True)
print(result)
```

---

## `return_excess`
Calculate excess returns by subtracting the risk-free rate.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_excess

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = return_excess(R, Rf=Rf)
print(result)
```

---

## `return_geltner`
Calculate Geltner liquidity-adjusted return series.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_geltner

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = return_geltner(R)
print(result)
```

---

## `return_portfolio`
Calculate weighted returns for a portfolio of assets.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `weights` (`pandas.Series | pandas.DataFrame | list | np.ndarray | None`, 默认值: `None`): 详细说明请参考源码
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `rebalance_on` (`str`, 默认值: `'none'`): 详细说明请参考源码

### 返回参数
- `<class 'pandas.Series'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import return_portfolio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = return_portfolio(R, weights=None, geometric=True, rebalance_on='none')
print(result)
```

---

## `semi_deviation`
Calculate semi-deviation.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import semi_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = semi_deviation(R)
print(result)
```

---

## `semi_variance`
Calculate semi-variance (MAR = mean, method = subset).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import semi_variance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = semi_variance(R)
print(result)
```

---

## `sharpe_ratio`
Calculate the Sharpe Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `FUN` (`str`, 默认值: `'StdDev'`): 详细说明请参考源码
- `annualize` (`bool`, 默认值: `False`): 详细说明请参考源码
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import sharpe_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = sharpe_ratio(R, Rf=Rf, p=0.95, FUN='StdDev', annualize=False, scale=None)
print(result)
```

---

## `sortino_ratio`
Calculate the Sortino Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import sortino_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = sortino_ratio(R, MAR=0.0)
print(result)
```

---

## `std_dev_annualized`
Calculate annualized standard deviation.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import std_dev_annualized

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = std_dev_annualized(R, scale=None)
print(result)
```

---

## `sterling_ratio`
Calculate Sterling Ratio.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)
- `excess` (`float`, 默认值: `0.1`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import sterling_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = sterling_ratio(R, scale=None, excess=0.1)
print(result)
```

---

## `to_period_contributions`
Aggregate high-frequency contributions to lower frequency.

### 输入参数
- `Contributions` (`pandas.Series | pandas.DataFrame`): 详细说明请参考源码
- `period` (`str`, 默认值: `'years'`): 周期

### 返回参数
- `<class 'pandas.DataFrame'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import to_period_contributions

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Contributions = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = to_period_contributions(Contributions=Contributions, period='years')
print(result)
```

---

## `treynor_ratio`
Calculate Treynor Ratio.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0`): 无风险收益率 (Risk-free rate)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import treynor_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = treynor_ratio(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `up_capture`
Calculate Up Capture Ratio.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import up_capture

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = up_capture(R, Rb=Rb)
print(result)
```

---

## `up_down_ratios`
Calculate metrics on up and down markets for the benchmark asset.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `method` (`str`, 默认值: `'Capture'`): 计算方法
- `side` (`str`, 默认值: `'Up'`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import up_down_ratios

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = up_down_ratios(R, Rb=Rb, method='Capture', side='Up')
print(result)
```

---

## `upside_potential`
Calculate upside potential.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import upside_potential

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = upside_potential(R, MAR=0.0)
print(result)
```

---

## `upside_potential_ratio`
Calculate Upside Potential Ratio of upside performance over downside risk.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0`): 最低可接受回报率 (Minimum Acceptable Return)
- `method` (`str`, 默认值: `'subset'`): 计算方法

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import upside_potential_ratio

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = upside_potential_ratio(R, MAR=0, method='subset')
print(result)
```

---

## `upside_risk`
Calculate upside risk, variance, or potential.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `method` (`str`, 默认值: `'full'`): 计算方法
- `stat` (`str`, 默认值: `'risk'`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import upside_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = upside_risk(R, MAR=0.0, method='full', stat='risk')
print(result)
```

---

## `volatility_skewness`
Calculate Volatility or Variability Skewness.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `MAR` (`float`, 默认值: `0.0`): 最低可接受回报率 (Minimum Acceptable Return)
- `stat` (`str`, 默认值: `'volatility'`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.returns import volatility_skewness

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = volatility_skewness(R, MAR=0.0, stat='volatility')
print(result)
```

---
