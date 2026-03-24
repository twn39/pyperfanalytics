# Drawdowns API 文档

该模块包含了 `drawdowns` 相关的核心功能函数。

## `average_drawdown`
Calculate the average depth of the observed drawdowns.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import average_drawdown

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = average_drawdown(R, geometric=True)
print(result)
```

---

## `average_length`
Calculate the average length (in periods) of the observed drawdowns.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import average_length

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = average_length(R, geometric=True)
print(result)
```

---

## `average_recovery`
Calculate the average length (in periods) of the observed recovery period.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import average_recovery

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = average_recovery(R, geometric=True)
print(result)
```

---

## `cdd`
Calculate Uryasev's proposed Conditional Drawdown at Risk (CDD or CDaR) measure.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `invert` (`bool`, 默认值: `True`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import cdd

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = cdd(R, p=0.95, geometric=True, invert=True)
print(result)
```

---

## `drawdown_deviation`
Calculate a standard deviation-type statistic using individual drawdowns.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import drawdown_deviation

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = drawdown_deviation(R, geometric=True)
print(result)
```

---

## `drawdown_peak`
Replicate R's DrawdownPeak logic.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import drawdown_peak

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = drawdown_peak(R)
print(result)
```

---

## `drawdowns`
Calculate the drawdown levels in a timeseries.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import drawdowns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = drawdowns(R, geometric=True)
print(result)
```

---

## `find_drawdowns`
Find drawdowns in a return series.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `dict[str, np.ndarray]`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import find_drawdowns

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = find_drawdowns(R, geometric=True)
print(result)
```

---

## `max_drawdown`
Calculate the maximum drawdown of a return series.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.drawdowns import max_drawdown

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = max_drawdown(R, geometric=True)
print(result)
```

---

## `sort_drawdowns`
Sort drawdowns from worst to best.

### 输入参数
- `runs` (`dict`): drawdowns 序列字典

### 返回参数
- `dict[str, np.ndarray]`: 计算结果

### 代码示例
```python
import numpy as np
from pyperfanalytics.drawdowns import sort_drawdowns

# 构造 runs 字典
runs = {"Return": np.array([0, -0.1, -0.2, 0])}

# 调用函数
result = sort_drawdowns(runs=runs)
print(result)
```

---
