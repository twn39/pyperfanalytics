# Risk API 文档

该模块包含了 `risk` 相关的核心功能函数。

## `capm_beta`
Calculate CAPM Beta of returns against a benchmark.

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
from pyperfanalytics.risk import capm_beta

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = capm_beta(R, Rb=Rb, Rf=Rf)
print(result)
```

---

## `cdar_alpha`
Calculate Conditional Drawdown Alpha (CDaR Alpha).

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `type` (`str | None`, 默认值: `None`): 详细说明请参考源码
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import cdar_alpha

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = cdar_alpha(R, Rb=Rb, p=0.95, geometric=True, type=None, scale=None)
print(result)
```

---

## `cdar_beta`
Calculate Conditional Drawdown Beta (CDaR Beta).

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `geometric` (`bool`, 默认值: `True`): 是否使用几何复合计算
- `type` (`str | None`, 默认值: `None`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import cdar_beta

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = cdar_beta(R, Rb=Rb, p=0.95, geometric=True, type=None)
print(result)
```

---

## `es_gaussian`
Calculate Gaussian Expected Shortfall (Conditional VaR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import es_gaussian

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = es_gaussian(R, p=0.95)
print(result)
```

---

## `es_historical`
Calculate Historical Expected Shortfall (Conditional VaR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import es_historical

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = es_historical(R, p=0.95)
print(result)
```

---

## `es_modified`
Calculate Modified (Cornish-Fisher) Expected Shortfall.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import es_modified

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = es_modified(R, p=0.95)
print(result)
```

---

## `fama_beta`
Calculate Fama Beta.

### 输入参数
- `Ra` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import fama_beta

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = fama_beta(R, Rb=Rb, scale=None)
print(result)
```

---

## `herfindahl_index`
Calculate Herfindahl Index based on autocorrelation.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import herfindahl_index

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = herfindahl_index(R)
print(result)
```

---

## `min_track_record`
Calculate the Minimum Track Record Length.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `refSR` (`float`): 详细说明请参考源码
- `Rf` (`float | pandas.Series | pandas.DataFrame`, 默认值: `0.0`): 无风险收益率 (Risk-free rate)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)
- `ignore_skewness` (`bool`, 默认值: `False`): 详细说明请参考源码
- `ignore_kurtosis` (`bool`, 默认值: `True`): 详细说明请参考源码

### 返回参数
- `dict | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import min_track_record

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rf = 0.0001

# 调用函数
result = min_track_record(R, refSR=None, Rf=Rf, p=0.95, ignore_skewness=False, ignore_kurtosis=True)
print(result)
```

---

## `pain_index`
Calculate the Pain Index.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import pain_index

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = pain_index(R)
print(result)
```

---

## `smoothing_index`
Calculate Normalized Getmansky Smoothing Index.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `neg_thetas` (`bool`, 默认值: `False`): 详细说明请参考源码
- `MAorder` (`int`, 默认值: `2`): 详细说明请参考源码

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import smoothing_index

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = smoothing_index(R, neg_thetas=False, MAorder=2)
print(result)
```

---

## `specific_risk`
Calculate Specific Risk.

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
from pyperfanalytics.risk import specific_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = specific_risk(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `systematic_risk`
Calculate Systematic Risk.

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
from pyperfanalytics.risk import systematic_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = systematic_risk(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `total_risk`
Calculate Total Risk (Systematic + Specific).

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
from pyperfanalytics.risk import total_risk

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
Rf = 0.0001

# 调用函数
result = total_risk(R, Rb=Rb, Rf=Rf, scale=None)
print(result)
```

---

## `tracking_error`
Calculate Tracking Error of returns against a benchmark.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`pandas.Series | pandas.DataFrame`): 基准收益率时间序列数据 (Benchmark returns)
- `scale` (`int | None`, 默认值: `None`): 年化缩放系数 (如日频=252, 月频=12)

### 返回参数
- `float | pandas.Series | pandas.DataFrame`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import tracking_error

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = tracking_error(R, Rb=Rb, scale=None)
print(result)
```

---

## `ulcer_index`
Calculate the Ulcer Index.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import ulcer_index

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = ulcer_index(R)
print(result)
```

---

## `var_gaussian`
Calculate Gaussian (Parametric) Value at Risk (VaR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import var_gaussian

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = var_gaussian(R, p=0.95)
print(result)
```

---

## `var_historical`
Calculate Historical Value at Risk (VaR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import var_historical

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = var_historical(R, p=0.95)
print(result)
```

---

## `var_modified`
Calculate Modified (Cornish-Fisher) Value at Risk (VaR).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `p` (`float`, 默认值: `0.95`): 置信区间、概率阈值或参数 (例如 0.95)

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.risk import var_modified

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = var_modified(R, p=0.95)
print(result)
```

---
