# Utils API 文档

该模块包含了 `utils` 相关的核心功能函数。

## `beta_co_kurtosis`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import beta_co_kurtosis

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = beta_co_kurtosis(R, Rb=Rb)
print(result)
```

---

## `beta_co_skewness`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import beta_co_skewness

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = beta_co_skewness(R, Rb=Rb)
print(result)
```

---

## `beta_co_variance`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import beta_co_variance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = beta_co_variance(R, Rb=Rb)
print(result)
```

---

## `centered_comoment`
Calculate the joint centered comoment of two series.

### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)
- `p1` (`int`): 置信区间、概率阈值或参数 (例如 0.95)
- `p2` (`int`): 置信区间、概率阈值或参数 (例如 0.95)
- `normalize` (`bool`, 默认值: `False`): 详细说明请参考源码

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import centered_comoment

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = centered_comoment(R, Rb=Rb, p1=3, p2=4, normalize=False)
print(result)
```

---

## `centered_moment`
Calculate the nth centered moment (population version, matching R's PerformanceAnalytics).

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `moment` (`int`): 矩的阶数

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import centered_moment

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = centered_moment(R, moment=None)
print(result)
```

---

## `co_kurtosis`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import co_kurtosis

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = co_kurtosis(R, Rb=Rb)
print(result)
```

---

## `co_skewness`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import co_skewness

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = co_skewness(R, Rb=Rb)
print(result)
```

---

## `co_variance`
### 输入参数
- `Ra` (`Series`): 资产收益率时间序列数据 (Asset returns)
- `Rb` (`Series`): 基准收益率时间序列数据 (Benchmark returns)

### 返回参数
- `<class 'float'>`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import co_variance

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
Rb = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)

# 调用函数
result = co_variance(R, Rb=Rb)
print(result)
```

---

## `kurtosis`
Calculate kurtosis of the return distribution.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `method` (`str`, 默认值: `'excess'`): 计算方法

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import kurtosis

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = kurtosis(R, method='excess')
print(result)
```

---

## `skewness`
Calculate skewness of the return distribution.

### 输入参数
- `R` (`pandas.Series | pandas.DataFrame`): 资产收益率时间序列数据 (Asset returns)
- `method` (`str`, 默认值: `'moment'`): 计算方法

### 返回参数
- `float | pandas.Series`: 计算结果

### 代码示例
```python
import pandas as pd
import numpy as np
from pyperfanalytics.utils import skewness

# 构造示例数据
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=10, freq="D")
R = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)

# 调用函数
result = skewness(R, method='moment')
print(result)
```

---
