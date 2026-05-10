# 数位DP (Digit DP)

## 1. 概念与定义

数位DP是一类用于**统计满足特定条件的数的个数**的动态规划方法。核心思想是逐位构造数字，利用记忆化搜索避免重复计算。

数位DP的典型应用场景：
- 统计 [L, R] 范围内满足某条件的数的个数
- 统计小于等于 N 的满足某条件的数的个数
- 某条件通常与数字的各个数位有关（如数位和、不含某些数字、相邻数位关系等）

关键概念：
- **limit**：当前位是否受到上界限制。如果之前的所有位都等于上界的对应位，则当前位不能超过上界的当前位
- **lead**：是否有前导零
- **state**：记录当前状态（如之前选了哪些数字、当前数位和等）

## 2. 状态定义与转移方程

### 2.1 通用框架

```
dfs(pos, state, limit, lead)
  pos:   当前处理第pos位（从高位到低位）
  state: 当前状态
  limit: 是否受上界限制
  lead:  是否有前导零

转移：枚举当前位可以填的数字 d，递归处理下一位
```

### 2.2 统计不含某数字的数的个数

```
dfs(pos, state, limit):
  如果 pos == len(digits): return 1
  如果 memo[pos][state] 有效且不限制：return memo[pos][state]
  上界 = digits[pos] if limit else 9
  for d in range(0, 上界+1):
    result += dfs(pos+1, new_state, limit and d==上界)
```

### 2.3 常见状态设计

- **数位和**：state = 当前数位和
- **数字集合**：state = 已用数字的位掩码
- **相邻关系**：state = 上一位数字
- **整除性**：state = 当前数对某数取模的结果

## 3. 算法实现

### 3.1 统计各位数字之和能被3整除的数

```python
def countDigitSumDivisible(n):
    """统计 [1, n] 中数位和能被3整除的数的个数"""
    digits = list(map(int, str(n)))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, mod, limit):
        if pos == len(digits):
            return 1 if mod == 0 else 0

        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            result += dfs(pos + 1, (mod + d) % 3, limit and d == upper)

        return result

    return dfs(0, 0, True)
```

### 3.2 统计不含数字4的数的个数

```python
def countWithout4(n):
    digits = list(map(int, str(n)))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, limit):
        if pos == len(digits):
            return 1

        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            if d == 4:
                continue
            result += dfs(pos + 1, limit and d == upper)

        return result

    return dfs(0, True) - 1  # 减去0
```

### 3.3 统计不含62和4的数的个数（HDU 2089）

```python
def countValid(n):
    """统计 [1, n] 中不含4和62的数的个数"""
    digits = list(map(int, str(n)))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, prev, limit):
        """
        prev: 上一位数字（-1表示还没有选过非零数字）
        """
        if pos == len(digits):
            return 1

        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            if d == 4:
                continue
            if prev == 6 and d == 2:
                continue
            result += dfs(pos + 1, d, limit and d == upper)

        return result

    return dfs(0, -1, True)
```

### 3.4 统计特殊整数（LeetCode 2376）

```python
def countSpecialNumbers(n):
    """统计 [1, n] 中各位数字互不相同的数的个数"""
    digits = list(map(int, str(n)))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, mask, limit, lead):
        if pos == len(digits):
            return 1 if not lead else 0  # 至少有一位非零

        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            if lead and d == 0:
                # 前导零
                result += dfs(pos + 1, mask, limit and d == upper, True)
            elif (mask >> d) & 1 == 0:
                # d没有被使用过
                result += dfs(pos + 1, mask | (1 << d), limit and d == upper, False)

        return result

    return dfs(0, 0, True, True)
```

### 3.5 C++ 实现

```cpp
// 统计不含4和62的数
int dp[20][2];
int digits[20];

int dfs(int pos, int prev6, bool limit) {
    if (pos == 0) return 1;
    if (!limit && dp[pos][prev6] != -1) return dp[pos][prev6];
    int upper = limit ? digits[pos] : 9;
    int res = 0;
    for (int d = 0; d <= upper; d++) {
        if (d == 4) continue;
        if (prev6 && d == 2) continue;
        res += dfs(pos - 1, d == 6, limit && d == upper);
    }
    if (!limit) dp[pos][prev6] = res;
    return res;
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 基本数位DP | O(len * 状态数 * 10) | O(len * 状态数) |
| 不含某些数字 | O(len * 10) | O(len) |
| 数位互不相同 | O(len * 2^10) | O(len * 2^10) |

其中 len 为数字的位数（最多约 19 位）。

## 5. 典型例题

### 例题1：统计整数的数目（LeetCode 2719）

```python
def count(num1, num2, min_sum, max_sum):
    MOD = 10**9 + 7

    def f(num):
        digits = list(map(int, str(num)))
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dfs(pos, s, limit):
            if pos == len(digits):
                return 1 if min_sum <= s <= max_sum else 0
            upper = digits[pos] if limit else 9
            result = 0
            for d in range(upper + 1):
                if s + d > max_sum:
                    break
                result = (result + dfs(pos + 1, s + d, limit and d == upper)) % MOD
            return result

        return dfs(0, 0, True)

    return (f(int(num2)) - f(int(num1) - 1)) % MOD
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **前导零处理**：0 是否合法需要根据题意判断
2. **范围问题**：[L, R] 用 f(R) - f(L-1)，注意 L=0 的情况
3. **记忆化条件**：只有不限制（limit=False）时才能记忆化
4. **状态设计**：状态必须完整，不能有后效性

### 6.2 数位DP模板

```python
def solve(n):
    digits = list(map(int, str(n)))
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, state, limit):
        if pos == len(digits):
            return 边界条件判断(state)
        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            if 合法(state, d):
                result += dfs(pos + 1, new_state(state, d), limit and d == upper)
        return result

    return dfs(0, 初始状态, True)
```

### 6.3 进阶：从低位到高位

某些问题从低位到高位更自然（如数位和取模），但需要注意限制条件的处理方式。
