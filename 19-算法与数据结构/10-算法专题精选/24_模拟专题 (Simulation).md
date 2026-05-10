# 模拟专题 (Simulation)

## 一、概念定义与原理

### 1.1 模拟题特征

模拟题不涉及复杂算法，但需要**严格按照题目描述**实现逻辑。常见类型：
- 机械运动模拟
- 大数运算
- 棋盘游戏模拟
- 过程模拟（如计算器）

### 1.2 解题要点

1. **仔细读题：** 每个细节都不能遗漏
2. **边界处理：** 注意数组越界、空输入等
3. **模块化：** 将复杂逻辑拆分为小函数
4. **对拍验证：** 用暴力数据验证

---

## 二、经典问题

### 2.1 螺旋矩阵

按螺旋顺序填充或遍历矩阵。

### 2.2 大数运算

处理超出基本类型范围的整数运算。

### 2.3 字符串解析

解析表达式、JSON 等。

---

## 三、代码实现

### 3.1 螺旋矩阵 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// LeetCode 54: 螺旋矩阵（按螺旋顺序输出）
vector<int> spiral_order(vector<vector<int>>& matrix) {
    vector<int> result;
    if (matrix.empty()) return result;
    int top = 0, bottom = matrix.size()-1;
    int left = 0, right = matrix[0].size()-1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; j++) result.push_back(top][j]);
        top++;
        for (int i = top; i <= bottom; i++) result.push_back(matrix[i][right]);
        right--;
        if (top <= bottom) {
            for (int j = right; j >= left; j--) result.push_back(matrix[bottom][j]);
            bottom--;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; i--) result.push_back(matrix[i][left]);
            left++;
        }
    }
    return result;
}
```

### 3.2 大数加法 - C++

```cpp
string add_bigint(string a, string b) {
    string result;
    int carry = 0, i = a.size()-1, j = b.size()-1;
    while (i >= 0 || j >= 0 || carry) {
        int sum = carry;
        if (i >= 0) sum += a[i--] - '0';
        if (j >= 0) sum += b[j--] - '0';
        result.push_back('0' + sum % 10);
        carry = sum / 10;
    }
    reverse(result.begin(), result.end());
    return result;
}
```

### 3.3 Python 实现

```python
def spiral_order(matrix):
    if not matrix: return []
    result = []
    top, bottom, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
    while top <= bottom and left <= right:
        for j in range(left, right+1): result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom+1): result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left-1, -1): result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1): result.append(matrix[i][left])
            left += 1
    return result

def add_bigint(a, b):
    return str(int(a) + int(b))  # Python 原生大数支持

# Python 直接支持大数
a = 10**100
b = 2**1000
print(a + b)

print(spiral_order([[1,2,3],[4,5,6],[7,8,9]]))  # [1,2,3,6,9,8,7,4,5]
```

### 3.4 行列转换（字符串模拟）

```cpp
// LeetCode 6: Z字形变换
string convert(string s, int numRows) {
    if (numRows == 1) return s;
    vector<string> rows(min(numRows, (int)s.size()));
    int cur_row = 0;
    bool going_down = false;
    for (char c : s) {
        rows[cur_row] += c;
        if (cur_row == 0 || cur_row == numRows-1) going_down = !going_down;
        cur_row += going_down ? 1 : -1;
    }
    string result;
    for (auto& row : rows) result += row;
    return result;
}
```

### 3.5 迷宫模拟

```cpp
// 在迷宫中按指令移动
pair<int,int> simulate_maze(vector<string>& maze, string& commands, int x, int y) {
    map<char, pair<int,int>> dirs = {{'U',{-1,0}},{'D',{1,0}},{'L',{0,-1}},{'R',{0,1}}};
    int m = maze.size(), n = maze[0].size();
    for (char cmd : commands) {
        auto [dx, dy] = dirs[cmd];
        int nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] != '#') {
            x = nx; y = ny;
        }
    }
    return {x, y};
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 螺旋矩阵 | $O(mn)$ | $O(1)$ |
| 大数加法 | $O(\max(m,n))$ | $O(\max(m,n))$ |
| Z字变换 | $O(n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 54：** 螞旋矩阵
2. **LeetCode 6：** Z字变换
3. **LeetCode 415：** 字符串相加（大数加法）
4. **LeetCode 67：** 二进制求和
5. **棋盘模拟：** 井字棋、扫雷等
