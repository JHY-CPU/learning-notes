# DFA与NFA - 有穷自动机


## 一、确定性有穷自动机 (DFA)


DFA是一个五元组 `M = (Q, Σ, δ, q₀, F)`，其中：


- **Q**
   — 有穷状态集合
- **Σ**
   — 有穷输入字母表
- **δ**
   — 转移函数
   `δ: Q × Σ → Q`
   （每个状态对每个输入符号恰好一个后继）
- **q₀**
   — 初始状态
- **F**
   — 接受状态集合 (F ⊆ Q)


DFA的关键特征：**确定性**——读取一个符号后，下一状态唯一确定。

DFA状态图示例：接受以01结尾的串

### 示例：接受以"01"结尾的二进制串

q₀
start
0
→
q₁
1
→
q₂
(接受)

q₀ --0--> q₁ --1--> q₂(接受)；q₀上读1留在q₀；q₁上读0留在q₁；q₂上读0回q₀，读1回q₂


## 二、DFA转移表


| 当前状态 | 输入 0 | 输入 1 |
| --- | --- | --- |
| → q₀ | q₁ | q₀ |
| q₁ | q₁ | *q₂ |
| *q₂ | q₀ | q₀ |


→ 初始状态 * 接受状态


## 三、非确定性有穷自动机 (NFA)


NFA与DFA类似，但转移函数可以映射到**状态的集合**：


- **δ: Q × (Σ ∪ {ε}) → P(Q)**
   — 后继可以是零个、一个或多个状态
- 允许
   **ε-转移**
   （不消耗输入的状态跳转）
- 若存在
   **某条路径**
   使输入被接受，则串被接受


### NFA状态图示意

q₀
0, 1
→
q₁
0
→
q₂
→
ε
↺
q₃

NFA允许一个状态对同一输入有多个转移，也允许ε-转移


## 四、DFA与NFA对比


### DFA


- 每个输入恰好
   **一个**
   后继状态
- 无 ε-转移
- 模拟无需回溯，实现高效
- 状态数可能
   **指数级增长**
- 转移函数：δ: Q × Σ → Q


### NFA


- 每个输入可有
   **零个或多个**
   后继
- 允许 ε-转移
- 设计更直观、简洁
- 状态数通常
   **较少**
- 转移函数：δ: Q × (Σ∪{ε}) → P(Q)


**核心定理：**对任意 NFA，存在等价的 DFA。即 DFA 与 NFA 描述的语言类相同——**正则语言**。（证明方法：子集构造法）


## 五、子集构造法（NFA → DFA）


将NFA转换为等价DFA的系统方法：


1. DFA的每个状态 = NFA状态的
   **子集**
2. DFA初始状态 =
   `ε-closure({q₀})`
3. 对DFA状态 S 和输入 a：δ'(S, a) = ε-closure(∪ δ(q, a))，对所有 q ∈ S
4. DFA接受状态 = 包含NFA任意接受状态的子集


```
// 子集构造法伪代码
function subset_construction(NFA):
    DFA_start = ε_closure({NFA.start})
    worklist = [DFA_start]
    visited = {DFA_start}

    while worklist is not empty:
        S = worklist.pop()
        for each symbol a in alphabet:
            T = ε_closure( { q | p in S, q in NFA.delta(p, a) } )
            DFA.delta(S, a) = T
            if T not in visited:
                visited.add(T)
                worklist.push(T)

    DFA.accept = { S | S ∩ NFA.accept ≠ ∅ }
    return DFA
```


## 六、DFA最小化（Hopcroft算法）


每个正则语言都有**唯一的最小DFA**（状态数最少），通过等价状态合并得到：


1. 初始划分：{接受状态}, {非接受状态}
2. 反复细化：若两个状态在同一组中，但对某个输入到达不同组，则将它们分开
3. 直到无法再细化为止
4. 合并每个等价类为一个状态


时间复杂度 O(n log n)，其中 n 为状态数。


## 七、简单DFA模拟器 (Python)


```
# DFA模拟器：判断二进制串是否以"01"结尾
class DFA:
    def __init__(self, states, alphabet, transitions, start, accept):
        self.states = states          # 状态集合
        self.alphabet = alphabet      # 字母表
        self.delta = transitions      # 转移函数 dict[(state, char)] -> state
        self.start = start            # 初始状态
        self.accept = accept          # 接受状态集合

    def run(self, input_string):
        current = self.start
        for ch in input_string:
            if ch not in self.alphabet:
                return False
            current = self.delta[(current, ch)]
        return current in self.accept

# 定义 DFA: 接受以 "01" 结尾的二进制串
dfa = DFA(
    states={'q0', 'q1', 'q2'},
    alphabet={'0', '1'},
    transitions={
        ('q0', '0'): 'q1', ('q0', '1'): 'q0',
        ('q1', '0'): 'q1', ('q1', '1'): 'q2',
        ('q2', '0'): 'q0', ('q2', '1'): 'q0',
    },
    start='q0',
    accept={'q2'}
)

# 测试
test_cases = ["101", "0101", "10", "01", "1101"]
for s in test_cases:
    result = "接受" if dfa.run(s) else "拒绝"
    print(f"  串 '{s}' -> {result}")
```


<!-- Converted from: 01_DFA与NFA.html -->
