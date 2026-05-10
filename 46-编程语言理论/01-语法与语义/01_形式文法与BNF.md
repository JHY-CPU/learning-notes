# 形式文法与BNF


## 一、Chomsky文法层级


诺姆·乔姆斯基在1956年提出的形式文法分类体系：


| 类型 | 名称 | 产生式限制 | 自动机等价 |
| --- | --- | --- | --- |
| 类型 0 | 无限制文法 | α → β (α至少一个变量) | 图灵机 |
| 类型 1 | 上下文有关文法 | αAβ → αγβ (\|γ\| ≥ 1) | 线性有界自动机 |
| 类型 2 | **上下文无关文法** | A → γ (A是变量) | 下推自动机 (PDA) |
| 类型 3 | 正则文法 | A → aB 或 A → a | 有限自动机 (DFA/NFA) |


重要 编程语言的语法主要使用**类型2（上下文无关文法）**描述。


## 二、BNF（巴科斯-诺尔范式）


BNF (Backus-Naur Form) 是描述上下文无关文法的标准记法。


### BNF基本符号


| 符号 | 含义 | 示例 |
| --- | --- | --- |
| `<name>` | 非终结符（语法变量） | `<expression>` |
| `::=` | "定义为" | `<expr> ::= <term>` |
| `\|` | "或者"（选择） | `A ::= B \| C` |
| 无标记的符号 | 终结符（字面值） | `+`, `if`, `;` |
| `"..."` | 终结符（明确标记） | `"+"` |


### BNF示例：简单算术表达式

<expression>
::=
<expression>
"+"
<term>
|
<term>
<term>
::=
<term>
"*"
<factor>
|
<factor>
<factor>
::=
"("
<expression>
")"
|
<number>
<number>
::=
<digit>
<number>
|
<digit>
<digit>
::=
"0"
|
"1"
|
"2"
|
...
|
"9"

## 三、EBNF（扩展BNF）


EBNF 添加了更多便利的语法糖：


| 构造 | 含义 | BNF等价 |
| --- | --- | --- |
| `[ ... ]` | 可选（0或1次） | A ::= B \| ε |
| `{ ... }` | 重复0或多次 | A ::= B A \| ε |
| `( ... )` | 分组 | 引入辅助非终结符 |
| `...` | 重复1或多次 | A ::= B { B } |


### EBNF版本的算术表达式

expression
=
term
, {
"+"
,
term
} ;
term
=
factor
, {
"*"
,
factor
} ;
factor
=
"("
,
expression
,
")"
|
number
;
number
=
digit
, {
digit
} ;

简洁性大幅提升，实际中更常用。


## 四、派生 (Derivation)


从起始符号出发，反复应用产生式，得到终结符串的过程。


### 最左派生示例


> **Example:** **输入串：**`3 + 5 * 2`
>
>
> expression
>
>
> ↓ expression → expression + term
>
>
> expression + term
>
>
> ↓ expression → term
>
>
> term + term
>
>
> ↓ term → factor
>
>
> factor + term
>
>
> ↓ factor → number
>
>
> number + term
>
>
> ↓ number → digit
>
>
> digit + term
>
>
> ↓ digit → "3"
>
>
> 3 + term
>
>
> ↓ term → term * factor
>
>
> 3 + term * factor
>
>
> ↓ 继续展开 term → 5, factor → 2
>
>
> 3 + 5 * 2


## 五、解析树与抽象语法树 (AST)


### 具体语法树 (CST / Parse Tree)


完整保留文法的所有细节，包括括号、分隔符等。


### 抽象语法树 (AST)


去掉语法糖，只保留语义结构：


```
// 具体语法树对应 "3 + 5 * 2"
    // AST：
              [+]
             /   \
           [3]   [*]
                /   \
              [5]   [2]

    // 编译器内部表示
    BinOpExpr(
        op: "+",
        left: Literal(3),
        right: BinOpExpr(op: "*", left: Literal(5), right: Literal(2))
    )
```


AST比解析树更紧凑，是编译器后续阶段（语义分析、代码生成）的输入。


## 六、实际编程语言的文法片段


### Python简化文法

statement
::=
assignment
|
if_stmt
|
while_stmt
|
expr_stmt
assignment
::=
identifier
"="
expression
if_stmt
::=
"if"
expression
":"
block
[
"else"
":"
block
]
block
::=
NEWLINE
INDENT
{
statement
}
DEDENT

### JSON简化文法

value
::=
object
|
array
|
string
|
number
|
"true"
|
"false"
|
"null"
object
::=
"{"
[
pair
{
","
pair
} ]
"}"
pair
::=
string
":"
value
array
::=
"["
[
value
{
","
value
} ]
"]"


<!-- Converted from: 01_形式文法与BNF.html -->
