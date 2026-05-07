# 44-编程语言理论

本目录系统整理编程语言理论（Programming Language Theory, PLT）的核心知识体系，涵盖从形式化语法定义到类型系统、从Lambda演算到多种编程范式的全面内容。

## 目录结构

| 序号 | 主题 | 说明 |
|------|------|------|
| 01 | [语法与语义](./01-语法与语义/) | 形式文法（BNF/EBNF）、操作语义、指称语义、公理语义、求值策略 |
| 02 | [类型系统](./02-类型系统/) | 静态/动态类型、强/弱类型、类型推导、多态、代数数据类型、依赖类型 |
| 03 | [Lambda演算](./03-Lambda演算/) | λ演算语法与归约、Church编码、有类型λ演算、System F、Currying |
| 04 | [编程范式](./04-编程范式/) | 命令式、函数式、逻辑式、面向对象、并发模型、FRP、元编程、DSL |

## 核心概念导览

### 形式化基础

编程语言理论的基石是对**语法**（Syntax）和**语义**（Semantics）的形式化描述。语法定义了"什么样的程序是合法的"，语义定义了"合法的程序做什么"。BNF范式提供了简洁的文法描述手段，而操作语义、指称语义和公理语义从不同角度刻画程序行为。

### 类型系统

类型系统是程序正确性的第一道防线。静态类型在编译时捕获错误，动态类型在运行时检查。Hindley-Milner类型系统实现了强大的类型推导，使得程序员无需手动标注即可享受静态类型的好处。代数数据类型（ADT）为数据建模提供了数学化的方式。

### Lambda演算

Lambda演算是所有函数式编程语言的理论基础。Alonzo Church在1930年代提出的这一计算模型证明了：仅通过函数抽象和应用即可实现所有可计算函数。Church编码展示了如何用纯λ表达式表示自然数、布尔值、列表等数据结构。

### 编程范式

不同的编程范式代表了不同的计算世界观。命令式编程关注"如何做"，函数式编程关注"做什么"，逻辑式编程关注"什么是真的"。理解各范式的核心思想、优势与局限，有助于在实际工程中做出合理的技术选型。

## 学习路径建议

```
语法与语义（形式化基础）
    ↓
Lambda演算（计算理论基础）
    ↓
类型系统（程序正确性保证）
    ↓
编程范式（实践方法论）
```

## 参考资料

- Benjamin C. Pierce, *Types and Programming Languages* (TAPL)
- Robert Harper, *Practical Foundations for Programming Languages* (PFPL)
- Harold Abelson & Gerald Jay Sussman, *Structure and Interpretation of Computer Programs* (SICP)
- Simon Peyton Jones, *The Implementation of Functional Programming Languages*
- Glynn Winskel, *The Formal Semantics of Programming Languages*
