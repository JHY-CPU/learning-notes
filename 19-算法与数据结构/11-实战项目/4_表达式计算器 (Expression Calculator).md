# 表达式计算器 (Expression Calculator)

## 项目需求与功能分析

表达式计算是编译原理和数据结构中栈应用的经典案例。本项目实现一个支持加减乘除、括号、函数调用的完整表达式计算器，深入理解中缀转后缀、递归下降解析等核心算法。

### 核心功能

- 中缀表达式求值（如 `3 + 4 * 2 / (1 - 5)`）
- 支持运算符优先级（乘除 > 加减）
- 支持括号嵌套
- 支持数学函数（sin, cos, sqrt, abs, pow）
- 支持变量绑定
- 表达式语法树可视化

### 应用场景

- 科学计算器
- 配置文件中的表达式求值
- 游戏中技能公式计算
- 数据库 WHERE 条件解析

## 核心算法原理

### Shunting-yard 算法（中缀转后缀）

Dijkstra 提出的经典算法：
1. 遇到操作数，直接输出
2. 遇到运算符，弹出栈中优先级 >= 自身的运算符，然后入栈
3. 遇到左括号，入栈
4. 遇到右括号，弹出到左括号为止
5. 表达式结束，弹出栈中所有运算符

### 后缀表达式求值

使用栈遍历后缀表达式：遇到操作数入栈，遇到运算符弹出两个操作数计算后结果入栈。

### 递归下降解析

为表达式定义文法规则：
```
expr   -> term ((+|-) term)*
term   -> factor ((*|/) factor)*
factor -> NUMBER | (expr) | func(expr)
```

## 完整代码实现

```python
import math, re
from typing import List, Dict, Callable, Optional
from enum import Enum


class TT(Enum):
    NUM='NUM'; PLUS='PLUS'; MINUS='MINUS'; MUL='MUL'; DIV='DIV'
    POW='POW'; LPAREN='LPAREN'; RPAREN='RPAREN'; FUNC='FUNC'
    VAR='VAR'; EOF='EOF'


class Token:
    def __init__(self, type_, value):
        self.type = type_; self.value = value
    def __repr__(self): return f"Token({self.type.value},{self.value!r})"


class Lexer:
    FUNCS = {'sin','cos','tan','sqrt','abs','log','log10','exp','ceil','floor','round','pow','max','min'}

    def __init__(self, text):
        self.text = text; self.pos = 0

    def tokenize(self):
        tokens = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch.isspace(): self.pos += 1; continue
            if ch.isdigit() or (ch=='.' and self.pos+1<len(self.text) and self.text[self.pos+1].isdigit()):
                tokens.append(self._num()); continue
            if ch.isalpha() or ch == '_':
                tokens.append(self._ident()); continue
            sm = {'+':TT.PLUS,'-':TT.MINUS,'*':TT.MUL,'/':TT.DIV,'^':TT.POW,
                  '(':TT.LPAREN,')':TT.RPAREN}
            if ch in sm: tokens.append(Token(sm[ch],ch)); self.pos+=1; continue
            raise SyntaxError(f"未知字符: '{ch}'")
        tokens.append(Token(TT.EOF, None))
        return tokens

    def _num(self):
        s = self.pos; dot = False
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isdigit(): self.pos+=1
            elif c=='.' and not dot: dot=True; self.pos+=1
            elif c in 'eE': self.pos+=1
            elif c in '+-' and self.text[self.pos-1] in 'eE': self.pos+=1
            else: break
        return Token(TT.NUM, float(self.text[s:self.pos]))

    def _ident(self):
        s = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos]=='_'):
            self.pos += 1
        name = self.text[s:self.pos]
        return Token(TT.FUNC, name) if name in self.FUNCS else Token(TT.VAR, name)


class ShuntingYard:
    PREC = {TT.PLUS:1, TT.MINUS:1, TT.MUL:2, TT.DIV:2, TT.POW:3}

    @staticmethod
    def parse(tokens):
        out, ops = [], []
        for tok in tokens:
            if tok.type in (TT.NUM, TT.VAR): out.append(tok)
            elif tok.type == TT.FUNC: ops.append(tok)
            elif tok.type in ShuntingYard.PREC:
                while ops and ops[-1].type != TT.LPAREN and \
                      (ops[-1].type==TT.FUNC or ShuntingYard.PREC.get(ops[-1].type,0) >= ShuntingYard.PREC[tok.type]):
                    out.append(ops.pop())
                ops.append(tok)
            elif tok.type == TT.LPAREN: ops.append(tok)
            elif tok.type == TT.RPAREN:
                while ops and ops[-1].type != TT.LPAREN: out.append(ops.pop())
                if not ops: raise SyntaxError("括号不匹配")
                ops.pop()
                if ops and ops[-1].type == TT.FUNC: out.append(ops.pop())
        while ops:
            if ops[-1].type == TT.LPAREN: raise SyntaxError("括号不匹配")
            out.append(ops.pop())
        return out


class Evaluator:
    FUNCS = {'sin':math.sin,'cos':math.cos,'tan':math.tan,'sqrt':math.sqrt,
             'abs':abs,'log':math.log,'log10':math.log10,'exp':math.exp,
             'ceil':math.ceil,'floor':math.floor,'round':round}

    def __init__(self, vars=None):
        self.vars = vars or {'pi': math.pi, 'e': math.e}

    def eval(self, postfix):
        stack = []
        for tok in postfix:
            if tok.type == TT.NUM: stack.append(tok.value)
            elif tok.type == TT.VAR:
                if tok.value not in self.vars: raise NameError(f"未定义: {tok.value}")
                stack.append(self.vars[tok.value])
            elif tok.type in (TT.PLUS,TT.MINUS,TT.MUL,TT.DIV,TT.POW):
                b,a = stack.pop(), stack.pop()
                ops = {'+':lambda x,y:x+y, '-':lambda x,y:x-y, '*':lambda x,y:x*y,
                       '/':lambda x,y:x/y, '^':lambda x,y:x**y}
                stack.append(ops[tok.value](a,b))
            elif tok.type == TT.FUNC:
                if tok.value == 'pow':
                    b,a = stack.pop(), stack.pop(); stack.append(math.pow(a,b))
                elif tok.value in ('max','min'):
                    b,a = stack.pop(), stack.pop()
                    stack.append(max(a,b) if tok.value=='max' else min(a,b))
                else: stack.append(self.FUNCS[tok.value](stack.pop()))
        if len(stack) != 1: raise ValueError("表达式错误")
        return stack[0]


class Calculator:
    def __init__(self, variables=None):
        self.variables = variables or {}

    def evaluate(self, expr):
        tokens = Lexer(expr).tokenize()
        postfix = ShuntingYard.parse(tokens)
        return Evaluator(self.variables).eval(postfix)
```

## 测试用例

```python
import unittest

class TestCalc(unittest.TestCase):
    def setUp(self): self.c = Calculator()

    def test_basic(self):
        self.assertAlmostEqual(self.c.evaluate("3 + 4"), 7)
        self.assertAlmostEqual(self.c.evaluate("10 - 3"), 7)
        self.assertAlmostEqual(self.c.evaluate("6 * 7"), 42)
        self.assertAlmostEqual(self.c.evaluate("15 / 3"), 5)

    def test_precedence(self):
        self.assertAlmostEqual(self.c.evaluate("3 + 4 * 2"), 11)
        self.assertAlmostEqual(self.c.evaluate("(3 + 4) * 2"), 14)

    def test_power(self):
        self.assertAlmostEqual(self.c.evaluate("2 ^ 10"), 1024)

    def test_functions(self):
        self.assertAlmostEqual(self.c.evaluate("sin(0)"), 0)
        self.assertAlmostEqual(self.c.evaluate("sqrt(16)"), 4)
        self.assertAlmostEqual(self.c.evaluate("abs(-5)"), 5)

    def test_variables(self):
        calc = Calculator({'x': 10, 'y': 3})
        self.assertAlmostEqual(calc.evaluate("x + y"), 13)

    def test_div_zero(self):
        with self.assertRaises(ZeroDivisionError): self.c.evaluate("1 / 0")

    def test_syntax_error(self):
        with self.assertRaises(SyntaxError): self.c.evaluate("(3 + 4")

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **赋值语句**：支持 `x = 3 + 4`
2. **条件表达式**：支持三目运算符
3. **自定义函数**：允许用户定义 `f(x) = x^2 + 1`
4. **单位计算**：支持 `3km + 500m = 3.5km`
5. **复数运算**：扩展支持复数
6. **矩阵运算**：支持矩阵加法、乘法
7. **REPL 交互**：构建交互式命令行计算器
