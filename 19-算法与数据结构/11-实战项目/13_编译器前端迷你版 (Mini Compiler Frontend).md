# 简易编译器前端 (Mini Compiler Frontend)

## 项目需求与功能分析

编译器是将高级语言翻译为机器语言的程序。本项目实现一个简易编译器的前端部分，包括词法分析器（Lexer）、语法分析器（Parser）和抽象语法树（AST），支持一个简单表达式语言。

### 核心功能

- 词法分析（Tokenization）
- 递归下降语法分析
- AST 构建与可视化
- AST 解释执行
- 类型检查

### 目标语言

支持变量声明、赋值、算术运算、条件判断和 print 语句：

```
let x = 10;
let y = x + 5 * 2;
if (x > 5) { print(x); }
```

## 核心算法原理

### 词法分析

将源代码字符流转换为 Token 流。每个 Token 包含类型和值。

### 递归下降解析

根据文法规则递归调用对应函数，自顶向下构建 AST。

文法定义：
```
program    -> statement*
statement  -> let_stmt | assign_stmt | if_stmt | print_stmt | expr_stmt
let_stmt   -> 'let' IDENT '=' expression ';'
expression -> term (('+'|'-') term)*
term       -> factor (('*'|'/') factor)*
factor     -> NUMBER | STRING | IDENT | '(' expression ')'
```

## 完整代码实现

```python
from enum import Enum, auto
from typing import List, Any, Dict, Optional
from dataclasses import dataclass


# ===== Token 类型 =====

class TK(Enum):
    NUMBER=auto(); STRING=auto(); IDENT=auto()
    LET=auto(); IF=auto(); ELSE=auto(); PRINT=auto()
    PLUS=auto(); MINUS=auto(); MUL=auto(); DIV=auto()
    EQ=auto(); NEQ=auto(); LT=auto(); GT=auto(); LE=auto(); GE=auto()
    ASSIGN=auto(); LPAREN=auto(); RPAREN=auto()
    LBRACE=auto(); RBRACE=auto(); SEMI=auto()
    EOF=auto()


@dataclass
class Token:
    type: TK; value: Any; line: int


# ===== 词法分析器 =====

class Lexer:
    KEYWORDS = {'let': TK.LET, 'if': TK.IF, 'else': TK.ELSE, 'print': TK.PRINT}

    def __init__(self, source: str):
        self.src = source; self.pos = 0; self.line = 1

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.src):
            ch = self.src[self.pos]
            if ch in ' \t': self.pos += 1; continue
            if ch == '\n': self.line += 1; self.pos += 1; continue
            if ch == '/' and self.pos+1 < len(self.src) and self.src[self.pos+1] == '/':
                while self.pos < len(self.src) and self.src[self.pos] != '\n': self.pos += 1
                continue
            if ch.isdigit(): tokens.append(self._number()); continue
            if ch == '"': tokens.append(self._string()); continue
            if ch.isalpha() or ch == '_': tokens.append(self._ident()); continue
            # 运算符和标点
            two = ch + (self.src[self.pos+1] if self.pos+1 < len(self.src) else '')
            two_map = {'==':TK.EQ,'!=':TK.NEQ,'<=':TK.LE,'>=':TK.GE}
            if two in two_map:
                tokens.append(Token(two_map[two], two, self.line)); self.pos += 2; continue
            one_map = {'+':TK.PLUS,'-':TK.MINUS,'*':TK.MUL,'/':TK.DIV,
                       '(':TK.LPAREN,')':TK.RPAREN,'{':TK.LBRACE,'}':TK.RBRACE,
                       ';':TK.SEMI,'=':TK.ASSIGN,'<':TK.LT,'>':TK.GT}
            if ch in one_map:
                tokens.append(Token(one_map[ch], ch, self.line)); self.pos += 1; continue
            raise SyntaxError(f"未知字符 '{ch}' (行 {self.line})")
        tokens.append(Token(TK.EOF, None, self.line))
        return tokens

    def _number(self):
        s = self.pos; dot = False
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c.isdigit(): self.pos += 1
            elif c == '.' and not dot: dot = True; self.pos += 1
            else: break
        return Token(TK.NUMBER, float(self.src[s:self.pos]), self.line)

    def _string(self):
        self.pos += 1; s = self.pos
        while self.pos < len(self.src) and self.src[self.pos] != '"': self.pos += 1
        val = self.src[s:self.pos]; self.pos += 1
        return Token(TK.STRING, val, self.line)

    def _ident(self):
        s = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == '_'):
            self.pos += 1
        name = self.src[s:self.pos]
        tk_type = self.KEYWORDS.get(name, TK.IDENT)
        return Token(tk_type, name, self.line)


# ===== AST 节点 =====

@dataclass
class NumberNode:
    value: float

@dataclass
class StringNode:
    value: str

@dataclass
class IdentNode:
    name: str

@dataclass
class BinOpNode:
    op: str; left: Any; right: Any

@dataclass
class LetNode:
    name: str; value: Any

@dataclass
class AssignNode:
    name: str; value: Any

@dataclass
class IfNode:
    condition: Any; body: List[Any]; else_body: Optional[List[Any]]

@dataclass
class PrintNode:
    value: Any

@dataclass
class ProgramNode:
    statements: List[Any]


# ===== 语法分析器 =====

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens; self.pos = 0

    def current(self): return self.tokens[self.pos]

    def eat(self, expected):
        tok = self.current()
        if tok.type != expected:
            raise SyntaxError(f"期望 {expected.name}, 得到 {tok.type.name} (行 {tok.line})")
        self.pos += 1; return tok

    def parse(self) -> ProgramNode:
        stmts = []
        while self.current().type != TK.EOF:
            stmts.append(self.statement())
        return ProgramNode(stmts)

    def statement(self):
        if self.current().type == TK.LET: return self.let_stmt()
        if self.current().type == TK.IF: return self.if_stmt()
        if self.current().type == TK.PRINT: return self.print_stmt()
        # 赋值或表达式语句
        if self.current().type == TK.IDENT:
            name = self.current().value
            self.pos += 1
            if self.current().type == TK.ASSIGN:
                self.pos += 1
                expr = self.expression()
                self.eat(TK.SEMI)
                return AssignNode(name, expr)
            self.pos -= 1  # 回退
        expr = self.expression()
        self.eat(TK.SEMI)
        return expr

    def let_stmt(self):
        self.eat(TK.LET)
        name = self.eat(TK.IDENT).value
        self.eat(TK.ASSIGN)
        expr = self.expression()
        self.eat(TK.SEMI)
        return LetNode(name, expr)

    def if_stmt(self):
        self.eat(TK.IF)
        self.eat(TK.LPAREN)
        cond = self.expression()
        self.eat(TK.RPAREN)
        self.eat(TK.LBRACE)
        body = []
        while self.current().type != TK.RBRACE:
            body.append(self.statement())
        self.eat(TK.RBRACE)
        else_body = None
        if self.current().type == TK.ELSE:
            self.pos += 1; self.eat(TK.LBRACE)
            else_body = []
            while self.current().type != TK.RBRACE:
                else_body.append(self.statement())
            self.eat(TK.RBRACE)
        return IfNode(cond, body, else_body)

    def print_stmt(self):
        self.eat(TK.PRINT); self.eat(TK.LPAREN)
        expr = self.expression()
        self.eat(TK.RPAREN); self.eat(TK.SEMI)
        return PrintNode(expr)

    def expression(self):
        left = self.term()
        while self.current().type in (TK.PLUS, TK.MINUS, TK.EQ, TK.NEQ, TK.LT, TK.GT, TK.LE, TK.GE):
            op = self.current().value; self.pos += 1
            right = self.term()
            left = BinOpNode(op, left, right)
        return left

    def term(self):
        left = self.factor()
        while self.current().type in (TK.MUL, TK.DIV):
            op = self.current().value; self.pos += 1
            right = self.factor()
            left = BinOpNode(op, left, right)
        return left

    def factor(self):
        tok = self.current()
        if tok.type == TK.NUMBER: self.pos += 1; return NumberNode(tok.value)
        if tok.type == TK.STRING: self.pos += 1; return StringNode(tok.value)
        if tok.type == TK.IDENT: self.pos += 1; return IdentNode(tok.value)
        if tok.type == TK.MINUS: self.pos += 1; return BinOpNode('-', NumberNode(0), self.factor())
        if tok.type == TK.LPAREN:
            self.pos += 1; expr = self.expression(); self.eat(TK.RPAREN); return expr
        raise SyntaxError(f"意外的 token: {tok.type.name} (行 {tok.line})")


# ===== 解释器 =====

class Interpreter:
    def __init__(self):
        self.env: Dict[str, Any] = {}

    def run(self, node):
        if isinstance(node, ProgramNode):
            for stmt in node.statements: self.run(stmt)
        elif isinstance(node, NumberNode): return node.value
        elif isinstance(node, StringNode): return node.value
        elif isinstance(node, IdentNode):
            if node.name not in self.env: raise NameError(f"未定义: {node.name}")
            return self.env[node.name]
        elif isinstance(node, BinOpNode):
            l, r = self.run(node.left), self.run(node.right)
            ops = {'+':lambda a,b:a+b, '-':lambda a,b:a-b, '*':lambda a,b:a*b,
                   '/':lambda a,b:a/b, '==':lambda a,b:a==b, '!=':lambda a,b:a!=b,
                   '<':lambda a,b:a<b, '>':lambda a,b:a>b, '<=':lambda a,b:a<=b, '>=':lambda a,b:a>=b}
            return ops[node.op](l, r)
        elif isinstance(node, LetNode):
            self.env[node.name] = self.run(node.value)
        elif isinstance(node, AssignNode):
            if node.name not in self.env: raise NameError(f"未定义: {node.name}")
            self.env[node.name] = self.run(node.value)
        elif isinstance(node, IfNode):
            if self.run(node.condition):
                for s in node.body: self.run(s)
            elif node.else_body:
                for s in node.else_body: self.run(s)
        elif isinstance(node, PrintNode):
            print(self.run(node.value))
```

## 测试用例

```python
import unittest

class TestCompiler(unittest.TestCase):
    def run_code(self, code):
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse()
        interp = Interpreter()
        interp.run(ast)
        return interp.env

    def test_let_assign(self):
        env = self.run_code("let x = 10; let y = x + 5;")
        self.assertEqual(env['x'], 10)
        self.assertEqual(env['y'], 15)

    def test_arithmetic(self):
        env = self.run_code("let r = (3 + 4) * 2 - 1;")
        self.assertEqual(env['r'], 13)

    def test_if(self):
        env = self.run_code("let x = 5; if (x > 3) { let y = 1; } else { let y = 0; }")
        self.assertEqual(env['y'], 1)

    def test_comparison(self):
        env = self.run_code("let a = 1 == 1; let b = 1 != 1;")
        self.assertTrue(env['a'])
        self.assertFalse(env['b'])

    def test_lexer_error(self):
        with self.assertRaises(SyntaxError):
            Lexer("let x = @;").tokenize()

    def test_parser_error(self):
        with self.assertRaises(SyntaxError):
            tokens = Lexer("let x = ;").tokenize()
            Parser(tokens).parse()

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **函数定义**：支持自定义函数和函数调用
2. **循环语句**：支持 while 和 for 循环
3. **数组和对象**：扩展数据类型
4. **代码生成**：将 AST 编译为字节码或机器码
5. **错误报告**：更精确的错误位置和信息
6. **优化遍**：AST 优化（常量折叠、死代码消除）
7. **REPL**：交互式解释器
