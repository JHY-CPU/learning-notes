# 解释器与JIT - 编译原理


# 解释器与JIT编译


Interpreters & Just-In-Time Compilation — 树遍历解释器、字节码VM、JIT、内联缓存

1. 执行方式总览

## 1. 代码执行方式


```
源代码执行的三种路径:

  源代码 ──▶ AOT编译器 ──▶ 机器码 ──▶ CPU直接执行 (C/C++/Rust)
     │
     ├──▶ 解释器 ──▶ 逐条解释执行 (早期Python/Ruby)
     │
     └──▶ 编译器 ──▶ 字节码 ──▶ 虚拟机执行 (Java/Python pyc)
                          │
                          └──▶ JIT编译 ──▶ 热点代码转机器码 (V8/PyPy)
```


| 方式 | 启动速度 | 执行速度 | 代表 |
| --- | --- | --- | --- |
| AOT编译 | 慢 (需编译) | 最快 | C/C++, Rust, Go |
| 纯解释 | 最快 | 慢 | CPython, MRI Ruby |
| 字节码VM | 中等 | 中等 | Java, C#, Lua |
| JIT | 中等 | 接近AOT | V8, PyPy, JVM C2 |

2. 树遍历解释器

## 2. 树遍历解释器 (Tree-Walking Interpreter)


直接在AST上递归求值。简单直观，但效率较低。


```
// 树遍历解释器核心 (Python风格伪代码)
class Interpreter:
    def eval(self, node, env):
        match node.type:

            case "Number":
                return node.value

            case "Variable":
                return env.lookup(node.name)

            case "BinaryOp":
                left = self.eval(node.left, env)
                right = self.eval(node.right, env)
                match node.op:
                    case "+": return left + right
                    case "*": return left * right

            case "Call":
                func = self.eval(node.func, env)
                args = [self.eval(a, env) for a in node.args]
                new_env = func.env.extend(func.params, args)
                return self.eval(func.body, new_env)

            case "If":
                if self.eval(node.cond, env):
                    return self.eval(node.then, env)
                elif node.else_branch:
                    return self.eval(node.else_branch, env)

            case "Function":
                return Closure(node.params, node.body, env)

// 缺点: 每次执行都要遍历AST, 分支预测差, 无法优化
// 优点: 实现简单, 调试方便, 启动快
```

3. 字节码虚拟机

## 3. 字节码虚拟机 (Bytecode VM)


先将AST编译为字节码 (Bytecode)，再由虚拟机解释执行字节码。比树遍历高效。


```
编译过程:

  源代码 → 词法分析 → 语法分析 → AST → 字节码编译器 → 字节码序列
                                          │
                                          ▼
                                    ┌──────────────┐
                                    │   VM主循环    │
                                    │   fetch      │
                                    │   decode     │
                                    │   execute    │
                                    │   loop       │
                                    └──────────────┘
```


```
// 字节码示例: 2 + 3 * 4
// 编译后的字节码:
CONST 2       // 压入2      栈: [2]
CONST 3       // 压入3      栈: [2, 3]
CONST 4       // 压入4      栈: [2, 3, 4]
MUL           // 3*4=12    栈: [2, 12]
ADD           // 2+12=14  栈: [14]
RETURN        // 返回14

// VM核心循环 (C语言)
typedef enum {
    OP_CONST, OP_ADD, OP_MUL, OP_RETURN, OP_CALL, OP_JUMP
} Opcode;

void run(VM* vm) {
    while (true) {
        uint8_t instruction = *vm->ip++;  // fetch
        switch (instruction) {            // decode + execute
            case OP_CONST: {
                Value constant = read_constant(vm);
                push(vm, constant);
                break;
            }
            case OP_ADD: {
                Value b = pop(vm);
                Value a = pop(vm);
                push(vm, a + b);
                break;
            }
            case OP_RETURN: {
                return;  // 退出
            }
        }
    }
}

// 常用指令:
// CONST n    — 压入常量表第n个常量
// LOAD  i    — 加载局部变量i
// STORE i    — 存储到局部变量i
// CALL  n    — 调用函数 (n个参数)
// JUMP  addr — 无条件跳转
// JUMP_IF addr — 条件跳转
```

4. JIT编译

## 4. JIT 编译 (Just-In-Time Compilation)


JIT在运行时将热点代码 (频繁执行的代码) 编译为机器码，兼具解释器的灵活性和编译器的速度。


```
JIT编译的分层执行:

  ┌─────────────────────────────────────────────┐
  │                代码执行路径                    │
  │                                             │
  │  源代码/字节码                                │
  │      │                                      │
  │      ▼                                      │
  │  ┌──────────┐   执行次数少                    │
  │  │ 解释执行   │ ──────────▶ 继续解释           │
  │  └────┬─────┘                                │
  │       │ 执行次数超过阈值                       │
  │       ▼                                      │
  │  ┌──────────┐   基础编译                      │
  │  │ Baseline │ ──────────▶ 执行机器码           │
  │  │ JIT      │            (未优化)              │
  │  └────┬─────┘                                │
  │       │ 继续热执行                            │
  │       ▼                                      │
  │  ┌──────────┐   高级优化                      │
  │  │ Opt. JIT │ ──────────▶ 执行优化机器码       │
  │  │ (C2/FG)  │            (高度优化)            │
  │  └──────────┘                                │
  └─────────────────────────────────────────────┘
```


### JVM的分层编译


```
// JVM分层编译 (Tiered Compilation)
// Level 0: 解释执行
// Level 1: C1编译 (无Profiling) — 快速编译, 基础优化
// Level 2: C1编译 (基本Profiling)
// Level 3: C1编译 (全Profiling)
// Level 4: C2编译 (完全优化) — 慢速编译, 激进优化

// 查看JIT编译信息
java -XX:+PrintCompilation MyApp
// 输出:
//  123   1       3       java.lang.String::hashCode (55 bytes)
//  124   2       3       MyApp::process (120 bytes)
//  125   3       4       MyApp::hotMethod (45 bytes)  ← 被C2优化

// 常见JIT优化:
// 1. 方法内联 (Method Inlining) — 消除调用开销
// 2. 逃逸分析 — 标量替换, 栈分配, 锁消除
// 3. 循环展开 — 减少分支开销
// 4. 常量折叠 — 编译时计算
// 5. 死代码消除 — 删除无用代码
```

5. 内联缓存

## 5. 内联缓存 (Inline Caching)


动态语言每次属性访问都需要查找。内联缓存记住上次查找结果，如果类型没变就直接使用。


```
单态内联缓存 (Monomorphic IC):

  obj.x 的访问:

  首次:   obj是Point类型? → 查找Point.x的偏移 → 缓存: {Point, offset_x}
  再次:   obj是Point类型? → 直接用缓存的offset_x → 跳过哈希查找!
  如果:   obj是其他类型? → 缓存未命中 → 查找并更新缓存

  ┌──────────────────────┐
  │   IC缓存槽            │
  │   Shape: Point       │  ← 上次的类型/形状
  │   Offset: 8          │  ← 上次的偏移量
  │   Hit count: 1000    │
  └──────────────────────┘
```


```
// 内联缓存的实现
struct InlineCache {
    Shape* cached_shape;    // 缓存的对象类型/形状
    size_t cached_offset;   // 缓存的属性偏移
    uint32_t hit_count;     // 命中次数
};

Value get_property(Object* obj, InlineCache* ic) {
    // 快速路径: 类型匹配
    if (obj->shape == ic->cached_shape) {
        return obj->slots[ic->cached_offset];  // 直接偏移访问!
    }
    // 慢速路径: 类型不匹配, 重新查找
    size_t offset = lookup_offset(obj->shape, property_name);
    ic->cached_shape = obj->shape;
    ic->cached_offset = offset;
    return obj->slots[offset];
}

// V8使用Shapes/Hidden Classes:
// { x: 1, y: 2 } 创建时分配Shape
// Shape记录属性名到偏移的映射
// 同结构的对象共享Shape

// 多态内联缓存 (Polymorphic IC):
// 缓存2-4种类型的结果, 超过则退化为哈希表 (Megamorphic)
```

6. 去优化 (Deoptimization)

## 6. 去优化 (Deoptimization)


JIT基于假设进行优化 (如类型不变量)。当假设失效时，需要回退到解释执行或更底层的编译代码。


```
// 去优化示例
// JIT假设: add函数的参数总是整数
function add(a, b) { return a + b; }
// 被优化为: 整数加法 (跳过类型检查)

// 突然调用: add("hello", "world")
// 假设失效! 触发去优化:
// 1. 保存当前机器状态 (寄存器、栈)
// 2. 从优化代码中提取解释器状态
// 3. 跳转到解释执行的入口
// 4. 后续可能用更保守的假设重新JIT

// V8的去优化:
// -bailout_reason 标记去优化原因
// 常见原因: unexpected_type, wrong_map, not_int32

// 查看V8去优化:
// node --trace-deopt app.js
// [deoptimizing (DEOPT eager): begin ... reason: wrong map]
```

7. 隐藏类 (Hidden Classes / Shapes)

## 7. 隐藏类 (Hidden Classes / Shapes)


V8等引擎为动态对象创建隐藏类，使动态语言的对象属性访问速度接近静态语言。


```
隐藏类转换链:

  new Point()          obj.x = 1              obj.y = 2
      │                    │                      │
      ▼                    ▼                      ▼
  ┌──────────┐       ┌──────────┐           ┌──────────┐
  │ Map 0    │──────▶│ Map 1    │──────────▶│ Map 2    │
  │ (空对象)  │       │ x: 偏移0 │           │ x: 偏移0 │
  └──────────┘       └──────────┘           │ y: 偏移1 │
                                            └──────────┘

  所有结构相同的Point对象共享Map 2
  IC缓存Map2 → 偏移, 实现快速属性访问
```


```
// V8 Shapes的优化
// 相同结构的对象共享Hidden Class
let a = { x: 1, y: 2 };  // Map A: {x→0, y→1}
let b = { x: 3, y: 4 };  // 共享 Map A!
let c = { y: 5, x: 6 };  // Map C: {y→0, x→1} ← 顺序不同!

// 建议: 以相同顺序初始化对象属性
// 以不同顺序创建的对象不能共享Shape
// JIT的IC缓存会频繁未命中
```


树遍历解释器
字节码
VM
JIT
内联缓存
去优化
隐藏类
分层编译


<!-- Converted from: 03_解释器与JIT.html -->
