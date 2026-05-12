# WASM模块结构

## 一、概念说明

WASM 模块是部署和加载的单位，包含类型、函数、内存、表等部分。

```javascript
(module
  (type $t0 (func (param i32) (result i32)))
  (import "env" "memory" (memory 1))
  (func $f0 (type $t0) ...)
  (table 1 funcref)
  (global $g0 (mut i32) (i32.const 0))
  (export "func" (func $f0))
  (data (i32.const 0) "hello"))
```

## 二、具体用法

### 2.1 模块部分

```javascript
;; WASM 模块的各个部分（按顺序）
(module
  ;; 1. 类型定义
  (type (func (param i32) (result i32)))

  ;; 2. 导入
  (import "env" "memory" (memory 1))
  (import "env" "log" (func $log (param i32)))

  ;; 3. 函数
  (func (type 0) ...)

  ;; 4. 表
  (table 10 funcref)

  ;; 5. 内存
  (memory 1)

  ;; 6. 全局变量
  (global (mut i32) (i32.const 0))

  ;; 7. 导出
  (export "func" (func 0))

  ;; 8. 元素（表初始化）
  (elem (i32.const 0) $func1 $func2)

  ;; 9. 数据（内存初始化）
  (data (i32.const 0) "hello"))
```

### 2.2 自定义段

```javascript
;; 自定义段用于调试信息
(module
  (@custom "name" ...)
  (func $my_func ...)
  (@custom "producers" ...)
  (@custom "sourceMappingURL" ...))
```

## 三、注意事项与常见陷阱

1. **部分顺序**：各部分有固定顺序
2. **自定义段**：可选，不影响执行
3. **导入优先级**：导入必须在定义之前
4. **模块大小**：大型模块需要分片加载
5. **版本兼容**：WASM 有版本号

## 四、WASM 二进制格式详解

WASM 二进制文件以魔数 `\0asm` 开头，后跟版本号：

```
偏移  |  值           |  含义
------+---------------+------------------
0x00  |  00 61 73 6d  |  魔数 "\0asm"
0x04  |  01 00 00 00  |  版本号 1
0x08  |  ...          |  各个段
```

每个段的结构：

```
段ID (1字节) + 段大小 (varuint32) + 段内容
```

常见段 ID：
- `1` = Type（类型定义段）
- `2` = Import（导入段）
- `3` = Function（函数声明段）
- `4` = Table（表段）
- `5` = Memory（内存段）
- `6` = Global（全局变量段）
- `7` = Export（导出段）
- `8` = Start（起始函数段）
- `9` = Element（表初始化段）
- `10` = Code（代码段）
- `11` = Data（数据段）
- `0` = Custom（自定义段）

## 五、使用 wasm-objdump 查看模块结构

```bash
# 查看模块所有段的摘要
wasm-objdump -h module.wasm

# 输出示例：
# Sections:
#   Type start=0x0000000a size=15
#   Function start=0x0000001b size=5
#   Table start=0x00000022 size=4
#   Memory start=0x00000028 size=3
#   Global start=0x0000002d size=9
#   Export start=0x00000038 size=15
#   Code start=0x00000045 size=38
#   Data start=0x0000006d size=15

# 查看详细的导入/导出信息
wasm-objdump -x module.wasm

# 反汇编查看所有函数的 WAT 代码
wasm-objdump -d module.wasm
```

## 六、完整的模块示例

以下是一个包含所有主要段的完整模块：

```wat
(module
  ;; 1. 类型段：定义函数签名
  (type $binop (func (param i32 i32) (result i32)))
  (type $unary (func (param i32) (result i32)))

  ;; 2. 导入段：从宿主环境引入功能
  (import "env" "memory" (memory 1))
  (import "env" "log_i32" (func $log (param i32)))
  (import "env" "abort" (func $abort (param i32 i32 i32 i32)))

  ;; 3. 函数段
  (func $add (type $binop)
    local.get 0
    local.get 1
    i32.add)

  (func $double (type $unary)
    local.get 0
    i32.const 2
    i32.mul)

  ;; 4. 表段：函数引用表
  (table 2 funcref)
  (elem (i32.const 0) $add $double)

  ;; 5. 全局变量段
  (global $counter (mut i32) (i32.const 0))
  (global $max_count i32 (i32.const 1000))

  ;; 6. 导出段
  (export "add" (func $add))
  (export "double" (func $double))
  (export "counter" (global $counter))
  (export "memory" (memory 0))

  ;; 7. 起始函数（模块加载时自动执行）
  ;; (start $init)

  ;; 8. 数据段：初始化内存
  (data (i32.const 0) "Hello, WASM!\00")
  (data (i32.const 100) "\48\65\6c\6c\6f"))  ;; "Hello" 的字节
```

## 七、模块实例化过程

```javascript
// 模块实例化的完整流程
async function instantiateModule(wasmUrl, imports) {
  // 1. 获取并编译模块
  const response = await fetch(wasmUrl);
  const module = await WebAssembly.compileStreaming(response);

  // 2. 检查模块的导入需求
  const requiredImports = WebAssembly.Module.imports(module);
  console.log('需要的导入:', requiredImports);
  // [{module: "env", name: "memory", kind: "memory"}, ...]

  // 3. 检查模块的导出
  const exports = WebAssembly.Module.exports(module);
  console.log('可用的导出:', exports);
  // [{name: "add", kind: "function"}, ...]

  // 4. 实例化
  const instance = await WebAssembly.instantiate(module, imports);

  // 5. 调用起始函数（如果有）并在实例化时自动执行
  return instance;
}

// 使用 WebAssembly.compile 和 WebAssembly.instantiate 的分离 API
// 适合需要在多个 Worker 中共享同一编译结果的场景
const module = await WebAssembly.compile(bytes);
const instance1 = await WebAssembly.instantiate(module, imports1);
const instance2 = await WebAssembly.instantiate(module, imports2);
// 两个实例共享同一编译的模块，但拥有独立的内存和全局变量
```

## 八、模块验证规则

WASM 模块在编译/实例化时必须通过验证：

- 所有类型引用必须指向已定义的类型
- 导入的类型签名必须与声明一致
- 函数体的指令序列必须类型安全
- 内存和表的索引访问不能越界（静态可检测部分）
- 全局变量的可变性必须匹配访问指令
- 元素段和数据段的初始化表达式必须是常量表达式
