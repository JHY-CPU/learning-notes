# WABT工具链

## 一、概念说明

WABT（WebAssembly Binary Toolkit）是 WASM 的官方工具集，提供 WAT 和 WASM 之间的转换。

```bash
# 安装 WABT
# Ubuntu
sudo apt install wabt

# macOS
brew install wabt

# 或从源码编译
git clone --recursive https://github.com/WebAssembly/wabt
cd wabt && mkdir build && cd build
cmake .. && make
```

## 二、具体用法

### 2.1 基本工具

```bash
# wat2wasm: WAT 转 WASM
wat2wasm example.wat -o example.wasm

# wasm2wat: WASM 转 WAT
wasm2wat example.wasm -o example.wat

# wasm-objdump: 查看 WASM 结构
wasm-objdump -x example.wasm
wasm-objdump -d example.wasm  # 反汇编

# wasm-validate: 验证 WASM
wasm-validate example.wasm

# wasm-strip: 移除调试信息
wasm-strip example.wasm -o stripped.wasm
```

### 2.2 高级工具

```bash
# wasm-interp: 解释执行
wasm-interp example.wasm

# wasm-decompile: 反编译为类 C 代码
wasm-decompile example.wasm -o example.c

# wasm2c: 转换为 C 代码
wasm2c example.wasm -o example.c

# wast2json: 转换测试套件
wast2json test.wast -o test.json
```

### 2.3 脚本示例

```bash
#!/bin/bash
# 构建脚本
wat2wasm src/module.wat -o dist/module.wasm
wasm2wat dist/module.wasm -o dist/module.wat  # 生成可读版本
wasm-objdump -x dist/module.wasm > dist/module.info
wasm-strip dist/module.wasm -o dist/module.min.wasm
```

## 三、注意事项与常见陷阱

1. **版本兼容**：确保 WABT 版本与 WASM 规范兼容
2. **调试信息**：strip 工具会移除调试信息
3. **验证重要性**：始终验证生成的 WASM
4. **性能**：大型 WASM 文件处理可能较慢
5. **平台支持**：WABT 跨平台支持良好

## 四、wasm-objdump 深入用法

```bash
# 查看模块头部信息
wasm-objdump -h module.wasm

# 查看完整的导入/导出表
wasm-objdump -x module.wasm

# 输出示例：
# Section Details:
#
# Type[2]:
#  - type[0] (i32, i32) -> i32
#  - type[1] (i32) -> i32
# Function[3]:
#  - func[0] sig=0 <add>
#  - func[1] sig=1 <factorial>
#  - func[2] sig=0 <multiply>
# Memory[1]:
#  - memory[0] pages: initial=1 max=10
# Export[2]:
#  - func[0] <add> -> "add"
#  - memory[0] -> "memory"

# 反汇编指定函数
wasm-objdump -d module.wasm | grep -A 20 "func\[0\]"

# 查看自定义段（调试信息）
wasm-objdump -j name -x module.wasm

# 查看数据段内容
wasm-objdump -j data -x module.wasm
```

## 五、wast2json 用于测试

```bash
# WAST 文件是 WASM 的测试格式
# test.wast
(assert_return (invoke "add" (i32.const 1) (i32.const 2)) (i32.const 3))
(assert_return (invoke "factorial" (i32.const 5)) (i32.const 120))
(assert_trap (invoke "divide" (i32.const 1) (i32.const 0)) "integer divide by zero")

# 转换为 JSON 格式
wast2json test.wast -o test.json

# 输出：test.json + 多个 .wasm 文件
# 适用于自动化测试框架

# 运行官方测试套件
# 克隆 spec 仓库
git clone https://github.com/WebAssembly/spec
cd spec
# 运行测试
make test
```

## 六、wasm-interp 解释执行

```bash
# 不经过编译，直接解释执行 WASM
wasm-interp module.wasm

# 带参数调用
wasm-interp module.wasm --invoke add 1 2
# 输出: 3

# 启用 WASI 支持
wasm-interp module.wasm --wasi

# 跟踪执行（调试用）
wasm-interp module.wasm --trace
# 输出每条指令的执行日志

# 运行多模块
wasm-interp module1.wasm module2.wasm
```

## 七、wasm2c 转换为 C 代码

```bash
# 将 WASM 转换为可编译的 C 代码
wasm2c module.wasm -o module.c

# 编译生成的 C 代码
gcc -O2 -o program module.c wasm-rt-impl.c -lm

# 用途：
# 1. 在不支持 WASM 的平台上运行
# 2. 使用 C 编译器进一步优化
# 3. 安全审计（检查生成的 C 代码）
```

## 八、WAT 语法参考速查

```wat
;; 注释
(; 块注释 ;)

;; 模块声明
(module ...)

;; 类型定义
(type (func (param i32 i32) (result i32)))

;; 导入
(import "module" "name" (func $name (param i32) (result i32)))
(import "env" "memory" (memory 1 10))
(import "env" "table" (table 10 funcref))
(import "env" "global" (global $g (mut i32)))

;; 导出
(export "name" (func $name))
(export "memory" (memory 0))
(export "table" (table 0))
(export "global" (global $g))

;; 全局变量
(global $g (mut i32) (i32.const 0))
(global $const f64 (f64.const 3.141592653589793))

;; 表和元素
(table 10 funcref)
(elem (i32.const 0) $func1 $func2)

;; 内存和数据
(memory 1 10)
(data (i32.const 0) "hello\00")
(data $d1 "world")

;; 起始函数
(start $init)
```
