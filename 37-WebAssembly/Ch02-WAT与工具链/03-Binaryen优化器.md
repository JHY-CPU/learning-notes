# Binaryen优化器

## 一、概念说明

Binaryen 是 WASM 的编译器基础设施，提供优化和代码生成工具。

```bash
# 安装 Binaryen
# Ubuntu
sudo apt install binaryen

# macOS
brew install binaryen

# 主要工具
# wasm-opt: WASM 优化器
# wasm-as: WAT 转 WASM
# wasm-dis: WASM 转 WAT
```

## 二、具体用法

### 2.1 优化级别

```bash
# 基本优化
wasm-opt -O1 input.wasm -o output.wasm

# 中等优化
wasm-opt -O2 input.wasm -o output.wasm

# 高级优化
wasm-opt -O3 input.wasm -o output.wasm

# 大小优化
wasm-opt -Oz input.wasm -o output.wasm

# 速度优化
wasm-opt -Os input.wasm -o output.wasm
```

### 2.2 特定优化

```bash
# 启用特定优化
wasm-opt --enable-simd input.wasm -o output.wasm
wasm-opt --enable-threads input.wasm -o output.wasm
wasm-opt --enable-bulk-memory input.wasm -o output.wasm

# 禁用特定优化
wasm-opt -O3 --no-fp16 input.wasm -o output.wasm
```

### 2.3 分析工具

```bash
# 查看优化效果
wasm-opt -O3 input.wasm -o optimized.wasm --print
wasm-opt input.wasm --print  # 查看中间表示

# 比较大小
wc -c input.wasm optimized.wasm
```

## 三、注意事项与常见陷阱

1. **优化时间**：高级优化可能很慢
2. **调试影响**：优化可能影响调试
3. **正确性**：优化不应改变程序行为
4. **版本更新**：Binaryen 更新可能改变优化行为
5. **测试**：优化后需要充分测试

## 四、优化级别详解

```bash
# -O0: 无优化（默认）
wasm-opt -O0 input.wasm -o output.wasm

# -O1: 轻度优化
# - 常量折叠
# - 死代码消除
# - 简单的表达式简化
wasm-opt -O1 input.wasm -o output.wasm

# -O2: 中等优化
# - 函数内联（小函数）
# - 循环优化
# - 公共子表达式消除
wasm-opt -O2 input.wasm -o output.wasm

# -O3: 高级优化（可能增大体积）
# - 激进的函数内联
# - 循环展开
# - 向量化
wasm-opt -O3 input.wasm -o output.wasm

# -Os: 优化大小
# - 减小代码体积
# - 适合 Web 传输
wasm-opt -Os input.wasm -o output.wasm

# -Oz: 极致大小优化
# - 比 -Os 更激进
# - 可能牺牲一些性能
wasm-opt -Oz input.wasm -o output.wasm

# -O4: 速度优先，可能增大体积
wasm-opt -O4 input.wasm -o output.wasm
```

## 五、高级优化选项

```bash
# 启用所有提案
wasm-opt -O3 \
  --enable-simd \
  --enable-threads \
  --enable-bulk-memory \
  --enable-mutable-globals \
  --enable-multivalue \
  --enable-tail-call \
  input.wasm -o output.wasm

# 查看所有优化 passes
wasm-opt --print input.wasm

# 查看每个 pass 之后的中间表示
wasm-opt --print-after-all input.wasm -o /dev/null

# 只运行特定 pass
wasm-opt --pass-arg=flatten -O3 input.wasm -o output.wasm

# 调试优化过程
wasm-opt --debug -O3 input.wasm -o output.wasm 2> debug.log
```

## 六、wasm-as 和 wasm-dis

```bash
# wasm-as: WAT → WASM（编译）
wasm-as input.wat -o output.wasm

# wasm-dis: WASM → WAT（反编译）
wasm-dis input.wasm -o output.wat

# 往返测试：确保编译/反编译不丢失信息
wasm-as input.wat -o temp.wasm
wasm-dis temp.wasm -o roundtrip.wat
# 比较 input.wat 和 roundtrip.wat
```

## 七、与 Emscripten 集成

```bash
# Emscripten 内置 Binaryen
# 在编译时自动优化
emcc main.c -O2 -o app.js  # 默认使用 Binaryen 优化

# 跳过 Binaryen 优化
emcc main.c -O2 -o app.js -s BINARYEN_IGNORE_IMPLICIT_TRAPS=1

# 单独对已有的 WASM 运行 Binaryen
wasm-opt app.wasm -O3 -o app.opt.wasm

# 在构建脚本中集成
#!/bin/bash
emcc main.c -O1 -o app.wasm
wasm-opt -O3 --enable-simd app.wasm -o app.opt.wasm
wasm-strip app.opt.wasm
echo "原始大小: $(wc -c < app.wasm) bytes"
echo "优化后: $(wc -c < app.opt.wasm) bytes"
```

## 八、常见优化效果示例

```bash
# 对典型项目运行优化并对比大小
echo "=== 优化前 ==="
wasm-objdump -h input.wasm | grep -E "section|total"

echo "=== -Os 优化 ==="
wasm-opt -Os input.wasm -o opt-s.wasm
wasm-objdump -h opt-s.wasm | grep -E "section|total"

echo "=== -O3 优化 ==="
wasm-opt -O3 input.wasm -o opt-3.wasm
wasm-objdump -h opt-3.wasm | grep -E "section|total"

# 典型结果：
# 原始: 125KB
# -Os: 89KB (-29%)
# -O3: 95KB (-24%，但运行更快)
```
