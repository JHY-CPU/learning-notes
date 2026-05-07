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
