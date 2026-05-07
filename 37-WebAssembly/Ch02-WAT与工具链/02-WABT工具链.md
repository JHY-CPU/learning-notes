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
