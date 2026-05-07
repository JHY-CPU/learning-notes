# Emscripten入门

## 一、概念说明

Emscripten 是将 C/C++ 编译为 WASM 的工具链，提供完整的 POSIX 兼容层。

```bash
# 安装 Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

## 二、具体用法

### 2.1 基本编译

```bash
# 编译 C 文件
emcc hello.c -o hello.html

# 生成 WASM 和 JavaScript
emcc hello.c -o hello.js -s WASM=1

# 只生成 WASM
emcc hello.c -o hello.wasm -s STANDALONE_WASM

# 优化编译
emcc hello.c -O2 -o hello.js
```

### 2.2 链接选项

```bash
# 链接数学库
emcc calc.c -lm -o calc.js

# 链接多个文件
emcc main.c utils.c -o app.js

# 使用 SDL
emcc game.c -s USE_SDL=2 -o game.js
```

### 2.3 高级选项

```bash
# 导出函数
emcc lib.c -s EXPORTED_FUNCTIONS='["_add","_subtract"]' -o lib.js

# 预加载文件
emcc game.c --preload-file assets -o game.js

# 允许内存增长
emcc app.c -s ALLOW_MEMORY_GROWTH=1 -o app.js

# 最大内存限制
emcc app.c -s MAXIMUM_MEMORY=1GB -o app.js
```

### 2.4 文件控制

```bash
# 生成独立 HTML
emcc app.c -o app.html --shell-file template.html

# 生成最小 HTML
emcc app.c -o app.html -s MINIMAL_RUNTIME=1

# 禁用文件系统
emcc calc.c -s NO_FILESYSTEM=1 -o calc.js
```

## 三、注意事项与常见陷阱

1. **C++ 支持**：使用 em++ 编译 C++ 文件
2. **异步加载**：使用 onRuntimeInitialized 回调
3. **内存管理**：注意内存泄漏
4. **文件大小**：优化编译减小文件大小
5. **兼容性**：测试不同浏览器兼容性
