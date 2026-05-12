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

## 四、C++ 示例

```cpp
// main.cpp
#include <emscripten.h>
#include <cmath>
#include <vector>
#include <string>

extern "C" {

// 导出函数供 JavaScript 调用
EMSCRIPTEN_KEEPALIVE
double calculate_pi(int iterations) {
    double pi = 0.0;
    for (int i = 0; i < iterations; i++) {
        pi += std::pow(-1, i) / (2 * i + 1);
    }
    return pi * 4;
}

// 处理数组
EMSCRIPTEN_KEEPALIVE
float* create_float_array(int size) {
    float* arr = new float[size];
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<float>(i * i);
    }
    return arr;
}

EMSCRIPTEN_KEEPALIVE
void free_float_array(float* arr) {
    delete[] arr;
}

// 字符串操作
EMSCRIPTEN_KEEPALIVE
const char* get_greeting() {
    static std::string greeting = "Hello from C++!";
    return greeting.c_str();
}

} // extern "C"
```

```bash
# 编译 C++ 项目
em++ main.cpp -O2 -o app.js \
  -s EXPORTED_FUNCTIONS='["_calculate_pi","_create_float_array","_free_float_array","_get_greeting","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1
```

## 五、JavaScript 胶水代码详解

```javascript
// Emscripten 生成的 Module 对象
var Module = {
  // 配置选项
  locateFile: (path) => `/wasm/${path}`,  // 定位 .wasm 文件

  onRuntimeInitialized: () => {
    console.log('Emscripten 运行时就绪');

    // 使用 ccall 调用 C 函数
    const pi = Module.ccall(
      'calculate_pi',    // 函数名
      'number',          // 返回类型
      ['number'],        // 参数类型
      [10000]            // 参数值
    );
    console.log('PI ≈', pi);

    // 使用 cwrap 创建函数包装
    const calculatePi = Module.cwrap(
      'calculate_pi',
      'number',
      ['number']
    );
    console.log('PI ≈', calculatePi(10000));
  },

  onAbort: (what) => {
    console.error('WASM 异常中止:', what);
  }
};
```

## 六、文件系统操作

```bash
# 预加载文件到虚拟文件系统
emcc main.c --preload-file assets -o app.js
# 生成 app.js + app.data

# 预加载特定目录
emcc main.c --preload-file assets/images@/images -o app.js

# 嵌入文件（编译时）
emcc main.c --embed-file config.json -o app.js

# 禁用文件系统（减小体积）
emcc main.c -s NO_FILESYSTEM=1 -o app.js

# 使用 WORKERFS（Worker 中挂载文件系统）
emcc main.c -s USE_WORKERFS=1 -o app.js
```

```javascript
// JavaScript 中访问虚拟文件系统
Module.onRuntimeInitialized = () => {
  // 读取预加载的文件
  const data = FS.readFile('/assets/data.txt', { encoding: 'utf8' });
  console.log(data);

  // 写入文件
  FS.writeFile('/output.txt', 'Hello from JS');

  // 创建目录
  FS.mkdir('/workspace');

  // 列出目录
  const files = FS.readdir('/');
  console.log(files);
};
```

## 七、异步操作

```c
// 使用 emscripten_async_call 实现异步
#include <emscripten.h>

void async_callback(void* arg) {
    int* data = (int*)arg;
    printf("异步回调: %d\n", *data);
    delete data;
}

void start_async_work() {
    int* data = new int(42);
    // 1000ms 后调用回调
    emscripten_async_call(async_callback, data, 1000);
}
```

```javascript
// JavaScript 端使用 Promise
async function loadEmscriptenModule() {
  const Module = await createModule({
    locateFile: (path) => `https://cdn.example.com/wasm/${path}`
  });
  return Module;
}

const module = await loadEmscriptenModule();
const result = module.ccall('calculate_pi', 'number', ['number'], [10000]);
```

## 八、编译优化建议

```bash
# Release 构建（最佳性能）
emcc main.c -O3 -o app.js \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s INITIAL_MEMORY=16777216 \
  -s ASSERTIONS=0 \
  -s DISABLE_EXCEPTION_CATCHING=1 \
  -flto

# Size 构建（最小体积）
emcc main.c -Oz -o app.js \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s ASSERTIONS=0 \
  -s NO_EXIT_RUNTIME=1 \
  -s NO_FILESYSTEM=1 \
  -s SINGLE_FILE=1

# Debug 枚举（完整调试信息）
emcc main.c -g4 -o app.js \
  -s ASSERTIONS=2 \
  -s SAFE_HEAP=1 \
  -s STACK_OVERFLOW_CHECK=2
```
