# Go与TinyGo

## 一、概念说明

TinyGo 是 Go 的精简实现，支持编译为 WASM。

```bash
# 安装 TinyGo
# macOS
brew tap tinygo-org/tools && brew install tinygo

# 下载二进制
# https://tinygo.org/getting-started/
```

## 二、具体用法

### 2.1 基本编译

```go
// main.go
package main

func main() {
    println("Hello from TinyGo!")
}

//go:export add
func add(a, b int32) int32 {
    return a + b
}
```

```bash
# 编译为 WASM
tinygo build -o main.wasm -target wasm main.go

# 优化编译
tinygo build -o main.wasm -target wasm -opt=2 main.go
```

### 2.2 JavaScript 交互

```go
package main

import "github.com/tinygo-org/js"

func main() {
    // 访问 JavaScript 对象
    doc := js.Global().Get("document")
    body := doc.Get("body")

    // 创建元素
    h1 := doc.Call("createElement", "h1")
    h1.Set("textContent", "Hello from TinyGo!")
    body.Call("appendChild", h1)
}
```

```javascript
// JavaScript 加载
const go = new Go();
WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject)
  .then((result) => {
    go.run(result.instance);
  });
```

### 2.3 回调函数

```go
package main

import "github.com/tinygo-org/js"

func main() {
    btn := js.Global().Get("document").Call("getElementById", "btn")
    btn.Call("addEventListener", "click", js.NewEventCallback(0, func(event js.Value) {
        println("按钮被点击！")
    }))
}
```

## 三、注意事项与常见陷阱

1. **Go 版本**：TinyGo 不完全兼容 Go
2. **包限制**：某些标准库包不可用
3. **性能**：TinyGo 性能可能不如 Go
4. **调试**：调试工具有限
5. **生态系统**：TinyGo 生态系统较小

## 四、标准 Go 与 TinyGo 的 WASM 对比

```bash
# 标准 Go 编译 WASM（实验性支持）
GOOS=js GOARCH=wasm go build -o main.wasm main.go
# 需要使用 Go 提供的 wasm_exec.js 加载
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .

# TinyGo 编译 WASM（推荐）
tinygo build -o main.wasm -target wasm -opt=2 main.go
# 需要使用 tinygo 提供的 wasm_exec.js
cp "$(tinygo env TINYGOROOT)/targets/wasm_exec.js" .

# 对比：
# 标准 Go: 二进制较大（~2MB+），但标准库完整
# TinyGo: 二进制较小（~100KB），但标准库受限
```

## 五、DOM 操作详细示例

```go
package main

import (
    "github.com/tinygo-org/js"
)

func main() {
    doc := js.Global().Get("document")

    // 创建按钮
    btn := doc.Call("createElement", "button")
    btn.Set("textContent", "点击我")
    btn.Set("id", "myButton")

    // 添加样式
    btn.Get("style").Set("padding", "10px 20px")
    btn.Get("style").Set("fontSize", "16px")

    // 添加到页面
    doc.Get("body").Call("appendChild", btn)

    // 绑定事件
    counter := 0
    btn.Call("addEventListener", "click", js.NewEventCallback(0, func(event js.Value) {
        counter++
        output := doc.Call("getElementById", "output")
        output.Set("textContent", fmt.Sprintf("点击次数: %d", counter))
    }))

    // 创建输出区域
    output := doc.Call("createElement", "div")
    output.Set("id", "output")
    output.Set("textContent", "点击次数: 0")
    doc.Get("body").Call("appendChild", output)

    // 阻止 main 函数退出
    <-make(chan struct{})
}

func fmt(format string, args ...interface{}) string {
    // 简化版格式化
    return fmt.Sprintf(format, args...)
}
```

## 六、Fetch API 使用

```go
package main

import (
    "github.com/tinygo-org/js"
)

func fetchJSON(url string, callback func(data js.Value)) {
    js.Global().Call("fetch", url).Call("then",
        js.FuncOf(func(this js.Value, args []js.Value) interface{} {
            response := args[0]
            return response.Call("json")
        }),
    ).Call("then",
        js.FuncOf(func(this js.Value, args []js.Value) interface{} {
            data := args[0]
            callback(data)
            return nil
        }),
    ).Call("catch",
        js.FuncOf(func(this js.Value, args []js.Value) interface{} {
            js.Global().Get("console").Call("error", args[0])
            return nil
        }),
    )
}

func main() {
    fetchJSON("https://api.example.com/data", func(data js.Value) {
        js.Global().Get("console").Call("log", "获取到数据:", data)
    })

    <-make(chan struct{})
}
```

## 七、Go WASM 的内存管理

```go
package main

import (
    "github.com/tinygo-org/js"
)

// 使用 FinalizationRegistry（如果浏览器支持）
// 或手动管理 JavaScript 对象引用

var refs []js.Func

func keepRef(fn js.Func) js.Func {
    refs = append(refs, fn)
    return fn
}

func main() {
    // 使用 keepRef 防止 GC 回调函数
    handler := keepRef(js.FuncOf(func(this js.Value, args []js.Value) interface{} {
        // 处理事件
        return nil
    }))

    js.Global().Get("document").Call("addEventListener", "click", handler)

    <-make(chan struct{})
}
```

## 八、TinyGo 的局限性

```go
// 以下功能在 TinyGo 中不可用或受限：
// 1. reflect 包（部分支持）
// 2. net/http 包（不支持）
// 3. encoding/json（部分支持）
// 4. sync 包（有限支持）
// 5. goroutine 调度（受限）

// 替代方案：
// - 使用 js 包直接调用 JavaScript 的 fetch
// - 使用 JavaScript 的 JSON 处理
// - 使用 js.NewEventCallback 处理异步
// - 考虑用 Rust 或 AssemblyScript 替代复杂需求
```
