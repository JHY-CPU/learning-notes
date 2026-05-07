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
