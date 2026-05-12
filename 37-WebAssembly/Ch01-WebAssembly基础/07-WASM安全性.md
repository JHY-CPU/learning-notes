# WASM安全性

## 一、概念说明

WASM 在沙箱环境中执行，提供了内存隔离和控制流完整性等安全特性。

```javascript
// WASM 安全特性
// 1. 沙箱隔离：无法直接访问宿主资源
// 2. 内存隔离：只能访问自己的线性内存
// 3. 控制流完整性：结构化控制流
// 4. 类型安全：验证阶段检查类型
```

## 二、具体用法

### 2.1 内存隔离

```javascript
// WASM 无法访问外部内存
// 所有内存访问通过索引
(module
  (memory 1)
  (func $write (param $addr i32) (param $value i32)
    local.get $addr
    local.get $value
    i32.store)

  (func $read (param $addr i32) (result i32)
    local.get $addr
    i32.load))
```

### 2.2 能力安全模型

```javascript
// 通过导入控制能力
const imports = {
  env: {
    memory: new WebAssembly.Memory({ initial: 1 }),
    // 只提供需要的函数
    log: (msg) => console.log(msg),
    // 不提供文件系统访问
  }
};
```

### 2.3 验证过程

```javascript
// WASM 模块在实例化前必须验证
// 验证检查：
// 1. 类型正确性
// 2. 内存访问边界
// 3. 控制流完整性
// 4. 栈平衡

try {
  const module = await WebAssembly.compile(bytes);
  // 验证通过
} catch (e) {
  console.error('验证失败:', e);
}
```

## 三、注意事项与常见陷阱

1. **不等于安全**：WASM 本身安全，但导入的函数可能不安全
2. **资源消耗**：WASM 可以消耗 CPU 和内存
3. **拒绝服务**：恶意 WASM 可能导致拒绝服务
4. **侧信道**：WASM 不能防止侧信道攻击
5. **未来增强**：WASM 安全特性还在演进中

## 四、沙箱隔离详解

WASM 的沙箱机制确保代码无法逃逸：

```javascript
// WASM 代码无法执行以下操作：
// 1. 直接调用 eval() 或 Function()
// 2. 访问 DOM（除非通过导入）
// 3. 发起网络请求（除非通过导入）
// 4. 读写文件系统（除非通过导入）
// 5. 访问其他 WASM 实例的内存

// 安全的导入配置示例
function createSandboxedImports() {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 10 });

  return {
    env: {
      // 只暴露必要的能力
      memory: memory,
      // 受限的日志函数
      console_log: (ptr, len) => {
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        const text = new TextDecoder().decode(bytes);
        console.log('[WASM]', text);
      },
      // 提供随机数（而非 Math.random）
      get_random: () => crypto.getRandomValues(new Uint32Array(1))[0],
      // 提供时间戳
      get_time: () => Date.now(),
      // 注意：不暴露 fetch, XMLHttpRequest, WebSocket 等
    }
  };
}
```

## 五、内存安全详解

WASM 的内存安全模型包含多层保护：

```wat
(module
  (memory 1)  ;; 64KB

  ;; 安全访问：编译器生成边界检查
  (func $safe_read (param $addr i32) (result i32)
    ;; 1. 检查地址是否在范围内
    local.get $addr
    i32.const 65532      ;; 最大有效地址 (64KB - 4)
    i32.gt_u
    if (result i32)
      i32.const -1       ;; 越界返回 -1
    else
      local.get $addr
      i32.load
    end)

  ;; 带偏移的访问（WASM 会检查 addr + offset < 内存大小）
  (func $read_with_offset (param $base i32) (result i32)
    local.get $base
    i32.load offset=16)  ;; 自动检查 base + 16 的边界
)
```

## 六、WASM 与同源策略

WASM 模块受浏览器的同源策略（CORS）保护：

```javascript
// 加载跨域 WASM 需要服务器设置正确的 CORS 头
// Access-Control-Allow-Origin: *

// 推荐使用 instantiateStreaming（自动处理 CORS）
WebAssembly.instantiateStreaming(fetch('module.wasm'), imports)
  .then(({ instance }) => {
    // 安全加载
  })
  .catch(error => {
    console.error('加载失败，可能是 CORS 问题:', error);
  });

// 内联 WASM（避免 CORS 问题，适合小型模块）
const wasmBytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d,  // 魔数
  0x01, 0x00, 0x00, 0x00,  // 版本
  // ... 完整的 WASM 字节
]);
const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
```

## 七、安全最佳实践

1. **最小权限原则**：只导入必要的宿主函数，避免暴露文件系统、网络等敏感 API
2. **输入验证**：在 WASM 入口函数中验证所有外部输入
3. **资源限制**：设置内存的 `maximum` 值，防止无限增长
4. **Content Security Policy**：在 HTML 中配置 CSP 限制 WASM 加载源

```html
<!-- 限制 WASM 只能从同源加载 -->
<meta http-equiv="Content-Security-Policy"
      content="default-src 'self'; script-src 'self' 'wasm-unsafe-eval'">
```

5. **定期更新**：关注 WASM 规范的安全更新
6. **审计导入**：定期审查所有 `import` 语句，确保没有不安全的依赖

## 八、与其他沙箱技术的对比

| 特性 | WASM | iframe sandbox | Service Worker |
|------|------|----------------|----------------|
| 隔离级别 | 内存级隔离 | 浏览器上下文隔离 | 网络请求拦截 |
| 性能开销 | 极低 | 中等 | 中等 |
| 粒度 | 函数级 | 页面级 | 请求级 |
| 调试支持 | 发展中 | 成熟 | 成熟 |
| 适用场景 | 计算密集 | 内容隔离 | 离线缓存 |
