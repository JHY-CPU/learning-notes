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
