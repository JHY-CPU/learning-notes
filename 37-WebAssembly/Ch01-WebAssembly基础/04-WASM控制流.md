# WASM控制流

## 一、概念说明

WASM 的控制流基于结构化控制流，使用 block、loop、if 等指令。

```javascript
// WASM 控制流指令
// block: 代码块
// loop: 循环
// if: 条件分支
// br: 无条件跳转
// br_if: 条件跳转
// br_table: 跳转表
// return: 返回
// unreachable: 不可达
```

## 二、具体用法

### 2.1 条件分支

```javascript
(module
  (func $abs (param $x i32) (result i32)
    local.get $x
    i32.const 0
    i32.lt_s
    (if (result i32)
      (then
        i32.const 0
        local.get $x
        i32.sub)
      (else
        local.get $x)))
  (export "abs" (func $abs)))
```

### 2.2 循环

```javascript
(module
  (func $sum (param $n i32) (result i32)
    (local $i i32)
    (local $sum i32)
    i32.const 0
    local.set $i
    i32.const 0
    local.set $sum
    (block $break
      (loop $continue
        local.get $i
        local.get $n
        i32.ge_s
        br_if $break
        local.get $sum
        local.get $i
        i32.add
        local.set $sum
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $continue))
    local.get $sum)
  (export "sum" (func $sum)))
```

### 2.3 跳转表

```javascript
(module
  (func $switch (param $x i32) (result i32)
    local.get $x
    (block $default
      (block $case2
        (block $case1
          (block $case0
            local.get $x
            br_table $case0 $case1 $case2 $default)
          ;; case 0
          i32.const 100
          return)
        ;; case 1
        i32.const 200
        return)
      ;; case 2
      i32.const 300
      return)
    ;; default
    i32.const 0)
  (export "switch" (func $switch)))
```

## 三、注意事项与常见陷阱

1. **结构化控制流**：不能任意跳转
2. **栈平衡**：控制流必须保持栈平衡
3. **循环条件**：loop 本身不循环，需要 br 回去
4. **break 语义**：br 跳出 block，br_if 条件跳出
5. **性能**：br_table 比多个 if-else 更快
