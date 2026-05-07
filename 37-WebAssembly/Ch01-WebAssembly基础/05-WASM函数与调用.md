# WASM函数与调用

## 一、概念说明

WASM 中的函数可以导出供外部调用，也可以从外部导入。

```javascript
(module
  ;; 导入函数
  (func $log (import "console" "log") (param i32))
  ;; 定义函数
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)
  ;; 导出函数
  (export "add" (func $add)))
```

## 二、具体用法

### 2.1 函数定义

```javascript
(module
  ;; 简单函数
  (func $square (param $x i32) (result i32)
    local.get $x
    local.get $x
    i32.mul)

  ;; 多返回值
  (func $divmod (param $a i32) (param $b i32) (result i32 i32)
    local.get $a
    local.get $b
    i32.div_s
    local.get $a
    local.get $b
    i32.rem_s)

  (export "square" (func $square))
  (export "divmod" (func $divmod)))
```

### 2.2 间接调用

```javascript
(module
  (table 2 funcref)
  (elem (i32.const 0) $add $sub)

  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)

  (func $sub (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.sub)

  (func $call_func (param $idx i32) (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    local.get $idx
    call_indirect (type $binop))

  (type $binop (func (param i32 i32) (result i32)))
  (export "call_func" (func $call_func)))
```

### 2.3 递归调用

```javascript
(module
  (func $fib (param $n i32) (result i32)
    (if (result i32)
      (i32.le_s (local.get $n) (i32.const 1))
      (then (local.get $n))
      (else
        (i32.add
          (call $fib (i32.sub (local.get $n) (i32.const 1)))
          (call $fib (i32.sub (local.get $n) (i32.const 2)))))))

  (export "fib" (func $fib)))
```

## 三、注意事项与常见陷阱

1. **调用开销**：WASM 调用有开销
2. **栈空间**：递归调用消耗栈空间
3. **间接调用**：call_indirect 需要类型检查
4. **导入函数**：导入的函数由宿主环境提供
5. **函数表**：函数表用于间接调用
