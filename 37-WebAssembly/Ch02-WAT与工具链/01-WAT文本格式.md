# WAT文本格式

## 一、概念说明

WAT（WebAssembly Text Format）是 WASM 的可读文本表示形式，用于调试和学习。

```javascript
;; WAT 基本结构
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)
  (export "add" (func $add)))
```

## 二、具体用法

### 2.1 基本语法

```javascript
;; 函数定义
(func $name (param $x i32) (result i32)
  local.get $x)

;; 多参数
(func $add3 (param i32 i32 i32) (result i32)
  local.get 0
  local.get 1
  i32.add
  local.get 2
  i32.add)

;; 局部变量
(func $example
  (local $x i32)
  (local $y f64)
  i32.const 42
  local.set $x)
```

### 2.2 控制结构

```javascript
;; 条件
(if (result i32)
  (i32.const 1)
  (then (i32.const 10))
  (else (i32.const 20)))

;; 循环
(block $break
  (loop $continue
    ;; 循环体
    br $continue  ;; 继续循环
    br $break))   ;; 跳出循环
```

### 2.3 内存和表

```javascript
;; 内存定义
(memory 1)  ;; 初始 1 页 (64KB)
(memory 1 10)  ;; 初始 1 页，最大 10 页

;; 表定义
(table 10 funcref)  ;; 10 个函数引用

;; 数据初始化
(data (i32.const 0) "hello")
```

## 三、注意事项与常见陷阱

1. **S-表达式**：WAT 使用 S-表达式语法
2. **索引从0开始**：所有索引从0开始
3. **类型注解**：参数和返回值需要类型注解
4. **调试用途**：WAT 主要用于调试和学习
5. **工具转换**：使用 wat2wasm 和 wasm2wat 转换
