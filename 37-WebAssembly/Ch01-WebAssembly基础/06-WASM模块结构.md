# WASM模块结构

## 一、概念说明

WASM 模块是部署和加载的单位，包含类型、函数、内存、表等部分。

```javascript
(module
  (type $t0 (func (param i32) (result i32)))
  (import "env" "memory" (memory 1))
  (func $f0 (type $t0) ...)
  (table 1 funcref)
  (global $g0 (mut i32) (i32.const 0))
  (export "func" (func $f0))
  (data (i32.const 0) "hello"))
```

## 二、具体用法

### 2.1 模块部分

```javascript
;; WASM 模块的各个部分（按顺序）
(module
  ;; 1. 类型定义
  (type (func (param i32) (result i32)))

  ;; 2. 导入
  (import "env" "memory" (memory 1))
  (import "env" "log" (func $log (param i32)))

  ;; 3. 函数
  (func (type 0) ...)

  ;; 4. 表
  (table 10 funcref)

  ;; 5. 内存
  (memory 1)

  ;; 6. 全局变量
  (global (mut i32) (i32.const 0))

  ;; 7. 导出
  (export "func" (func 0))

  ;; 8. 元素（表初始化）
  (elem (i32.const 0) $func1 $func2)

  ;; 9. 数据（内存初始化）
  (data (i32.const 0) "hello"))
```

### 2.2 自定义段

```javascript
;; 自定义段用于调试信息
(module
  (@custom "name" ...)
  (func $my_func ...)
  (@custom "producers" ...)
  (@custom "sourceMappingURL" ...))
```

## 三、注意事项与常见陷阱

1. **部分顺序**：各部分有固定顺序
2. **自定义段**：可选，不影响执行
3. **导入优先级**：导入必须在定义之前
4. **模块大小**：大型模块需要分片加载
5. **版本兼容**：WASM 有版本号
