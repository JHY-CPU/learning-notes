# WASM类型系统

## 一、概念说明

WASM 有基本的数值类型，没有对象或引用类型（WASM 2.0 增加了引用类型）。

```javascript
// WASM 基本类型
// i32: 32位整数
// i64: 64位整数
// f32: 32位浮点数
// f64: 64位浮点数
// v128: 128位向量 (SIMD)
```

## 二、具体用法

### 2.1 类型操作

```javascript
// 整数运算
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)
  (export "add" (func $add)))
```

### 2.2 类型转换

```javascript
// 类型转换指令
i32.wrap_i64      // i64 -> i32
i64.extend_i32_s  // i32 -> i64 (符号扩展)
f32.demote_f64    // f64 -> f32
f64.promote_f32   // f32 -> f64
i32.trunc_f32_s   // f32 -> i32 (截断)
f32.convert_i32_s // i32 -> f32
```

### 2.3 SIMD 类型

```javascript
// WASM SIMD (v128)
(module
  (func $add_vectors (param $a v128) (param $b v128) (result v128)
    local.get $a
    local.get $b
    f32x4.add)
  (export "add_vectors" (func $add_vectors)))
```

### 2.4 引用类型 (WASM 2.0)

```javascript
// 引用类型
// externref: 外部引用
// funcref: 函数引用

(module
  (func $call_func (param $f funcref)
    local.get $f
    call_indirect)
  (export "call_func" (func $call_func)))
```

## 三、注意事项与常见陷阱

1. **类型严格**：WASM 类型检查是严格的
2. **转换安全**：类型转换可能丢失精度
3. **SIMD 支持**：SIMD 需要浏览器支持
4. **引用类型**：引用类型还在标准化中
5. **无符号类型**：WASM 没有无符号类型，由指令区分
