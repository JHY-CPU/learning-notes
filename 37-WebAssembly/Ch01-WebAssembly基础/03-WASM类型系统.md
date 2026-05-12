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

## 五、整数运算指令详解

WASM 提供丰富的整数运算指令：

```wat
(module
  ;; 算术运算
  (func $arithmetic (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add          ;; 加法: a + b
    local.get $a
    local.get $b
    i32.sub          ;; 减法: a - b
    i32.mul          ;; 乘法: (a+b) * (a-b)
    drop
    local.get $a
    local.get $b
    i32.div_s        ;; 有符号除法: a / b
    )

  ;; 位运算
  (func $bitwise (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and          ;; 按位与
    local.get $a
    local.get $b
    i32.or           ;; 按位或
    i32.xor          ;; 异或后与或结果异或
    )

  ;; 比较运算
  (func $compare (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.eq           ;; 相等返回 1，否则 0
    )

  ;; 移位运算
  (func $shift (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.shl          ;; 左移
    )

  ;; 有符号 vs 无符号的区别
  (func $signed_vs_unsigned (param $a i32) (param $b i32)
    ;; 有符号比较：-1 > 0 为 false
    local.get $a
    local.get $b
    i32.gt_s         ;; 有符号大于
    drop

    ;; 无符号比较：0xFFFFFFFF > 0 为 true
    local.get $a
    local.get $b
    i32.gt_u         ;; 无符号大于
    drop))
```

## 六、浮点运算详解

```wat
(module
  ;; 浮点运算
  (func $float_ops (param $x f64) (param $y f64) (result f64)
    local.get $x
    local.get $y
    f64.add)

  ;; 特殊值处理
  (func $special_values (result f32)
    f32.const nan          ;; NaN
    f32.const inf          ;; 正无穷
    f32.const -inf         ;; 负无穷
    f32.const 0x1.fffffep127  ;; f32 最大正规化数
    drop
    drop
    drop
    f32.const 1.0)

  ;; 浮点取整
  (func $rounding (param $x f64) (result f64)
    local.get $x
    f64.nearest)     ;; 四舍五入到最近整数

  ;; 浮点数分类
  (func $is_nan (param $x f64) (result i32)
    local.get $x
    local.get $x
    f64.ne)          ;; NaN != NaN 为 true
    )
)
```

## 七、SIMD（v128）详细用法

SIMD 指令允许单条指令处理多个数据，显著提升并行计算性能：

```wat
(module
  ;; 四个 f32 向量加法
  (func $vec4_add (param $a v128) (param $b v128) (result v128)
    local.get $a
    local.get $b
    f32x4.add)

  ;; 从标量构造向量
  (func $make_vec (param $x f32) (param $y f32) (param $z f32) (param $w f32) (result v128)
    local.get $x
    local.get $y
    local.get $z
    local.get $w
    f32x4.splat  ;; 广播单个值到所有通道
    drop
    local.get $x
    local.get $y
    local.get $z
    local.get $w
    f32x4.make)

  ;; 向量点积（优化示例）
  (func $dot_product (param $a v128) (param $b v128) (result f32)
    local.get $a
    local.get $b
    f32x4.mul       ;; 逐元素相乘
    f32x4.extract_lane 0
    local.get $a
    local.get $b
    f32x4.mul
    f32x4.extract_lane 1
    f32.add
    local.get $a
    local.get $b
    f32x4.mul
    f32x4.extract_lane 2
    f32.add
    local.get $a
    local.get $b
    f32x4.mul
    f32x4.extract_lane 3
    f32.add)

  ;; 整数 SIMD
  (func $int_simd (param $a v128) (param $b v128) (result v128)
    local.get $a
    local.get $b
    i32x4.add          ;; 4 个 i32 并行加法
    local.get $a
    local.get $b
    i32x4.mul          ;; 4 个 i32 并行乘法
    i32x4.min_s        ;; 取有符号最小值
    )

  (export "vec4_add" (func $vec4_add))
  (export "dot_product" (func $dot_product))
)
```

## 八、函数类型与间接调用

```wat
(module
  ;; 定义函数类型
  (type $binop (func (param i32 i32) (result i32)))

  ;; 函数表：存储函数引用
  (table 3 funcref)
  (elem (i32.const 0) $add $sub $mul)

  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)

  (func $sub (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.sub)

  (func $mul (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.mul)

  ;; 间接调用：通过索引调用表中的函数
  (func $call_op (param $op_idx i32) (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    local.get $op_idx
    call_indirect (type $binop))

  (export "call_op" (func $call_op))
)
```

## 九、类型安全与验证

WASM 在模块加载时进行严格的类型验证：

- 栈上的值类型必须与指令期望的类型匹配
- 控制流结构（block/loop/if）必须保证栈平衡
- 函数签名的参数和返回类型必须精确匹配
- 验证失败时模块加载会抛出 `CompileError`
