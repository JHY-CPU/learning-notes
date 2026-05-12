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

## 四、完整示例：字符串处理模块

```wat
(module
  ;; 定义 1 页内存
  (memory 1)
  (export "memory" (memory 0))

  ;; 数据段：在偏移 0 存储 "Hello, World!"
  (data (i32.const 0) "Hello, World!\00")

  ;; 获取字符串指针
  (func $get_string_ptr (result i32)
    i32.const 0)
  (export "get_string_ptr" (func $get_string_ptr))

  ;; 获取字符串长度（不包括 null 终止符）
  (func $get_string_length (result i32)
    i32.const 13)
  (export "get_string_length" (func $get_string_length))

  ;; 计算字符串长度（动态）
  (func $strlen (param $ptr i32) (result i32)
    (local $len i32)
    (local.set $len (i32.const 0))
    (block $break
      (loop $continue
        ;; 如果当前字节为 0，退出
        (br_if $break
          (i32.eqz
            (i32.load8_u (local.get $ptr))))
        ;; 长度加 1
        (local.set $len (i32.add (local.get $len) (i32.const 1)))
        ;; 指针加 1
        (local.set $ptr (i32.add (local.get $ptr) (i32.const 1)))
        (br $continue)))
    (local.get $len))
  (export "strlen" (func $strlen))

  ;; 字符串反转（原地）
  (func $reverse_string (param $ptr i32) (param $len i32)
    (local $left i32)
    (local $right i32)
    (local $temp i32)
    (local.set $left (local.get $ptr))
    (local.set $right
      (i32.sub
        (i32.add (local.get $ptr) (local.get $len))
        (i32.const 1)))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $left) (local.get $right)))
        ;; 交换 left 和 right 指向的字节
        (local.set $temp (i32.load8_u (local.get $left)))
        (i32.store8 (local.get $left) (i32.load8_u (local.get $right)))
        (i32.store8 (local.get $right) (local.get $temp))
        ;; 移动指针
        (local.set $left (i32.add (local.get $left) (i32.const 1)))
        (local.set $right (i32.sub (local.get $right) (i32.const 1)))
        (br $continue))))
  (export "reverse_string" (func $reverse_string))
)
```

```javascript
// 浏览器中使用
const memory = new WebAssembly.Memory({ initial: 1 });
const { instance } = await WebAssembly.instantiate(bytes, { env: { memory } });

const ptr = instance.exports.get_string_ptr();
const len = instance.exports.get_string_length();
const view = new Uint8Array(memory.buffer, ptr, len);
console.log(new TextDecoder().decode(view)); // "Hello, World!"

// 反转字符串
instance.exports.reverse_string(ptr, len);
console.log(new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len)));
// "!dlroW ,olleH"
```

## 五、WAT 中的批量内存操作

```wat
(module
  (memory 1)

  ;; 使用 bulk memory 提案（WASM 1.1+）
  ;; 内存拷贝
  (func $memcpy (param $dst i32) (param $src i32) (param $len i32)
    local.get $dst
    local.get $src
    local.get $len
    memory.copy)

  ;; 内存填充
  (func $memset (param $dst i32) (param $value i32) (param $len i32)
    local.get $dst
    local.get $value
    local.get $len
    memory.fill)

  ;; 内存初始化（从数据段）
  (data $init_data "\01\02\03\04\05")
  (func $init_from_data (param $dst i32) (param $offset i32) (param $len i32)
    local.get $dst
    local.get $offset
    local.get $len
    memory.init $init_data)

  (export "memcpy" (func $memcpy))
  (export "memset" (func $memset))
)
```

## 六、内联导入和导出语法

```wat
(module
  ;; 内联导入
  (import "math" "PI" (global $pi f64))
  (import "console" "log" (func $log (param i32)))

  ;; 内联导出
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)

  (global (export "counter") (mut i32) (i32.const 0))
  (memory (export "memory") 1)
)
```

## 七、常见 WAT 模式速查

```wat
;; 1. 条件赋值
(func $max (param $a i32) (param $b i32) (result i32)
  local.get $a
  local.get $b
  local.get $a
  local.get $b
  i32.gt_s
  select)  ;; 如果 a > b 返回 a，否则返回 b

;; 2. while 循环
(func $sum_1_to_n (param $n i32) (result i32)
  (local $i i32)
  (local $sum i32)
  (block $break
    (loop $continue
      (br_if $break (i32.gt_u (local.get $i) (local.get $n)))
      (local.set $sum (i32.add (local.get $sum) (local.get $i)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $continue)))
  (local.get $sum))

;; 3. switch/match 模式
(func $switch_example (param $x i32) (result i32)
  (block $default
    (block $case2
      (block $case1
        (br_table $case1 $case2 $default
          (local.get $x))
      ) ;; case 1
      (return (i32.const 10))
    ) ;; case 2
    (return (i32.const 20))
  ) ;; default
  (i32.const 0))

;; 4. 多返回值
(func $divmod (param $a i32) (param $b i32) (result i32 i32)
  local.get $a
  local.get $b
  i32.div_s
  local.get $a
  local.get $b
  i32.rem_s)
```
