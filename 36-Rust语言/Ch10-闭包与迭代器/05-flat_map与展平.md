# flat_map与展平

## 一、概念说明

`flat_map` 是 map 和 flatten 的结合：先对每个元素应用映射函数（返回迭代器），然后将所有结果展平。`flatten` 直接展平嵌套迭代器。

```rust
// flat_map = map + flatten
let result: Vec<i32> = vec![1, 2, 3]
    .iter()
    .flat_map(|x| vec![*x, *x * 10])
    .collect();
// [1, 10, 2, 20, 3, 30]
```

## 二、具体用法

### 2.1 基本用法

```rust
// 展开字符串为字符
let chars: Vec<char> = vec!["hello", "world"]
    .iter()
    .flat_map(|s| s.chars())
    .collect();
// ['h','e','l','l','o','w','o','r','l','d']

// 展开嵌套集合
let nested = vec![vec![1, 2], vec![3], vec![4, 5]];
let flat: Vec<i32> = nested
    .into_iter()
    .flatten()
    .collect();
// [1, 2, 3, 4, 5]

// 空集合处理
let result: Vec<i32> = vec![vec![1, 2], vec![], vec![3, 4]]
    .into_iter()
    .flatten()
    .collect();
// [1, 2, 3, 4]（空vec被忽略）
```

### 2.2 Option 和 Result 的展平

```rust
// Option 展平
let options: Vec<Option<i32>> = vec![Some(1), None, Some(3)];
let values: Vec<i32> = options
    .into_iter()
    .flatten()
    .collect();
// [1, 3]

// Result 展平
let results: Vec<Result<i32, &str>> = vec![Ok(1), Err("错误"), Ok(3)];
let successes: Vec<i32> = results
    .into_iter()
    .flatten()
    .collect();
// [1, 3]

// flat_map 处理 Option
let inputs = vec!["1", "hello", "3", "world"];
let numbers: Vec<i32> = inputs
    .iter()
    .flat_map(|s| s.parse::<i32>().ok())
    .collect();
// [1, 3]
```

### 2.3 高级模式

```rust
// 展开关系数据
struct User { name: String, orders: Vec<String> }

let users = vec![
    User { name: "张三".into(), orders: vec!["订单1".into(), "订单2".into()] },
    User { name: "李四".into(), orders: vec!["订单3".into()] },
];

let all_orders: Vec<String> = users
    .into_iter()
    .flat_map(|u| u.orders)
    .collect();
// ["订单1", "订单2", "订单3"]
```

## 三、注意事项与常见陷阱

1. **惰性求值**：flat_map 也是惰性的，需要消费操作
2. **多次迭代**：每个元素的返回值可能包含多个结果
3. **性能注意**：嵌套层级深时考虑链式 flatten
4. **None/Err 过滤**：自动过滤 None 和 Err 值
5. **替代方案**：简单的展平用 flatten，需要变换用 flat_map
