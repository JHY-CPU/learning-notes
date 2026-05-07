# zip与enumerate

## 一、概念说明

`zip` 将两个迭代器合并为一个产生元组的迭代器。`enumerate` 为迭代器的每个元素添加索引。两者都是常用的数据组合工具。

```rust
// zip 合并两个迭代器
let names = vec!["张三", "李四", "王五"];
let scores = vec![90, 85, 92];
let paired: Vec<(&str, i32)> = names.iter()
    .zip(scores.iter())
    .map(|(n, s)| (*n, *s))
    .collect();
// [("张三", 90), ("李四", 85), ("王五", 92)]
```

## 二、具体用法

### 2.1 zip 基本用法

```rust
// 等长迭代器
let a = vec![1, 2, 3];
let b = vec!['a', 'b', 'c'];
let combined: Vec<(i32, char)> = a.into_iter().zip(b).collect();
// [(1, 'a'), (2, 'b'), (3, 'c')]

// 不等长（以短的为准）
let a = vec![1, 2, 3, 4];
let b = vec!['a', 'b'];
let combined: Vec<(i32, char)> = a.into_iter().zip(b).collect();
// [(1, 'a'), (2, 'b')]

// 链式 zip
let result: Vec<(i32, char, bool)> = vec![1, 2, 3]
    .into_iter()
    .zip(vec!['a', 'b', 'c'])
    .zip(vec![true, false, true])
    .map(|((n, c), b)| (n, c, b))
    .collect();
```

### 2.2 enumerate 索引

```rust
// 基本 enumerate
let fruits = vec!["苹果", "香蕉", "橘子"];
for (index, fruit) in fruits.iter().enumerate() {
    println!("{}: {}", index, fruit);
}
// 0: 苹果
// 1: 香蕉
// 2: 橘子

// 指定起始索引
let data = vec!["a", "b", "c"];
let indexed: Vec<(usize, &str)> = data.iter()
    .enumerate()
    .map(|(i, s)| (i + 1, *s))
    .collect();
// [(1, "a"), (2, "b"), (3, "c")]
```

### 2.3 实际应用

```rust
// 矩阵行列处理
let matrix = vec![
    vec![1, 2, 3],
    vec![4, 5, 6],
    vec![7, 8, 9],
];

// 获取对角线元素
let diagonal: Vec<i32> = matrix.iter()
    .enumerate()
    .map(|(i, row)| row[i])
    .collect();
// [1, 5, 9]

// 查找并返回索引
fn find_with_index<T: PartialEq>(data: &[T], target: &T) -> Option<(usize, &T)> {
    data.iter()
        .enumerate()
        .find(|(_, item)| *item == target)
        .map(|(i, item)| (i, item))
}
```

## 三、注意事项与常见陷阱

1. **长度不等**：zip 以较短迭代器为准，多余的元素被忽略
2. **枚举类型**：enumerate 产生的索引类型是 usize
3. **所有权**：zip 后的元组可能包含引用或所有权值
4. **性能**：zip 和 enumerate 都是零成本抽象
5. **unzip**：元组迭代器可以用 unzip 分离
