# Tag Dispatch惯用法

## 一、概念说明

Tag Dispatch（标签分发）是一种模板技术，通过空标签类（Tag Class）在编译期选择不同的函数重载。它利用类型信息在编译时进行分支选择，是实现STL算法优化（如随机访问迭代器vs前向迭代器）的经典惯用法。

## 二、具体用法

### 2.1 基本标签分发

```cpp
// 标签类型（空结构体）
struct FastPath {};
struct SlowPath {};

// 内部实现：根据标签选择不同算法
template <typename T>
T accumulate_impl(T* data, std::size_t n, FastPath) {
    std::cout << "快速路径（SIMD友好）" << std::endl;
    T sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += data[i];
    return sum;
}

template <typename T>
T accumulate_impl(T* data, std::size_t n, SlowPath) {
    std::cout << "慢速路径（逐个处理）" << std::endl;
    T sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] > 0) sum += data[i];  // 过滤负数
    }
    return sum;
}

// 根据类型特征选择路径
template <typename T>
T smart_accumulate(T* data, std::size_t n) {
    if constexpr (std::is_integral_v<T>) {
        return accumulate_impl(data, n, FastPath{});
    } else {
        return accumulate_impl(data, n, SlowPath{});
    }
}
```

### 2.2 迭代器类别标签分发

```cpp
// advance实现：根据迭代器类别优化
template <typename Iter>
void advance_impl(Iter& it, std::ptrdiff_t n, std::random_access_iterator_tag) {
    std::cout << "随机访问：直接跳转" << std::endl;
    it += n;  // O(1)
}

template <typename Iter>
void advance_impl(Iter& it, std::ptrdiff_t n, std::bidirectional_iterator_tag) {
    std::cout << "双向：逐个前进" << std::endl;
    for (std::ptrdiff_t i = 0; i < n; ++i) ++it;  // O(n)
}

template <typename Iter>
void smart_advance(Iter& it, std::ptrdiff_t n) {
    // iterator_category是迭代器的标签类型
    using Category = typename std::iterator_traits<Iter>::iterator_category;
    advance_impl(it, n, Category{});
}

int main() {
    std::vector<int> v{1, 2, 3, 4, 5};
    auto vit = v.begin();
    smart_advance(vit, 3);   // 随机访问：直接跳转
    std::cout << *vit << std::endl;  // 4

    std::list<int> lst{1, 2, 3, 4, 5};
    auto lit = lst.begin();
    smart_advance(lit, 3);   // 双向：逐个前进
    std::cout << *lit << std::endl;  // 4
}
```

### 2.3 编译期策略选择

```cpp
struct PolicyA { static constexpr const char* name = "策略A"; };
struct PolicyB { static constexpr const char* name = "策略B"; };

template <typename Policy>
void execute(Policy) {
    std::cout << "执行: " << Policy::name << std::endl;
}

int main() {
    execute(PolicyA{});  // 执行: 策略A
    execute(PolicyB{});  // 执行: 策略B
}
```

## 三、注意事项与常见陷阱

- 标签类型应该是空结构体，零运行时开销
- 标签参数通常按值传递（`Tag{}`），不需要变量
- C++17的`if constexpr`可以替代部分tag dispatch场景
- STL大量使用此技术：`std::advance`、`std::distance`、`std::copy`等
- 标签类型命名应清晰表达其含义（如`forward_iterator_tag`）
- 使用`std::iterator_traits<Iter>::iterator_category`获取迭代器类别
