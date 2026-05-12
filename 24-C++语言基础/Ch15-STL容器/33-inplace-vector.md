# inplace_vector（C++26）

## 一、概念说明

`std::inplace_vector`是C++26提议的固定容量动态数组（P0843），所有元素存储在栈上（或对象内部），提供动态大小但固定容量的容器。适用于嵌入式系统、实时系统和性能关键路径，消除堆分配的不确定性和开销。

### 1.1 设计动机

```
vector的问题：
- 堆分配带来不确定性（时间、失败）
- 嵌入式/实时系统可能没有堆
- 频繁分配/释放导致碎片

array的问题：
- 大小编译时固定，不够灵活

inplace_vector的优势：
- 动态大小（运行时）
- 固定容量（编译时）
- 无堆分配（全部栈/对象内）
- 确定性性能
```

### 1.2 相关方案对比

| 方案 | 动态大小 | 堆分配 | 容量 |
|------|---------|-------|------|
| vector | 是 | 是 | 动态 |
| array | 否 | 否 | 固定 |
| inplace_vector | 是 | 否 | 固定 |
| boost::static_vector | 是 | 否 | 固定 |

## 二、概念设计

### 2.1 基本接口

```cpp
// C++26 <inplace_vector>（设计草案/P0843R6）
// #include <inplace_vector>

void concept_demo() {
    // 最多存储10个int，所有内存栈上分配
    // std::inplace_vector<int, 10> iv;

    // iv.push_back(1);
    // iv.push_back(2);
    // iv.push_back(3);

    // std::cout << iv.size() << std::endl;      // 3
    // std::cout << iv.capacity() << std::endl;  // 10

    // 超出容量抛异常
    // for (int i = 0; i < 11; ++i) iv.push_back(i);
    // 抛出 std::bad_alloc 或类似异常
}
```

### 2.2 当前替代方案

```cpp
#include <array>
#include <stdexcept>
#include <iostream>

// 手动实现简化版
template<typename T, size_t N>
class FixedBuffer {
    alignas(T) unsigned char storage[N * sizeof(T)];
    size_t sz = 0;

public:
    ~FixedBuffer() { clear(); }

    void push_back(const T& val) {
        if (sz >= N) throw std::bad_alloc();
        new (&storage[sz * sizeof(T)]) T(val);
        ++sz;
    }

    void pop_back() {
        if (sz == 0) return;
        --sz;
        reinterpret_cast<T*>(&storage[sz * sizeof(T)])->~T();
    }

    T& operator[](size_t i) {
        return *reinterpret_cast<T*>(&storage[i * sizeof(T)]);
    }

    size_t size() const { return sz; }
    size_t capacity() const { return N; }

    void clear() {
        while (sz > 0) pop_back();
    }
};

void alternative_demo() {
    FixedBuffer<int, 10> buf;
    buf.push_back(1);
    buf.push_back(2);
    std::cout << buf[0] << std::endl;  // 1
}
```

## 三、适用场景

```cpp
/*
适用场景：
1. 嵌入式系统（无堆或堆受限）
2. 实时系统（需要确定性内存分配时间）
3. 性能关键路径（避免堆分配开销）
4. 小缓冲区优化（SSO）
5. 临时缓冲区（已知最大大小）

替代方案：
- std::array（固定大小）
- std::vector（动态但堆分配）
- boost::static_vector（成熟实现）
- 自定义固定缓冲区
- std::pmr::monotonic_buffer_resource（栈缓冲区）
*/
```

### 3.1 使用pmr作为当前替代

```cpp
#include <memory_resource>
#include <vector>

void pmr_alternative() {
    // pmr的monotonic_buffer_resource可以模拟inplace_vector
    char buffer[1024];
    std::pmr::monotonic_buffer_resource pool{buffer, sizeof(buffer)};
    std::pmr::vector<int> v{&pool};

    for (int i = 0; i < 100; ++i) v.push_back(i);
    // 小数据从buffer分配，超出后回退到new
}
```

## 四、注意事项与常见陷阱

1. **C++26特性**：目前编译器支持有限，属于提案阶段
2. **`push_back`超出容量时抛异常**：需要确保容量足够
3. **所有操作不涉及堆分配**：确定性性能
4. **可能导致栈溢出**：容量过大时（栈空间通常1-8MB）
5. **需要手动管理元素生命周期**：placement new和显式析构
6. **替代方案**：`boost::static_vector`、`std::pmr::monotonic_buffer_resource`、自定义实现
