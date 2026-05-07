# inplace_vector（C++26）

## 一、概念说明

`std::inplace_vector`是C++26提议的固定容量动态数组，所有元素存储在栈上（或对象内部），提供动态大小但固定容量的容器。适用于嵌入式系统和性能关键路径。

## 二、概念设计

```cpp
// C++26 <inplace_vector>（设计草案）
// #include <inplace_vector>

void concept_demo() {
    // 最多存储10个int，所有内存栈上分配
    // std::inplace_vector<int, 10> iv;

    // iv.push_back(1);
    // iv.push_back(2);
    // iv.push_back(3);

    // 当前替代方案
    struct FixedBuffer {
        int data[10];
        size_t size = 0;

        void push_back(int val) {
            if (size < 10) data[size++] = val;
        }

        int& operator[](size_t i) { return data[i]; }
        size_t sz() const { return size; }
    };
}
```

## 三、适用场景

```cpp
/*
适用场景：
1. 嵌入式系统（无堆或堆受限）
2. 实时系统（需要确定性内存）
3. 性能关键路径（避免堆分配）
4. 小缓冲区优化

替代方案：
- std::array（固定大小）
- std::vector（动态但堆分配）
- boost::static_vector
- 自定义固定缓冲区
*/
```

## 四、注意事项

- C++26特性，目前编译器支持有限
- push_back超出容量时抛异常
- 所有操作不涉及堆分配
- 可能导致栈溢出（容量过大时）
