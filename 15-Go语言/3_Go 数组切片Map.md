# Go 数组切片 Map


## 📦 Go 数组、切片与 Map


数组固定长度、切片动态扩容 (make/append/copy)、子切片操作、map 声明与操作、comma ok 模式、nil slice vs empty slice。


## 数组


```
// ========== 数组 ==========
// 固定长度, 值类型 (赋值/传参会复制)
// 长度是类型的一部分: [3]int ≠ [5]int

package main

import "fmt"

func main() {
    // ========== 声明 ==========
    var arr1 [3]int               // [0, 0, 0]
    arr2 := [5]int{1, 2, 3, 4, 5}
    arr3 := [...]int{10, 20, 30} // 自动计算长度

    // 指定索引初始化
    arr4 := [5]int{0: 10, 2: 30, 4: 50}
    // [10, 0, 30, 0, 50]

    // ========== 访问与修改 ==========
    fmt.Println(arr2[0])          // 1
    arr2[2] = 99
    fmt.Println(len(arr2))        // 5
    fmt.Println(cap(arr2))        // 5 (数组 cap = len)

    // ========== 遍历 ==========
    for i := 0; i < len(arr2); i++ {
        fmt.Println(arr2[i])
    }

    for index, value := range arr2 {
        fmt.Printf("arr[%d] = %d\n", index, value)
    }

    // ========== 数组是值类型 ==========
    original := [3]int{1, 2, 3}
    copy_arr := original          // 复制整个数组
    copy_arr[0] = 99
    fmt.Println(original[0])      // 1 (不受影响)
    fmt.Println(copy_arr[0])      // 99

    // 传参也是值复制
    // func process(arr [1000]int) { }  // 复制 1000 个元素!
    // 推荐传指针或使用切片
}

// ========== 多维数组 ==========
func multiDim() {
    var matrix [3][3]int
    matrix[0][0] = 1
    matrix[1][1] = 1

    board := [2][3]int{
        {1, 2, 3},
        {4, 5, 6},
    }
}
```


## 切片 (Slice)


```
// ========== 切片 ==========
// 动态数组, 引用类型
// 结构: 指针 + 长度 (len) + 容量 (cap)

func main() {
    // ========== 创建切片 ==========
    // 方式 1: 字面量
    s1 := []int{1, 2, 3}           // len=3, cap=3

    // 方式 2: make (指定长度和容量)
    s2 := make([]int, 5)            // len=5, cap=5, [0,0,0,0,0]
    s3 := make([]int, 3, 5)         // len=3, cap=5, [0,0,0]

    // 方式 3: 从数组/切片创建子切片
    arr := [5]int{10, 20, 30, 40, 50}
    sl := arr[1:4]                  // [20, 30, 40], len=3, cap=4

    // 方式 4: nil 切片
    var s4 []int                    // nil, len=0, cap=0

    // ========== 切片操作 ==========
    nums := []int{0, 1, 2, 3, 4, 5}

    fmt.Println(nums[1:3])          // [1 2]
    fmt.Println(nums[:3])           // [0 1 2]
    fmt.Println(nums[3:])           // [3 4 5]
    fmt.Println(nums[:])            // [0 1 2 3 4 5]

    // ========== append ==========
    // 追加元素, 自动扩容 (cap 不足时翻倍)
    var items []int
    items = append(items, 1)        // [1]
    items = append(items, 2, 3)     // [1, 2, 3]

    // 追加另一个切片
    items = append(items, []int{4, 5}...)
    // [1, 2, 3, 4, 5]

    // ========== copy ==========
    src := []int{1, 2, 3}
    dst := make([]int, len(src))
    n := copy(dst, src)             // 返回复制的元素数
    fmt.Println(dst)                // [1, 2, 3]
    fmt.Println(n)                  // 3

    // ========== 切片扩容机制 ==========
    s := make([]int, 0, 2)
    fmt.Println(cap(s))             // 2
    s = append(s, 1)
    s = append(s, 2)
    fmt.Println(cap(s))             // 2
    s = append(s, 3)                // 触发扩容
    fmt.Println(cap(s))             // 4 (翻倍)

    // Go 1.18+: 小于 256 翻倍, 大于 256 增长 25%

    // ========== 切片作为引用类型 ==========
    a := []int{1, 2, 3}
    b := a
    b[0] = 99
    fmt.Println(a[0])               // 99 (共享底层数组!)
}

// ========== 常见切片操作 ==========

// 删除索引 i 的元素 (保持顺序)
func remove(s []int, i int) []int {
    return append(s[:i], s[i+1:]...)
}

// 删除索引 i 的元素 (不保持顺序, 快)
func removeFast(s []int, i int) []int {
    s[i] = s[len(s)-1]
    return s[:len(s)-1]
}

// 插入元素
func insert(s []int, i, v int) []int {
    s = append(s, 0)                // 扩容
    copy(s[i+1:], s[i:])            // 右移
    s[i] = v
    return s
}

// 过滤
func filter(s []int, fn func(int) bool) []int {
    var result []int
    for _, v := range s {
        if fn(v) {
            result = append(result, v)
        }
    }
    return result
}
// 使用: filter([]int{1,2,3,4}, func(n int) bool { return n%2 == 0 })
// → [2, 4]
```


## Map


```
// ========== Map ==========
// 键值对, 引用类型
// 键必须可比较 (comparable)

func main() {
    // ========== 创建 Map ==========
    // 方式 1: 字面量
    scores := map[string]int{
        "Alice": 95,
        "Bob":   87,
    }

    // 方式 2: make
    users := make(map[string]string)
    users["admin"] = "管理员"

    // 方式 3: nil map (不能写入!)
    // var m map[string]int    // nil, 写入会 panic

    // ========== 增删改查 ==========
    // 添加/修改
    scores["Charlie"] = 92

    // 获取 (comma ok 模式)
    score, ok := scores["Alice"]
    if ok {
        fmt.Println("Alice 分数:", score)
    } else {
        fmt.Println("Alice 不存在")
    }

    // 获取不存在 key: 返回零值
    fmt.Println(scores["Unknown"])     // 0

    // 删除
    delete(scores, "Bob")

    // 检查 key 是否存在
    if _, exists := scores["Bob"]; !exists {
        fmt.Println("Bob 已被删除")
    }

    // ========== 遍历 ==========
    for name, score := range scores {
        fmt.Printf("%s: %d\n", name, score)
    }

    // 只遍历 key
    for name := range scores {
        fmt.Println(name)
    }

    // map 遍历顺序是不确定的!
    // 需要顺序: 先提取 key 排序, 再遍历

    keys := make([]string, 0, len(scores))
    for k := range scores {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    for _, k := range keys {
        fmt.Println(k, scores[k])
    }

    // ========== 长度 ==========
    fmt.Println(len(scores))            // key 数量

    // ========== Map 是引用类型 ==========
    m1 := map[string]int{"a": 1}
    m2 := m1
    m2["a"] = 99
    fmt.Println(m1["a"])                // 99

    // ========== 常用模式 ==========

    // 使用 map 实现 Set (集合)
    set := make(map[string]bool)
    set["apple"] = true
    set["banana"] = true

    if set["apple"] {
        fmt.Println("apple 存在")
    }
    delete(set, "apple")

    // 使用 map 计数
    words := []string{"a", "b", "a", "c", "b", "a"}
    counts := make(map[string]int)
    for _, w := range words {
        counts[w]++
    }
    // counts["a"] = 3, counts["b"] = 2, counts["c"] = 1

    // map 作为缓存
    type Cache struct {
        mu    sync.Mutex
        data  map[string]interface{}
    }

    // map + struct 组合
    type User struct {
        Name string
        Age  int
    }
    userMap := make(map[int64]User)
    userMap[1001] = User{Name: "Alice", Age: 30}
}
```


## nil 切片与空切片


```
// ========== nil 切片 vs 空切片 ==========

var s1 []int              // nil 切片, len=0, cap=0
s2 := []int{}             // 空切片, len=0, cap=0
s3 := make([]int, 0)      // 空切片, len=0, cap=0

// 大多数情况下行为相同
fmt.Println(len(s1))      // 0
fmt.Println(len(s2))      // 0
fmt.Println(len(s3))      // 0

// 区别:
fmt.Println(s1 == nil)    // true
fmt.Println(s2 == nil)    // false
fmt.Println(s3 == nil)    // false

// append nil 切片正常工作
s1 = append(s1, 1)        // [1]

// JSON 序列化:
// var s []int          → null
// s := []int{}         → []

// ========== nil map 注意事项 ==========
var m map[string]int
// m["key"] = 1     // panic: assignment to nil map
// len(m) == 0      // 安全
// delete(m, "key") // 安全
// _, ok := m["key"] // 安全 (ok = false)

// 读取 nil map 返回零值
fmt.Println(m["anything"])  // 0

// 必须先 make 才能写入:
m = make(map[string]int)
m["key"] = 1                // 安全

// ========== 切片内存优化 ==========
// 大数组的切片持有整个数组的引用

// 问题:
var bigArray [1000000]int
slice := bigArray[:]        // 持有大数组引用, GC 无法回收

// 解决: 复制需要的部分
result := make([]int, 100)
copy(result, bigArray[:100])

// ========== 切片扩容性能 ==========
// 预分配容量避免多次扩容
// ❌ 慢:
var s []int
for i := 0; i < 10000; i++ {
    s = append(s, i)        // 多次扩容
}

// ✅ 快:
s := make([]int, 0, 10000)
for i := 0; i < 10000; i++ {
    s = append(s, i)        // 无需扩容
}
```


> **Note:** 💡 数据容器要点: 数组固定长度值类型; 切片动态引用类型; make 预分配容量; append 自动扩容; slice 共享底层数组; map 引用类型 comma ok; nil 与空切片区别; map 遍历无序; sort.Strings 排序 key 遍历; set 用 map[T]bool 实现。


## 练习


<!-- Converted from: 3_Go 数组切片Map.html -->
