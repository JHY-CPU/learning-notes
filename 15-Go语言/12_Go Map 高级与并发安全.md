# Go Map 高级与并发安全


## 🗺️ Go Map 高级与并发安全


Map 底层实现（哈希表/Bucket/溢出链）、并发安全模型（sync.Mutex/RWMutex/sync.Map）、顺序遍历技巧、Set 与映射模式。


## Map 底层实现


```
// ========== Map 底层结构 ==========
// Go map 基于哈希表, 运行时表示为 hmap:
// type hmap struct {
//     count     int     // 元素数量
//     B         uint8   // 桶数量 = 2^B
//     buckets   unsafe.Pointer // 桶数组指针
//     oldbuckets unsafe.Pointer // 扩容时旧桶
//     nevacuate uintptr  // 扩容迁移进度
//     ...
// }
//
// 每个桶 (bmap) 存 8 个键值对 + 溢出链指针

// ========== Map 创建与初始化 ==========
func mapCreation() {
    // 方式 1: 字面量
    m1 := map[string]int{
        "a": 1,
        "b": 2,
    }

    // 方式 2: make
    m2 := make(map[string]int)         // 零容量
    m3 := make(map[string]int, 100)    // 预分配, 减少扩容

    // 方式 3: var (nil map)
    var m4 map[string]int              // nil, 不能写入!
    // m4["key"] = 1  // panic: assignment to entry in nil map

    // 正确: 先初始化
    m4 = make(map[string]int)
    m4["key"] = 1 // OK
}

// ========== Map 操作模式 ==========
func mapOps() {
    m := make(map[string]int)

    // 增/改
    m["key"] = 1
    m["key"] = 2  // 覆盖

    // 查 (comma ok 模式)
    v, ok := m["key"]
    if ok {
        fmt.Println("找到:", v)
    }

    // 删
    delete(m, "key")

    // 清空 (Go 1.21+)
    clear(m)  // 清空所有元素

    // 长度
    fmt.Println(len(m))

    // 遍历 (无序!)
    for k, v := range m {
        fmt.Println(k, v)
    }
}

// ========== 遍历顺序随机化 ==========
// Go map 遍历故意随机化起始位置, 防止依赖顺序

func randomOrder() {
    m := map[int]string{
        1: "a", 2: "b", 3: "c", 4: "d", 5: "e",
    }

    // 每次运行顺序不同!
    for k, v := range m {
        fmt.Printf("(%d,%s) ", k, v)
    }
    // 可能: (5,e) (1,a) (2,b) (3,c) (4,d)
    // 也可能: (3,c) (1,a) (4,d) (5,e) (2,b)

    // 需要有序: 提取 key 排序
    keys := make([]int, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    sort.Ints(keys)
    for _, k := range keys {
        fmt.Printf("(%d,%s) ", k, m[k])
    }
    // 始终: (1,a) (2,b) (3,c) (4,d) (5,e)
}
```


## 并发安全


```
// ========== Map 非并发安全 ==========
// 并发读写 map 会 panic: concurrent map read and map write

// ❌ 错误: 并发写
// func concurrentWrite() {
//     m := make(map[int]int)
//     for i := 0; i < 100; i++ {
//         go func() {
//             m[1]++  // 竞态!
//         }()
//     }
// }

// ========== 方案 1: sync.RWMutex ==========
type SafeMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (m *SafeMap) Get(key string) interface{} {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return m.data[key]
}

func (m *SafeMap) Set(key string, val interface{}) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = val
}

// 原子操作: 不存在才设置
func (m *SafeMap) GetOrSet(key string, fn func() interface{}) interface{} {
    m.mu.Lock()
    defer m.mu.Unlock()
    if v, ok := m.data[key]; ok {
        return v
    }
    v := fn()
    m.data[key] = v
    return v
}

// ========== 方案 2: sync.Map ==========
// 读多写少场景优化 (空间换时间)
// 内部采用双 map: 读 (atomic) + 写 (mutex)

func syncMapDemo() {
    var m sync.Map

    // 写
    m.Store("key", "value")

    // 读
    v, ok := m.Load("key")

    // 删除
    m.Delete("key")

    // 存在则加载, 否则写入
    v, loaded := m.LoadOrStore("key", "default")

    // 原子交换
    old, loaded := m.Swap("key", "new")

    // 不存在则写入 (CAS)
    actual, loaded := m.CompareAndSwap("key", "old", "new")

    // 遍历
    m.Range(func(key, value interface{}) bool {
        fmt.Println(key, value)
        return true  // false 停止遍历
    })

    // 注意:
    // 1. 类型为 interface{}, 需断言
    // 2. 不适合有明显 key 类型的场景
    // 3. 写多场景不如 RWMutex
}

// ========== 方案 3: Sharding (分片) ==========
// 多个小 map 减少锁竞争

type ShardMap struct {
    shards []*SafeMap
    size   int
}

func NewShardMap(n int) *ShardMap {
    sm := &ShardMap{shards: make([]*SafeMap, n), size: n}
    for i := 0; i < n; i++ {
        sm.shards[i] = &SafeMap{data: make(map[string]interface{})}
    }
    return sm
}

func (sm *ShardMap) getShard(key string) *SafeMap {
    h := fnv.New32()
    h.Write([]byte(key))
    return sm.shards[int(h.Sum32())%sm.size]
}

func (sm *ShardMap) Get(key string) interface{} {
    return sm.getShard(key).Get(key)
}

func (sm *ShardMap) Set(key string, val interface{}) {
    sm.getShard(key).Set(key, val)
}
```


## Set 与映射模式


```
// ========== Set (集合) ==========
// Go 无内置 Set, 用 map[T]struct{} 实现

type Set[T comparable] struct {
    m map[T]struct{}
}

func NewSet[T comparable]() *Set[T] {
    return &Set[T]{m: make(map[T]struct{})}
}

func (s *Set[T]) Add(v T)    { s.m[v] = struct{}{} }
func (s *Set[T]) Remove(v T) { delete(s.m, v) }
func (s *Set[T]) Has(v T) bool {
    _, ok := s.m[v]
    return ok
}
func (s *Set[T]) Len() int     { return len(s.m) }
func (s *Set[T]) Clear()       { s.m = make(map[T]struct{}) }
func (s *Set[T]) Values() []T {
    out := make([]T, 0, len(s.m))
    for k := range s.m {
        out = append(out, k)
    }
    return out
}

// 集合运算
func (s *Set[T]) Union(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for v := range s.m { result.Add(v) }
    for v := range other.m { result.Add(v) }
    return result
}

func (s *Set[T]) Intersect(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for v := range s.m {
        if other.Has(v) { result.Add(v) }
    }
    return result
}

// ========== 实用映射模式 ==========

// 1. 计数器
func wordCount(words []string) map[string]int {
    m := make(map[string]int)
    for _, w := range words {
        m[w]++
    }
    return m
}

// 2. 映射值到键反转
func invertMap(m map[string]int) map[int]string {
    inv := make(map[int]string, len(m))
    for k, v := range m {
        inv[v] = k  // 注意: 值冲突时后者覆盖前者
    }
    return inv
}

// 3. 分组
func groupBy[T any, K comparable](items []T, keyFn func(T) K) map[K][]T {
    m := make(map[K][]T)
    for _, item := range items {
        k := keyFn(item)
        m[k] = append(m[k], item)
    }
    return m
}
// 使用: groupBy(users, func(u User) string { return u.City })

// 4. 缓存/记忆化 (Memoization)
func memoize[T, U comparable](fn func(T) U) func(T) U {
    cache := make(map[T]U)
    return func(x T) U {
        if v, ok := cache[x]; ok {
            return v
        }
        v := fn(x)
        cache[x] = v
        return v
    }
}

// 5. 多值映射 (MultiMap)
type MultiMap[K comparable, V any] struct {
    data map[K][]V
}

func (mm *MultiMap[K, V]) Add(key K, val V) {
    mm.data[key] = append(mm.data[key], val)
}

func (mm *MultiMap[K, V]) Get(key K) []V {
    return mm.data[key]
}
```


> **Note:** 💡 Map 要点: 哈希表实现, 桶 (bmap) 存 8 键值对 + 溢出链; nil map 写入 panic; 遍历随机化 (防依赖顺序); 需要有序: 提取 key 排序; 并发读写 panic → 用 sync.Mutex/RWMutex/sync.Map/Sharding; sync.Map 适合读多写少; Set 用 map[T]struct{} 实现 (零内存值类型); clear() 清空 (Go 1.21+); 预分配减少扩容。


## 练习


<!-- Converted from: 12_Go Map 高级与并发安全.html -->
