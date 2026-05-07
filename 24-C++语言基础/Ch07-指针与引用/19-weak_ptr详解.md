# weak_ptr详解

## 一、概念说明

`std::weak_ptr`是`shared_ptr`的伴生类，提供对`shared_ptr`管理对象的**弱引用**。`weak_ptr`不增加引用计数，不影响对象生命周期。主要用于解决`shared_ptr`的**循环引用**问题和缓存实现。

## 二、具体用法

### 2.1 基本使用

```cpp
std::weak_ptr<int> wp;
{
    auto sp = std::make_shared<int>(42);
    wp = sp;

    std::cout << "expired: " << wp.expired() << std::endl;  // 输出: expired: 0

    // 使用lock()获取shared_ptr
    if (auto locked = wp.lock()) {
        std::cout << *locked << std::endl;  // 输出: 42
        std::cout << "计数: " << locked.use_count() << std::endl;  // 输出: 计数: 2
    }
}
std::cout << "expired: " << wp.expired() << std::endl;  // 输出: expired: 1
```

### 2.2 打破循环引用

```cpp
struct Node {
    std::string name;
    std::shared_ptr<Node> next;   // 持有后继
    std::weak_ptr<Node> prev;     // 持有前驱（弱引用避免循环）

    Node(const std::string& n) : name(n) {
        std::cout << name << " 构造\n";
    }
    ~Node() { std::cout << name << " 析构\n"; }
};

auto a = std::make_shared<Node>("A");
auto b = std::make_shared<Node>("B");
a->next = b;
b->prev = a;  // weak_ptr不增加引用计数
// 输出:
// A 构造
// B 构造
// 离开作用域后:
// B 析构
// A 析构
```

### 2.3 缓存实现

```cpp
class Cache {
    std::unordered_map<std::string, std::weak_ptr<std::string>> cache;
public:
    std::shared_ptr<std::string> get(const std::string& key) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (auto ptr = it->second.lock()) return ptr;  // 未过期
            cache.erase(it);  // 已过期，清理
        }
        auto val = std::make_shared<std::string>("data_" + key);
        cache[key] = val;
        return val;
    }
};
```

### 2.4 观察者模式

```cpp
class Observer {
public:
    void onEvent() { std::cout << "收到事件\n"; }
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers;
public:
    void addObserver(std::shared_ptr<Observer> obs) {
        observers.push_back(obs);
    }
    void notify() {
        for (auto it = observers.begin(); it != observers.end(); ) {
            if (auto obs = it->lock()) {
                obs->onEvent();
                ++it;
            } else {
                it = observers.erase(it);  // 观察者已销毁
            }
        }
    }
};
```

## 三、注意事项与常见陷阱

- `weak_ptr`不能直接访问对象，必须通过`lock()`获取`shared_ptr`
- `expired()`检查对象是否已被销毁（但可能有竞态条件）
- 使用`lock()`而非`expired()`+访问，避免TOCTOU问题
- `weak_ptr`可从`shared_ptr`赋值，反之不行
- 循环引用是`shared_ptr`最常见的陷阱
