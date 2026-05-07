# weak_ptr详解

## 一、概念说明

`std::weak_ptr`是`shared_ptr`的观察者，不增加引用计数。它用于解决`shared_ptr`的循环引用问题，通过`lock()`方法获取一个临时的`shared_ptr`来安全访问对象。

`weak_ptr`不拥有所指向的对象，不能直接解引用，必须先转换为`shared_ptr`。

```cpp
#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> wp;

    {
        auto sp = std::make_shared<int>(42);
        wp = sp;

        std::cout << "sp引用计数: " << sp.use_count() << std::endl;
        std::cout << "wp.expired(): " << wp.expired() << std::endl;

        // lock()获取shared_ptr
        if (auto locked = wp.lock()) {
            std::cout << "lock成功，值: " << *locked << std::endl;
            std::cout << "sp引用计数: " << sp.use_count() << std::endl;
        }
    } // sp销毁，对象释放

    std::cout << "sp销毁后 wp.expired(): " << wp.expired() << std::endl;
    if (auto locked = wp.lock()) {
        std::cout << "lock成功" << std::endl;
    } else {
        std::cout << "lock失败，对象已销毁" << std::endl;
    }

    return 0;
}
```

**输出：**
```
sp引用计数: 1
wp.expired(): 0
lock成功，值: 42
sp引用计数: 2
sp销毁后 wp.expired(): 1
lock失败，对象已销毁
```

## 二、具体用法

### 2.1 用weak_ptr实现缓存

```cpp
#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

class ExpensiveResource {
    std::string data;
public:
    ExpensiveResource(std::string d) : data(std::move(d)) {
        std::cout << "创建资源: " << data << std::endl;
    }
    ~ExpensiveResource() {
        std::cout << "销毁资源: " << data << std::endl;
    }
    const std::string& getData() const { return data; }
};

class Cache {
    std::unordered_map<std::string, std::weak_ptr<ExpensiveResource>> cache;
public:
    std::shared_ptr<ExpensiveResource> get(const std::string& key) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (auto locked = it->second.lock()) {
                std::cout << "缓存命中: " << key << std::endl;
                return locked;
            }
            cache.erase(it);
        }
        auto resource = std::make_shared<ExpensiveResource>(key);
        cache[key] = resource;
        return resource;
    }
};

int main() {
    Cache cache;
    auto r1 = cache.get("config");
    auto r2 = cache.get("config"); // 命中缓存
    r1.reset();
    r2.reset();
    auto r3 = cache.get("config"); // 已过期，重新创建
    return 0;
}
```

**输出：**
```
创建资源: config
缓存命中: config
销毁资源: config
创建资源: config
销毁资源: config
```

## 三、注意事项与常见陷阱

- **`weak_ptr`不能直接解引用**：必须先`lock()`。
- **`expired()`不是原子的**：在`expired()`检查和`lock()`之间对象可能被释放。
- **`lock()`是安全的**：返回的`shared_ptr`要么有效要么为空。
- **用于观察者模式**：观察者持有被观察者的`weak_ptr`。
- **`use_count()`返回关联`shared_ptr`的数量**：不包含`weak_ptr`自身的计数。
