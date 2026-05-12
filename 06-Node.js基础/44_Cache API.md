# Cache API


## Cache API


caches.open/add/delete/match、缓存版本管理、离线存储。


## Cache API 方法


```
// ========== 打开缓存 ==========
caches.open('my-cache').then(cache => {
    // cache 对象可用
});

// ========== 添加资源 ==========
cache.add('/api/data');          // fetch + put
cache.addAll(['/', '/style.css']);

// ========== 存储响应 ==========
cache.put('/api/data', response);

// ========== 读取缓存 ==========
cache.match('/api/data');        // 匹配单个
caches.match('/api/data');       // 所有缓存中查找

// ========== 删除 ==========
cache.delete('/api/data');
caches.delete('my-cache');       // 删除整个缓存

// ========== 遍历 ==========
cache.keys();                    // 所有请求
cache.keys('/api');              // 匹配模式
```


## 演示：Cache API

点击按钮查看


## 什么是 Cache API

Cache API 是浏览器提供的缓存接口，与Service Worker配合实现离线应用和资源缓存。基于Promise，比localStorage更强大，可存储Response对象。

## 工作流程

1. `caches.open(name)` 打开/创建命名缓存
2. `cache.add(url)` 发起fetch并存储Response
3. `cache.match(request)` 从缓存中匹配Response
4. 返回缓存的Response或发起网络请求（离线优先/网络优先策略）

## 缓存策略

- **Cache First**：先查缓存，没有则网络请求（适合静态资源）
- **Network First**：先网络，失败再查缓存（适合动态数据）
- **Stale While Revalidate**：返回缓存同时后台更新（平衡速度与新鲜度）

## 注意事项

- Cache API只存储Response，无法缓存非HTTP请求
- 同源限制，不能跨域缓存
- 需定期清理旧版本缓存避免占用存储空间
- 存储配额受浏览器限制（通常可用容量为磁盘的一定比例）
- 与Service Worker的fetch事件配合使用效果最佳

<!-- Converted from: 44_Cache API.html -->
