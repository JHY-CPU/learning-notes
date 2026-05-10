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


<!-- Converted from: 44_Cache API.html -->
