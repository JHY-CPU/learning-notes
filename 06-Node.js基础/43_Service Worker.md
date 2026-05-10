# Service Worker


## Service Worker


注册/安装/激活、fetch 事件拦截、离线策略(NetworkFirst/CacheFirst)。


## Service Worker API


```
// ========== 注册 ==========
navigator.serviceWorker.register('/sw.js')
    .then(reg => console.log('注册成功:', reg.scope))
    .catch(err => console.error('注册失败:', err));

// ========== 生命周期 ==========
// 安装 → 激活 → 空闲 → 拦截请求
self.addEventListener('install', event => {
    event.waitUntil(self.skipWaiting());
});
self.addEventListener('activate', event => {
    event.waitUntil(self.clients.claim());
});

// ========== 拦截请求 ==========
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(cached => {
            return cached || fetch(event.request);
        })
    );
});

// ========== 离线策略 ==========
// NetworkFirst: 优先网络, 失败用缓存
// CacheFirst: 优先缓存, 无缓存查网络
// StaleWhileRevalidate: 缓存返回 + 后台更新
// NetworkOnly: 仅网络
// CacheOnly: 仅缓存
```


## 演示：Service Worker

点击按钮查看


<!-- Converted from: 43_Service Worker.html -->
