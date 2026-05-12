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


## 完整 Service Worker 示例

```javascript
// ========== sw.js 完整实现 ==========
const CACHE_NAME = 'app-cache-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/styles/main.css',
    '/scripts/app.js',
    '/images/logo.png',
];

// 安装阶段: 预缓存静态资源
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting(); // 立即激活
});

// 激活阶段: 清理旧缓存
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys
                    .filter(key => key !== CACHE_NAME)
                    .map(key => caches.delete(key))
            );
        })
    );
    self.clients.claim(); // 立即接管页面
});

// 请求拦截: 实现离线策略
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // API 请求: NetworkFirst
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(networkFirst(request));
        return;
    }

    // 静态资源: CacheFirst
    if (request.destination === 'image' || request.destination === 'style' || request.destination === 'script') {
        event.respondWith(cacheFirst(request));
        return;
    }

    // 页面: StaleWhileRevalidate
    event.respondWith(staleWhileRevalidate(request));
});

// 策略实现
async function networkFirst(request) {
    try {
        const response = await fetch(request);
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, response.clone());
        return response;
    } catch {
        const cached = await caches.match(request);
        return cached || new Response('Offline', { status: 503 });
    }
}

async function cacheFirst(request) {
    const cached = await caches.match(request);
    return cached || fetch(request);
}

async function staleWhileRevalidate(request) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    const fetchPromise = fetch(request).then(response => {
        cache.put(request, response.clone());
        return response;
    });
    return cached || fetchPromise;
}

// ========== 前端注册 ==========
if ('serviceWorker' in navigator) {
    window.addEventListener('load', async () => {
        try {
            const reg = await navigator.serviceWorker.register('/sw.js');
            console.log('SW 注册成功:', reg.scope);

            // 监听更新
            reg.addEventListener('updatefound', () => {
                const newWorker = reg.installing;
                newWorker.addEventListener('statechange', () => {
                    if (newWorker.state === 'activated') {
                        console.log('新版本已激活');
                    }
                });
            });
        } catch (err) {
            console.error('SW 注册失败:', err);
        }
    });
}
```

<!-- Converted from: 43_Service Worker.html -->
