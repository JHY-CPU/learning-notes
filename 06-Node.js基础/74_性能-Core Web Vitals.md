# 性能-Core Web Vitals


## 性能-Core Web Vitals


LCP/FID/CLS/INP/FCP/TTFB 指标定义、测量、优化策略。


## Web Vitals API


```
// ========== Core Web Vitals ==========
// LCP — Largest Contentful Paint   (≤2.5s)
// FID — First Input Delay          (≤100ms) → INP
// CLS — Cumulative Layout Shift    (≤0.1)

// ========== 其他指标 ==========
// FCP  — First Contentful Paint   (≤1.8s)
// TTFB — Time to First Byte       (≤800ms)
// INP  — Interaction to Next Paint (≤200ms) 新取代 FID

// ========== JS 测量 ==========
new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        console.log('LCP:', entry.startTime);
    }
}).observe({ type: 'largest-contentful-paint', buffered: true });
```


## 演示：Web Vitals

点击按钮查看


<!-- Converted from: 74_性能-Core Web Vitals.html -->
