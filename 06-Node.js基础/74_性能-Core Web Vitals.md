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


## 什么是 Core Web Vitals

Core Web Vitals 是Google定义的一组衡量网页用户体验的核心指标，直接影响搜索排名（SEO）。

## 三大核心指标

| 指标 | 含义 | 良好阈值 | 优化方向 |
|------|------|---------|---------|
| **LCP** | 最大内容绘制时间 | ≤2.5s | 优化服务器响应、CDN、图片优化、预加载关键资源 |
| **INP** | 交互到下次绘制延迟 | ≤200ms | 拆分长任务、减少JS执行、使用Web Worker |
| **CLS** | 累积布局偏移 | ≤0.1 | 为图片/视频预留尺寸、避免动态插入内容、使用CSS `contain` |

## INP 替代 FID

INP（Interaction to Next Paint）是2024年正式取代FID的新指标，测量所有交互的响应延迟，而FID只测量首次交互。更能反映整体交互体验。

## 测量方法

1. **实验室测量**：Lighthouse、Chrome DevTools Performance面板
2. **真实用户数据**：Chrome UX Report（CrUX）、`web-vitals` 库
3. **JS API**：PerformanceObserver 监听各指标

## Node.js 后端配合

- 优化TTFB：服务端渲染缓存、数据库查询优化、CDN边缘缓存
- 设置合理的缓存头减少重复请求
- 压缩响应体减少传输时间

<!-- Converted from: 74_性能-Core Web Vitals.html -->
