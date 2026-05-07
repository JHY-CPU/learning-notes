# 常见的 DOM 操作场景

React 推荐声明式编程，但有些场景仍然需要直接操作 DOM。本文总结最常见的 DOM 操作场景及其实现方式。

---

## 一、焦点管理

### 1.1 自动聚焦

```jsx
function SearchInput() {
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return <input ref={inputRef} type="search" placeholder="搜索..." />;
}
```

### 1.2 条件聚焦

```jsx
function LoginForm() {
  const [showPassword, setShowPassword] = useState(false);
  const passwordRef = useRef(null);

  useEffect(() => {
    if (showPassword) {
      passwordRef.current?.focus();
    }
  }, [showPassword]);

  return (
    <form>
      <input type="text" placeholder="用户名" />
      {showPassword && (
        <input ref={passwordRef} type="password" placeholder="密码" />
      )}
      <button type="button" onClick={() => setShowPassword(true)}>
        下一步
      </button>
    </form>
  );
}
```

### 1.3 焦点陷阱（模态框）

```jsx
function Modal({ isOpen, onClose, children }) {
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      // 保存当前聚焦的元素
      previousFocusRef.current = document.activeElement;

      // 聚焦到模态框
      modalRef.current?.focus();
    } else {
      // 关闭时恢复之前的焦点
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);

  // Tab 键焦点限制在模态框内
  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      const focusable = modalRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        ref={modalRef}
        className="modal"
        tabIndex={-1}
        onKeyDown={handleKeyDown}
        onClick={(e) => e.stopPropagation()}
      >
        <button onClick={onClose}>关闭</button>
        {children}
      </div>
    </div>
  );
}
```

### 1.4 可访问性焦点管理

```jsx
// 通知屏幕阅读器的 live region
function Announcer({ message }) {
  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      style={{ position: 'absolute', left: '-9999px' }}
    >
      {message}
    </div>
  );
}

// 错误消息聚焦
function FormField({ label, error, children }) {
  const errorRef = useRef(null);

  useEffect(() => {
    if (error) {
      errorRef.current?.focus();
    }
  }, [error]);

  return (
    <div>
      <label>{label}</label>
      {children}
      {error && (
        <span ref={errorRef} role="alert" tabIndex={-1} className="error">
          {error}
        </span>
      )}
    </div>
  );
}
```

---

## 二、滚动控制

### 2.1 滚动到指定位置

```jsx
function ChatWindow({ messages }) {
  const bottomRef = useRef(null);

  // 新消息时自动滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  return (
    <div className="chat" style={{ height: 400, overflow: 'auto' }}>
      {messages.map(msg => (
        <div key={msg.id} className="message">{msg.text}</div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

### 2.2 平滑滚动到锚点

```jsx
function TableOfContents({ sections }) {
  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  };

  return (
    <nav>
      {sections.map(section => (
        <button key={section.id} onClick={() => scrollToSection(section.id)}>
          {section.title}
        </button>
      ))}
    </nav>
  );
}
```

### 2.3 滚动位置记忆

```jsx
function ScrollRestoration({ children }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    const savedPosition = sessionStorage.getItem('scrollPosition');

    if (savedPosition) {
      container.scrollTop = parseInt(savedPosition, 10);
    }

    const handleScroll = () => {
      sessionStorage.setItem('scrollPosition', container.scrollTop);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div ref={containerRef} style={{ height: '100vh', overflow: 'auto' }}>
      {children}
    </div>
  );
}
```

### 2.4 无限滚动检测

```jsx
function InfiniteList({ loadMore, hasMore, children }) {
  const sentinelRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && hasMore) {
          loadMore();
        }
      },
      { rootMargin: '200px' }
    );

    if (sentinelRef.current) {
      observer.observe(sentinelRef.current);
    }

    return () => observer.disconnect();
  }, [loadMore, hasMore]);

  return (
    <div>
      {children}
      <div ref={sentinelRef} style={{ height: 1 }} />
      {hasMore && <div>加载中...</div>}
    </div>
  );
}
```

---

## 三、测量元素

### 3.1 getBoundingClientRect

```jsx
function Tooltip({ children, content }) {
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);
  const [position, setPosition] = useState(null);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (show && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setPosition({
        top: rect.bottom + window.scrollY + 8,
        left: rect.left + window.scrollX,
      });
    }
  }, [show]);

  return (
    <div
      ref={triggerRef}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      style={{ display: 'inline-block' }}
    >
      {children}
      {show && position && (
        <div
          ref={tooltipRef}
          className="tooltip"
          style={{ position: 'absolute', ...position }}
        >
          {content}
        </div>
      )}
    </div>
  );
}
```

### 3.2 ResizeObserver

```jsx
function useResizeObserver(ref) {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setDimensions({ width, height });
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, [ref]);

  return dimensions;
}

// 使用
function ResponsiveComponent() {
  const containerRef = useRef(null);
  const { width } = useResizeObserver(containerRef);

  return (
    <div ref={containerRef}>
      {width < 600 ? <MobileLayout /> : <DesktopLayout />}
      <p>容器宽度: {Math.round(width)}px</p>
    </div>
  );
}
```

### 3.3 元素尺寸测量 Hook

```jsx
function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;

    const updateSize = () => {
      const { width, height } = element.getBoundingClientRect();
      setSize({ width, height });
    };

    updateSize();

    const observer = new ResizeObserver(updateSize);
    observer.observe(element);

    return () => observer.disconnect();
  }, [ref]);

  return size;
}
```

---

## 四、Canvas / WebGL 集成

### 4.1 Canvas 2D

```jsx
function DrawingCanvas() {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e) => {
    isDrawing.current = true;
    lastPos.current = getPos(e);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const pos = getPos(e);

    ctx.beginPath();
    ctx.moveTo(lastPos.current.x, lastPos.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    lastPos.current = pos;
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      style={{ border: '1px solid #ccc' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
}
```

### 4.2 与 Three.js 集成

```jsx
import { useEffect, useRef } from 'react';
import * as THREE from 'three';

function ThreeScene() {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Three.js 场景初始化
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();

    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // 创建立方体
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    camera.position.z = 5;

    // 动画循环
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      cube.rotation.x += 0.01;
      cube.rotation.y += 0.01;
      renderer.render(scene, camera);
    };
    animate();

    sceneRef.current = { scene, camera, renderer };

    // 清理
    return () => {
      cancelAnimationFrame(animationId);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={containerRef} style={{ width: '100%', height: 400 }} />;
}
```

---

## 五、媒体播放控制

```jsx
function VideoPlayer({ src }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => setDuration(video.duration);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, []);

  const togglePlay = () => {
    if (isPlaying) videoRef.current.pause();
    else videoRef.current.play();
  };

  const seek = (time) => {
    videoRef.current.currentTime = time;
  };

  return (
    <div>
      <video ref={videoRef} src={src} width="640" />
      <div className="controls">
        <button onClick={togglePlay}>{isPlaying ? '暂停' : '播放'}</button>
        <input
          type="range"
          min={0}
          max={duration}
          value={currentTime}
          onChange={(e) => seek(parseFloat(e.target.value))}
        />
        <span>{Math.floor(currentTime)}s / {Math.floor(duration)}s</span>
      </div>
    </div>
  );
}
```

---

## 六、点击外部检测

```jsx
function useClickOutside(ref, handler) {
  useEffect(() => {
    const listener = (event) => {
      // 点击的是 ref 内部元素，忽略
      if (!ref.current || ref.current.contains(event.target)) {
        return;
      }
      handler(event);
    };

    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}

// 使用
function Dropdown({ items, onSelect }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useClickOutside(dropdownRef, () => setIsOpen(false));

  return (
    <div ref={dropdownRef} className="dropdown">
      <button onClick={() => setIsOpen(!isOpen)}>选择</button>
      {isOpen && (
        <ul className="dropdown-menu">
          {items.map(item => (
            <li key={item.id} onClick={() => {
              onSelect(item);
              setIsOpen(false);
            }}>
              {item.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

---

## 七、第三方 DOM 库集成

### 7.1 集成 jQuery 插件

```jsx
function DatePicker({ onChange }) {
  const inputRef = useRef(null);

  useEffect(() => {
    // 假设使用 jQuery datepicker
    const $input = $(inputRef.current);
    $input.datepicker({
      onSelect: (dateText) => onChange(dateText),
    });

    return () => {
      $input.datepicker('destroy');
    };
  }, [onChange]);

  return <input ref={inputRef} type="text" />;
}
```

### 7.2 集成 D3.js

```jsx
import * as d3 from 'd3';

function BarChart({ data }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    const width = 500;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    svg.selectAll('*').remove();  // 清空

    const x = d3.scaleBand()
      .domain(data.map(d => d.label))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .range([height - margin.bottom, margin.top]);

    svg.append('g')
      .attr('fill', 'steelblue')
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', d => x(d.label))
      .attr('y', d => y(d.value))
      .attr('height', d => y(0) - y(d.value))
      .attr('width', x.bandwidth());

    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));
  }, [data]);

  return <svg ref={svgRef} width={500} height={300} />;
}
```

---

## 八、全局事件监听

```jsx
function useGlobalEvent(eventName, handler, options) {
  const handlerRef = useRef(handler);

  useEffect(() => {
    handlerRef.current = handler;
  });

  useEffect(() => {
    const listener = (event) => handlerRef.current(event);
    window.addEventListener(eventName, listener, options);
    return () => window.removeEventListener(eventName, listener, options);
  }, [eventName]);
}

// 使用
function App() {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useGlobalEvent('mousemove', (e) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  });

  useGlobalEvent('keydown', (e) => {
    if (e.key === 'Escape') {
      // 关闭弹窗等
    }
  });

  return <div>鼠标位置: {mousePos.x}, {mousePos.y}</div>;
}
```
