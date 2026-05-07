# Canvas绑定

## 一、概念说明

WASM 可通过 web-sys 操作 Canvas 2D 或 WebGL 上下文，实现高性能图形渲染。

```rust
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, CanvasRenderingContext2d};

#[wasm_bindgen]
pub fn draw(canvas: HtmlCanvasElement) {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    ctx.set_fill_style(&JsValue::from_str("blue"));
    ctx.fill_rect(10.0, 10.0, 100.0, 100.0);
}
```

## 二、具体用法

### 2.1 Canvas 2D 绑定

```rust
use web_sys::CanvasRenderingContext2d;

pub fn draw_shapes(ctx: &CanvasRenderingContext2d) {
    // 矩形
    ctx.set_fill_style(&JsValue::from_str("#FF0000"));
    ctx.fill_rect(20.0, 20.0, 150.0, 100.0);

    // 圆形
    ctx.begin_path();
    ctx.arc(200.0, 200.0, 50.0, 0.0, std::f64::consts::PI * 2.0)
        .unwrap();
    ctx.set_fill_style(&JsValue::from_str("green"));
    ctx.fill();

    // 文字
    ctx.set_font("24px Arial");
    ctx.set_fill_style(&JsValue::from_str("black"));
    ctx.fill_text("Hello WASM", 50.0, 300.0).unwrap();
}
```

### 2.2 路径与渐变

```rust
pub fn draw_with_gradient(ctx: &CanvasRenderingContext2d) {
    // 创建渐变
    let gradient = ctx
        .create_linear_gradient(0.0, 0.0, 300.0, 0.0)
        .unwrap();
    gradient.add_color_stop(0.0, "red").unwrap();
    gradient.add_color_stop(1.0, "blue").unwrap();

    ctx.set_fill_style(&gradient);
    ctx.fill_rect(0.0, 0.0, 300.0, 150.0);

    // 贝塞尔曲线
    ctx.begin_path();
    ctx.move_to(0.0, 300.0);
    ctx.bezier_curve_to(100.0, 100.0, 200.0, 500.0, 300.0, 300.0);
    ctx.stroke();
}
```

### 2.3 像素级操作

```rust
pub fn pixel_manipulation(ctx: &CanvasRenderingContext2d, width: u32, height: u32) {
    let image_data = ctx.get_image_data(0.0, 0.0, width as f64, height as f64).unwrap();
    let mut data = image_data.data();

    // 反色处理
    for i in (0..data.0.len()).step_by(4) {
        data.0[i] = 255 - data.0[i];       // R
        data.0[i + 1] = 255 - data.0[i + 1]; // G
        data.0[i + 2] = 255 - data.0[i + 2]; // B
    }

    ctx.put_image_data(&image_data, 0.0, 0.0).unwrap();
}
```

### 2.4 requestAnimationFrame 集成

```rust
use wasm_bindgen::JsCast;

pub fn animation_loop() {
    let f: std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // 每帧绘制逻辑
        draw_frame();

        // 请求下一帧
        web_sys::window()
            .unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }) as Box<dyn FnMut()>));

    web_sys::window()
        .unwrap()
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())
        .unwrap();
}
```

## 三、注意事项与常见陷阱

1. **类型转换**：Canvas 上下文需要正确的 `dyn_into` 转换
2. **浮点精度**：Canvas 坐标使用 f64，注意精度问题
3. **性能优化**：尽量减少 WASM-JS 边界调用次数
4. **内存管理**：Closure 需妥善管理避免内存泄漏
5. **WebGL 选择**：高性能场景考虑使用 WebGL 而非 2D Context
