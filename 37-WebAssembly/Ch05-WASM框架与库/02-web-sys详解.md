# web-sys详解

## 一、概念说明

web-sys 为所有 Web API 提供了 Rust 绑定，覆盖 DOM、Fetch、Canvas、WebGL、Web Audio 等。需通过 feature flags 按需启用。

```toml
# Cargo.toml
[dependencies.web-sys]
version = "0.3"
features = [
    "Window",
    "Document",
    "HtmlElement",
    "HtmlCanvasElement",
    "CanvasRenderingContext2d",
    "console",
]
```

## 二、具体用法

### 2.1 DOM API 完整示例

```rust
use web_sys::{Document, Element, HtmlElement, Window};

fn dom_example() -> Result<(), JsValue> {
    let window: Window = web_sys::window().expect("无 window");
    let document: Document = window.document().expect("无 document");

    let body = document.body().expect("无 body");

    let div: Element = document.create_element("div")?;
    let html_div: HtmlElement = div.dyn_into()?;

    html_div.set_inner_text("由 WASM 生成");
    html_div.set_class_name("wasm-content");
    html_div.style().set_property("color", "blue")?;

    body.append_child(&html_div)?;
    Ok(())
}
```

### 2.2 Fetch API 高级用法

```rust
use web_sys::{Request, RequestInit, RequestMode, Response};
use wasm_bindgen_futures::JsFuture;

pub async fn fetch_json(url: &str) -> Result<JsValue, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;

    request.headers().set("Accept", "application/json")?;

    let window = web_sys::window().unwrap();
    let resp_promise = window.fetch_with_request(&request);
    let resp_val = JsFuture::from(resp_promise).await?;
    let resp: Response = resp_val.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!("HTTP {}", resp.status())));
    }

    let json = JsFuture::from(resp.json()?).await?;
    Ok(json)
}
```

### 2.3 WebGL 基础绑定

```rust
use web_sys::{WebGlProgram, WebGlShader, WebGlRenderingContext};

fn init_gl(canvas: &web_sys::HtmlCanvasElement) -> Result<WebGlRenderingContext, JsValue> {
    let gl: WebGlRenderingContext = canvas
        .get_context("webgl")?
        .unwrap()
        .dyn_into()?;

    let vert_shader = compile_shader(
        &gl,
        WebGlRenderingContext::VERTEX_SHADER,
        "attribute vec4 position; void main() { gl_Position = position; }",
    )?;

    let frag_shader = compile_shader(
        &gl,
        WebGlRenderingContext::FRAGMENT_SHADER,
        "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
    )?;

    let program = link_program(&gl, &vert_shader, &frag_shader)?;
    gl.use_program(Some(&program));

    Ok(gl)
}

fn compile_shader(gl: &WebGlRenderingContext, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl.create_shader(shader_type).ok_or("无法创建 shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl.get_shader_parameter(&shader, WebGlRenderingContext::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        Err(gl.get_shader_info_log(&shader).unwrap_or_else(|| "未知错误".into()))
    }
}
```

### 2.4 Web Audio

```rust
use web_sys::AudioContext;

pub fn play_tone(freq: f32, duration: f32) -> Result<(), JsValue> {
    let ctx = AudioContext::new()?;

    let oscillator = ctx.create_oscillator()?;
    oscillator.frequency().set_value(freq);
    oscillator.start()?;

    let gain = ctx.create_gain()?;
    gain.gain().set_value(0.5);
    oscillator.connect_with_audio_node(&gain)?;
    gain.connect_with_audio_node(&ctx.destination())?;

    oscillator.stop_with_when(ctx.current_time() as f64 + duration as f64)?;
    Ok(())
}
```

## 三、注意事项与常见陷阱

1. **Feature Flags**：必须在 Cargo.toml 中启用需要的 feature，否则编译报错
2. **可选方法**：许多 API 返回 Option，因为某些浏览器不支持
3. **类型转换**：Element → HtmlElement 需要 `dyn_into()`
4. **异步 API**：大部分 Fetch/WebSocket 等 API 是异步的
5. **包体积**：启用太多 feature 会增大 WASM 二进制大小
