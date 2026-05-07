# wgpu图形库

## 一、概念说明

wgpu 是跨平台图形 API 封装，支持 Vulkan/Metal/DX12/WebGPU 后端。通过 WASM 可在浏览器中使用 WebGPU。

```rust
use wgpu::Surface;

async fn init_gpu(canvas: &web_sys::HtmlCanvasElement) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..Default::default()
    });

    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .unwrap();

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }).await.unwrap();
}
```

## 二、具体用法

### 2.1 Web 环境初始化

```rust
pub async fn setup(canvas: web_sys::HtmlCanvasElement) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..Default::default()
    });

    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .unwrap();

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }).await.unwrap();

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            ..Default::default()
        },
        None,
    ).await.unwrap();
}
```

### 2.2 着色器与渲染管线

```rust
fn create_render_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        ..Default::default()
    });

    pipeline
}
```

### 2.3 WGSL 着色器

```wgsl
// shader.wgsl
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

### 2.4 requestAnimationFrame 渲染循环

```rust
pub fn run_render_loop(device: wgpu::Device, queue: wgpu::Queue, surface: wgpu::Surface<'static>) {
    let f: std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let output = surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        // 渲染命令...

        queue.submit(std::iter::once(encoder.finish()));
        output.present();

        web_sys::window().unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }) as Box<dyn FnMut()>));

    web_sys::window().unwrap()
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())
        .unwrap();
}
```

## 三、注意事项与常见陷阱

1. **WebGPU 支持**：Chrome/Edge 支持，Safari/Firefox 有限支持
2. **WebGL 后端**：wgpu 可降级到 WebGL，但功能受限
3. **Limits 限制**：WebGL 有更严格的 buffer/texture 大小限制
4. **性能差异**：Web 端 wgpu 比原生慢，适合中等复杂度场景
5. **浏览器兼容**：需检测 WebGPU 可用性，提供降级方案
