# 嵌入式与IoT

## 一、概念说明

WASM 作为轻量级运行时可用于嵌入式设备和 IoT 场景，提供安全的沙箱执行环境。

```rust
// WASI 应用可在嵌入式设备上运行
use std::io::{self, Write};

fn main() {
    let sensor_data = read_sensor();
    let processed = process_data(&sensor_data);
    io::stdout().write_all(processed.as_bytes()).unwrap();
}
```

## 二、具体用法

### 2.1 传感器数据处理

```rust
#[wasm_bindgen]
pub struct SensorProcessor {
    window_size: usize,
    buffer: Vec<f64>,
    threshold: f64,
}

#[wasm_bindgen]
impl SensorProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize, threshold: f64) -> SensorProcessor {
        SensorProcessor {
            window_size,
            buffer: Vec::with_capacity(window_size),
            threshold,
        }
    }

    pub fn add_reading(&mut self, value: f64) -> bool {
        self.buffer.push(value);
        if self.buffer.len() > self.window_size {
            self.buffer.remove(0);
        }

        let avg = self.buffer.iter().sum::<f64>() / self.buffer.len() as f64;
        avg > self.threshold
    }

    pub fn get_average(&self) -> f64 {
        if self.buffer.is_empty() { return 0.0; }
        self.buffer.iter().sum::<f64>() / self.buffer.len() as f64
    }

    pub fn get_variance(&self) -> f64 {
        let avg = self.get_average();
        self.buffer.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / self.buffer.len() as f64
    }
}
```

### 2.2 协议解析

```rust
#[wasm_bindgen]
pub fn parse_modbus_frame(frame: &[u8]) -> JsValue {
    if frame.len() < 8 {
        return JsValue::null();
    }

    let slave_id = frame[0];
    let function_code = frame[1];
    let address = u16::from_be_bytes([frame[2], frame[3]]);
    let count = u16::from_be_bytes([frame[4], frame[5]]);
    let crc = u16::from_le_bytes([frame[frame.len()-2], frame[frame.len()-1]]);

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"slaveId".into(), &JsValue::from(slave_id)).unwrap();
    js_sys::Reflect::set(&result, &"functionCode".into(), &JsValue::from(function_code)).unwrap();
    js_sys::Reflect::set(&result, &"address".into(), &JsValue::from(address)).unwrap();
    js_sys::Reflect::set(&result, &"count".into(), &JsValue::from(count)).unwrap();
    result.into()
}
```

### 2.3 控制逻辑

```rust
#[wasm_bindgen]
pub struct PidController {
    kp: f64,
    ki: f64,
    kd: f64,
    integral: f64,
    prev_error: f64,
}

#[wasm_bindgen]
impl PidController {
    #[wasm_bindgen(constructor)]
    pub fn new(kp: f64, ki: f64, kd: f64) -> PidController {
        PidController { kp, ki, kd, integral: 0.0, prev_error: 0.0 }
    }

    pub fn update(&mut self, setpoint: f64, measured: f64, dt: f64) -> f64 {
        let error = setpoint - measured;
        self.integral += error * dt;
        let derivative = (error - self.prev_error) / dt;
        self.prev_error = error;

        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}
```

### 2.4 配置管理

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_id: String,
    pub sampling_rate: u32,
    pub thresholds: Vec<f64>,
    pub enabled_sensors: Vec<String>,
}

#[wasm_bindgen]
pub fn load_config(data: &[u8]) -> Result<JsValue, JsValue> {
    let config: DeviceConfig = serde_json::from_slice(data)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&config)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

### 2.5 OTA 更新

```rust
#[wasm_bindgen]
pub fn verify_firmware(data: &[u8], expected_hash: &[u8]) -> bool {
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(data);
    hash.as_slice() == expected_hash
}

#[wasm_bindgen]
pub fn apply_patch(base: &[u8], patch: &[u8]) -> Vec<u8> {
    // 简易差分补丁应用
    let mut result = base.to_vec();
    let mut i = 0;
    while i + 2 < patch.len() {
        let offset = u16::from_le_bytes([patch[i], patch[i+1]]) as usize;
        let len = patch[i+2] as usize;
        if offset + len <= result.len() && i + 3 + len <= patch.len() {
            result[offset..offset+len].copy_from_slice(&patch[i+3..i+3+len]);
        }
        i += 3 + len;
    }
    result
}
```

## 三、注意事项与常见陷阱

1. **资源限制**：嵌入式设备内存和计算能力有限
2. **实时性要求**：WASM 执行时间可预测性不如原生代码
3. **安全沙箱**：WASM 沙箱可防止恶意固件
4. **跨平台**：同一 WASM 可在不同架构设备上运行
5. **调试支持**：嵌入式 WASM 调试工具链尚不完善
