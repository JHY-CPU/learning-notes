# js-sys使用

## 一、概念说明

js-sys 提供了 JavaScript 内置对象的 Rust 绑定。

```rust
use wasm_bindgen::prelude::*;
use js_sys::{Array, Object, Promise, Date};

#[wasm_bindgen]
pub fn create_array() -> Array {
    let arr = Array::new();
    arr.push(&"Hello".into());
    arr.push(&42.into());
    arr
}
```

## 二、具体用法

### 2.1 数组操作

```rust
use js_sys::Array;

#[wasm_bindgen]
pub fn process_array(arr: &Array) -> Array {
    let result = Array::new();
    for i in 0..arr.length() {
        let item = arr.get(i);
        result.push(&item);
    }
    result
}

#[wasm_bindgen]
pub fn map_array(arr: &Array, f: &js_sys::Function) -> Array {
    arr.map(&mut |val, _, _| f.call1(&JsValue::NULL, &val).unwrap())
}
```

### 2.2 对象操作

```rust
use js_sys::Object;

#[wasm_bindgen]
pub fn create_object() -> Object {
    let obj = Object::new();
    js_sys::Reflect::set(&obj, &"name".into(), &"张三".into()).unwrap();
    js_sys::Reflect::set(&obj, &"age".into(), &25.into()).unwrap();
    obj
}

#[wasm_bindgen]
pub fn get_property(obj: &Object, key: &str) -> JsValue {
    js_sys::Reflect::get(obj, &key.into()).unwrap()
}
```

### 2.3 Promise 操作

```rust
use js_sys::Promise;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
pub async fn await_promise(promise: &Promise) -> Result<JsValue, JsValue> {
    JsFuture::from(promise.clone()).await
}

#[wasm_bindgen]
pub fn create_promise() -> Promise {
    Promise::new(&mut |resolve, _reject| {
        resolve.call0(&JsValue::NULL).unwrap();
    })
}
```

### 2.4 日期和时间

```rust
use js_sys::Date;

#[wasm_bindgen]
pub fn get_timestamp() -> f64 {
    Date::now()
}

#[wasm_bindgen]
pub fn format_date(timestamp: f64) -> String {
    let date = Date::new(&timestamp.into());
    date.to_locale_string("zh-CN", &JsValue::NULL)
        .as_string()
        .unwrap()
}
```

## 三、注意事项与常见陷阱

1. **类型安全**：js-sys 提供类型安全的绑定
2. **错误处理**：许多方法返回 Result
3. **性能**：频繁调用有性能开销
4. **兼容性**：确保浏览器支持
5. **文档**：参考 js-sys 文档
