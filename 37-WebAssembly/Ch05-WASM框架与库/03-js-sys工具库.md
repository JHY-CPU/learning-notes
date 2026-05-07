# js-sys工具库

## 一、概念说明

js-sys 提供了 JavaScript 内置对象（Array, Object, Map, Set, Promise 等）的 Rust 绑定。

```rust
use js_sys::{Array, Object, Map, Set, Promise};

// Array 操作
let arr = Array::new();
arr.push(&JsValue::from(42));
arr.push(&JsValue::from_str("hello"));

// Map 操作
let map = Map::new();
map.set(&"key".into(), &"value".into());
```

## 二、具体用法

### 2.1 Array 操作

```rust
use js_sys::Array;

#[wasm_bindgen]
pub fn process_array(input: &Array) -> Array {
    let output = Array::new();

    for i in 0..input.length() {
        let val = input.get(i);
        if let Some(num) = val.as_f64() {
            output.push(&JsValue::from_f64(num * 2.0));
        }
    }

    // 其他操作
    // input.push(&val);
    // input.pop();
    // input.shift();
    // input.unshift(&val);
    // input.splice(1, 2, &Array::of3(&a, &b, &c));

    output
}
```

### 2.2 Object 操作

```rust
use js_sys::Object;

pub fn create_config(host: &str, port: u16) -> Object {
    let config = Object::new();

    Reflect::set(&config, &"host".into(), &host.into()).unwrap();
    Reflect::set(&config, &"port".into(), &JsValue::from(port)).unwrap();
    Reflect::set(&config, &"ssl".into(), &true.into()).unwrap();

    config
}

pub fn read_config(config: &Object) -> (String, u16) {
    let host = Reflect::get(config, &"host".into())
        .unwrap()
        .as_string()
        .unwrap();
    let port = Reflect::get(config, &"port".into())
        .unwrap()
        .as_f64()
        .unwrap() as u16;
    (host, port)
}
```

### 2.3 Map 与 Set

```rust
use js_sys::{Map, Set};

pub fn demo_map() -> Map {
    let map = Map::new();
    map.set(&"name".into(), &"Alice".into());
    map.set(&"age".into(), &30.into());

    // 遍历
    map.for_each(&mut |value, key| {
        web_sys::console::log_2(&key, &value);
    });

    map
}

pub fn demo_set() -> Set {
    let set = Set::new(&js_sys::Array::of3(
        &1.into(), &2.into(), &3.into()
    ));

    set.add(&4.into());
    set.has(&2.into()) // true
}
```

### 2.4 Reflect API

```rust
use js_sys::Reflect;

pub fn dynamic_access(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &key.into()).unwrap_or(JsValue::undefined())
}

pub fn dynamic_set(obj: &JsValue, key: &str, val: &JsValue) -> bool {
    Reflect::set(obj, &key.into(), val).unwrap_or(false)
}

pub fn has_property(obj: &JsValue, key: &str) -> bool {
    Reflect::has(obj, &key.into()).unwrap_or(false)
}
```

### 2.5 JSON 操作

```rust
use js_sys::JSON;

pub fn stringify(value: &JsValue) -> Result<String, JsValue> {
    let result = JSON::stringify(value)?;
    Ok(result.as_string().unwrap())
}

pub fn parse(text: &str) -> Result<JsValue, JsValue> {
    JSON::parse(text)
}
```

## 三、注意事项与常见陷阱

1. **类型安全**：js-sys 使用 JsValue，需要手动检查类型
2. **Reflect 优先**：修改对象属性使用 Reflect::set 而非直接赋值
3. **遍历方式**：Map/Set 使用 for_each，Array 使用索引遍历
4. **性能对比**：频繁操作大量数据考虑使用 Rust 集合而非 js-sys
5. **序列化**：复杂对象考虑使用 serde-wasm-bindgen
