# DOM操作

## 一、概念说明

WASM 不能直接访问 DOM，必须通过 JavaScript 互操作层。web-sys crate 提供了 Rust 对 Web API 的安全封装。

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlElement};

#[wasm_bindgen]
pub fn greet(name: &str) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let div = document.create_element("div").unwrap();
    div.set_text_content(Some(&format!("Hello, {}!", name)));
    body.append_child(&div).unwrap();
}
```

## 二、具体用法

### 2.1 元素查找与操作

```rust
use web_sys::Document;

fn get_document() -> Document {
    web_sys::window().unwrap().document().unwrap()
}

pub fn manipulate_dom() {
    let doc = get_document();

    // 按 ID 查找
    let element = doc.get_element_by_id("app").unwrap();

    // 查询选择器
    let btn: web_sys::HtmlButtonElement = doc
        .query_selector(".submit-btn")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    // 修改属性
    btn.set_inner_text("提交中...");
    btn.set_disabled(true);

    // 修改样式
    let style = element.style();
    style.set_property("color", "red").unwrap();
}
```

### 2.2 创建和插入元素

```rust
pub fn build_list(items: &[String]) {
    let doc = web_sys::window().unwrap().document().unwrap();
    let ul = doc.create_element("ul").unwrap();

    for item in items {
        let li = doc.create_element("li").unwrap();
        li.set_text_content(Some(item));
        ul.append_child(&li).unwrap();
    }

    let container = doc.get_element_by_id("list-container").unwrap();
    container.append_child(&ul).unwrap();
}
```

### 2.3 属性与数据集

```rust
pub fn set_data_attributes(element: &web_sys::Element) {
    element.set_attribute("data-id", "123").unwrap();
    element.set_attribute("data-type", "user").unwrap();

    // 获取属性
    let id = element.get_attribute("data-id").unwrap();
    // id = "123"

    // 移除属性
    element.remove_attribute("data-type").unwrap();
}
```

### 2.4 classList 操作

```rust
pub fn toggle_class(element: &web_sys::Element, class_name: &str) {
    let class_list = element.class_list();

    if class_list.contains(class_name) {
        class_list.remove_1(class_name).unwrap();
    } else {
        class_list.add_1(class_name).unwrap();
    }

    // 多个 class
    class_list.add_2("active", "visible").unwrap();
}
```

## 三、注意事项与常见陷阱

1. **类型转换**：`dyn_into()` 失败会 panic，可用 `dyn_ref()` 安全转换
2. **DOM 性能**：频繁 DOM 操作性能差，应批量更新
3. **跨语言开销**：每次 DOM 调用都有 WASM-JS 边界开销
4. **空值处理**：DOM API 返回 Option，需处理 None 情况
5. **内存泄漏**：Closure 作为事件监听器需要正确释放
