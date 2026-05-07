# Dioxus框架

## 一、概念说明

Dioxus 是跨平台 Rust UI 框架，支持 WASM Web、桌面（Tauri）、移动端。语法类似 React，使用 RSX 宏。

```rust
use dioxus::prelude::*;

fn App(cx: Scope) -> Element {
    let count = use_state(cx, || 0);

    cx.render(rsx! {
        div {
            h1 { "计数器: {count}" }
            button { onclick: move |_| count.set(count + 1), "点击" }
        }
    })
}
```

## 二、具体用法

### 2.1 组件定义

```rust
use dioxus::prelude::*;

#[derive(Props, PartialEq)]
struct UserCardProps {
    name: String,
    #[props(optional)]
    email: Option<String>,
}

fn UserCard(cx: Scope<UserCardProps>) -> Element {
    cx.render(rsx! {
        div { class: "card",
            h3 { "{cx.props.name}" }
            cx.props.email.as_ref().map(|e| rsx! {
                p { "邮箱: {e}" }
            })
        }
    })
}
```

### 2.2 状态管理

```rust
fn TodoApp(cx: Scope) -> Element {
    let todos = use_ref(cx, Vec::<String>::new);
    let input = use_state(cx, String::new);

    cx.render(rsx! {
        div {
            input {
                value: "{input}",
                oninput: move |evt| input.set(evt.value.clone()),
            }
            button {
                onclick: move |_| {
                    todos.write().push(input.get().clone());
                    input.set(String::new());
                },
                "添加"
            }
            ul {
                todos.read().iter().enumerate().map(|(i, t)| rsx! {
                    li { key: "{i}",
                        "{t}"
                        button { onclick: move |_| { todos.write().remove(i); }, "删除" }
                    }
                })
            }
        }
    })
}
```

### 2.3 副作用

```rust
fn DataLoader(cx: Scope) -> Element {
    let data = use_state(cx, || None::<String>);

    use_effect(cx, (), |_| {
        let data = data.to_owned();
        async move {
            let resp = reqwest::get("https://api.example.com/data")
                .await.unwrap()
                .text().await.unwrap();
            data.set(Some(resp));
        }
    });

    match data.get() {
        Some(d) => cx.render(rsx! { pre { "{d}" } }),
        None => cx.render(rsx! { p { "加载中..." } }),
    }
}
```

### 2.4 路由

```rust
use dioxus::prelude::*;
use dioxus_router::prelude::*;

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    Home {},
    #[route("/about")]
    About {},
    #[route("/user/:id")]
    User { id: String },
}

fn App(cx: Scope) -> Element {
    cx.render(rsx! {
        Router::<Route> {}
    })
}

fn Home(cx: Scope) -> Element {
    cx.render(rsx! { h1 { "首页" } })
}
```

## 三、注意事项与常见陷阱

1. **Scope 生命周期**：所有组件函数需要 cx: Scope 参数
2. **RSX 语法**：类似 JSX 但有 Rust 特有的语法限制
3. **跨平台**：同一代码可编译为 WASM Web 或桌面应用
4. **热重载**：Dioxus CLI 支持热重载，开发体验较好
5. **成熟度**：框架仍在快速迭代，API 可能有 breaking changes
