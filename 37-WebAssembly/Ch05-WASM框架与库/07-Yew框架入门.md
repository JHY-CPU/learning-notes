# Yew框架入门

## 一、概念说明

Yew 是 Rust 版的 React，使用组件化架构构建 WASM 前端应用。支持 JSX 风格的 html! 宏。

```rust
use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    let counter = use_state(|| 0);
    let onclick = {
        let counter = counter.clone();
        Callback::from(move |_| counter.set(*counter + 1))
    };

    html! {
        <div>
            <h1>{ "计数器: " } { *counter }</h1>
            <button {onclick}>{ "点击" }</button>
        </div>
    }
}
```

## 二、具体用法

### 2.1 组件定义

```rust
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub name: String,
    pub count: Option<u32>,
}

#[function_component(Greeting)]
pub fn greeting(props: &Props) -> Html {
    let count = props.count.unwrap_or(1);
    html! {
        <div class="greeting">
            <h2>{ &props.name }</h2>
            <p>{ format!("你好 {} 次!", count) }</p>
        </div>
    }
}
```

### 2.2 状态管理

```rust
use yew::prelude::*;

#[function_component(TodoList)]
fn todo_list() -> Html {
    let todos = use_state(Vec::new);
    let input = use_state(String::new);

    let oninput = {
        let input = input.clone();
        Callback::from(move |e: InputEvent| {
            let input_el: web_sys::HtmlInputElement = e.target_dyn_into().unwrap();
            input.set(input_el.value());
        })
    };

    let onclick = {
        let todos = todos.clone();
        let input = input.clone();
        Callback::from(move |_| {
            let mut new_todos = (*todos).clone();
            new_todos.push((*input).clone());
            todos.set(new_todos);
            input.set(String::new());
        })
    };

    html! {
        <div>
            <input {oninput} value={(*input).clone()} />
            <button {onclick}>{ "添加" }</button>
            <ul>
                { for todos.iter().map(|t| html! { <li>{ t }</li> }) }
            </ul>
        </div>
    }
}
```

### 2.3 Hooks

```rust
use yew::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[function_component(DataFetcher)]
fn data_fetcher() -> Html {
    let data = use_state(|| None::<String>);
    let loading = use_state(|| false);

    {
        let data = data.clone();
        let loading = loading.clone();
        use_effect(move || {
            loading.set(true);
            spawn_local(async move {
                let resp = reqwest::get("https://api.example.com/data")
                    .await.unwrap()
                    .text().await.unwrap();
                data.set(Some(resp));
                loading.set(false);
            });
            || {}
        });
    }

    if *loading {
        html! { <p>{ "加载中..." }</p> }
    } else if let Some(ref d) = *data {
        html! { <pre>{ d }</pre> }
    } else {
        html! { <p>{ "无数据" }</p> }
    }
}
```

### 2.4 路由

```rust
use yew::prelude::*;
use yew_router::prelude::*;

#[derive(Clone, Routable, PartialEq)]
enum Route {
    #[at("/")]
    Home,
    #[at("/about")]
    About,
    #[at("/user/:id")]
    User { id: String },
}

fn switch(routes: Route) -> Html {
    match routes {
        Route::Home => html! { <h1>{ "首页" }</h1> },
        Route::About => html! { <h1>{ "关于" }</h1> },
        Route::User { id } => html! { <h1>{ format!("用户: {}", id) }</h1> },
    }
}
```

## 三、注意事项与常见陷阱

1. **学习曲线**：html! 宏语法与 JSX 有差异，需适应
2. **编译时间**：Yew 项目编译较慢，建议使用 cargo-watch
3. **包体积**：生成的 WASM 较大，需要优化配置
4. **生态差异**：部分 JS 库需要额外绑定
5. **Server-side**：Yew 主要面向客户端渲染，SSR 支持有限
