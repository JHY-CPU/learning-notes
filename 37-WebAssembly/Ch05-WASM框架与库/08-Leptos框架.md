# Leptos框架

## 一、概念说明

Leptos 是新一代 Rust WASM 框架，借鉴 SolidJS 的细粒度响应式设计，编译更快、运行时更轻。

```rust
use leptos::*;

#[component]
fn App() -> impl IntoView {
    let (count, set_count) = create_signal(0);

    view! {
        <button on:click=move |_| set_count.update(|n| *n += 1)>
            "点击: " {count}
        </button>
    }
}
```

## 二、具体用法

### 2.1 信号与派生

```rust
use leptos::*;

#[component]
fn Counter() -> impl IntoView {
    let (count, set_count) = create_signal(0);
    let doubled = create_memo(move |_| count.get() * 2);
    let is_even = create_memo(move |_| count.get() % 2 == 0);

    view! {
        <div>
            <p>"计数: " {count}</p>
            <p>"双倍: " {doubled}</p>
            <p>"奇偶: " {move || if is_even.get() { "偶数" } else { "奇数" }}</p>
            <button on:click=move |_| set_count.update(|n| *n += 1)>
                "+1"
            </button>
            <button on:click=move |_| set_count.update(|n| *n -= 1)>
                "-1"
            </button>
        </div>
    }
}
```

### 2.2 组件与 Props

```rust
use leptos::*;

#[component]
fn UserCard(
    name: String,
    #[prop(optional)] email: Option<String>,
    #[prop(default = 0)] score: u32,
) -> impl IntoView {
    view! {
        <div class="card">
            <h3>{name}</h3>
            {email.map(|e| view! { <p>"邮箱: " {e}</p> })}
            <p>"分数: " {score}</p>
        </div>
    }
}

#[component]
fn App() -> impl IntoView {
    view! {
        <UserCard name="Alice".into() email=Some("alice@example.com".into()) score=95 />
        <UserCard name="Bob".into() />
    }
}
```

### 2.3 异步资源

```rust
use leptos::*;

#[component]
fn DataLoader() -> impl IntoView {
    let (url, set_url) = create_signal("https://api.example.com/data".to_string());

    let data = create_resource(
        move || url.get(),
        |url| async move {
            reqwest::get(&url).await.unwrap().text().await.unwrap()
        },
    );

    view! {
        <Suspense fallback=move || view! { <p>"加载中..."</p> }>
            {move || {
                data.get().map(|d| view! { <pre>{d}</pre> })
            }}
        </Suspense>
    }
}
```

### 2.4 表单处理

```rust
use leptos::*;

#[component]
fn LoginForm() -> impl IntoView {
    let (username, set_username) = create_signal(String::new());
    let (password, set_password) = create_signal(String::new());
    let (result, set_result) = create_signal(String::new());

    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        set_result.set(format!(
            "登录: {} / {}",
            username.get(),
            "*".repeat(password.get().len())
        ));
    };

    view! {
        <form on:submit=on_submit>
            <input
                type="text"
                placeholder="用户名"
                on:input=move |e| set_username.set(event_target_value(&e))
            />
            <input
                type="password"
                placeholder="密码"
                on:input=move |e| set_password.set(event_target_value(&e))
            />
            <button type="submit">"登录"</button>
        </form>
        <p>{result}</p>
    }
}
```

### 2.5 路由

```rust
use leptos::*;
use leptos_router::*;

#[component]
fn App() -> impl IntoView {
    view! {
        <Router>
            <nav>
                <A href="/">"首页"</A>
                <A href="/about">"关于"</A>
            </nav>
            <main>
                <Routes>
                    <Route path="/" view=|| view! { <h1>"首页"</h1> } />
                    <Route path="/about" view=|| view! { <h1>"关于"</h1> } />
                    <Route path="/user/:id" view=UserPage />
                </Routes>
            </main>
        </Router>
    }
}
```

## 三、注意事项与常见陷阱

1. **响应式模型**：与 Yew 的 virtual DOM 不同，Leptos 使用信号驱动更新
2. **编译速度**：比 Yew 快，但 Rust WASM 编译仍较慢
3. **生态成熟度**：比 Yew 更新，生态库较少
4. **SSR 支持**：Leptos 原生支持 SSR 和 hydration
5. **学习资源**：文档完善，但社区资源相对较少
