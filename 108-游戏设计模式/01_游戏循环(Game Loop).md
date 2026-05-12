# 游戏循环（Game Loop）

## 核心概念

游戏循环是游戏引擎的心脏，是驱动整个游戏世界运转的无限循环。它不断重复"处理输入→更新逻辑→渲染画面"的过程，直到游戏退出。理解游戏循环是理解游戏引擎运作方式的第一步。

游戏循环与普通应用程序的事件驱动模型不同。普通GUI程序将控制权交给操作系统，由操作系统决定何时调用回调函数。而游戏循环主动夺取控制权，自行管理每一帧的执行节奏，这样才能保证持续的实时渲染和物理模拟。

### 可变步长（Variable Timestep）

每帧消耗真实时间，dt 不固定。简单直接，但物理模拟不稳定。

```cpp
// 最简单的游戏循环——可变步长
while (gameRunning)
{
    double dt = GetDeltaTime();  // 距上一帧的真实时间
    ProcessInput();
    UpdateGame(dt);              // 直接用真实dt更新
    Render();
}
```

**问题**：假设物理引擎说"每秒移动10个单位"，60FPS时dt=0.0167，每帧移动0.167单位；但当帧率骤降到20FPS时，dt=0.05，每帧移动0.5单位——物体一帧跨越的距离是之前的3倍，可能导致穿墙。

### 固定步长（Fixed Timestep）与累加器

逻辑以固定 dt 更新，与渲染频率解耦。物理确定性好，跨帧率行为一致。

```cpp
// 经典固定步长游戏循环（来自 Glenn Fiedler 的经典文章）
const double FIXED_DT = 1.0 / 60.0; // 60Hz逻辑更新
double accumulator = 0.0;
double currentTime = GetTime();

while (gameRunning)
{
    double newTime = GetTime();
    double frameTime = newTime - currentTime;
    currentTime = newTime;

    // 防止"螺旋死亡"（Spiral of Death）
    // 当某帧特别慢（如断点调试），如果不做限制，
    // 累加器会积累大量时间导致需要执行几十次逻辑更新
    if (frameTime > 0.25) frameTime = 0.25;

    accumulator += frameTime;

    // 以固定步长更新逻辑（可能一帧内执行多次）
    while (accumulator >= FIXED_DT)
    {
        ProcessInput();
        UpdateGame(FIXED_DT);
        accumulator -= FIXED_DT;
    }

    // 渲染插值：消除视觉抖动
    // alpha ∈ [0, 1]，表示"在上一逻辑帧和当前逻辑帧之间的位置"
    double alpha = accumulator / FIXED_DT;
    RenderInterpolated(alpha);
}
```

**累加器的关键理解**：
- `accumulator` 记录"还没来得及处理的剩余时间"
- 当帧率很高（如120FPS），可能2帧才攒够一个FIXED_DT，此时每2帧才更新一次逻辑
- 当帧率很低（如30FPS），一帧可能攒够2个FIXED_DT，此时一帧执行2次逻辑更新
- `alpha` 用于插值渲染，避免"逻辑已更新但还没渲染"导致的画面抖动

### 渲染插值详解

```cpp
// 渲染插值：在两个逻辑帧之间平滑过渡
class GameObject
{
    Vec3 previousPosition;  // 上一逻辑帧的位置
    Vec3 currentPosition;   // 当前逻辑帧的位置

    // 逻辑更新时保存上一帧状态
    void Update(float dt)
    {
        previousPosition = currentPosition;
        currentPosition += velocity * dt;
    }

    // 渲染时插值
    Vec3 GetInterpolatedPosition(float alpha)
    {
        // alpha=0 → previousPosition（上一逻辑帧）
        // alpha=1 → currentPosition（当前逻辑帧）
        return Lerp(previousPosition, currentPosition, alpha);
    }

    void Render(float alpha)
    {
        Vec3 renderPos = GetInterpolatedPosition(alpha);
        DrawSprite(sprite, renderPos);
    }
};
```

### Update/Render 分离架构

将逻辑更新与渲染分离是现代引擎的标准做法，允许各自以不同频率运行。

```cpp
class GameEngine
{
    Timer timer;
    InputSystem inputSystem;
    PhysicsWorld physicsWorld;
    Scene scene;
    Renderer renderer;
    AudioSystem audioSystem;

    void MainLoop()
    {
        while (running)
        {
            double dt = timer.GetDeltaTime();

            // ===== 逻辑层 =====
            // 输入处理（必须最先执行）
            inputSystem.PollEvents();
            inputSystem.ProcessBindings();

            // 场景逻辑更新
            scene.Update(dt);

            // 物理模拟（固定步长）
            physicsWorld.Step(dt);

            // AI决策
            scene.UpdateAI(dt);

            // 音频更新（3D音效位置等）
            audioSystem.Update();

            // 碰撞检测后处理
            scene.ResolveCollisions();

            // ===== 渲染层 =====
            renderer.BeginFrame();

            // 可见性剔除
            auto visibleObjects = scene.CullObjects(renderer.GetCamera());

            // 排序（从前到后、透明物体后排序等）
            SortRenderables(visibleObjects);

            // 提交渲染命令
            for (auto& obj : visibleObjects)
                obj.SubmitRenderCommands(renderer);

            renderer.ExecuteCommands();
            renderer.EndFrame();
            renderer.Present();

            // ===== 帧尾处理 =====
            scene.DestroyMarkedObjects();
            scene.ProcessPendingSpawns();
        }
    }
};
```

### 多线程游戏循环

现代引擎将不同系统分配到独立线程，最大化硬件利用率。

```
主线程（Game Thread）：
    输入 → 逻辑更新 → 构建渲染命令 → 等待上一帧渲染完成 → 下一帧

渲染线程（Render Thread）：
    接收渲染命令 → 状态排序 → 提交GPU指令 → 驱动GPU

工作线程池（Worker Threads）：
    物理模拟 | AI计算 | 资源加载 | 动画蒙皮 | 遮挡剔除

音频线程（Audio Thread）：
    独立的音频混音、3D音效计算
```

```cpp
// 多线程游戏循环（简化版）
class MultiThreadedEngine
{
    RenderThread renderThread;
    ThreadPool workers;

    void MainThreadLoop()
    {
        while (running)
        {
            // 输入处理（必须在主线程）
            input.Poll();

            // 并行提交工作任务
            auto physicsFuture  = workers.Enqueue([&]{ physics.Step(dt); });
            auto aiFuture       = workers.Enqueue([&]{ scene.UpdateAI(dt); });
            auto animFuture     = workers.Enqueue([&]{ scene.UpdateAnimations(dt); });

            // 等待所有工作完成
            physicsFuture.wait();
            aiFuture.wait();
            animFuture.wait();

            // 主线程整合结果
            scene.ResolveCollisions();

            // 构建渲染命令
            RenderCommandList cmdList = scene.BuildRenderCommands();

            // 提交给渲染线程（双缓冲）
            renderThread.WaitForPreviousFrame(); // 等待上一帧GPU完成
            renderThread.SubmitCommands(std::move(cmdList));
        }
    }
};
```

### 帧预算（Frame Budget）

在固定帧率目标下，每一帧有严格的时间预算。超出预算就会掉帧。

```
60FPS → 每帧16.67ms预算：
    输入处理：    ~0.5ms
    逻辑更新：    ~3ms
    物理模拟：    ~2ms
    动画更新：    ~2ms
    AI决策：      ~1.5ms
    渲染提交：    ~4ms
    GPU等待：     ~2ms
    缓冲余量：    ~1.67ms（留给GC、系统波动）

30FPS → 每帧33.33ms，可以做更多工作
```

```cpp
// 帧预算监控
class FrameBudgetTracker
{
    double frameStartTime;
    double budgetMs;

    void BeginFrame() { frameStartTime = GetTime(); }

    void Checkpoint(const char* systemName)
    {
        double elapsed = (GetTime() - frameStartTime) * 1000.0;
        if (elapsed > budgetMs * 0.8) // 超过80%预算
        {
            LogWarning("%s 耗时过长: %.2fms (预算%.2fms)",
                       systemName, elapsed, budgetMs);
        }
    }
};
```

### 半固定步长（Semi-Fixed Timestep）

某些系统（如相机平滑、UI动画）不需要固定步长，可以使用可变步长更新。

```cpp
// 混合更新循环
while (gameRunning)
{
    double frameTime = GetDeltaTime();
    if (frameTime > 0.25) frameTime = 0.25;
    accumulator += frameTime;

    // 固定步长：物理、逻辑
    while (accumulator >= FIXED_DT)
    {
        UpdatePhysics(FIXED_DT);
        UpdateGameLogic(FIXED_DT);
        accumulator -= FIXED_DT;
    }

    // 可变步长：相机、UI、特效（对输入响应更即时）
    double alpha = accumulator / FIXED_DT;
    UpdateCamera(frameTime);
    UpdateUI(frameTime);
    UpdateEffects(frameTime);

    Render(alpha);
}
```

## 方案对比

| 方案 | 复杂度 | 物理稳定性 | 帧率一致性 | 输入延迟 | 适用场景 |
|------|--------|-----------|-----------|---------|---------|
| 可变步长 | 低 | 差 | 不一致 | 最低 | 原型开发、回合制 |
| 固定步长+累加器 | 中 | 优秀 | 一致 | 中等 | 动作、格斗、竞速 |
| 多线程分离 | 高 | 优秀 | 一致 | 低 | 3A级引擎 |
| 半固定步长 | 中 | 好 | 一致 | 低 | FPS、RTS |

## 常见陷阱与解决方案

1. **螺旋死亡**：极端卡顿时累加器无限增长。解决方案：限制 `frameTime` 上限（如0.25秒）
2. **输入延迟**：固定步长导致输入要等下一个逻辑tick才处理。解决方案：在渲染前重新采样输入
3. **线程安全**：多线程中逻辑和渲染共享数据。解决方案：双缓冲、命令队列
4. **帧率不稳**：某些帧特别慢。解决方案：使用滑动窗口平均帧率，动态调整画质
5. **时间精度**：不同平台的计时器精度不同。解决方案：使用高精度计时器（Windows: `QueryPerformanceCounter`）

## Unity/Unreal 实现

### Unity

```csharp
// Unity 的时间管理
void Update()
{
    // Time.deltaTime —— 可变步长，每帧的真实间隔
    // 适用于：非物理逻辑、相机、UI动画
    transform.Rotate(Vector3.up, 90f * Time.deltaTime);
}

void FixedUpdate()
{
    // Time.fixedDeltaTime —— 固定步长，默认0.02s（50Hz）
    // 适用于：物理模拟、刚体操作
    rb.AddForce(Vector3.forward * 10f);
}

void LateUpdate()
{
    // 所有Update执行完后调用
    // 适用于：相机跟随（确保在角色移动后再更新相机）
    cameraTransform.position = target.position + offset;
}
```

### Unreal Engine

```cpp
// UE 的 Tick 系统提供了更精细的控制
void AMyActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // PrimaryActorTick.TickGroup 控制更新阶段：
    // TG_PrePhysics  —— 物理前（输入处理）
    // TG_DuringPhysics —— 物理模拟中
    // TG_PostPhysics —— 物理后（角色移动结果）
    // TG_PostUpdateWork —— 渲染前最后更新（相机）

    // PrimaryActorTick.TickInterval 可以降低更新频率
    // 设为0.1 = 每0.1秒更新一次，节省CPU
}
```

## 实际使用案例

- **Unity** 的 `FixedUpdate` 以固定步长运行物理和逻辑，`Update` 以可变步长运行
- **《星际争霸2》** 使用固定步长确保所有客户端的游戏状态完全一致，支持精确回放
- **Source引擎**（《半条命2》）使用固定步长+插值，是业界经典的实现范例
- **Unreal Engine** 的 Game Thread 和 Rendering Thread 分离，渲染延迟一帧以保证流畅
- **《守望先锋》** 使用60Hz固定tick率，配合确定性锁步同步实现网络对战
