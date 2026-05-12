# AI调试与可视化

## 1. 核心理论

### 1.1 为什么AI调试困难

游戏AI的调试与传统软件调试有本质区别：

1. **非确定性**：AI行为依赖于随机数、感知数据、时间等变化因素
2. **涌现性**：单个AI看起来正常，群体AI可能产生异常行为
3. **时间依赖**：Bug只在特定时间序列下出现（"5秒后如果敌人在左侧且血量低于30%..."）
4. **主观判断**：Bug可能是"AI看起来太蠢"而非崩溃或断言失败
5. **Heisenbug**：添加调试代码（如打印日志）可能改变AI的行为（时序变化）

传统的断点调试对AI效率极低，因为：
- 断点暂停游戏时，时间仍然流逝，恢复后状态可能已变
- AI的Tick频率高（每帧或每0.1秒），单步调试不现实
- Bug可能只在连续运行几分钟后才出现

### 1.2 可视化调试的核心价值

可视化调试让开发者**看到**AI的内部状态，而非通过断点**猜测**：

- **实时性**：不暂停游戏运行即可观察AI状态
- **全局性**：同时观察多个AI的行为和交互
- **可解释性**：直观展示AI决策的原因（为什么选择了这个行为）
- **历史追踪**：记录AI的行为历史，回溯Bug原因

### 1.3 调试基础设施设计原则

1. **零开销原则**：Release构建中调试代码不产生成本
2. **分层原则**：调试信息分详细等级，按需显示
3. **非侵入原则**：调试代码不改变AI的行为
4. **可搜索原则**：大量调试信息支持过滤和搜索

## 2. 调试绘制系统

### 2.1 Debug Drawer接口

```cpp
// 调试绘制抽象接口
class DebugDrawer {
public:
    virtual ~DebugDrawer() = default;

    // 基本图元
    virtual void DrawLine(const Vec3& start, const Vec3& end,
                          const Color& color, float duration = 0) = 0;
    virtual void DrawPoint(const Vec3& pos, float size,
                           const Color& color, float duration = 0) = 0;
    virtual void DrawSphere(const Vec3& center, float radius,
                            const Color& color, float duration = 0) = 0;
    virtual void DrawWireSphere(const Vec3& center, float radius,
                                const Color& color, float duration = 0) = 0;
    virtual void DrawBox(const Vec3& center, const Vec3& extents,
                         const Color& color, float duration = 0) = 0;
    virtual void DrawWireBox(const Vec3& center, const Vec3& extents,
                             const Color& color, float duration = 0) = 0;
    virtual void DrawCylinder(const Vec3& base, float radius, float height,
                              const Color& color, float duration = 0) = 0;

    // 高级图元
    virtual void DrawWireCone(const Vec3& apex, const Vec3& direction,
                              float angle, float range,
                              const Color& color, float duration = 0) = 0;
    virtual void DrawArrow(const Vec3& start, const Vec3& end,
                           const Color& color, float headSize = 0.2f,
                           float duration = 0) = 0;
    virtual void DrawText3D(const Vec3& position, const std::string& text,
                            const Color& color, float duration = 0) = 0;
    virtual void DrawText2D(const Vec2& screenPos, const std::string& text,
                            const Color& color, float fontSize = 12.0f) = 0;

    // 面/填充
    virtual void DrawTriangle(const Vec3& a, const Vec3& b, const Vec3& c,
                              const Color& color, float duration = 0) = 0;
    virtual void DrawCircle(const Vec3& center, float radius,
                            const Vec3& normal, const Color& color,
                            float duration = 0) = 0;
    virtual void DrawCapsule(const Vec3& start, const Vec3& end, float radius,
                             const Color& color, float duration = 0) = 0;
};

// 空实现（Release模式使用）
class NullDebugDrawer : public DebugDrawer {
public:
    void DrawLine(const Vec3&, const Vec3&, const Color&, float) override {}
    void DrawPoint(const Vec3&, float, const Color&, float) override {}
    void DrawSphere(const Vec3&, float, const Color&, float) override {}
    void DrawWireSphere(const Vec3&, float, const Color&, float) override {}
    void DrawBox(const Vec3&, const Vec3&, const Color&, float) override {}
    void DrawWireBox(const Vec3&, const Vec3&, const Color&, float) override {}
    void DrawCylinder(const Vec3&, float, float, const Color&, float) override {}
    void DrawWireCone(const Vec3&, const Vec3&, float, float, const Color&, float) override {}
    void DrawArrow(const Vec3&, const Vec3&, const Color&, float, float) override {}
    void DrawText3D(const Vec3&, const std::string&, const Color&, float) override {}
    void DrawText2D(const Vec2&, const std::string&, const Color&, float) override {}
    void DrawTriangle(const Vec3&, const Vec3&, const Vec3&, const Color&, float) override {}
    void DrawCircle(const Vec3&, float, const Vec3&, const Color&, float) override {}
    void DrawCapsule(const Vec3&, const Vec3&, float, const Color&, float) override {}
};

// 全局调试绘制器（编译时可切换为Null实现）
#ifdef _DEBUG
extern DebugDrawer* g_DebugDrawer;
#define DEBUG_DRAW g_DebugDrawer
#else
#define DEBUG_DRAW (static_cast<DebugDrawer*>(nullptr))
#endif

// 安全调用宏
#ifdef _DEBUG
#define DD_LINE(s, e, c, d)     if(DEBUG_DRAW) DEBUG_DRAW->DrawLine(s, e, c, d)
#define DD_SPHERE(c, r, cl, d)  if(DEBUG_DRAW) DEBUG_DRAW->DrawSphere(c, r, cl, d)
#define DD_TEXT(p, t, c, d)     if(DEBUG_DRAW) DEBUG_DRAW->DrawText3D(p, t, c, d)
#define DD_ARROW(s, e, c, d)    if(DEBUG_DRAW) DEBUG_DRAW->DrawArrow(s, e, c, 0.2f, d)
#else
#define DD_LINE(s, e, c, d)
#define DD_SPHERE(c, r, cl, d)
#define DD_TEXT(p, t, c, d)
#define DD_ARROW(s, e, c, d)
#endif
```

### 2.2 路径可视化

```cpp
class PathVisualizer {
public:
    static void DrawAStarPath(const std::vector<Vec3>& path,
                               const Color& color = Color::Green,
                               float duration = 0.5f) {
        if (path.size() < 2) return;

        for (size_t i = 0; i < path.size() - 1; i++) {
            DD_LINE(path[i], path[i+1], color, duration);
            DD_SPHERE(path[i], 0.15f, Color::Yellow, duration);
        }
        // 终点特殊标记
        DD_SPHERE(path.back(), 0.2f, Color::Red, duration);
    }

    static void DrawPathSearchProcess(
        const std::unordered_set<Vec3>& openSet,
        const std::unordered_set<Vec3>& closedSet,
        float cellSize = 1.0f)
    {
        for (const auto& pos : openSet) {
            DD_SPHERE(pos, cellSize * 0.3f, Color(1, 1, 0, 0.3f), 0.1f);
        }
        for (const auto& pos : closedSet) {
            DD_SPHERE(pos, cellSize * 0.3f, Color(1, 0, 0, 0.2f), 0.1f);
        }
    }

    static void DrawNavMeshPath(const std::vector<Vec3>& waypoints,
                                 const std::vector<int>& polyPath,
                                 const NavMesh& navMesh) {
        // 绘制路径点
        for (size_t i = 0; i < waypoints.size() - 1; i++) {
            DD_LINE(waypoints[i], waypoints[i+1], Color::Green, 0.5f);
            DD_SPHERE(waypoints[i], 0.1f, Color::Yellow, 0.5f);
        }

        // 绘制经过的多边形
        for (int polyId : polyPath) {
            const auto& poly = navMesh.polys[polyId];
            // 半透明填充多边形
            for (size_t i = 1; i < poly.vertices.size() - 1; i++) {
                DD_TRIANGLE(poly.vertices[0], poly.vertices[i], poly.vertices[i+1],
                           Color(0, 1, 0, 0.15f), 0.5f);
            }
        }
    }
};
```

### 2.3 感知可视化

```cpp
class PerceptionVisualizer {
public:
    static void DrawVisionCone(const Vec3& eyePos, const Vec3& forward,
                                float fovAngle, float range,
                                const Color& color = Color(1, 1, 0, 0.2f)) {
        DD_SPHERE(eyePos, 0.1f, Color::White, 0.1f);
        DD_WIRECONE(eyePos, forward, fovAngle, range, color, 0.1f);

        // 视锥边缘线
        float halfAngleRad = fovAngle * 3.14159f / 180.0f;
        Vec3 right = RotateAroundAxis(forward, Vec3{0,1,0}, halfAngleRad);
        Vec3 left = RotateAroundAxis(forward, Vec3{0,1,0}, -halfAngleRad);
        DD_LINE(eyePos, eyePos + right * range, Color(1, 1, 0, 0.5f), 0.1f);
        DD_LINE(eyePos, eyePos + left * range, Color(1, 1, 0, 0.5f), 0.1f);
    }

    static void DrawHearingRange(const Vec3& position, float range,
                                  const std::vector<const SoundEvent*>& audibleSounds) {
        DD_WIRESPHERE(position, range, Color(0, 0.5f, 1, 0.1f), 0.1f);

        for (const auto* sound : audibleSounds) {
            // 从声音来源画一条线到听者
            DD_LINE(sound->origin, position, Color(0, 1, 1, 0.5f), 0.5f);
            // 声音来源标记
            DD_SPHERE(sound->origin, 0.2f, Color::Cyan, 0.5f);
            DD_TEXT(sound->origin + Vec3{0, 1, 0},
                   "Sound: " + SoundTypeToString(sound->type), Color::White, 0.5f);
        }
    }

    static void DrawMemoryMap(const PerceptionMemoryManager& memories,
                               const std::string& aiName) {
        auto allMem = memories.GetAllMemories();
        for (const auto* mem : allMem) {
            // 记忆位置：用球体表示，大小表示置信度
            float radius = 0.3f + mem->confidence * 0.5f;
            Color color = mem->confidence > 0.7f ? Color::Red :
                          mem->confidence > 0.3f ? Color::Yellow : Color::Gray;
            color.a = mem->confidence;

            DD_SPHERE(mem->lastKnownPosition, radius, color, 0.1f);
            DD_TEXT(mem->lastKnownPosition + Vec3{0, 1.5f, 0},
                   "Conf: " + std::to_string((int)(mem->confidence * 100)) + "%",
                   Color::White, 0.1f);
        }
    }
};
```

### 2.4 行为树可视化

```cpp
class BehaviorTreeVisualizer {
public:
    static void DrawBTStatus(const BTNode* rootNode, const Vec3& aiPosition,
                              float maxHeight = 3.0f) {
        // 在AI头顶显示行为树状态
        Vec3 basePos = aiPosition + Vec3{0, 2.0f, 0};

        // 绘制当前执行的节点高亮
        DrawNodeHighlight(rootNode, basePos, 0.5f);
    }

    static void DrawNodeHighlight(const BTNode* node, const Vec3& pos, float spacing) {
        if (!node) return;

        Color nodeColor;
        switch (node->GetLastStatus()) {
            case Status::Success: nodeColor = Color::Green; break;
            case Status::Failure: nodeColor = Color::Red; break;
            case Status::Running: nodeColor = Color::Yellow; break;
            default: nodeColor = Color::Gray;
        }

        DD_SPHERE(pos, 0.1f, nodeColor, 0.1f);
        DD_TEXT(pos + Vec3{0.2f, 0, 0}, node->GetName(), nodeColor, 0.1f);

        // 递归绘制子节点
        float childX = pos.x - spacing;
        for (const auto* child : node->GetChildren()) {
            Vec3 childPos = {childX, pos.y - 0.3f, pos.z};
            DD_LINE(pos, childPos, Color(1, 1, 1, 0.3f), 0.1f);
            DrawNodeHighlight(child, childPos, spacing * 0.5f);
            childX += spacing;
        }
    }
};
```

## 3. 决策日志系统

### 3.1 日志记录器

```cpp
#include <deque>
#include <string>
#include <chrono>
#include <fstream>

class AIDecisionLogger {
public:
    enum class LogLevel {
        Verbose,   // 详细：所有决策输入
        Normal,    // 正常：主要决策
        Important, // 重要：状态转换、关键事件
        Error      // 错误：异常行为
    };

    struct LogEntry {
        float timestamp;
        LogLevel level;
        std::string category;    // "FSM", "BT", "GOAP", "Perception", "Path"
        std::string agentId;     // AI实例标识
        std::string message;     // 日志消息
        std::string details;     // 详细信息

        std::string ToString() const {
            char buf[64];
            sprintf(buf, "[%.2f]", timestamp);
            return std::string(buf) + " [" + category + "] " + agentId + ": " + message;
        }
    };

private:
    std::deque<LogEntry> history;
    size_t maxEntries = 1000;
    LogLevel minLevel = LogLevel::Normal;
    std::string filterCategory;
    std::string filterAgent;
    bool bConsoleOutput = true;
    bool bFileOutput = false;
    std::ofstream logFile;

public:
    AIDecisionLogger() = default;

    void Init(const std::string& logFilePath = "", LogLevel minLvl = LogLevel::Normal) {
        minLevel = minLvl;
        if (!logFilePath.empty()) {
            logFile.open(logFilePath);
            bFileOutput = true;
        }
    }

    void Log(LogLevel level, const std::string& category,
             const std::string& agentId, const std::string& message,
             const std::string& details = "") {
        if (level < minLevel) return;

        LogEntry entry;
        entry.timestamp = GetGameTime();
        entry.level = level;
        entry.category = category;
        entry.agentId = agentId;
        entry.message = message;
        entry.details = details;

        history.push_back(entry);
        if (history.size() > maxEntries) history.pop_front();

        if (bConsoleOutput && MatchesFilter(entry)) {
            std::cout << entry.ToString() << "\n";
        }
        if (bFileOutput && logFile.is_open()) {
            logFile << entry.ToString() << "\n";
            if (!details.empty()) logFile << "  Details: " << details << "\n";
        }
    }

    // 便捷方法
    void LogStateTransition(const std::string& agentId,
                             const std::string& from, const std::string& to,
                             const std::string& reason) {
        Log(LogLevel::Important, "FSM", agentId,
            from + " -> " + to + " (原因: " + reason + ")");
    }

    void LogBTNode(const std::string& agentId, const std::string& nodeName,
                    Status result, const std::string& reason = "") {
        std::string statusStr = (result == Status::Success) ? "SUCCESS" :
                                (result == Status::Failure) ? "FAILURE" : "RUNNING";
        Log(LogLevel::Verbose, "BT", agentId,
            nodeName + " = " + statusStr + (reason.empty() ? "" : " (" + reason + ")"));
    }

    void LogPerception(const std::string& agentId, const std::string& targetType,
                        PerceptionType type, bool bDetected) {
        std::string typeStr = (type == PERCEPTION_VISUAL) ? "视觉" :
                              (type == PERCEPTION_AUDITORY) ? "听觉" : "伤害";
        Log(LogLevel::Verbose, "Perception", agentId,
            (bDetected ? "感知到 " : "失去 ") + targetType + " (" + typeStr + ")");
    }

    void LogPathfinding(const std::string& agentId, const Vec3& from, const Vec3& to,
                         int nodesExplored, float pathLength) {
        Log(LogLevel::Normal, "Path", agentId,
            "寻路完成: 节点数=" + std::to_string(nodesExplored) +
            " 路径长度=" + std::to_string(pathLength));
    }

    // 查询
    std::vector<LogEntry> GetRecentEntries(int count = 20) const {
        int start = std::max(0, (int)history.size() - count);
        return std::vector<LogEntry>(history.begin() + start, history.end());
    }

    std::vector<LogEntry> FilterByAgent(const std::string& agentId) const {
        std::vector<LogEntry> result;
        for (const auto& e : history) {
            if (e.agentId == agentId) result.push_back(e);
        }
        return result;
    }

    std::vector<LogEntry> FilterByCategory(const std::string& category) const {
        std::vector<LogEntry> result;
        for (const auto& e : history) {
            if (e.category == category) result.push_back(e);
        }
        return result;
    }

    // 清除
    void Clear() { history.clear(); }
    void SetMaxEntries(size_t max) { maxEntries = max; }
    void SetFilter(const std::string& category, const std::string& agent = "") {
        filterCategory = category;
        filterAgent = agent;
    }

private:
    bool MatchesFilter(const LogEntry& entry) const {
        if (!filterCategory.empty() && entry.category != filterCategory) return false;
        if (!filterAgent.empty() && entry.agentId != filterAgent) return false;
        return true;
    }

    float GetGameTime() const {
        // 返回游戏时间（秒）
        static auto startTime = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<float>(now - startTime).count();
    }
};
```

### 3.2 决策时间线

```cpp
class DecisionTimeline {
public:
    struct TimelineEvent {
        float time;
        std::string agentId;
        std::string label;
        Color color;
    };

private:
    std::vector<TimelineEvent> events;
    float windowStart = 0;
    float windowDuration = 10.0f; // 显示最近10秒

public:
    void AddEvent(const std::string& agentId, const std::string& label,
                  const Color& color) {
        events.push_back({GetGameTime(), agentId, label, color});
        // 清理旧事件
        float cutoff = GetGameTime() - windowDuration - 5.0f;
        events.erase(
            std::remove_if(events.begin(), events.end(),
                [cutoff](const TimelineEvent& e) { return e.time < cutoff; }),
            events.end());
    }

    // 渲染到屏幕
    void Draw(float screenX, float screenY, float width, float height) {
        float currentTime = GetGameTime();
        windowStart = currentTime - windowDuration;

        // 背景
        DD_TEXT2D({screenX, screenY - 15}, "AI Decision Timeline", Color::White, 14);

        // 时间轴
        for (int i = 0; i <= (int)windowDuration; i++) {
            float x = screenX + (float)i / windowDuration * width;
            std::string label = std::to_string((int)(windowStart + i)) + "s";
            DD_TEXT2D({x, screenY + height + 5}, label, Color::Gray, 10);
        }

        // 绘制事件
        for (const auto& event : events) {
            if (event.time < windowStart || event.time > currentTime) continue;
            float x = screenX + (event.time - windowStart) / windowDuration * width;
            float y = screenY + height * 0.5f; // 简化：所有事件在同一行
            DD_POINT({x, y}, 3.0f, event.color);
            DD_TEXT2D({x, y + 5}, event.label, event.color, 9);
        }
    }
};
```

## 4. AI性能分析器

### 4.1 帧时间分析

```cpp
class AIPerformanceProfiler {
public:
    struct ProfileEntry {
        std::string name;
        float totalTime = 0;
        float maxTime = 0;
        int callCount = 0;
        std::vector<float> recentTimes;

        void AddSample(float time) {
            totalTime += time;
            maxTime = std::max(maxTime, time);
            callCount++;
            recentTimes.push_back(time);
            if (recentTimes.size() > 60) recentTimes.erase(recentTimes.begin());
        }

        float GetAverage() const {
            return callCount > 0 ? totalTime / callCount : 0;
        }

        float GetRecentAverage() const {
            if (recentTimes.empty()) return 0;
            float sum = 0;
            for (float t : recentTimes) sum += t;
            return sum / recentTimes.size();
        }
    };

private:
    std::unordered_map<std::string, ProfileEntry> profiles;
    std::stack<std::pair<std::string, std::chrono::high_resolution_clock::time_point>> activeScopes;

public:
    // 计时作用域
    class ScopedTimer {
        AIPerformanceProfiler& profiler;
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
    public:
        ScopedTimer(AIPerformanceProfiler& p, const std::string& n)
            : profiler(p), name(n),
              start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(end - start).count();
            profiler.profiles[name].AddSample(ms);
        }
    };

    // 宏简化使用
    #define AI_PROFILE(profiler, name) \
        AIPerformanceProfiler::ScopedTimer _timer_##__LINE__(profiler, name)

    // 重置每帧数据
    void BeginFrame() {
        // 不清除历史，只清除每帧最大值
        for (auto& [name, entry] : profiles) {
            entry.maxTime = 0;
        }
    }

    // 输出报告
    void PrintReport() const {
        std::cout << "\n=== AI性能报告 ===\n";
        std::cout << std::left << std::setw(30) << "系统"
                  << std::setw(12) << "平均(ms)"
                  << std::setw(12) << "近期(ms)"
                  << std::setw(12) << "最大(ms)"
                  << std::setw(10) << "调用次数" << "\n";
        std::cout << std::string(76, '-') << "\n";

        std::vector<const ProfileEntry*> sorted;
        for (const auto& [name, entry] : profiles) sorted.push_back(&entry);
        std::sort(sorted.begin(), sorted.end(),
                  [](const ProfileEntry* a, const ProfileEntry* b) {
                      return a->totalTime > b->totalTime;
                  });

        for (const auto* entry : sorted) {
            std::cout << std::left << std::setw(30) << entry->name
                      << std::setw(12) << std::fixed << std::setprecision(3) << entry->GetAverage()
                      << std::setw(12) << entry->GetRecentAverage()
                      << std::setw(12) << entry->maxTime
                      << std::setw(10) << entry->callCount << "\n";
        }
        std::cout << "==================\n";
    }

    // 获取热力图数据
    std::unordered_map<std::string, float> GetTimings() const {
        std::unordered_map<std::string, float> result;
        for (const auto& [name, entry] : profiles) {
            result[name] = entry.GetRecentAverage();
        }
        return result;
    }
};
```

### 4.2 使用示例

```cpp
AIPerformanceProfiler g_AIProfiler;

void UpdateAllAI(float dt) {
    g_AIProfiler.BeginFrame();

    {
        AI_PROFILE(g_AIProfiler, "AI_Total");

        {
            AI_PROFILE(g_AIProfiler, "Perception_Update");
            for (auto* ai : allAIs) {
                ai->UpdatePerception(dt);
            }
        }

        {
            AI_PROFILE(g_AIProfiler, "Decision_Making");
            for (auto* ai : allAIs) {
                ai->UpdateDecision(dt);
            }
        }

        {
            AI_PROFILE(g_AIProfiler, "Pathfinding");
            for (auto* ai : allAIs) {
                ai->UpdatePathfinding(dt);
            }
        }

        {
            AI_PROFILE(g_AIProfiler, "Movement");
            for (auto* ai : allAIs) {
                ai->UpdateMovement(dt);
            }
        }
    }

    // 每60帧输出一次报告
    static int frameCount = 0;
    if (++frameCount % 60 == 0) {
        g_AIProfiler.PrintReport();
    }
}
```

**输出示例**：
```
=== AI性能报告 ===
系统                            平均(ms)    近期(ms)    最大(ms)    调用次数
----------------------------------------------------------------------------
AI_Total                        2.345       2.210       4.567       3600
Perception_Update               0.890       0.830       2.100       3600
Decision_Making                 0.650       0.620       1.500       3600
Pathfinding                     0.520       0.490       3.200       3600
Movement                        0.285       0.270       0.800       3600
==================
```

## 5. AI Sandbox测试环境

### 5.1 沙箱环境设计

```cpp
class AISandbox {
    struct TestScenario {
        std::string name;
        std::string description;
        std::function<void()> setup;       // 场景初始化
        std::function<bool()> condition;   // 通过条件
        std::function<void()> teardown;    // 清理
        float timeout = 30.0f;             // 超时时间
    };

    std::vector<TestScenario> scenarios;
    int currentScenario = -1;
    float scenarioTimer = 0;
    bool bRunning = false;

    // 环境控制
    std::vector<Vec3> spawnPoints;
    std::vector<Obstacle> testObstacles;

public:
    void AddScenario(const std::string& name, std::function<void()> setup,
                     std::function<bool()> condition, float timeout = 30.0f) {
        scenarios.push_back({name, "", setup, condition, nullptr, timeout});
    }

    void RunAll() {
        std::cout << "\n=== AI沙箱测试 ===\n";
        int passed = 0, failed = 0;

        for (size_t i = 0; i < scenarios.size(); i++) {
            std::cout << "运行: " << scenarios[i].name << " ... ";
            bool result = RunScenario(i);
            if (result) { passed++; std::cout << "通过\n"; }
            else { failed++; std::cout << "失败\n"; }
        }

        std::cout << "结果: " << passed << " 通过, " << failed << " 失败\n";
        std::cout << "==================\n";
    }

    bool RunScenario(int index) {
        auto& scenario = scenarios[index];
        scenarioTimer = 0;

        if (scenario.setup) scenario.setup();
        bRunning = true;

        while (bRunning && scenarioTimer < scenario.timeout) {
            // 更新游戏（简化）
            float dt = 1.0f / 60.0f;
            scenarioTimer += dt;

            // 检查通过条件
            if (scenario.condition && scenario.condition()) {
                bRunning = false;
                if (scenario.teardown) scenario.teardown();
                return true; // 通过
            }
        }

        bRunning = false;
        if (scenario.teardown) scenario.teardown();
        return false; // 超时或失败
    }

    // 注入事件
    void InjectDamage(Entity* target, float amount) {
        target->TakeDamage(amount);
    }

    void InjectSound(const Vec3& origin, SoundType type, float intensity) {
        SoundEvent event{origin, type, intensity, 50.0f, GetGameTime(), nullptr};
        soundManager->RegisterSound(event);
    }

    void SpawnEnemy(const Vec3& position) {
        auto* enemy = entityFactory->CreateEnemy(position);
        entityManager->AddEntity(enemy);
    }

    void SetObstacle(const Vec3& position, const Vec3& size) {
        testObstacles.push_back({position, size});
    }

    // 录制与回放
    void StartRecording(const std::string& filename);
    void StopRecording();
    void PlayRecording(const std::string& filename);
};
```

### 5.2 回归测试

```cpp
class AIRegressionTester {
    struct TestCase {
        std::string name;
        WorldState initialState;
        std::vector<Event> injectedEvents;
        std::function<bool(const std::vector<AIAction>&)> validator;
    };

    std::vector<TestCase> testCases;

public:
    void AddTest(const std::string& name,
                 WorldState initial,
                 std::vector<Event> events,
                 std::function<bool(const std::vector<AIAction>&)> validator) {
        testCases.push_back({name, initial, events, validator});
    }

    void RunRegressionTests() {
        std::cout << "\n=== AI回归测试 ===\n";
        int passed = 0, failed = 0;

        for (const auto& test : testCases) {
            // 重置环境
            WorldStateManager::Reset(test.initialState);

            // 记录AI行为
            std::vector<AIAction> recordedActions;
            ActionRecorder recorder(recordedActions);

            // 注入事件
            for (const auto& event : test.injectedEvents) {
                EventManager::Inject(event);
            }

            // 运行指定时间
            SimulateTime(5.0f);

            // 验证
            bool result = test.validator(recordedActions);
            std::cout << test.name << ": " << (result ? "PASS" : "FAIL") << "\n";
            result ? passed++ : failed++;
        }

        std::cout << "总计: " << passed << "/" << (passed + failed) << " 通过\n";
    }
};
```

## 6. Unity可视化实现

```csharp
using UnityEngine;
using System.Collections.Generic;

public class AIDebugVisualizer : MonoBehaviour {
    [Header("可视化开关")]
    public bool showVisionCone = true;
    public bool showHearingRange = true;
    public bool showPathfinding = true;
    public bool showBehaviorTree = true;
    public bool showDecisionLog = true;

    [Header("颜色")]
    public Color visionColor = new Color(1, 1, 0, 0.2f);
    public Color hearingColor = new Color(0, 0.5f, 1, 0.1f);
    public Color pathColor = Color.green;
    public Color memoryColor = Color.red;

    private PerceptionSystem perception;
    private AIBehaviorTree bt;

    void Start() {
        perception = GetComponent<PerceptionSystem>();
        bt = GetComponent<AIBehaviorTree>();
    }

    void OnDrawGizmosSelected() {
        if (!Application.isPlaying) return;

        // 视锥
        if (showVisionCone && perception != null) {
            Gizmos.color = visionColor;
            Vector3 forward = transform.forward;
            float halfFov = perception.fovAngle * Mathf.Deg2Rad;

            // 绘制视锥扇形
            int segments = 20;
            Vector3 prev = transform.position + forward * perception.sightRange;
            for (int i = 1; i <= segments; i++) {
                float angle = -perception.fovAngle + (2 * perception.fovAngle * i / segments);
                Quaternion rot = Quaternion.Euler(0, angle, 0);
                Vector3 dir = rot * forward;
                Vector3 point = transform.position + dir * perception.sightRange;
                Gizmos.DrawLine(transform.position, point);
                if (i > 0) Gizmos.DrawLine(prev, point);
                prev = point;
            }
        }

        // 听觉范围
        if (showHearingRange && perception != null) {
            Gizmos.color = hearingColor;
            Gizmos.DrawWireSphere(transform.position, perception.hearingRange);
        }

        // 记忆可视化
        if (perception != null) {
            foreach (var mem in perception.GetAllMemories()) {
                Gizmos.color = new Color(1, 0, 0, mem.confidence);
                Gizmos.DrawSphere(mem.lastKnownPosition, 0.3f * mem.confidence);
            }
        }

        // 行为树当前节点
        if (showBehaviorTree && bt != null) {
            Vector3 pos = transform.position + Vector3.up * 2.5f;
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(pos, 0.1f);
            #if UNITY_EDITOR
            UnityEditor.Handles.Label(pos, bt.GetCurrentNodeName());
            #endif
        }
    }

    // 屏幕UI调试信息
    void OnGUI() {
        if (!showDecisionLog) return;

        GUILayout.BeginArea(new Rect(10, 10, 400, 300));
        GUILayout.Box("AI Debug: " + gameObject.name);

        if (perception != null) {
            GUILayout.Label("感知到的实体: " + perception.GetMemoryCount());
            foreach (var mem in perception.GetAllMemories()) {
                GUILayout.Label($"  {mem.target.name}: 置信度={mem.confidence:F2}");
            }
        }

        if (bt != null) {
            GUILayout.Label("行为树: " + bt.GetCurrentNodeName());
            GUILayout.Label("状态: " + bt.GetLastStatus());
        }

        GUILayout.EndArea();
    }
}
```

## 7. UE可视化实现

```cpp
// UE5 AI调试HUD
UCLASS()
class AAIDebugHUD : public AHUD {
    GENERATED_BODY()

public:
    virtual void DrawHUD() override {
        Super::DrawHUD();

        if (!GEngine->GetDebugCanvas()) return;

        // 绘制选中AI的调试信息
        AAIController* SelectedAI = GetSelectedAIController();
        if (!SelectedAI) return;

        // 行为树状态
        UBehaviorTreeComponent* BTComp = SelectedAI->FindComponentByClass<UBehaviorTreeComponent>();
        if (BTComp) {
            DrawBTDebugInfo(BTComp);
        }

        // 感知信息
        UAIPerceptionComponent* PercComp = SelectedAI->FindComponentByClass<UAIPerceptionComponent>();
        if (PercComp) {
            DrawPerceptionDebugInfo(PercComp);
        }
    }

    void DrawBTDebugInfo(UBehaviorTreeComponent* BTComp) {
        float X = 50, Y = 50;
        UBlackboardComponent* BB = BTComp->GetBlackboardComponent();

        DrawText(FString::Printf(TEXT("Behavior Tree: %s"),
            *BTComp->GetBehaviorTreeAsset()->GetName()),
            FLinearColor::White, X, Y);

        Y += 25;
        // 黑板键值
        if (BB) {
            TArray<FName> Keys;
            BB->GetKeys(Keys);
            for (const FName& Key : Keys) {
                FString Value = BB->GetValueAsString(Key);
                DrawText(FString::Printf(TEXT("  %s = %s"), *Key.ToString(), *Value),
                    FLinearColor::Gray, X, Y);
                Y += 20;
            }
        }
    }

    void DrawPerceptionDebugInfo(UAIPerceptionComponent* PercComp) {
        TArray<AActor*> PerceivedActors;
        PercComp->GetCurrentlyPerceivedActors(UAISense_Sight::StaticClass(), PerceivedActors);

        for (AActor* Actor : PerceivedActors) {
            FVector ScreenPos;
            if (Project(Actor->GetActorLocation(), ScreenPos)) {
                DrawText(TEXT("SEEN"), FLinearColor::Yellow, ScreenPos.X, ScreenPos.Y);
            }
        }
    }
};
```

## 8. 常见陷阱与最佳实践

### 8.1 调试陷阱

1. **调试绘制影响性能**：大量Debug.DrawLine在Release中未剥离
   - 解决：编译宏控制 + NullDebugDrawer

2. **信息过载**：同时显示所有AI的调试信息导致画面混乱
   - 解决：选中AI显示详情，其他AI只显示基本标记
   - 解决：使用LOD：近处AI显示详情，远处AI只显示颜色标记

3. **Heisenbug**：调试代码改变AI时序
   - 解决：避免在调试代码中使用互斥锁或阻塞操作
   - 解决：调试信息异步收集，不同步显示

4. **日志爆炸**：日志量过大难以定位关键决策
   - 解决：分级日志 + 过滤器
   - 解决：只记录状态转换，不记录每帧的中间状态

5. **可视化误导**：调试绘制的几何形状与实际逻辑不符
   - 解决：定期校验调试可视化与代码逻辑的一致性

### 8.2 最佳实践

1. **Debug菜单**：提供统一的调试开关面板，按系统分组
2. **快捷键**：常用调试可视化绑定快捷键（如F1切路径、F2切感知）
3. **截图标记**：Bug报告时自动附带当前调试可视化状态
4. **性能热力图**：用颜色表示各AI子系统的开销
5. **对比视图**：同时显示"期望行为"和"实际行为"便于发现问题
6. **录制回放**：录制AI行为序列，精确复现Bug场景

## 9. 性能分析

### 9.1 调试系统的开销

| 调试操作 | 开销（每AI） | 说明 |
|---------|-------------|------|
| 绘制视锥 | ~0.01ms | 10-20条线段 |
| 绘制路径 | ~0.005ms | 路径点数相关 |
| 日志记录 | ~0.001ms | 字符串构建 |
| 性能采样 | ~0.0001ms | 高精度时钟 |
| 行为树状态 | ~0.005ms | 树遍历 |

50个AI同时调试可视化：约1ms额外开销（15fps预算的6.7%）。

### 9.2 Release构建处理

```cpp
// Release中零开销的调试系统
#ifdef _DEBUG
    #define AI_LOG(logger, ...) logger.Log(__VA_ARGS__)
    #define AI_PROFILE(profiler, name) AIPerformanceProfiler::ScopedTimer _t(profiler, name)
    #define AI_DRAW(drawer, ...) drawer.Draw(__VA_ARGS__)
#else
    #define AI_LOG(logger, ...)
    #define AI_PROFILE(profiler, name)
    #define AI_DRAW(drawer, ...)
#endif
```

## 10. 实际游戏案例

### 案例1：Unreal Engine AI调试器

UE内置的AI调试工具（F11键）提供：
- 行为树实时执行可视化：绿色=成功，红色=失败，黄色=运行中
- 黑板键值实时显示
- 感知系统可视化：视锥、听觉范围、记忆点
- NavMesh显示：可行走区域、路径、Agent位置
- AI日志：可过滤的决策日志时间线

### 案例2：Unity AI Navigation可视化

Unity的AI Navigation窗口提供：
- NavMesh烘焙结果可视化（蓝色区域=可行走）
- Agent路径实时绘制
- Off-Mesh Link显示
- NavMesh Obstacle的切割效果预览

### 案例3：Halo系列内部工具

Halo开发团队使用的AI Sandbox：
- 隔离测试场景：固定环境变量测试AI行为
- 多AI对比：同时运行不同参数的AI实例
- 事件注入：通过UI制造伤害、放置障碍
- 行为录制回放：精确复现Bug场景

### 案例4：Valve Source引擎

Source引擎的AI调试命令：
- `ai_debug_sleepevents`：显示AI休眠/唤醒事件
- `ai_debug_shoot_positions`：显示AI射击位置计算
- `ai_debug_los`：显示视线检测射线
- `npc_select`：选中特定NPC显示详细调试信息
- `ai_show_connect`：显示NPC之间的关系图
