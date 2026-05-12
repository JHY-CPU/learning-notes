# Motion Matching

## 核心概念

Motion Matching 是一种数据驱动的动画技术，从大量预录制的动作捕捉数据库中，根据当前角色状态和目标轨迹，实时搜索最匹配的动画帧并平滑过渡。相比传统状态机，它产生更自然、更流畅的运动，减少手工状态和转换的维护工作量。

## 数学基础

### 特征向量

Motion Matching 的核心是为动画的每一帧构建一个特征向量（Feature Vector），用于描述该帧的运动特征：

$$\mathbf{f}_i = [\mathbf{p}_{root}, \mathbf{v}_{root}, \mathbf{p}_{foot_L}, \mathbf{p}_{foot_R}, \mathbf{v}_{foot_L}, \mathbf{v}_{foot_R}, h_{pelvis}, \mathbf{traj}_1, \mathbf{traj}_2, \mathbf{traj}_3]$$

每个特征分量在搜索前需要归一化到统一尺度，否则不同量纲的特征会导致权重被淹没。

### 代价函数

搜索最优匹配帧时，计算查询特征与数据库中每帧特征之间的代价：

$$cost(\mathbf{q}, \mathbf{f}_i) = \sum_{k} w_k \cdot \|q_k - f_{i,k}\|^2 + transition\_penalty(i)$$

其中 $w_k$ 是第 $k$ 个特征分量的权重，$transition\_penalty(i)$ 是跳帧惩罚，防止搜索到时间上很远的帧导致动作突变。

### 轨迹预测

根据玩家输入（摇杆方向和力度）预测未来轨迹：

$$\mathbf{p}_{t+\Delta t} = \mathbf{p}_t + \mathbf{v}_{desired} \cdot \Delta t$$

$$\theta_{t+\Delta t} = \theta_t + \omega_{desired} \cdot \Delta t$$

## 特征向量设计

```cpp
struct FeatureWeights {
    float trajectoryPos = 1.0f;    // 轨迹位置权重
    float trajectoryDir = 0.5f;    // 轨迹方向权重
    float rootVel = 0.3f;          // 根关节速度权重
    float footPos = 0.8f;          // 脚部位置权重
    float footVel = 0.2f;          // 脚部速度权重
    float pelvisHeight = 0.1f;     // 骨盆高度权重
};

struct MotionFeature {
    // 轨迹特征 (未来预测点，相对于当前根关节)
    Vec3 trajectoryPositions[3];     // 未来 0.2s, 0.4s, 0.6s 的位置
    float trajectoryDirections[3];   // 对应的朝向角度（弧度）

    // 姿势特征（相对于根关节）
    Vec3 rootVelocity;               // 根关节速度
    Vec3 footPositions[2];           // 双脚位置（世界空间）
    Vec3 footVelocities[2];          // 双脚速度
    float pelvisHeight;              // 骨盆高度

    // 计算与另一个特征向量的距离代价
    float DistanceTo(const MotionFeature& other,
                     const FeatureWeights& weights) const {
        float cost = 0;

        // 轨迹位置代价
        for (int i = 0; i < 3; i++) {
            Vec3 diff = trajectoryPositions[i] - other.trajectoryPositions[i];
            cost += weights.trajectoryPos * glm::dot(diff, diff);
        }

        // 轨迹方向代价
        for (int i = 0; i < 3; i++) {
            float diff = trajectoryDirections[i] - other.trajectoryDirections[i];
            cost += weights.trajectoryDir * diff * diff;
        }

        // 根关节速度代价
        Vec3 velDiff = rootVelocity - other.rootVelocity;
        cost += weights.rootVel * glm::dot(velDiff, velDiff);

        // 脚部位置代价
        for (int i = 0; i < 2; i++) {
            Vec3 footDiff = footPositions[i] - other.footPositions[i];
            cost += weights.footPos * glm::dot(footDiff, footDiff);
        }

        // 脚部速度代价
        for (int i = 0; i < 2; i++) {
            Vec3 fvDiff = footVelocities[i] - other.footVelocities[i];
            cost += weights.footVel * glm::dot(fvDiff, fvDiff);
        }

        // 骨盆高度代价
        float heightDiff = pelvisHeight - other.pelvisHeight;
        cost += weights.pelvisHeight * heightDiff * heightDiff;

        return cost;
    }
};
```

## 数据库构建

```cpp
struct MotionFrame {
    int clipIndex;                   // 所属动画片段索引
    int frameIndex;                  // 帧索引
    float time;                      // 时间戳
    MotionFeature feature;           // 该帧的特征向量

    // 原始姿势数据（用于过渡时的混合）
    std::vector<Vec3> jointPositions;
    std::vector<Quat> jointRotations;
};

class MotionDatabase {
public:
    std::vector<MotionFrame> frames;

    // 从动捕片段构建数据库
    void BuildFromClips(const std::vector<AnimationClip*>& clips) {
        frames.clear();

        for (int ci = 0; ci < (int)clips.size(); ci++) {
            const auto* clip = clips[ci];
            int frameCount = (int)(clip->duration * clip->frameRate);

            for (int f = 0; f < frameCount; f++) {
                MotionFrame mf;
                mf.clipIndex = ci;
                mf.frameIndex = f;
                mf.time = f / clip->frameRate;

                ExtractFeature(clip, f, frameCount, mf.feature);

                auto pose = clip->Sample(mf.time);
                for (const auto& j : pose.joints) {
                    mf.jointPositions.push_back(j.position);
                    mf.jointRotations.push_back(j.rotation);
                }

                frames.push_back(mf);
            }
        }
    }

private:
    void ExtractFeature(const AnimationClip* clip, int frameIndex,
                        int totalFrames, MotionFeature& feature) {
        float time = frameIndex / clip->frameRate;
        auto pose = clip->Sample(time);
        if (pose.joints.empty()) return;

        // 根关节（索引 0）
        feature.rootVelocity = ComputeVelocity(clip, frameIndex, 0);

        // 脚部
        feature.footPositions[0] = pose.joints[16].position;  // 左脚
        feature.footPositions[1] = pose.joints[19].position;  // 右脚
        feature.footVelocities[0] = ComputeVelocity(clip, frameIndex, 16);
        feature.footVelocities[1] = ComputeVelocity(clip, frameIndex, 19);

        // 骨盆高度
        feature.pelvisHeight = pose.joints[1].position.y;

        // 轨迹预测
        for (int i = 0; i < 3; i++) {
            float futureTime = time + (i + 1) * 0.2f;
            if (futureTime < clip->duration) {
                auto futurePose = clip->Sample(futureTime);
                feature.trajectoryPositions[i] = futurePose.joints[0].position;
                Vec3 dir = futurePose.joints[0].position - pose.joints[0].position;
                feature.trajectoryDirections[i] = std::atan2(dir.x, dir.z);
            }
        }
    }

    Vec3 ComputeVelocity(const AnimationClip* clip, int frame, int joint) const {
        float dt = 1.0f / clip->frameRate;
        auto p0 = clip->Sample(std::max(0.0f, (frame - 1) * dt));
        auto p1 = clip->Sample(frame * dt);
        if (joint < (int)p0.joints.size() && joint < (int)p1.joints.size()) {
            return (p1.joints[joint].position - p0.joints[joint].position) / dt;
        }
        return Vec3(0);
    }
};
```

## 运行时搜索系统

```cpp
class MotionMatchingSystem {
public:
    MotionDatabase database;
    FeatureWeights weights;

    int currentFrameIndex = 0;
    MotionFeature desiredFeature;
    float searchInterval = 0.1f;       // 搜索间隔（秒）
    float searchTimer = 0;
    float transitionPenalty = 0.5f;    // 跳帧惩罚
    int transitionPenaltyFrames = 30;  // 惩罚范围（帧）
    float blendDuration = 0.15f;       // 过渡混合时长
    float blendProgress = 0;
    bool isBlending = false;
    int blendTargetFrame = 0;

    void Update(float dt, const Vec3& playerInputDir, float playerInputSpeed) {
        // 1. 从玩家输入预测目标轨迹
        PredictTrajectory(playerInputDir, playerInputSpeed, desiredFeature);

        // 2. 定期搜索最优匹配
        searchTimer += dt;
        if (searchTimer >= searchInterval) {
            searchTimer = 0;
            int bestFrame = FindBestMatch(desiredFeature);

            if (bestFrame != currentFrameIndex && !isBlending) {
                StartTransition(bestFrame);
            }
        }

        // 3. 更新当前播放
        if (isBlending) {
            blendProgress += dt / blendDuration;
            if (blendProgress >= 1.0f) {
                isBlending = false;
                currentFrameIndex = blendTargetFrame;
            }
        } else {
            currentFrameIndex = (currentFrameIndex + 1) % (int)database.frames.size();
        }
    }

    // 获取当前姿势（带混合过渡）
    SkeletonPose GetCurrentPose() const {
        if (isBlending) {
            float t = SmoothStep(blendProgress);
            const auto& src = database.frames[currentFrameIndex];
            const auto& dst = database.frames[blendTargetFrame];

            SkeletonPose blended;
            int count = std::min((int)src.jointPositions.size(),
                                 (int)dst.jointPositions.size());
            blended.joints.resize(count);
            for (int j = 0; j < count; j++) {
                blended.joints[j].position = glm::mix(
                    src.jointPositions[j], dst.jointPositions[j], t);
                blended.joints[j].rotation = glm::slerp(
                    src.jointRotations[j], dst.jointRotations[j], t);
            }
            return blended;
        }

        const auto& frame = database.frames[currentFrameIndex];
        SkeletonPose pose;
        pose.joints.resize(frame.jointPositions.size());
        for (size_t j = 0; j < frame.jointPositions.size(); j++) {
            pose.joints[j].position = frame.jointPositions[j];
            pose.joints[j].rotation = frame.jointRotations[j];
        }
        return pose;
    }

private:
    // 暴力搜索最优匹配
    int FindBestMatch(const MotionFeature& desired) const {
        float bestCost = FLT_MAX;
        int bestIdx = currentFrameIndex;

        for (int i = 0; i < (int)database.frames.size(); i++) {
            float cost = desired.DistanceTo(database.frames[i].feature, weights);

            // 跳帧惩罚
            if (std::abs(i - currentFrameIndex) < transitionPenaltyFrames) {
                cost += transitionPenalty;
            }

            if (cost < bestCost) {
                bestCost = cost;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    void PredictTrajectory(const Vec3& inputDir, float inputSpeed,
                           MotionFeature& feature) const {
        Vec3 currentPos = database.frames[currentFrameIndex].jointPositions[0];
        for (int i = 0; i < 3; i++) {
            float dt = (i + 1) * 0.2f;
            feature.trajectoryPositions[i] = currentPos + inputDir * inputSpeed * dt;
            feature.trajectoryDirections[i] = std::atan2(inputDir.x, inputDir.z);
        }
        feature.rootVelocity = inputDir * inputSpeed;
    }

    void StartTransition(int targetFrame) {
        isBlending = true;
        blendTargetFrame = targetFrame;
        blendProgress = 0;
    }

    float SmoothStep(float t) const {
        return t * t * (3.0f - 2.0f * t);
    }
};
```

## 搜索加速

```cpp
// 空间分区搜索：按方向将数据库分桶
class OptimizedMotionMatching {
public:
    static constexpr int NUM_BINS = 36;
    static constexpr float BIN_SIZE = 10.0f;  // 每桶 10 度

    std::vector<std::vector<int>> directionBins;

    void BuildIndex(const MotionDatabase& db) {
        directionBins.resize(NUM_BINS);
        for (int i = 0; i < (int)db.frames.size(); i++) {
            float dir = db.frames[i].feature.trajectoryDirections[0];
            int bin = ((int)glm::degrees(dir) % 360 + 360) % 360 / (int)BIN_SIZE;
            directionBins[bin].push_back(i);
        }
    }

    int FindBestMatchOptimized(const MotionFeature& desired,
                                const MotionDatabase& db,
                                const FeatureWeights& weights,
                                int currentFrame) {
        float desiredDir = glm::degrees(desired.trajectoryDirections[0]);
        int centerBin = ((int)desiredDir % 360 + 360) % 360 / (int)BIN_SIZE;

        float bestCost = FLT_MAX;
        int bestIdx = currentFrame;

        // 只搜索相邻的 bin（约 60 度范围）
        for (int offset = -3; offset <= 3; offset++) {
            int bin = (centerBin + offset + NUM_BINS) % NUM_BINS;
            for (int idx : directionBins[bin]) {
                float cost = desired.DistanceTo(db.frames[idx].feature, weights);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestIdx = idx;
                }
            }
        }
        return bestIdx;
    }
};
```

## 与传统状态机对比

| 特性 | 传统状态机 | Motion Matching |
|------|-----------|----------------|
| 数据需求 | 分类动画片段 | 大量连续动捕数据 |
| 转换设计 | 手工定义每个转换 | 自动搜索最优帧 |
| 自然度 | 转换处可能生硬 | 天然流畅 |
| 可控性 | 完全可控 | 需要设计搜索约束 |
| 调试难度 | 状态图直观 | 搜索是"黑盒"，需要可视化 |
| 扩展成本 | 新状态需要新转换 | 添加数据即可 |
| 内存占用 | 小（只存使用的片段） | 大（需要完整数据库） |
| CPU 开销 | 低（条件检查） | 较高（搜索算法） |

## 常见陷阱与解决方案

1. **数据库太小导致动作重复**：需要至少 5-10 分钟连续动捕数据。使用动画混合扩充数据库或程序化生成变体。

2. **特征权重设置不当**：权重错误会导致搜索到不匹配的帧。使用可视化工具对比查询和结果的特征值，逐步调整。

3. **搜索频率过高**：每帧搜索消耗大量 CPU。降低搜索频率到每 0.1-0.2 秒一次，中间使用自然推进。

4. **数据库覆盖不全**：某些输入组合没有对应的动捕数据。标记"不可搜索区域"，或用混合树补充缺失方向。

## 实际游戏案例

- **The Last of Us Part II**：广泛使用 Motion Matching 实现角色移动的自然过渡，包含步行、跑步、转身等多种运动
- **Unreal Engine 5**：Motion Warping + Motion Matching 支持，允许运行时搜索和过渡动画
- **For Honor**：角色移动的流畅性得益于 Motion Matching，不同武器架势有独立的数据库
- **Guilty Gear Strive**：格斗游戏中使用 Motion Matching 过渡，站立到蹲下的自然过渡
- **EA Sports FC (FIFA)**：球员的移动动画大量使用 Motion Matching 技术
- **荣耀战魂**：战斗姿态切换通过 Motion Matching 实现平滑的架势变化
