# 混合树（Blend Tree）

## 核心概念

混合树根据一个或多个混合参数（Blend Parameter）在多个动画片段之间进行插值混合，实现运动的平滑过渡。与离散的状态机不同，混合树提供连续的运动空间，参数的每个值都对应一个唯一的混合姿势。最常见的应用是角色移动：根据速度和方向参数混合站立、走路、跑步、侧移等动画。

## 数学基础

### 1D 线性混合

最简单的混合形式，沿一个参数轴在两个或多个动画之间插值：

$$P_{blend}(\theta) = \sum_{i} w_i(\theta) \cdot P_i$$

其中权重 $w_i(\theta)$ 是参数 $\theta$ 的分段线性函数。对于两个动画 A 和 B，参数在 $[t_A, t_B]$ 区间内：

$$\alpha = \frac{\theta - t_A}{t_B - t_A}, \quad \alpha \in [0, 1]$$

$$w_A = 1 - \alpha, \quad w_B = \alpha$$

对于多动画情况，先定位参数所在的区间，再在该区间的两个动画间插值。

### 2D 自由形式笛卡尔混合

在二维参数空间 $(x, y)$ 中，每个动画对应一个位置点 $(p_i^x, p_i^y)$，权重由逆距离加权（Inverse Distance Weighting）计算：

$$w_i = \frac{1}{d_i^2 + \epsilon} / \sum_j \frac{1}{d_j^2 + \epsilon}$$

其中 $d_i = \|(\theta_x, \theta_y) - (p_i^x, p_i^y)\|$ 是参数点到第 $i$ 个动画位置的距离，$\epsilon$ 是防止除零的小常数。

### 2D 简单方向混合

用于角色移动的方向混合。每个动画对应一个方向向量 $\mathbf{v}_i$，参数为当前移动方向 $\mathbf{d}$ 和速度 $s$：

$$w_i = \max(0, \frac{\mathbf{d} \cdot \mathbf{v}_i}{|\mathbf{d}| \cdot |\mathbf{v}_i|})^k$$

其中 $k$ 是锐化因子（通常 $k=2$），使权重分布更集中于最接近的动画。

### 方向混合的角度方法

```
              前 (0°)
               |
    前左(-45°) | 前右(45°)
               |
   左(-90°) ---+--- 右(90°)
               |
    后左(-135°)| 后右(135°)
               |
              后 (180°)
```

对于 8 方向混合，每个方向覆盖 45° 的扇区。设参数角度为 $\theta$，第 $i$ 个动画方向为 $\theta_i$，扇区宽度为 $\Delta = 45°$：

$$w_i = \max(0, 1 - \frac{|\theta - \theta_i|}{\Delta/2})$$

### 加性混合

加性混合将动画的增量（相对于参考姿势的偏移）叠加到基础姿势上：

$$P_{result} = P_{base} + w \cdot (P_{additive} - P_{reference})$$

对于旋转，使用四元数表示：

$$rot_{result} = rot_{base} \cdot \text{Slerp}(I, rot_{additive} \cdot rot_{ref}^{-1}, w)$$

其中 $I$ 是单位四元数，$rot_{ref}$ 是参考姿势的旋转。

## 完整 C++ 混合树实现

### 1D 混合树

```cpp
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

using Vec3 = glm::vec3;
using Quat = glm::quat;

struct JointPose {
    Vec3 position;
    Quat rotation;
    Vec3 scale = Vec3(1.0f);
};

struct SkeletonPose {
    std::vector<JointPose> joints;

    static SkeletonPose Lerp(const SkeletonPose& a, const SkeletonPose& b, float t) {
        SkeletonPose result;
        result.joints.resize(a.joints.size());
        for (size_t i = 0; i < a.joints.size(); i++) {
            result.joints[i].position = glm::mix(a.joints[i].position, b.joints[i].position, t);
            result.joints[i].rotation = glm::slerp(a.joints[i].rotation, b.joints[i].rotation, t);
            result.joints[i].scale = glm::mix(a.joints[i].scale, b.joints[i].scale, t);
        }
        return result;
    }

    // 加性混合
    static SkeletonPose AdditiveBlend(const SkeletonPose& base,
                                       const SkeletonPose& additive,
                                       const SkeletonPose& reference,
                                       float weight) {
        SkeletonPose result = base;
        for (size_t i = 0; i < base.joints.size(); i++) {
            // 位置增量
            Vec3 posDelta = additive.joints[i].position - reference.joints[i].position;
            result.joints[i].position += posDelta * weight;

            // 旋转增量
            Quat rotDelta = additive.joints[i].rotation * glm::inverse(reference.joints[i].rotation);
            result.joints[i].rotation = base.joints[i].rotation *
                glm::slerp(Quat(1, 0, 0, 0), rotDelta, weight);

            // 缩放增量
            Vec3 scaleDelta = additive.joints[i].scale - reference.joints[i].scale;
            result.joints[i].scale += scaleDelta * weight;
        }
        return result;
    }
};

// 采样动画片段（简化版）
class AnimationClip {
public:
    std::string name;
    float duration;
    float frameRate = 30.0f;
    std::vector<SkeletonPose> frames;

    SkeletonPose Sample(float time) const {
        if (frames.empty()) return SkeletonPose();
        float t = fmod(time, duration);
        float frameFloat = t * frameRate;
        int f0 = (int)frameFloat % frames.size();
        int f1 = (f0 + 1) % frames.size();
        float frac = frameFloat - (int)frameFloat;
        return SkeletonPose::Lerp(frames[f0], frames[f1], frac);
    }
};

// ===== 1D 混合树 =====
class BlendTree1D {
public:
    struct ChildMotion {
        AnimationClip* clip;
        float threshold;    // 参数阈值位置
        float timeOffset = 0.0f;  // 时间偏移（用于同步不同长度的动画）
    };

    std::vector<ChildMotion> children;
    float parameter = 0.0f;  // 混合参数（如速度）

    SkeletonPose Sample(float time) const {
        if (children.empty()) return SkeletonPose();
        if (children.size() == 1) return children[0].clip->Sample(time);

        // 找到参数所在的区间
        int lowerIdx = 0, upperIdx = 0;

        if (parameter <= children.front().threshold) {
            // 参数低于所有阈值：使用第一个动画
            return children.front().clip->Sample(time);
        }
        if (parameter >= children.back().threshold) {
            // 参数高于所有阈值：使用最后一个动画
            return children.back().clip->Sample(time);
        }

        for (int i = 0; i < (int)children.size() - 1; i++) {
            if (parameter >= children[i].threshold &&
                parameter <= children[i + 1].threshold) {
                lowerIdx = i;
                upperIdx = i + 1;
                break;
            }
        }

        // 计算插值权重
        float range = children[upperIdx].threshold - children[lowerIdx].threshold;
        float t = (range > 0) ? (parameter - children[lowerIdx].threshold) / range : 0.0f;

        // 采样两个动画并混合
        SkeletonPose poseA = children[lowerIdx].clip->Sample(time + children[lowerIdx].timeOffset);
        SkeletonPose poseB = children[upperIdx].clip->Sample(time + children[upperIdx].timeOffset);

        return SkeletonPose::Lerp(poseA, poseB, t);
    }

    // 获取每个子动画的权重（用于调试可视化）
    std::vector<float> GetWeights() const {
        std::vector<float> weights(children.size(), 0.0f);
        if (children.size() == 1) { weights[0] = 1.0f; return weights; }

        for (int i = 0; i < (int)children.size() - 1; i++) {
            if (parameter >= children[i].threshold &&
                parameter <= children[i + 1].threshold) {
                float range = children[i + 1].threshold - children[i].threshold;
                float t = (range > 0) ? (parameter - children[i].threshold) / range : 0.0f;
                weights[i] = 1.0f - t;
                weights[i + 1] = t;
                return weights;
            }
        }
        return weights;
    }
};
```

### 2D 方向混合树

```cpp
// ===== 2D 简单方向混合树 =====
class BlendTree2DDirectional {
public:
    struct ChildMotion {
        AnimationClip* clip;
        Vec2 position;  // 在参数空间中的位置 (方向, 速度)
    };

    std::vector<ChildMotion> children;
    Vec2 parameter = Vec2(0.0f);  // (方向角度, 速度)

    SkeletonPose Sample(float time) const {
        if (children.empty()) return SkeletonPose();

        // 计算每个子动画的权重
        std::vector<float> weights = ComputeWeights();

        // 加权混合所有子动画
        SkeletonPose result;
        bool first = true;

        for (size_t i = 0; i < children.size(); i++) {
            if (weights[i] < 0.001f) continue;

            SkeletonPose pose = children[i].clip->Sample(time);
            if (first) {
                result = pose;
                // 初始化为零
                for (auto& j : result.joints) {
                    j.position = Vec3(0);
                    j.rotation = Quat(0, 0, 0, 0);
                    j.scale = Vec3(0);
                }
                first = false;
            }

            for (size_t j = 0; j < result.joints.size(); j++) {
                result.joints[j].position += pose.joints[j].position * weights[i];

                // 四元数加权累积（需处理符号）
                float sign = (glm::dot(result.joints[j].rotation, pose.joints[j].rotation) < 0) ? -1.0f : 1.0f;
                result.joints[j].rotation.x += pose.joints[j].rotation.x * weights[i] * sign;
                result.joints[j].rotation.y += pose.joints[j].rotation.y * weights[i] * sign;
                result.joints[j].rotation.z += pose.joints[j].rotation.z * weights[i] * sign;
                result.joints[j].rotation.w += pose.joints[j].rotation.w * weights[i] * sign;

                result.joints[j].scale += pose.joints[j].scale * weights[i];
            }
        }

        // 归一化四元数
        for (auto& j : result.joints) {
            j.rotation = glm::normalize(j.rotation);
        }

        return result;
    }

private:
    std::vector<float> ComputeWeights() const {
        std::vector<float> weights(children.size(), 0.0f);
        constexpr float epsilon = 0.001f;

        float totalWeight = 0;
        for (size_t i = 0; i < children.size(); i++) {
            Vec2 diff = parameter - children[i].position;
            float distSq = glm::dot(diff, diff);
            float w = 1.0f / (distSq + epsilon);
            weights[i] = w;
            totalWeight += w;
        }

        // 归一化
        if (totalWeight > 0) {
            for (auto& w : weights) w /= totalWeight;
        }

        return weights;
    }
};

// ===== 2D 自由形式笛卡尔混合 =====
class BlendTree2DCartesian {
public:
    struct ChildMotion {
        AnimationClip* clip;
        Vec2 position;  // (x, y) 参数空间位置
    };

    std::vector<ChildMotion> children;
    Vec2 parameter = Vec2(0.0f);

    // 使用三角剖分 + 重心坐标混合
    // 首先将所有子动画位置进行 Delaunay 三角剖分
    struct Triangle {
        int indices[3];
    };
    std::vector<Triangle> triangles;

    void Triangulate() {
        // 简化：使用贪婪三角剖分
        // 实际项目中应使用 Delaunay 三角剖分库
        for (size_t i = 0; i + 2 < children.size(); i++) {
            triangles.push_back({(int)i, (int)i + 1, (int)i + 2});
        }
    }

    SkeletonPose Sample(float time) const {
        // 找到参数所在的三角形
        for (const auto& tri : triangles) {
            Vec2 p0 = children[tri.indices[0]].position;
            Vec2 p1 = children[tri.indices[1]].position;
            Vec2 p2 = children[tri.indices[2]].position;

            // 计算重心坐标
            float w0, w1, w2;
            if (Barycentric(parameter, p0, p1, p2, w0, w1, w2)) {
                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    // 用重心坐标混合三个动画
                    SkeletonPose pose0 = children[tri.indices[0]].clip->Sample(time);
                    SkeletonPose pose1 = children[tri.indices[1]].clip->Sample(time);
                    SkeletonPose pose2 = children[tri.indices[2]].clip->Sample(time);

                    return BlendThreePoses(pose0, pose1, pose2, w0, w1, w2);
                }
            }
        }

        // 不在任何三角形内：使用最近的动画
        return FindNearest(time);
    }

private:
    static bool Barycentric(Vec2 p, Vec2 a, Vec2 b, Vec2 c,
                            float& w0, float& w1, float& w2) {
        Vec2 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = glm::dot(v0, v0);
        float d01 = glm::dot(v0, v1);
        float d11 = glm::dot(v1, v1);
        float d20 = glm::dot(v2, v0);
        float d21 = glm::dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;

        if (std::abs(denom) < 0.0001f) return false;

        w1 = (d11 * d20 - d01 * d21) / denom;
        w2 = (d00 * d21 - d01 * d20) / denom;
        w0 = 1.0f - w1 - w2;
        return true;
    }

    static SkeletonPose BlendThreePoses(const SkeletonPose& a, const SkeletonPose& b,
                                         const SkeletonPose& c, float wa, float wb, float wc) {
        SkeletonPose result;
        result.joints.resize(a.joints.size());
        for (size_t i = 0; i < a.joints.size(); i++) {
            result.joints[i].position = a.joints[i].position * wa +
                                         b.joints[i].position * wb +
                                         c.joints[i].position * wc;

            // 四元数加权球面混合
            Quat q = a.joints[i].rotation * wa;
            if (glm::dot(q, b.joints[i].rotation) < 0) {
                q += b.joints[i].rotation * (-wb);
            } else {
                q += b.joints[i].rotation * wb;
            }
            if (glm::dot(q, c.joints[i].rotation) < 0) {
                q += c.joints[i].rotation * (-wc);
            } else {
                q += c.joints[i].rotation * wc;
            }
            result.joints[i].rotation = glm::normalize(q);

            result.joints[i].scale = a.joints[i].scale * wa +
                                      b.joints[i].scale * wb +
                                      c.joints[i].scale * wc;
        }
        return result;
    }

    SkeletonPose FindNearest(float time) const {
        float minDist = FLT_MAX;
        int nearest = 0;
        for (int i = 0; i < (int)children.size(); i++) {
            float dist = glm::length(parameter - children[i].position);
            if (dist < minDist) { minDist = dist; nearest = i; }
        }
        return children[nearest].clip->Sample(time);
    }
};
```

### 方向混合树（8方向移动）

```cpp
// ===== 8方向移动混合树 =====
class DirectionalBlendTree {
public:
    AnimationClip* forward;       // 0°
    AnimationClip* forwardRight;  // 45°
    AnimationClip* right;         // 90°
    AnimationClip* backRight;     // 135°
    AnimationClip* back;          // 180°
    AnimationClip* backLeft;      // -135°
    AnimationClip* left;          // -90°
    AnimationClip* forwardLeft;   // -45°

    float moveDirection = 0.0f;   // 移动方向（度），0=前, 90=右
    float moveSpeed = 0.0f;       // 移动速度

    SkeletonPose Sample(float time, float idleBlendFactor = 0.1f) const {
        AnimationClip* clips[8] = {
            forward, forwardRight, right, backRight,
            back, backLeft, left, forwardLeft
        };
        float angles[8] = {0, 45, 90, 135, 180, -135, -90, -45};

        // 计算每个方向动画的权重
        float weights[8] = {0};
        float totalWeight = 0;

        for (int i = 0; i < 8; i++) {
            float angleDiff = AngleDifference(moveDirection, angles[i]);
            float w = std::max(0.0f, 1.0f - std::abs(angleDiff) / 45.0f);
            weights[i] = w * w;  // 平方使权重更集中
            totalWeight += weights[i];
        }

        // 归一化方向权重
        if (totalWeight > 0) {
            for (auto& w : weights) w /= totalWeight;
        }

        // 混合方向动画
        SkeletonPose result;
        bool first = true;
        for (int i = 0; i < 8; i++) {
            if (weights[i] < 0.001f) continue;
            SkeletonPose pose = clips[i]->Sample(time);

            if (first) {
                result = pose;
                for (auto& j : result.joints) {
                    j.position = Vec3(0); j.rotation = Quat(0,0,0,0); j.scale = Vec3(0);
                }
                first = false;
            }
            for (size_t j = 0; j < result.joints.size(); j++) {
                result.joints[j].position += pose.joints[j].position * weights[i];
                result.joints[j].rotation += pose.joints[j].rotation * weights[i];
                result.joints[j].scale += pose.joints[j].scale * weights[i];
            }
        }
        for (auto& j : result.joints) j.rotation = glm::normalize(j.rotation);

        // 与站立动画混合（低速时）
        // moveSpeed 从 0 到 walkThreshold 之间混合站立和移动
        return result;  // 实际需要与 idle 动画做速度混合
    }

private:
    static float AngleDifference(float a, float b) {
        float diff = fmod(a - b + 180.0f, 360.0f) - 180.0f;
        return diff < -180.0f ? diff + 360.0f : diff;
    }
};
```

## 加性混合

```cpp
// ===== 加性动画混合 =====
class AdditiveBlending {
public:
    // 将加性动画叠加到基础姿势
    // referencePose 是加性动画的参考姿态（通常是 T-Pose）
    static SkeletonPose Apply(
        const SkeletonPose& basePose,
        const SkeletonPose& additivePose,
        const SkeletonPose& referencePose,
        float weight)
    {
        SkeletonPose result = basePose;

        for (size_t i = 0; i < basePose.joints.size(); i++) {
            // 位置：基础 + (加性 - 参考) * 权重
            Vec3 posDelta = additivePose.joints[i].position - referencePose.joints[i].position;
            result.joints[i].position += posDelta * weight;

            // 旋转：基础 * Slerp(单位四元数, 增量旋转, 权重)
            Quat additiveRot = additivePose.joints[i].rotation *
                glm::inverse(referencePose.joints[i].rotation);
            result.joints[i].rotation = basePose.joints[i].rotation *
                glm::slerp(Quat(1, 0, 0, 0), additiveRot, weight);

            // 缩放
            Vec3 scaleDelta = additivePose.joints[i].scale - referencePose.joints[i].scale;
            result.joints[i].scale += scaleDelta * weight;
        }

        return result;
    }
};

// 典型用例：呼吸动画叠加
void SetupBreathingAnimation() {
    // 呼吸动画是一个加性动画，参考姿态是 T-Pose
    // 呼吸动画本身只包含胸腔的微小上下位移
    // 叠加到任何基础动画上（站立、跑步、攻击等）都能看到呼吸效果

    SkeletonPose idlePose = idleClip->Sample(time);
    SkeletonPose breathPose = breathClip->Sample(time);
    SkeletonPose tPose = tPoseReference;  // 呼吸动画的参考姿态

    // 叠加呼吸效果
    float breathWeight = 0.5f;
    SkeletonPose finalPose = AdditiveBlending::Apply(idlePose, breathPose, tPose, breathWeight);
}
```

## 动画同步

混合不同长度的动画时需要同步它们的时间轴：

```cpp
// ===== 动画同步 =====
class AnimationSynchronizer {
public:
    // 基于速度的同步
    // 将不同步长的走路和跑步动画按实际位移同步
    struct SyncedClip {
        AnimationClip* clip;
        float distancePerCycle;  // 每个循环周期的位移距离
        float cycleDuration;     // 循环时长
    };

    static float ComputeSyncedTime(
        float baseTime,
        float baseDistance,
        const SyncedClip& targetClip)
    {
        // baseTime 为基础动画的当前时间
        // baseDistance 为基础动画每周期的位移
        // 计算目标动画对应的同步时间

        float baseCycles = baseTime / baseDistance;  // 基础动画走了几个"步长"
        return baseCycles * targetClip.distancePerCycle;
    }

    // 相位同步：确保左脚和右脚的步态一致
    static float PhaseSync(float timeA, float clipADuration,
                           float timeB, float clipBDuration) {
        float phaseA = fmod(timeA, clipADuration) / clipADuration;
        return phaseA * clipBDuration;
    }
};
```

## Unity Blend Tree 实战

```csharp
using UnityEngine;

public class PlayerBlendTree : MonoBehaviour {
    Animator animator;

    static readonly int SpeedHash = Animator.StringToHash("Speed");
    static readonly int DirectionHash = Animator.StringToHash("Direction");

    void Start() {
        animator = GetComponent<Animator>();
    }

    void Update() {
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");

        Vector3 input = new Vector3(h, 0, v);
        float speed = input.magnitude;

        // 将输入方向转换为相对于角色朝向的角度
        float direction = 0;
        if (speed > 0.1f) {
            Vector3 localDir = transform.InverseTransformDirection(input);
            direction = Mathf.Atan2(localDir.x, localDir.z) * Mathf.Rad2Deg;
        }

        // 设置混合树参数
        // 1D 混合：只用 Speed
        animator.SetFloat(SpeedHash, speed, 0.1f, Time.deltaTime);

        // 2D 混合：Speed + Direction
        // animator.SetFloat(SpeedHash, speed, 0.1f, Time.deltaTime);
        // animator.SetFloat(DirectionHash, direction, 0.1f, Time.deltaTime);
    }
}

// 代码中创建 BlendTree
public class RuntimeBlendTreeCreator {
    public static BlendTree Create1DMoveBlend(AnimatorController controller,
        AnimationClip idle, AnimationClip walk, AnimationClip run) {
        // 注意：运行时无法直接创建 BlendTree，通常在编辑器中设置
        // 这里展示概念

        BlendTree tree = new BlendTree();
        tree.blendType = BlendTreeType.Simple1D;
        tree.blendParameter = "Speed";

        tree.AddChild(idle, 0.0f);    // 速度 0 -> 站立
        tree.AddChild(walk, 1.0f);    // 速度 1 -> 走路
        tree.AddChild(run, 4.0f);     // 速度 4 -> 跑步

        return tree;
    }
}
```

## Unreal Engine Blend Space

```cpp
// UE5 C++ - Blend Space 使用
#include "Animation/BlendSpace.h"
#include "Animation/BlendSpace1D.h"

// 在 Animation Blueprint 中使用 Blend Space
UCLASS()
class UMyAnimInstance : public UAnimInstance {
    GENERATED_BODY()

public:
    // 引用 Blend Space 资源
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BlendSpace")
    UBlendSpace1D* MoveBlendSpace1D;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BlendSpace")
    UBlendSpace* MoveBlendSpace2D;

    // 混合参数
    UPROPERTY(BlueprintReadWrite, Category = "BlendSpace")
    float Speed;

    UPROPERTY(BlueprintReadWrite, Category = "BlendSpace")
    float Direction;

    // 在 AnimGraph 中，Blend Space 节点会自动使用这些参数
    // 也可以通过代码直接采样
    void SampleBlendSpace(float DeltaTime) {
        if (MoveBlendSpace2D) {
            // 获取当前混合权重
            FCompactPose OutPose;
            FBlendedCurve OutCurve;
            float CurrentTime = GetWorld()->GetTimeSeconds();

            // BlendSpace 的采样通常由 AnimGraph 自动处理
            // 这里展示手动采样的概念
        }
    }
};
```

## 性能考量

| 操作 | 开销 | 优化建议 |
|------|------|---------|
| 1D 混合（2动画） | 约 50 次四元数操作/帧 | 可以每帧计算 |
| 1D 混合（5+动画） | 约 200 次四元数操作/帧 | 注意动画数量 |
| 2D 方向混合（8动画） | 约 350 次四元数操作/帧 | 可接受 |
| 2D 笛卡尔混合 | 约 500-1000 次四元数操作/帧 | 减少子动画数量 |
| 加性混合 | 额外 50 次/层 | 限制加性层数量 |

## 常见陷阱与解决方案

### 陷阱 1：动画长度差异导致不自然拉伸

**问题**：走路动画 0.8 秒，跑步动画 0.6 秒，混合时动画被拉伸或压缩。

**解决方案**：使用相位同步或基于速度的同步，确保混合的动画在相同步态相位。

### 陷阱 2：Slerp 未处理最短路径

**问题**：四元数 $q_1$ 和 $-q_1$ 代表相同旋转，但 Slerp 会绕远路。

**解决方案**：在 Slerp 前检查点积，如果为负则取反其中一个四元数。

### 陷阱 3：2D 混合参数空间覆盖不全

**问题**：参数空间中存在空白区域，没有动画覆盖，导致姿势异常。

**解决方案**：确保三角剖分覆盖整个参数空间，或者在空白区域使用最近邻。

### 陷阱 4：根运动混合后位置跳变

**问题**：混合树中各动画的根运动位移不同，混合后角色脚底滑步或跳变。

**解决方案**：将根运动分离处理，位置混合使用单独的逻辑（详见动画同步）。

## 与替代方案对比

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| 混合树 | 连续平滑过渡，参数驱动 | 需要多个方向的动画数据 | 角色移动系统 |
| 离散状态切换 | 实现简单，每个动作独立 | 切换时有生硬过渡 | 固定动作（攻击、翻滚） |
| Motion Matching | 更自然，自动搜索 | 需要大量动捕数据 | 高品质角色移动 |
| 程序化动画 | 无需预录动画 | 实现复杂，难以控制艺术效果 | 非人类角色，物理效果 |

## 实际游戏案例

- **Unity Blend Tree**：内置 1D 和 2D 混合树，支持 Simple/Freeform/Direct 模式，参数可绑定到脚本
- **Unreal Blend Space**：Blend Space 1D/2D 可视化编辑，支持方向混合和自由形式混合
- **荒野大镖客2**：马匹的移动动画使用复杂 2D 混合树，包含步行/小跑/疾驰 × 前/左/右/后
- **GTA5**：角色移动的八方向混合 + 速度渐变，结合碰撞混合（斜坡上的姿势调整）
- **黑暗之魂**：移动混合树控制步态和武器架势的组合
- **最后生还者 Part II**：高品质移动混合，包含大量运动方向和速度的动画数据
