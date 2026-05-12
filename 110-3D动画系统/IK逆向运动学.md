# IK 逆向运动学

## 核心概念

逆向运动学（Inverse Kinematics, IK）是从末端执行器（如手、脚）的期望位置反推各关节旋转的计算方法。与正向运动学（FK，从根到末端逐级计算关节变换）相反，IK 使角色能自适应地接触目标点，如脚踏地面、手抓物体、眼睛注视目标等。

## 数学基础

### 正向运动学回顾

给定关节角度 $\theta_1, \theta_2, ..., \theta_n$，末端执行器位置 $\mathbf{p}$ 通过链式变换计算：

$$\mathbf{p} = f(\theta_1, \theta_2, ..., \theta_n) = T_1(\theta_1) \cdot T_2(\theta_2) \cdots T_n(\theta_n) \cdot \mathbf{p}_{local}$$

### 逆向运动学问题

给定期望位置 $\mathbf{p}_{target}$，求解关节角度 $\theta$：

$$f(\theta_1, \theta_2, ..., \theta_n) = \mathbf{p}_{target}$$

这是一个非线性方程组，通常有无穷多解（冗余自由度）或无解（目标超出可达范围）。

### 雅可比矩阵法

对于 $n$ 自由度的关节链，雅可比矩阵 $J$ 描述末端速度与关节角速度的关系：

$$\dot{\mathbf{p}} = J(\theta) \cdot \dot{\theta}$$

其中 $J$ 是 $6 \times n$ 矩阵（3 个平移 + 3 个旋转自由度）。IK 求解通过迭代更新关节角度：

$$\dot{\theta} = J^{-1} \cdot \dot{\mathbf{p}}_{error}$$

实际中使用伪逆（Damped Least Squares）：

$$\dot{\theta} = J^T (J J^T + \lambda^2 I)^{-1} \cdot \dot{\mathbf{p}}_{error}$$

其中 $\lambda$ 是阻尼因子，防止奇异点附近的数值不稳定。

### 余弦定理（Two-Bone IK）

对于二节骨骼（如手臂），使用余弦定理直接求解：

给定三角形三边长度 $a$（上臂）、$b$（前臂）、$c$（髋到目标距离）：

$$\cos(\angle_{elbow}) = \frac{a^2 + b^2 - c^2}{2ab}$$

$$\cos(\angle_{shoulder\_target}) = \frac{a^2 + c^2 - b^2}{2ac}$$

## Two-Bone IK 完整推导与实现

### 几何推导

```
      Shoulder (S)
       / \
      /   \ a (上臂)
     /     \
    / θ0    \        θ0 = ∠(S-T方向, S-E方向)
   /         \
  Target(T)   Elbow(E)
     \       /
   c  \    / b (前臂)
       \ /
        Ankle/Hand (A)
```

已知：S（肩部位置）、T（目标位置）、a（上臂长度）、b（前臂长度）

### 完整 C++ 实现

```cpp
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cmath>

using Vec3 = glm::vec3;
using Quat = glm::quat;
using Mat4 = glm::mat4;

// Two-Bone IK 结果
struct TwoBoneIKResult {
    Quat hipRotation;      // 髋关节旋转
    Quat kneeRotation;     // 膝关节旋转
    Quat ankleRotation;    // 踝关节旋转
};

class TwoBoneIKSolver {
public:
    static TwoBoneIKResult Solve(
        Vec3 hipPos,           // 髋关节位置
        Vec3 targetPos,        // 目标位置（脚要放的位置）
        Vec3 kneeHint,         // 膝关节弯曲方向提示
        Vec3 hipForward,       // 髋关节的前方向
        float upperLength,     // 大腿长度
        float lowerLength,     // 小腿长度
        float ankleOffset = 0.0f)  // 踝关节偏移
    {
        TwoBoneIKResult result;

        // 1. 计算目标方向和距离
        Vec3 targetDir = targetPos - hipPos;
        float targetDist = glm::length(targetDir);

        if (targetDist < 0.001f) {
            // 目标与髋关节重合，无法求解
            result.hipRotation = Quat(1, 0, 0, 0);
            result.kneeRotation = Quat(1, 0, 0, 0);
            result.ankleRotation = Quat(1, 0, 0, 0);
            return result;
        }

        // 2. 限制目标距离在可及范围内
        float maxReach = upperLength + lowerLength;
        float minReach = std::abs(upperLength - lowerLength);

        if (targetDist > maxReach * 0.999f) {
            targetDist = maxReach * 0.999f;
            targetDir = glm::normalize(targetDir) * targetDist;
            targetPos = hipPos + targetDir;
        }
        if (targetDist < minReach * 1.001f) {
            targetDist = minReach * 1.001f;
            targetDir = glm::normalize(targetDir) * targetDist;
            targetPos = hipPos + targetDir;
        }

        // 3. 余弦定理求关节角度
        float a = upperLength;
        float b = lowerLength;
        float c = targetDist;

        // 髋关节弯曲角（肩膀-目标角度）
        float cosHipAngle = (a * a + c * c - b * b) / (2.0f * a * c);
        cosHipAngle = glm::clamp(cosHipAngle, -1.0f, 1.0f);
        float hipAngle = std::acos(cosHipAngle);

        // 膝关节弯曲角
        float cosKneeAngle = (a * a + b * b - c * c) / (2.0f * a * b);
        cosKneeAngle = glm::clamp(cosKneeAngle, -1.0f, 1.0f);
        float kneeAngle = std::acos(cosKneeAngle);

        // 4. 计算弯曲轴
        // 髋关节的旋转轴 = (目标方向) × (膝关节提示方向)
        Vec3 normalizedTarget = glm::normalize(targetDir);
        Vec3 bendAxis = glm::cross(normalizedTarget, glm::normalize(kneeHint));

        if (glm::length(bendAxis) < 0.001f) {
            // 提示方向与目标方向平行，使用前方向作为备选
            bendAxis = glm::cross(normalizedTarget, hipForward);
        }
        bendAxis = glm::normalize(bendAxis);

        // 5. 计算髋关节旋转
        // 从髋关节到目标方向的旋转
        Vec3 boneDir = Vec3(0, -1, 0);  // 假设骨骼默认朝下
        Quat baseRotation = RotationBetweenVectors(boneDir, normalizedTarget);

        // 加上弯曲角度
        Quat bendRotation = glm::angleAxis(hipAngle, bendAxis);
        result.hipRotation = baseRotation * bendRotation;

        // 6. 计算膝关节旋转
        // 膝关节弯曲角度 = PI - kneeAngle（因为kneeAngle是内角）
        float kneeBendAngle = glm::pi<float>() - kneeAngle;
        result.kneeRotation = glm::angleAxis(-kneeBendAngle, bendAxis);

        // 7. 计算踝关节旋转（使脚平行于地面）
        result.ankleRotation = Quat(1, 0, 0, 0);  // 简化，实际需要根据地面法线调整

        return result;
    }

private:
    // 计算将 v1 旋转到 v2 方向的四元数
    static Quat RotationBetweenVectors(Vec3 v1, Vec3 v2) {
        v1 = glm::normalize(v1);
        v2 = glm::normalize(v2);

        float dot = glm::dot(v1, v2);

        if (dot > 0.9999f) {
            return Quat(1, 0, 0, 0);  // 几乎相同，无需旋转
        }
        if (dot < -0.9999f) {
            // 方向相反，找一个垂直轴旋转180度
            Vec3 axis = glm::cross(v1, Vec3(1, 0, 0));
            if (glm::length(axis) < 0.001f) {
                axis = glm::cross(v1, Vec3(0, 1, 0));
            }
            return glm::angleAxis(glm::pi<float>(), glm::normalize(axis));
        }

        Vec3 axis = glm::cross(v1, v2);
        float angle = std::acos(glm::clamp(dot, -1.0f, 1.0f));
        return glm::angleAxis(angle, glm::normalize(axis));
    }
};
```

## CCD 算法（Cyclic Coordinate Descent）

CCD 从末端关节开始，逐个关节调整旋转，使末端趋近目标：

```cpp
// ===== CCD IK 求解器 =====
class CCDSolver {
public:
    struct Joint {
        Vec3 position;        // 关节位置（世界空间）
        Quat rotation;        // 关节旋转
        Vec3 localOffset;     // 到下一个关节的偏移
        float minAngle = -glm::pi<float>();  // 最小旋转限制
        float maxAngle = glm::pi<float>();   // 最大旋转限制
    };

    std::vector<Joint> joints;
    int maxIterations = 20;
    float tolerance = 0.001f;

    // 执行 CCD 求解
    bool Solve(const Vec3& target) {
        for (int iter = 0; iter < maxIterations; iter++) {
            // 从末端到根部逐关节调整
            for (int i = (int)joints.size() - 2; i >= 0; i--) {
                // 重新计算末端位置
                UpdateForwardKinematics();
                Vec3 endEffector = joints.back().position;

                // 检查是否已达到目标
                if (glm::length(endEffector - target) < tolerance) {
                    return true;
                }

                // 计算从当前关节到末端和目标的方向
                Vec3 toEnd = endEffector - joints[i].position;
                Vec3 toTarget = target - joints[i].position;

                if (glm::length(toEnd) < 0.001f || glm::length(toTarget) < 0.001f) {
                    continue;
                }

                toEnd = glm::normalize(toEnd);
                toTarget = glm::normalize(toTarget);

                // 计算需要的旋转
                Vec3 axis = glm::cross(toEnd, toTarget);
                float angle = std::acos(glm::clamp(glm::dot(toEnd, toTarget), -1.0f, 1.0f));

                if (glm::length(axis) > 0.001f) {
                    axis = glm::normalize(axis);
                    Quat deltaRot = glm::angleAxis(angle, axis);

                    // 应用旋转并限制角度
                    joints[i].rotation = deltaRot * joints[i].rotation;
                    joints[i].rotation = glm::normalize(joints[i].rotation);

                    // 应用关节角度限制
                    ClampJointRotation(i);
                }
            }
        }

        // 最终更新
        UpdateForwardKinematics();
        return glm::length(joints.back().position - target) < tolerance * 10;
    }

private:
    void UpdateForwardKinematics() {
        if (joints.empty()) return;

        joints[0].position = joints[0].position;  // 根关节固定

        for (int i = 1; i < (int)joints.size(); i++) {
            // 累积旋转
            Quat accumulatedRot = joints[i - 1].rotation;
            if (i > 1) {
                // 需要累积父级旋转
                accumulatedRot = Quat(1, 0, 0, 0);
                for (int j = 0; j < i; j++) {
                    accumulatedRot = accumulatedRot * joints[j].rotation;
                }
            }

            Vec3 offset = accumulatedRot * joints[i].localOffset;
            joints[i].position = joints[i - 1].position + offset;
        }
    }

    void ClampJointRotation(int jointIndex) {
        // 简化的角度限制实现
        // 实际项目中可能需要限制特定轴的旋转
        Vec3 euler = glm::eulerAngles(joints[jointIndex].rotation);
        euler.x = glm::clamp(euler.x, joints[jointIndex].minAngle, joints[jointIndex].maxAngle);
        joints[jointIndex].rotation = glm::quat(euler);
    }
};
```

## FABRIK 算法（Forward And Backward Reaching IK）

FABRIK 是一种迭代位置-based 的 IK 算法，具有极好的收敛性和简单性：

```cpp
// ===== FABRIK IK 求解器 =====
class FABRIKSolver {
public:
    struct Bone {
        Vec3 position;     // 关节位置
        float length;      // 到下一个关节的骨骼长度
    };

    std::vector<Bone> bones;
    std::vector<float> minAngles;  // 每个关节的最小弯曲限制
    std::vector<float> maxAngles;  // 每个关节的最大弯曲限制
    int maxIterations = 10;
    float tolerance = 0.001f;

    // FABRIK 求解
    bool Solve(const Vec3& target, const std::vector<Vec3>& constraints = {}) {
        if (bones.size() < 2) return false;

        int n = (int)bones.size();
        Vec3 rootPos = bones[0].position;  // 保存根关节位置

        for (int iter = 0; iter < maxIterations; iter++) {
            // ===== 前向阶段：从末端到根 =====
            // 将末端移动到目标位置
            bones[n - 1].position = target;

            for (int i = n - 2; i >= 0; i--) {
                // 计算当前关节到下一个关节的方向
                Vec3 direction = bones[i].position - bones[i + 1].position;
                direction = glm::normalize(direction);

                // 保持骨骼长度不变
                bones[i].position = bones[i + 1].position + direction * bones[i].length;

                // 应用角度约束
                if (i > 0 && !constraints.empty()) {
                    ApplyAngleConstraint(i, constraints);
                }
            }

            // ===== 后向阶段：从根到末端 =====
            // 将根关节移回原始位置
            bones[0].position = rootPos;

            for (int i = 0; i < n - 1; i++) {
                Vec3 direction = bones[i + 1].position - bones[i].position;
                direction = glm::normalize(direction);

                bones[i + 1].position = bones[i].position + direction * bones[i].length;

                if (!constraints.empty()) {
                    ApplyAngleConstraint(i, constraints);
                }
            }

            // 检查收敛
            float dist = glm::length(bones[n - 1].position - target);
            if (dist < tolerance) {
                return true;
            }
        }

        return glm::length(bones[n - 1].position - target) < tolerance * 10;
    }

    // 从求解结果提取旋转四元数
    std::vector<Quat> ExtractRotations() const {
        std::vector<Quat> rotations;
        rotations.reserve(bones.size());

        Vec3 defaultDir(0, 1, 0);  // 默认骨骼方向

        for (size_t i = 0; i < bones.size() - 1; i++) {
            Vec3 currentDir = glm::normalize(bones[i + 1].position - bones[i].position);
            Quat rot = RotationBetweenVectors(defaultDir, currentDir);
            rotations.push_back(rot);
        }

        return rotations;
    }

private:
    void ApplyAngleConstraint(int jointIndex, const std::vector<Vec3>& constraints) {
        // 简化的角度约束：将关节限制在指定平面内
        if (jointIndex >= (int)constraints.size()) return;

        Vec3 constraintDir = constraints[jointIndex];
        Vec3 currentDir = glm::normalize(bones[jointIndex + 1].position - bones[jointIndex].position);

        // 将方向投影到约束平面
        float dot = glm::dot(currentDir, constraintDir);
        Vec3 projected = currentDir - constraintDir * dot;

        if (glm::length(projected) > 0.001f) {
            projected = glm::normalize(projected);
            bones[jointIndex + 1].position = bones[jointIndex].position + projected * bones[jointIndex].length;
        }
    }

    static Quat RotationBetweenVectors(Vec3 v1, Vec3 v2) {
        v1 = glm::normalize(v1);
        v2 = glm::normalize(v2);
        float dot = glm::clamp(glm::dot(v1, v2), -1.0f, 1.0f);
        if (dot > 0.9999f) return Quat(1, 0, 0, 0);
        if (dot < -0.9999f) {
            Vec3 axis = glm::cross(v1, Vec3(1, 0, 0));
            if (glm::length(axis) < 0.001f) axis = glm::cross(v1, Vec3(0, 1, 0));
            return glm::angleAxis(glm::pi<float>(), glm::normalize(axis));
        }
        Vec3 axis = glm::normalize(glm::cross(v1, v2));
        return glm::angleAxis(std::acos(dot), axis);
    }
};
```

## Foot IK 系统

```cpp
// ===== 完整的脚部 IK 系统 =====
class FootIKSystem {
public:
    struct FootIKData {
        Vec3 currentPos;          // 当前脚部位置
        Vec3 targetPos;           // 目标位置（地面）
        Vec3 groundNormal;        // 地面法线
        float groundOffset = 0;   // 地面偏移
        float ikWeight = 0.0f;    // IK 权重（平滑过渡）
        bool grounded = false;    // 是否触地
    };

    FootIKData leftFoot, rightFoot;
    float raycastHeight = 0.5f;     // 射线起点高度
    float raycastDistance = 1.0f;   // 射线长度
    float ikSmoothSpeed = 10.0f;    // IK 平滑速度
    float pelvisAdjustSpeed = 8.0f; // 盆骨调整速度
    float maxPelvisDrop = 0.3f;     // 盆骨最大下沉量

    void Update(Character* character, float dt) {
        // 1. 射线检测每只脚的地面
        UpdateFootRaycast(character, leftFoot, "LeftFoot", dt);
        UpdateFootRaycast(character, rightFoot, "RightFoot", dt);

        // 2. 计算盆骨高度调整（防止两脚都在高处时角色浮空）
        float pelvisAdjust = CalculatePelvisAdjustment();

        // 3. 应用 Two-Bone IK 到腿部
        ApplyFootIK(character, leftFoot, "LeftUpLeg", "LeftLeg", "LeftFoot", pelvisAdjust);
        ApplyFootIK(character, rightFoot, "RightUpLeg", "RightLeg", "RightFoot", pelvisAdjust);

        // 4. 调整角色根关节位置
        character->AdjustRootPosition(Vec3(0, pelvisAdjust, 0));
    }

private:
    void UpdateFootRaycast(Character* character, FootIKData& foot,
                           const std::string& footBone, float dt) {
        Vec3 footPos = character->GetBoneWorldPosition(footBone);
        Vec3 rayStart = footPos + Vec3(0, raycastHeight, 0);
        Vec3 rayDir = Vec3(0, -1, 0);

        RaycastHit hit;
        foot.grounded = false;

        if (Physics::Raycast(rayStart, rayDir, raycastDistance, hit)) {
            foot.targetPos = hit.point;
            foot.groundNormal = hit.normal;
            foot.grounded = true;

            // 平滑 IK 权重
            foot.ikWeight = glm::clamp(foot.ikWeight + dt * ikSmoothSpeed, 0.0f, 1.0f);
        } else {
            foot.ikWeight = glm::clamp(foot.ikWeight - dt * ikSmoothSpeed, 0.0f, 1.0f);
        }

        foot.currentPos = footPos;
    }

    float CalculatePelvisAdjustment() {
        // 如果两只脚都在地面上，取较低的那只脚来调整盆骨
        if (leftFoot.grounded && rightFoot.grounded) {
            float leftDrop = leftFoot.targetPos.y - leftFoot.currentPos.y;
            float rightDrop = rightFoot.targetPos.y - rightFoot.currentPos.y;
            float minDrop = std::min(leftDrop, rightDrop);
            return glm::clamp(minDrop, -maxPelvisDrop, 0.0f);
        } else if (leftFoot.grounded) {
            return glm::clamp(leftFoot.targetPos.y - leftFoot.currentPos.y, -maxPelvisDrop, 0.0f);
        } else if (rightFoot.grounded) {
            return glm::clamp(rightFoot.targetPos.y - rightFoot.currentPos.y, -maxPelvisDrop, 0.0f);
        }
        return 0.0f;
    }

    void ApplyFootIK(Character* character, FootIKData& foot,
                     const std::string& hipBone, const std::string& kneeBone,
                     const std::string& footBone, float pelvisAdjust) {
        if (foot.ikWeight < 0.01f) return;

        Vec3 hipPos = character->GetBoneWorldPosition(hipBone);
        Vec3 kneePos = character->GetBoneWorldPosition(kneeBone);
        Vec3 anklePos = character->GetBoneWorldPosition(footBone);

        // 计算腿部骨骼长度
        float upperLen = glm::length(kneePos - hipPos);
        float lowerLen = glm::length(anklePos - kneePos);

        // 目标位置（考虑盆骨调整）
        Vec3 targetPos = foot.targetPos + Vec3(0, pelvisAdjust, 0);

        // 计算膝关节提示方向（膝盖弯曲方向）
        Vec3 kneeHint = character->GetForward();  // 膝盖向前弯

        // 求解 Two-Bone IK
        auto result = TwoBoneIKSolver::Solve(
            hipPos, targetPos, kneeHint, character->GetForward(),
            upperLen, lowerLen);

        // 应用 IK 结果（带权重插值）
        character->SetBoneRotation(hipBone,
            Quat::Slerp(character->GetBoneRotation(hipBone), result.hipRotation, foot.ikWeight));
        character->SetBoneRotation(kneeBone,
            Quat::Slerp(character->GetBoneRotation(kneeBone), result.kneeRotation, foot.ikWeight));

        // 根据地面法线调整脚的旋转
        if (foot.grounded) {
            Quat footBaseRot = character->GetBoneRotation(footBone);
            Quat surfaceRot = MatchSurfaceNormal(footBaseRot, foot.groundNormal);
            character->SetBoneRotation(footBone,
                Quat::Slerp(footBaseRot, surfaceRot, foot.ikWeight));
        }
    }

    Quat MatchSurfaceNormal(Quat baseRotation, Vec3 surfaceNormal) {
        // 计算脚底朝向地面法线的旋转
        Vec3 footUp(0, 1, 0);
        Quat normalRotation = TwoBoneIKSolver::RotationBetweenVectors(footUp, surfaceNormal);
        return normalRotation * baseRotation;
    }
};
```

## LookAt IK

```cpp
// ===== LookAt IK：头部/眼睛追踪目标 =====
class LookAtIK {
public:
    struct SpineChain {
        std::string boneName;
        float weight;           // 该骨骼参与追踪的权重
        float maxYaw = 60.0f;   // 最大左右转角度
        float maxPitch = 30.0f; // 最大上下转角度
    };

    std::vector<SpineChain> chain;  // 从胸到头的骨骼链
    Vec3 targetPosition;
    float globalWeight = 1.0f;
    float smoothSpeed = 5.0f;

    Vec3 currentLookDir;
    Vec3 desiredLookDir;

    void Update(Character* character, const Vec3& target, float dt) {
        targetPosition = target;

        // 计算从头部到目标的方向
        Vec3 headPos = character->GetBoneWorldPosition("Head");
        desiredLookDir = glm::normalize(target - headPos);

        // 平滑过渡
        currentLookDir = glm::normalize(
            glm::mix(currentLookDir, desiredLookDir, glm::clamp(smoothSpeed * dt, 0.0f, 1.0f)));

        // 计算角度偏转
        Vec3 baseForward = character->GetForward();
        float totalYaw = AngleBetween(currentLookDir, baseForward, Vec3(0, 1, 0));
        float totalPitch = std::asin(currentLookDir.y) * 180.0f / glm::pi<float>();

        // 将角度分布到骨骼链上
        float remainingYaw = totalYaw * globalWeight;
        float remainingPitch = totalPitch * globalWeight;

        for (auto& link : chain) {
            float linkYaw = remainingYaw * link.weight;
            float linkPitch = remainingPitch * link.weight;

            // 限制角度
            linkYaw = glm::clamp(linkYaw, -link.maxYaw, link.maxYaw);
            linkPitch = glm::clamp(linkPitch, -link.maxPitch, link.maxPitch);

            // 计算旋转
            Quat yawRot = glm::angleAxis(glm::radians(linkYaw), Vec3(0, 1, 0));
            Quat pitchRot = glm::angleAxis(glm::radians(linkPitch), Vec3(1, 0, 0));
            Quat totalRot = yawRot * pitchRot;

            // 应用到骨骼
            Quat currentRot = character->GetBoneRotation(link.boneName);
            Quat newRot = totalRot * currentRot;
            character->SetBoneRotation(link.boneName, newRot);

            remainingYaw -= linkYaw;
            remainingPitch -= linkPitch;
        }
    }

private:
    float AngleBetween(Vec3 dir, Vec3 forward, Vec3 up) {
        Vec3 flatDir = glm::normalize(Vec3(dir.x, 0, dir.z));
        Vec3 flatFwd = glm::normalize(Vec3(forward.x, 0, forward.z));
        float dot = glm::clamp(glm::dot(flatDir, flatFwd), -1.0f, 1.0f);
        float angle = std::acos(dot);
        // 判断左右
        if (glm::dot(glm::cross(flatFwd, flatDir), up) < 0) angle = -angle;
        return angle * 180.0f / glm::pi<float>();
    }
};
```

## IK 算法对比

| 算法 | 收敛速度 | 结果质量 | 实现难度 | 适用场景 |
|------|---------|---------|---------|---------|
| Two-Bone IK | 解析解，一次收敛 | 精确（二节骨骼） | 简单 | 手臂、腿部 |
| CCD | 快（5-15次迭代） | 可能不自然 | 简单 | 长关节链 |
| FABRIK | 快（3-10次迭代） | 自然 | 简单 | 长关节链 |
| 雅可比法 | 慢（10-50次迭代） | 最优（有约束） | 复杂 | 多自由度 |
| Analytical (6DOF) | 解析解 | 精确 | 复杂 | 工业机器人 |

## Unity Final IK 集成

```csharp
using UnityEngine;

// 使用 Final IK 插件（或 Unity 内置 IK）
public class CharacterIK : MonoBehaviour {
    Animator animator;

    [Header("Foot IK")]
    public float footRaycastHeight = 0.5f;
    public float footRaycastDist = 1.0f;
    public LayerMask groundLayer;
    [Range(0, 1)] public float footIKWeight = 1.0f;

    [Header("Look At")]
    public Transform lookAtTarget;
    [Range(0, 1)] public float lookAtWeight = 1.0f;
    public float lookAtBodyWeight = 0.3f;
    public float lookAtHeadWeight = 1.0f;
    public float lookAtClamp = 0.5f;

    [Header("Hand IK")]
    public Transform leftHandTarget;
    public Transform rightHandTarget;
    [Range(0, 1)] public float handIKWeight = 0.0f;

    void Start() {
        animator = GetComponent<Animator>();
    }

    // Unity 内置 IK 回调（需要在 Animator 中开启 IK Pass）
    void OnAnimatorIK(int layerIndex) {
        // 脚部 IK
        ApplyFootIK(HumanBodyBones.LeftFoot, AvatarIKGoal.LeftFoot, layerIndex);
        ApplyFootIK(HumanBodyBones.RightFoot, AvatarIKGoal.RightFoot, layerIndex);

        // 手部 IK
        if (leftHandTarget) {
            animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, handIKWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, handIKWeight);
            animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
            animator.SetIKRotation(AvatarIKGoal.LeftHand, leftHandTarget.rotation);
        }

        // Look At
        if (lookAtTarget) {
            animator.SetLookAtWeight(lookAtWeight, lookAtBodyWeight, lookAtHeadWeight, 0, lookAtClamp);
            animator.SetLookAtPosition(lookAtTarget.position);
        }
    }

    void ApplyFootIK(HumanBodyBones bone, AvatarIKGoal ikGoal, int layerIndex) {
        Transform foot = animator.GetBoneTransform(bone);
        if (!foot) return;

        RaycastHit hit;
        Vector3 rayStart = foot.position + Vector3.up * footRaycastHeight;

        if (Physics.Raycast(rayStart, Vector3.down, out hit, footRaycastDist + footRaycastHeight, groundLayer)) {
            // IK 目标位置
            Vector3 targetPos = hit.point;
            // 稍微抬高脚底，避免穿模
            targetPos.y += 0.05f;

            // 根据地面法线调整脚的旋转
            Quaternion targetRot = Quaternion.FromToRotation(Vector3.up, hit.normal) * foot.rotation;

            animator.SetIKPositionWeight(ikGoal, footIKWeight);
            animator.SetIKRotationWeight(ikGoal, footIKWeight);
            animator.SetIKPosition(ikGoal, targetPos);
            animator.SetIKRotation(ikGoal, targetRot);
        } else {
            animator.SetIKPositionWeight(ikGoal, 0);
            animator.SetIKRotationWeight(ikGoal, 0);
        }
    }
}
```

## Unreal Engine IK 节点

```cpp
// UE5 C++ - IK 系统
#include "Animation/AnimNode_TwoBoneIK.h"
#include "Animation/AnimNode_Fabrik.h"

// 在 Animation Blueprint 中使用 IK 节点
UCLASS()
class UMyAnimInstance : public UAnimInstance {
    GENERATED_BODY()

public:
    // Foot IK 目标
    UPROPERTY(BlueprintReadWrite, Category = "IK")
    FVector LeftFootIKTarget;

    UPROPERTY(BlueprintReadWrite, Category = "IK")
    FVector RightFootIKTarget;

    UPROPERTY(BlueprintReadWrite, Category = "IK")
    FRotator LeftFootIKRotation;

    UPROPERTY(BlueprintReadWrite, Category = "IK")
    FRotator RightFootIKRotation;

    UPROPERTY(BlueprintReadWrite, Category = "IK")
    float LeftFootIKAlpha = 1.0f;

    UPROPERTY(BlueprintReadWrite, Category = "IK")
    float RightFootIKAlpha = 1.0f;

    virtual void NativeUpdateAnimation(float DeltaSeconds) override {
        Super::NativeUpdateAnimation(DeltaSeconds);

        ACharacter* Character = Cast<ACharacter>(TryGetPawnOwner());
        if (!Character) return;

        // 射线检测地面
        PerformFootIK(Character, FName("foot_l"), LeftFootIKTarget, LeftFootIKRotation, LeftFootIKAlpha);
        PerformFootIK(Character, FName("foot_r"), RightFootIKTarget, RightFootIKRotation, RightFootIKAlpha);
    }

private:
    void PerformFootIK(ACharacter* Character, FName BoneName,
                       FVector& OutTarget, FRotator& OutRotation, float& OutAlpha) {
        USkeletalMeshComponent* MeshComp = Character->GetMesh();
        FVector BoneLoc = MeshComp->GetBoneLocation(BoneName);

        FHitResult Hit;
        FVector Start = BoneLoc + FVector(0, 0, 50);
        FVector End = BoneLoc - FVector(0, 0, 50);

        if (Character->GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_WorldStatic)) {
            OutTarget = Hit.ImpactPoint + FVector(0, 0, 2);  // 微调高度
            OutRotation = FRotationMatrix::MakeFromZ(Hit.ImpactNormal).Rotator();
            OutAlpha = 1.0f;
        } else {
            OutAlpha = 0.0f;
        }
    }
};
```

## 常见陷阱与解决方案

### 陷阱 1：目标距离超出骨骼链长度

**问题**：目标太远，IK 无法到达，产生拉伸效果。

**解决方案**：限制目标距离在最大可达范围内，或使用伸缩因子（非自然但有时可接受）。

### 陷阱 2：膝关节反转

**问题**：Two-Bone IK 的膝关节弯曲方向错误（如膝盖向后弯）。

**解决方案**：使用 Hint Target（提示目标）指定膝盖弯曲方向。Hint 是一个虚拟目标点，引导弯曲轴的方向。

### 陷阱 3：IK 权重过渡不平滑

**问题**：IK 权重从 0 到 1 快速变化时，脚部滑步或跳跃。

**解决方案**：使用平滑插值控制权重变化速度，或在动画状态机中将 IK 作为层叠加。

### 陷阱 4：多层 IK 叠加顺序

**问题**：Foot IK + LookAt IK + Hand IK 同时作用，执行顺序不同导致结果不一致。

**解决方案**：固定执行顺序：先 Foot IK（影响盆骨），再 Hand IK（影响手臂），最后 LookAt IK（影响头部）。

## 实际游戏案例

- **Unreal Engine**：内置 Two-Bone IK 和 FABRIK 蓝图节点，IK Rig 系统（UE5）提供更灵活的 IK 设置
- **Unity**：Final IK 插件是业界标准的 IK 解决方案，Unity 2023+ 内置 Animation Rigging 包
- **荒野大镖客2**：手部抓取物体（门把手、马鞍）和脚部贴合地形都使用高质量 IK
- **对马岛之魂**：角色在不平地形上脚部自适应贴合，武器交互使用 Hand IK
- **最后生还者 Part II**：角色攀爬和身体接触系统大量使用多骨骼链 IK
- **蜘蛛侠**：蛛丝摆荡时的手臂 IK 和着陆时的脚部 IK
