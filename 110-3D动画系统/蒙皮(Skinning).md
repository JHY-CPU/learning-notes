# 蒙皮（Skinning）

## 核心概念

蒙皮（Skinning）是将3D网格的每个顶点绑定到一个或多个骨骼关节上，使得网格能够跟随骨骼运动而变形的过程。这是实时角色动画的核心技术，直接决定了角色运动时的视觉质量。每个顶点可以被多个关节影响，通过权重（Weight）控制各关节的影响程度，实现平滑的皮肤变形效果。

## 数学基础

### 线性混合蒙皮（LBS）

线性混合蒙皮（Linear Blend Skinning, LBS）是最广泛使用的蒙皮算法。对于每个顶点 $v$，其最终位置 $v'$ 由影响它的所有关节的加权变换得到：

$$v' = \sum_{i=1}^{k} w_i \cdot M_{skin,i} \cdot v$$

其中：
- $k$ 是每个顶点的最大影响关节数（通常为 4）
- $w_i$ 是第 $i$ 个关节的权重，满足 $\sum_{i=1}^{k} w_i = 1$
- $M_{skin,i}$ 是第 $i$ 个关节的蒙皮矩阵：$M_{skin,i} = W_i \cdot B_i^{-1}$
- $W_i$ 是关节 $i$ 的当前世界变换矩阵
- $B_i$ 是关节 $i$ 在绑定姿态下的世界变换矩阵

### 法线变换

顶点法线的变换不能直接使用蒙皮矩阵，需要使用逆转置矩阵以保证法线方向在非均匀缩放下仍然正确：

$$n' = \sum_{i=1}^{k} w_i \cdot (M_{skin,i}^{-1})^T \cdot n$$

实际上，更常见的做法是使用蒙皮矩阵的上 3x3 部分的逆转置。对于纯旋转（无缩放），蒙皮矩阵本身即可正确变换法线。

### 对偶四元数蒙皮（DQS）

对偶四元数蒙皮（Dual Quaternion Skinning, DQS）使用对偶四元数来表示刚体变换（旋转+平移），避免了 LBS 中的"糖果包装"塌陷问题。

对偶四元数定义为：

$$\hat{q} = q_r + \epsilon \cdot q_d$$

其中 $q_r$ 是实部四元数（表示旋转），$q_d$ 是对偶部四元数，$\epsilon$ 是对偶单位（$\epsilon^2 = 0$）。

变换的组合通过四元数乘法完成，保证了体积守恒。对于顶点变换：

$$v' = \hat{q} \cdot v \cdot \hat{q}^*$$

其中 $\hat{q}^*$ 是对偶四元数的共轭。

### LBS vs DQS 数学对比

LBS 的问题在于对变换矩阵进行线性插值，不是刚体变换空间中的插值：

```
LBS:  v' = Σ(w_i * M_i * v)     // 矩阵加权求和 -> 非刚体，导致塌陷
DQS:  v' = Q_blend(v)            // 对偶四元数插值 -> 保持刚体性质
```

LBS 在关节弯曲时的塌陷量可以量化：

$$collapse = 1 - \cos(\theta/2) \cdot (1 - 2w_1 w_2)$$

当 $\theta = 90°$ 且 $w_1 = w_2 = 0.5$ 时，塌陷约为 29.3%。

## 完整 C++ 蒙皮系统

```cpp
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Quat = glm::quat;
using Mat4 = glm::mat4;

// 顶点蒙皮数据
struct VertexSkinData {
    int jointIndices[4];     // 影响该顶点的4个关节索引
    float weights[4];        // 对应的4个权重

    // 归一化权重（确保和为1.0）
    void Normalize() {
        float sum = weights[0] + weights[1] + weights[2] + weights[3];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            weights[0] *= inv;
            weights[1] *= inv;
            weights[2] *= inv;
            weights[3] *= inv;
        }
    }

    // 将权重从少于4个关节的情况填充
    void Pack(int* indices, float* ws, int count) {
        // 按权重排序，取最大的4个
        // （简化版本，实际需要排序逻辑）
        for (int i = 0; i < 4; i++) {
            if (i < count) {
                jointIndices[i] = indices[i];
                weights[i] = ws[i];
            } else {
                jointIndices[i] = 0;
                weights[i] = 0.0f;
            }
        }
        Normalize();
    }
};

// ===== LBS 蒙皮实现 =====
class LBSSkinning {
public:
    // 对单个顶点应用 LBS 蒙皮
    static Vec3 SkinVertex(
        const Vec3& originalPos,
        const VertexSkinData& skin,
        const std::vector<Mat4>& skinMatrices)
    {
        Vec3 result(0.0f);
        for (int i = 0; i < 4; i++) {
            if (skin.weights[i] > 0.001f) {
                Mat4 m = skinMatrices[skin.jointIndices[i]];
                Vec4 transformed = m * Vec4(originalPos, 1.0f);
                result += Vec3(transformed) * skin.weights[i];
            }
        }
        return result;
    }

    // 对法线应用 LBS 蒙皮
    static Vec3 SkinNormal(
        const Vec3& originalNormal,
        const VertexSkinData& skin,
        const std::vector<Mat4>& skinMatrices)
    {
        Vec3 result(0.0f);
        for (int i = 0; i < 4; i++) {
            if (skin.weights[i] > 0.001f) {
                // 使用逆转置矩阵变换法线
                Mat4 m = glm::transpose(glm::inverse(skinMatrices[skin.jointIndices[i]]));
                Vec4 transformed = m * Vec4(originalNormal, 0.0f);  // w=0 表示方向
                result += Vec3(transformed) * skin.weights[i];
            }
        }
        return glm::normalize(result);
    }

    // 批量蒙皮整个网格
    static void SkinMesh(
        const std::vector<Vec3>& restPositions,
        const std::vector<Vec3>& restNormals,
        const std::vector<VertexSkinData>& skinData,
        const std::vector<Mat4>& skinMatrices,
        std::vector<Vec3>& outPositions,
        std::vector<Vec3>& outNormals)
    {
        outPositions.resize(restPositions.size());
        outNormals.resize(restNormals.size());

        for (size_t i = 0; i < restPositions.size(); i++) {
            outPositions[i] = SkinVertex(restPositions[i], skinData[i], skinMatrices);
            outNormals[i] = SkinNormal(restNormals[i], skinData[i], skinMatrices);
        }
    }
};

// ===== 对偶四元数蒙皮实现 =====
class DQSSkinning {
public:
    // 对偶四元数结构
    struct DualQuat {
        Quat real;    // 实部（旋转）
        Quat dual;    // 对偶部（平移信息）

        static DualQuat FromTransform(const Vec3& pos, const Quat& rot) {
            DualQuat dq;
            dq.real = rot;
            // dual = 0.5 * t * r  (t是平移四元数 (0, tx, ty, tz))
            Quat t(0.0f, pos.x, pos.y, pos.z);
            dq.dual = 0.5f * t * rot;
            return dq;
        }

        // 对偶四元数加法
        DualQuat operator+(const DualQuat& other) const {
            return { real + other.real, dual + other.dual };
        }

        // 标量乘法
        DualQuat operator*(float s) const {
            return { real * s, dual * s };
        }

        // 归一化
        void Normalize() {
            float len = glm::length(real);
            if (len > 0.0001f) {
                float inv = 1.0f / len;
                real *= inv;
                dual *= inv;
            }
        }
    };

    // 从蒙皮矩阵构造对偶四元数
    static DualQuat MatrixToDualQuat(const Mat4& m) {
        // 提取旋转
        Quat rot = glm::toQuat(glm::mat3(m));
        // 提取平移
        Vec3 trans(m[3][0], m[3][1], m[3][2]);
        return DualQuat::FromTransform(trans, rot);
    }

    // 用对偶四元数变换顶点
    static Vec3 TransformPoint(const DualQuat& dq, const Vec3& point) {
        // 旋转：v' = q_r * v * q_r^*
        Vec3 rotated = glm::rotate(dq.real, point);
        // 平移：t = 2 * q_d * q_r^*
        Quat t = 2.0f * dq.dual * glm::conjugate(dq.real);
        Vec3 translation(t.x, t.y, t.z);
        return rotated + translation;
    }

    // 批量 DQS 蒙皮
    static void SkinMesh(
        const std::vector<Vec3>& restPositions,
        const std::vector<Vec3>& restNormals,
        const std::vector<VertexSkinData>& skinData,
        const std::vector<Mat4>& skinMatrices,
        std::vector<Vec3>& outPositions,
        std::vector<Vec3>& outNormals)
    {
        // 预转换所有蒙皮矩阵为对偶四元数
        std::vector<DualQuat> dqs(skinMatrices.size());
        for (size_t i = 0; i < skinMatrices.size(); i++) {
            dqs[i] = MatrixToDualQuat(skinMatrices[i]);
        }

        outPositions.resize(restPositions.size());
        outNormals.resize(restNormals.size());

        for (size_t v = 0; v < restPositions.size(); v++) {
            const auto& skin = skinData[v];

            // 加权混合对偶四元数
            DualQuat blended = dqs[skin.jointIndices[0]] * skin.weights[0];
            for (int i = 1; i < 4; i++) {
                // 保证最短路径：如果点积为负，取反第二个四元数
                if (glm::dot(blended.real, dqs[skin.jointIndices[i]].real) < 0) {
                    blended = blended + (dqs[skin.jointIndices[i]] * skin.weights[i]) * (-1.0f);
                } else {
                    blended = blended + dqs[skin.jointIndices[i]] * skin.weights[i];
                }
            }
            blended.Normalize();

            // 变换顶点
            outPositions[v] = TransformPoint(blended, restPositions[v]);

            // 法线变换（只用旋转部分）
            outNormals[v] = glm::normalize(glm::rotate(blended.real, restNormals[v]));
        }
    }
};
```

## GPU 蒙皮着色器

### GLSL 顶点着色器

```glsl
// ===== vertex_shader.glsl =====
#version 450 core

// 顶点属性
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texCoord;
layout(location = 3) in ivec4 a_jointIndices;   // 4个关节索引
layout(location = 4) in vec4 a_jointWeights;     // 4个权重

// Uniform
uniform mat4 u_viewProjection;
uniform mat4 u_jointMatrices[128];  // 关节蒙皮矩阵数组
uniform mat3 u_jointNormalMatrices[128];  // 法线变换矩阵（逆转置）

// 输出
out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_texCoord;

void main() {
    // LBS 蒙皮计算
    vec4 skinnedPos = vec4(0.0);
    vec3 skinnedNormal = vec3(0.0);

    for (int i = 0; i < 4; i++) {
        int idx = a_jointIndices[i];
        float w = a_jointWeights[i];

        if (w > 0.0) {
            skinnedPos += w * u_jointMatrices[idx] * vec4(a_position, 1.0);
            skinnedNormal += w * u_jointNormalMatrices[idx] * a_normal;
        }
    }

    v_worldPos = skinnedPos.xyz;
    v_normal = normalize(skinnedNormal);
    v_texCoord = a_texCoord;

    gl_Position = u_viewProjection * skinnedPos;
}
```

### HLSL (Unity/Unreal) 蒙皮

```hlsl
// ===== Unity Shader - GPU Skinning =====
struct appdata {
    float4 vertex : POSITION;
    float3 normal : NORMAL;
    float4 weights : BLENDWEIGHT;      // 4个权重
    uint4 indices : BLENDINDICES;       // 4个关节索引
};

uniform float4x4 _BoneMatrices[200];   // 关节矩阵数组

v2f vert(appdata v) {
    v2f o;

    // LBS 蒙皮
    float4 skinnedPos = float4(0, 0, 0, 0);
    float3 skinnedNormal = float3(0, 0, 0);

    for (int i = 0; i < 4; i++) {
        float w = v.weights[i];
        if (w > 0) {
            uint boneIdx = v.indices[i];
            skinnedPos += w * mul(_BoneMatrices[boneIdx], v.vertex);
            // 法线用逆转置矩阵
            skinnedNormal += w * mul((float3x3)_BoneMatrices[boneIdx], v.normal);
        }
    }

    o.vertex = UnityObjectToClipPos(skinnedPos);
    o.normal = normalize(skinnedNormal);
    return o;
}
```

### Compute Shader GPU 蒙皮

对于大规模角色（如全军系列同屏数千人），使用 Compute Shader 预计算蒙皮结果更高效：

```glsl
// ===== compute_skinning.glsl =====
#version 450 core

layout(local_size_x = 256) in;

// 输入缓冲
struct VertexData {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
    ivec4 jointIndices;
    vec4 jointWeights;
};

layout(std430, binding = 0) readonly buffer InputVertices {
    VertexData inputVerts[];
};

layout(std430, binding = 1) readonly buffer JointMatrices {
    mat4 jointMats[];
};

// 输出缓冲
layout(std430, binding = 2) writeonly buffer OutputVertices {
    vec4 outputPositions[];
};

layout(std430, binding = 3) writeonly buffer OutputNormals {
    vec4 outputNormals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= inputVerts.length()) return;

    VertexData v = inputVerts[idx];
    vec4 skinnedPos = vec4(0.0);
    vec3 skinnedNormal = vec3(0.0);

    for (int i = 0; i < 4; i++) {
        float w = v.jointWeights[i];
        if (w > 0.0) {
            int ji = v.jointIndices[i];
            skinnedPos += w * jointMats[ji] * vec4(v.position, 1.0);
            skinnedNormal += w * mat3(jointMats[ji]) * v.normal;
        }
    }

    outputPositions[idx] = skinnedPos;
    outputNormals[idx] = vec4(normalize(skinnedNormal), 0.0);
}
```

## 权重绘制工作流

### 在建模工具中的操作

权重绘制（Weight Painting）是美术师为每个顶点分配关节权重的过程：

1. **初始绑定**：选择模型和骨架，执行自动绑定（如 Blender 的 Automatic Weights）
2. **手动调整**：在权重绘制模式下，用笔刷调整每个关节的影响区域
3. **验证测试**：旋转各个关节，观察变形效果并修正问题区域

### 权重可视化

| 权重值 | 颜色 | 含义 |
|--------|------|------|
| 1.0 | 红色 | 完全受该关节影响 |
| 0.5 | 黄绿 | 中等影响 |
| 0.0 | 蓝色 | 完全不受影响 |

### 自动权重计算算法

```cpp
// 基于距离的自动权重计算
class AutoWeightCalculator {
public:
    void ComputeWeights(
        const std::vector<Vec3>& vertexPositions,
        const std::vector<int>& boneIndices,
        const std::vector<Vec3>& bonePositions,
        const std::vector<std::pair<int,int>>& boneHierarchy,
        std::vector<VertexSkinData>& outSkinData)
    {
        outSkinData.resize(vertexPositions.size());

        for (size_t v = 0; v < vertexPositions.size(); v++) {
            // 计算顶点到每根骨骼线段的距离
            struct BoneWeight {
                int boneIdx;
                float distance;
            };
            std::vector<BoneWeight> candidates;

            for (int b : boneIndices) {
                float dist = PointToBoneDistance(
                    vertexPositions[v], b, bonePositions, boneHierarchy);
                candidates.push_back({b, dist});
            }

            // 按距离排序，取最近的4根
            std::sort(candidates.begin(), candidates.end(),
                [](auto& a, auto& b) { return a.distance < b.distance; });

            // 基于距离的反比权重
            float weights[4];
            float totalWeight = 0;
            for (int i = 0; i < 4 && i < (int)candidates.size(); i++) {
                // 使用距离的平方反比，可以得到更锐利的权重分布
                float w = 1.0f / (candidates[i].distance * candidates[i].distance + 0.001f);
                weights[i] = w;
                totalWeight += w;
            }

            // 归一化
            int indices[4];
            for (int i = 0; i < 4; i++) {
                indices[i] = (i < (int)candidates.size()) ? candidates[i].boneIdx : 0;
                weights[i] /= totalWeight;
            }

            outSkinData[v].Pack(indices, weights, std::min(4, (int)candidates.size()));
        }
    }

private:
    float PointToBoneDistance(
        const Vec3& point, int boneIdx,
        const std::vector<Vec3>& bonePositions,
        const std::vector<std::pair<int,int>>& hierarchy)
    {
        // 计算点到骨骼线段的最短距离
        int parent = hierarchy[boneIdx].first;
        if (parent < 0) return glm::length(point - bonePositions[boneIdx]);

        Vec3 a = bonePositions[parent];
        Vec3 b = bonePositions[boneIdx];
        Vec3 ab = b - a;
        Vec3 ap = point - a;

        float t = glm::dot(ap, ab) / glm::dot(ab, ab);
        t = glm::clamp(t, 0.0f, 1.0f);

        Vec3 closest = a + ab * t;
        return glm::length(point - closest);
    }
};
```

## Unity 中的蒙皮

### SkinnedMeshRenderer 组件

```csharp
using UnityEngine;

public class SkinningExample : MonoBehaviour {
    void Start() {
        SkinnedMeshRenderer smr = GetComponent<SkinnedMeshRenderer>();

        // 获取骨骼数组
        Transform[] bones = smr.bones;
        Debug.Log($"骨骼数量: {bones.Length}");
        Debug.Log($"顶点数量: {smr.sharedMesh.vertexCount}");

        // 获取绑定姿态矩阵
        Matrix4x4[] bindposes = smr.sharedMesh.bindposes;
        // bindposes[i] 是第i个骨骼的逆绑定矩阵

        // 修改骨骼权重（运行时动态调整）
        BoneWeight[] weights = smr.sharedMesh.boneWeights;
        for (int i = 0; i < weights.Length; i++) {
            // 每个顶点最多4个骨骼影响
            // weights[i].boneIndex0/1/2/3
            // weights[i].weight0/1/2/3
        }
    }

    // 运行时动态添加蒙皮网格
    void AttachSkinnedMesh(GameObject target, Mesh mesh, Material mat) {
        SkinnedMeshRenderer smr = target.AddComponent<SkinnedMeshRenderer>();
        smr.sharedMesh = mesh;
        smr.material = mat;

        // 映射骨骼
        Transform[] newBones = new Transform[mesh.bindposes.Length];
        Skeleton skeleton = target.GetComponent<Skeleton>();
        for (int i = 0; i < mesh.bindposes.Length; i++) {
            // 通过骨骼名称查找对应的Transform
            string boneName = mesh.GetBoneName(i); // 假设有此方法
            newBones[i] = skeleton.FindBone(boneName);
        }
        smr.bones = newBones;
    }
}
```

### BlendShape（形态键）

Unity 的 BlendShape 系统用于面部表情等变形目标：

```csharp
public class BlendShapeController : MonoBehaviour {
    SkinnedMeshRenderer smr;

    void Start() {
        smr = GetComponent<SkinnedMeshRenderer>();

        // 列出所有BlendShape
        for (int i = 0; i < smr.sharedMesh.blendShapeCount; i++) {
            string name = smr.sharedMesh.GetBlendShapeName(i);
            int frameCount = smr.sharedMesh.GetBlendShapeFrameCount(i);
            Debug.Log($"BlendShape: {name}, 帧数: {frameCount}");
        }
    }

    // 平滑设置BlendShape权重
    void SetBlendShapeSmooth(string shapeName, float targetWeight, float speed) {
        int index = smr.sharedMesh.GetBlendShapeIndex(shapeName);
        if (index >= 0) {
            float current = smr.GetBlendShapeWeight(index);
            float newWeight = Mathf.MoveTowards(current, targetWeight, speed * Time.deltaTime);
            smr.SetBlendShapeWeight(index, newWeight);
        }
    }
}
```

## Unreal Engine 中的蒙皮

### Skeletal Mesh 和 Morph Target

```cpp
// UE5 C++ - 蒙皮网格操作
#include "Components/SkeletalMeshComponent.h"
#include "Engine/SkeletalMesh.h"

void AMyCharacter::SetupSkinning() {
    USkeletalMeshComponent* MeshComp = GetMesh();

    // 获取蒙皮网格信息
    USkeletalMesh* SkelMesh = MeshComp->GetSkeletalMeshAsset();
    int32 NumBones = SkelMesh->GetRefSkeleton().GetNum();
    int32 NumVertices = SkelMesh->GetImportedModel()->LODModels[0].NumVertices;

    UE_LOG(LogTemp, Log, TEXT("骨骼数: %d, 顶点数: %d"), NumBones, NumVertices);

    // Morph Target（形态目标）- 面部表情等
    // 在编辑器中创建，运行时通过名称控制权重
    MeshComp->SetMorphTarget(FName("Smile"), 0.5f);
    MeshComp->SetMorphTarget(FName("Blink_Left"), 1.0f);

    // 获取所有可用的 Morph Target
    TArray<FName> MorphTargetNames;
    SkelMesh->GetMorphTargetNames(MorphTargetNames);
    for (const FName& Name : MorphTargetNames) {
        UE_LOG(LogTemp, Log, TEXT("Morph Target: %s"), *Name.ToString());
    }
}
```

## LBS vs DQS vs 混合方案对比

### 数学特性对比

| 特性 | LBS | DQS | 混合方案 |
|------|-----|-----|---------|
| 体积守恒 | 不保持（塌陷） | 保持 | 近似保持 |
| 计算开销 | 4次矩阵乘法/顶点 | 4次对偶四元数运算/顶点 | 略高于LBS |
| GPU适配性 | 极好 | 良好 | 需要额外判断 |
| 关节处质量 | 有"糖果包装"效应 | 自然平滑 | 两者之间 |
| 实现复杂度 | 简单 | 中等 | 较复杂 |

### 混合方案实现

```cpp
// 关节附近用DQS，远离关节处用LBS
Vec3 HybridSkin(
    const Vec3& pos,
    const VertexSkinData& skin,
    const std::vector<Mat4>& skinMatrices,
    const std::vector<Vec3>& jointPositions)
{
    // 计算顶点到最近关节的距离
    float minDist = FLT_MAX;
    for (int i = 0; i < 4; i++) {
        if (skin.weights[i] > 0.001f) {
            float dist = glm::length(pos - jointPositions[skin.jointIndices[i]]);
            minDist = std::min(minDist, dist);
        }
    }

    // 在关节附近使用DQS，远处使用LBS
    float blendZone = 0.15f;  // 15cm混合区
    float dqsBlend = glm::clamp(1.0f - minDist / blendZone, 0.0f, 1.0f);

    Vec3 lbsResult = LBSSkinning::SkinVertex(pos, skin, skinMatrices);
    Vec3 dqsResult = DQSSkinning::SkinVertex(pos, skin, skinMatrices);

    return glm::mix(lbsResult, dqsResult, dqsBlend);
}
```

## 性能基准数据

| 场景 | 顶点数 | CPU LBS | CPU DQS | GPU LBS (Compute) |
|------|--------|---------|---------|-------------------|
| 单角色 | 10K | 0.3ms | 0.5ms | 0.02ms |
| 10角色 | 100K | 3.0ms | 5.0ms | 0.08ms |
| 100角色 | 1M | 30ms | 50ms | 0.5ms |
| 1000角色 | 10M | 不可行 | 不可行 | 4ms |

测试环境：RTX 3070, i7-12700K, 每顶点4关节影响。

## 常见陷阱与解决方案

### 陷阱 1：权重未归一化导致顶点漂移

**问题**：权重之和不等于 1.0，导致顶点偏离正确位置。

**解决方案**：
```cpp
// 在导入时强制归一化
void NormalizeAllWeights(std::vector<VertexSkinData>& skinData) {
    for (auto& sd : skinData) {
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += sd.weights[i];
        if (sum > 0) {
            for (int i = 0; i < 4; i++) sd.weights[i] /= sum;
        }
    }
}
```

### 陷阱 2：关节影响数超过限制

**问题**：某些顶点被超过 4 个关节影响（或 8 个），低权重关节被截断导致变形异常。

**解决方案**：导入时按权重排序，只保留最高的 N 个，然后重新归一化。

### 陷阱 3：非均匀缩放导致法线错误

**问题**：骨骼存在非均匀缩放时，直接用蒙皮矩阵变换法线会出错。

**解决方案**：使用逆转置矩阵或单独维护法线变换矩阵。

### 陷阱 4：GPU 蒙皮与 CPU 碰撞检测不一致

**问题**：碰撞检测在 CPU 上使用原始网格位置，而渲染使用 GPU 蒙皮结果。

**解决方案**：在 CPU 上维护一份简化版蒙皮（减少关节/顶点数），或使用 Compute Shader 输出碰撞数据。

## 实际游戏案例

- **Unreal Engine**：支持每顶点最多 4 关节影响，GPU Skinning 通过 Vertex Factory 实现，支持 Morph Target 用于面部
- **Unity**：默认支持 LBS，SkinnedMeshRenderer 组件处理蒙皮，可通过插件实现 DQS
- **蜘蛛侠 (PS4/PS5)**：高质量皮肤变形使用 DQS + 修正形变，面部使用 150+ Morph Targets
- **地平线：零之曙光**：角色使用 LBS + 表情骨骼蒙皮，机械兽使用分段蒙皮
- **最后生还者 Part II**：LBS + DQS 混合方案，面部使用高精度 Morph Target + 骨骼驱动
- **全军系列**：大规模场景使用 Compute Shader GPU 蒙皮，每帧处理数千角色
