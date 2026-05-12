# 103-游戏引擎Unity

## 模块概述

本模块系统学习Unity游戏引擎的核心知识体系，涵盖从编辑器基础到高级功能的完整路径。Unity是全球使用最广泛的游戏引擎之一，支持2D/3D游戏开发，可发布到PC、移动端、主机及Web平台。

## 知识体系

| 编号 | 主题 | 核心内容 |
|------|------|----------|
| 01 | Unity架构与编辑器概览 | 编辑器界面布局、核心视图、Prefab系统 |
| 02 | C#脚本与MonoBehaviour | 生命周期函数、SerializeField、协程 |
| 03 | GameObject与Component模型 | 组件化架构、Transform层级、AddComponent |
| 04 | 物理系统与碰撞 | Rigidbody、Collider、Trigger/Collision、物理材质 |
| 05 | 输入系统 | 旧Input Manager vs 新Input System、轴映射 |
| 06 | UI系统(UGUI) | Canvas渲染模式、RectTransform、Layout Group |
| 07 | 动画系统(Animator) | Animator Controller、状态机、Blend Tree |
| 08 | 资源管理与AssetBundle | Resources加载、Addressable、热更新方案 |
| 09 | 导航与寻路 | NavMesh烘焙、NavMeshAgent、Off-Mesh Link |
| 10 | 粒子系统 | Particle System模块、Shuriken、碰撞/子发射器 |
| 11 | ShaderGraph与URP | 可视化Shader编辑、Universal Render Pipeline |
| 12 | Lua热更新与xLua | xLua集成、Lua与C#交互、热更流程 |

## 学习路径建议

1. **基础阶段** (01-03): 掌握编辑器操作和脚本编程基础
2. **交互阶段** (04-06): 学习物理、输入和UI交互
3. **表现阶段** (07-10): 掌握动画、资源、导航和粒子效果
4. **进阶阶段** (11-12): Shader可视化和热更新方案

## 技术栈

- 引擎版本: Unity 2022 LTS / Unity 6
- 编程语言: C# 10+、Lua 5.x (热更新)
- 渲染管线: URP (Universal Render Pipeline)
- 脚本后端: Mono / IL2CPP
