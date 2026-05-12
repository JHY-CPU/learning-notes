# 107-Shader编程

本模块系统讲解Shader编程的核心知识，涵盖从渲染管线基础到高级特效实现。

## 目录

| 编号 | 主题 | 说明 |
|------|------|------|
| 01 | 渲染管线回顾 | CPU/GPU分工、顶点/光栅化/片元阶段、Draw Call |
| 02 | GLSL基础语法 | 变量类型、uniform/varying、精度限定符 |
| 03 | HLSL基础语法 | 与GLSL的差异、语义、Constant Buffer |
| 04 | 顶点着色器 | 顶点变换、法线变换、顶点动画 |
| 05 | 片元着色器与光照模型 | Phong/Blinn-Phong/PBR、法线贴图 |
| 06 | 纹理采样与UV | 纹理坐标、Wrap模式、Mipmap、各向异性过滤 |
| 07 | 透明与混合 | Alpha Blend、Alpha Test、渲染队列 |
| 08 | 后处理特效 | Bloom、运动模糊、色调映射 |
| 09 | 阴影技术 | Shadow Map、PCF、CSM级联阴影 |
| 10 | 卡通渲染(Toon Shading) | Ramp贴图、描边、Rim Light |
| 11 | Compute Shader | GPGPU、线程组、粒子系统 |
| 12 | Shader性能优化 | 过度绘制、数学优化、带宽优化 |

## 学习建议

建议按照编号顺序学习，先掌握渲染管线和基础语法，再逐步深入光照、特效和优化。每篇笔记均包含核心概念、代码示例和实际案例。
