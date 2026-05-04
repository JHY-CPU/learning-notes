<div align="center">

# 📚 深度学习与全栈学习笔记

**一份持续更新的个人知识库：全栈 / 算法（HTML） + 深度学习（Markdown）**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Deep Learning Notes](https://img.shields.io/badge/Deep%20Learning-390%2B-blue)
![Full-Stack Modules](https://img.shields.io/badge/Modules-18-green)

</div>

---

## 📖 概览 | Overview

本仓库是**单体知识库**：前 18 个编号目录为**前端 → 后端 → 运维 → 算法**的渐进式专题（以 **HTML** 页面为主，便于浏览器直接打开）；**`19-深度学习/`** 为**深度学习专题**，以 **Markdown** 为主，按「数学基础 → 传统 ML → 神经网络 → CV / NLP / 生成式 / 强化学习 → 工程部署」组织，含公式推导与代码示例。

> **目的**：用固定结构（核心概念、推导、直觉、代码、关联）把学过的东西写清楚，方便日后检索与复盘。

---

## 🗂️ 仓库结构 | Repository Layout

克隆后你会看到**按序号排列的模块**，与 README 中的描述一一对应：

| 路径 | 内容 | 主要格式 |
|------|------|----------|
| `01-HTML基础` … `18-算法与数据结构` | 全栈与算法数据结构学习路径 | 多为 `.html`（可直接双击或用 Live Server 打开） |
| `19-深度学习/` | 深度学习系统化笔记（8 个子目录） | `.md`（GitHub / VS Code / Typora 均可） |

深度学习子目录与**领域索引**详见 **[`19-深度学习/README.md`](19-深度学习/README.md)**（内含各目录篇数、学习路径 A/B/C、按问题速查表）。

---

## 🧠 深度学习笔记 | Deep Learning

路径：**`19-深度学习/`**（勿与旧文档中的 `DeepLearningNotes/` 混淆；本仓库已合并为上述目录）。

### 目录结构（与仓库内一致）

```
19-深度学习/
├── 01_Math_Foundations/          # 数学基础
├── 02_ML_Basics/                 # 机器学习基础
├── 03_NN_Core/                   # 神经网络核心（优化器、正则化、训练技巧等）
├── 04_Computer_Vision/           # 计算机视觉（CNN、检测、分割、ViT 等）
├── 05_NLP_Sequence/              # NLP 与序列模型（嵌入、RNN、Transformer、LLM 相关）
├── 06_Generative_AI/             # 生成式 AI（GAN、VAE、扩散等）
├── 07_Reinforcement_Learning/    # 强化学习
└── 08_Engineering_Deployment/    # 工程与部署（PyTorch、分布式、量化、服务等）
```

各子目录的**篇数统计与 README 索引**以 [`19-深度学习/README.md`](19-深度学习/README.md) 为准（会随笔记增减更新）。

---

## 🌐 全栈与算法 | Full-Stack & Algorithms

共 **18 个模块**（`01` … `18`），从基础到项目与算法：

| 序号 | 目录 | 说明 |
|------|------|------|
| 01 | `01-HTML基础` | HTML5 与页面结构 |
| 02 | `02-CSS基础` | 样式与布局入门 |
| 03 | `03-CSS进阶` | Flex、动画、响应式等 |
| 04 | `04-JavaScript基础` | JS 语言基础 |
| 05 | `05-JavaScript进阶` | 异步、模块化、进阶 API 等 |
| 06 | `06-Node.js基础` | Node 与 npm 生态 |
| 07 | `07-计算机网络` | 协议栈、HTTP、缓存等 |
| 08 | `08-Linux与Shell` | 命令行与脚本 |
| 09 | `09-Python` | Python 语言与常用库 |
| 10 | `10-数据库` | SQL 与 NoSQL |
| 11 | `11-Express与后端框架` | Express / Koa 等 |
| 12 | `12-Java` | Java 基础与集合等 |
| 13 | `13-SpringBoot` | Spring Boot 与生态 |
| 14 | `14-Go语言` | Go 语言与并发 |
| 15 | `15-Docker与DevOps` | 容器与交付 |
| 16 | `16-后端架构与安全` | 架构与安全实践 |
| 17 | `17-项目实战与总结` | 综合项目笔记 |
| 18 | `18-算法与数据结构` | 算法、数据结构、专题与实战 |

---

## ✨ 特点 | Features

- **深度学习**：五段式模板（核心概念 → 数学推导 → 直观理解 → 代码示例 → 与 DL 的关联），技术名词中英对照。
- **全栈 / 算法**：按专题分目录，单页 HTML 便于对照练习与演示。
- **可追溯**：重要专题在 `19-深度学习` 各子目录配有 `README.md` 作目录与导航。

---

## 🚀 快速开始 | Getting Started

1. **克隆本仓库**

```bash
git clone git@github.com:JHY-CPU/learning-notes.git
cd learning-notes
```

（若使用 HTTPS，将地址改为 `https://github.com/JHY-CPU/learning-notes.git` 即可。）

2. **阅读深度学习笔记**：从数学基础或 [`19-深度学习/README.md`](19-深度学习/README.md) 中的学习路径进入，例如：

```bash
# Windows（PowerShell）— 用默认程序打开示例文件
start "" "19-深度学习\01_Math_Foundations\01_向量空间与线性组合的物理意义.md"
```

3. **阅读全栈 HTML**：在资源管理器中进入对应 `01-` … `18-` 目录，用浏览器打开 `.html` 文件，或在 VS Code 中使用 Live Server。

---

## 📎 相关文件 | Related

| 文件 | 作用 |
|------|------|
| [`19-深度学习/README.md`](19-深度学习/README.md) | 深度学习总索引、篇数、学习路径、速查 |
| 各 `19-深度学习/0x_*/README.md` | 分领域目录与说明 |

---

## 📄 License

内容以 **MIT 精神**（可自由阅读、分享与二次学习）整理；若你希望 GitHub 显示标准协议栏，可在仓库根目录添加 `LICENSE` 文本文件（例如从 [Choose a License](https://choosealicense.com/licenses/mit/) 复制 MIT 全文）。

---

<div align="center">

若笔记对你有帮助，欢迎点个 ⭐

Happy Learning!

</div>
