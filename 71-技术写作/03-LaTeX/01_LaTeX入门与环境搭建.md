# LaTeX 入门与环境搭建

## 什么是 TeX 与 LaTeX

TeX 是由 Donald Knuth 在 1970 年代末开发的排版系统，专门用于高质量的科技文档排版。LaTeX 是 Leslie Lamport 在 TeX 基础上开发的一套宏集，它将排版的底层细节封装起来，让用户可以专注于文档内容而非格式。

LaTeX 的核心优势：
- **数学公式排版**：业界标准，无出其右
- **参考文献管理**：BibTeX/biblatex 自动化处理
- **交叉引用**：图表、章节、公式的自动编号与引用
- **高质量输出**：专业的排版效果，适合学术出版

## TeX 发行版选择

### TeX Live（推荐）

TeX Live 是跨平台的 TeX 发行版，由 TeX 用户组（TUG）维护，是目前最推荐的选择。

```bash
# Linux 安装
sudo apt-get install texlive-full    # Debian/Ubuntu
sudo dnf install texlive-scheme-full # Fedora

# macOS 安装
brew install --cask mactex           # 完整版（约 5GB）
brew install --cask basictex         # 精简版

# Windows
# 从 https://tug.org/texlive/ 下载安装程序
```

TeX Live 的版本每年更新一次（如 TeX Live 2024、TeX Live 2025），建议保持每年更新。

### MiKTeX

MiKTeX 是 Windows 平台的另一选择，支持按需安装包（on-the-fly package installation），初始安装体积较小。

```bash
# 从 https://miktex.org/download 下载安装程序
# 安装时选择 "Install missing packages on the fly: Yes"
```

### Overleaf（在线编辑器）

Overleaf 是最流行的在线 LaTeX 编辑器，无需本地安装，适合协作和快速入门。

- 地址：https://www.overleaf.com
- 支持实时协作、版本历史、丰富的模板库
- 免费版编译时间限制为 20 秒，适合大多数文档

## 第一个 LaTeX 文档

### 基本文档结构

```latex
% 这是注释，以 % 开头
\documentclass{article}          % 文档类：article, report, book

% 导言区：加载宏包、设置全局参数
\usepackage[UTF8]{ctex}          % 中文支持
\usepackage{amsmath}             % 数学公式
\usepackage{geometry}            % 页面布局
\geometry{a4paper, margin=2.5cm}

\title{我的第一篇 LaTeX 文档}
\author{张三}
\date{\today}

% 正文区
\begin{document}

\maketitle                       % 生成标题

\section{引言}
这是我的第一个 LaTeX 文档。LaTeX 可以轻松排版数学公式，例如 $E = mc^2$。

\section{正文}
LaTeX 会自动处理段落缩进和行间距。两个换行符表示新段落。

第二段文字会自动缩进。

\end{document}
```

### 文档类说明

| 文档类   | 适用场景             | 默认字号 | 默认纸张 |
|----------|----------------------|----------|----------|
| article  | 短文、期刊论文       | 10pt     | letter   |
| report   | 长报告、学位论文     | 12pt     | letter   |
| book     | 书籍                 | 12pt     | letter   |
| beamer   | 演示文稿             | —        | —        |
| letter   | 信函                 | 12pt     | letter   |

## 编译工作流程

### 命令行编译

```bash
# 基本编译（使用 pdflatex）
pdflatex document.tex

# 涉及参考文献时的完整编译流程
pdflatex document.tex    # 第一次：生成 .aux 引用信息
bibtex document          # 处理参考文献
pdflatex document.tex    # 第二次：插入参考文献
pdflatex document.tex    # 第三次：解决交叉引用

# 使用 latexmk 自动处理依赖
latexmk -pdf document.tex         # 自动判断需要编译几次
latexmk -c                       # 清理辅助文件
```

### 编译引擎对比

| 引擎       | 输入编码   | 字体支持         | 推荐场景           |
|------------|------------|------------------|--------------------|
| pdflatex   | UTF-8/Latin| Type1/OTF        | 英文文档           |
| xelatex    | UTF-8      | 系统字体（TTF/OTF）| 中文文档（推荐） |
| lualatex   | UTF-8      | 系统字体+Lua 扩展 | 复杂排版需求       |

对于中文文档，推荐使用 **XeLaTeX** 引擎：

```bash
xelatex document.tex
```

## VS Code + LaTeX Workshop 环境搭建

```json
// .vscode/settings.json 中的 LaTeX Workshop 配置
{
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": ["-synctex=1", "-interaction=nonstopmode", "%DOC%"]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": ["%DOCFILE%"]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": ["xelatex"]
        },
        {
            "name": "XeLaTeX + BibTeX + XeLaTeX x2",
            "tools": ["xelatex", "bibtex", "xelatex", "xelatex"]
        }
    ],
    "latex-workshop.latex.autoBuild.run": "onSave",
    "latex-workshop.latex.outputDir": "%DIR%/build"
}
```

## 辅助文件说明

编译过程中会生成多种辅助文件，不必纳入版本控制：

| 扩展名     | 用途                         |
|------------|------------------------------|
| .aux       | 交叉引用信息                 |
| .log       | 编译日志                     |
| .toc       | 目录信息                     |
| .bbl       | BibTeX 生成的参考文献        |
| .blg       | BibTeX 日志                  |
| .synctex.gz| 正向/反向搜索同步信息        |
| .out       | PDF 书签信息                 |

建议在 `.gitignore` 中添加：

```gitignore
*.aux
*.log
*.toc
*.bbl
*.blg
*.synctex.gz
*.out
*.fls
*.fdb_latexmk
build/
```
