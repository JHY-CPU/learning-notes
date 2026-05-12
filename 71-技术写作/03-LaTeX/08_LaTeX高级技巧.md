# LaTeX 高级技巧

## TikZ 绘图基础

TikZ 是 LaTeX 中最强大的绘图工具，可以直接在文档中创建矢量图形。

### 基本图形

```latex
\usepackage{tikz}

% 直线
\begin{tikzpicture}
    \draw (0,0) -- (2,1) -- (4,0) -- cycle;   % 三角形
    \draw[->] (0,0) -- (3,0);                  % 带箭头的线
    \draw[thick, red] (0,1) -- (3,1);          % 粗红线
\end{tikzpicture}

% 矩形和圆
\begin{tikzpicture}
    \draw (0,0) rectangle (2,1);               % 矩形
    \draw[fill=blue!20] (4,0.5) circle (0.5);  % 填充的圆
    \draw (7,0.5) ellipse (1 and 0.5);         % 椭圆
\end{tikzpicture}

% 节点和边
\begin{tikzpicture}
    \node[draw, circle] (a) at (0,0) {A};
    \node[draw, circle] (b) at (2,0) {B};
    \node[draw, circle] (c) at (1,1.5) {C};
    \draw (a) -- (b);
    \draw (a) -- (c);
    \draw (b) -- (c);
\end{tikzpicture}
```

### 流程图

```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\tikzstyle{startstop} = [rectangle, rounded corners,
    minimum width=3cm, minimum height=1cm,
    text centered, draw=black, fill=red!30]
\tikzstyle{process} = [rectangle,
    minimum width=3cm, minimum height=1cm,
    text centered, draw=black, fill=orange!30]
\tikzstyle{decision} = [diamond,
    minimum width=3cm, minimum height=1cm,
    text centered, draw=black, fill=green!30]
\tikzstyle{arrow} = [thick, ->, >=Stealth]

\begin{tikzpicture}[node distance=2cm]
    \node (start) [startstop] {开始};
    \node (input) [process, below of=start] {输入数据};
    \node (decide) [decision, below of=input, yshift=-0.5cm]
        {数据有效?};
    \node (proc) [process, below of=decide, yshift=-0.5cm]
        {处理数据};
    \node (output) [process, right of=decide, xshift=3cm]
        {报错};
    \node (stop) [startstop, below of=proc] {结束};

    \draw [arrow] (start) -- (input);
    \draw [arrow] (input) -- (decide);
    \draw [arrow] (decide) -- node[anchor=east] {是} (proc);
    \draw [arrow] (decide) -- node[anchor=south] {否} (output);
    \draw [arrow] (proc) -- (stop);
\end{tikzpicture}
```

### 神经网络图

```latex
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{tikzpicture}[
    neuron/.style={circle, draw, minimum size=0.8cm},
    layer/.style={draw, dashed, minimum width=2.5cm, minimum height=4cm}
]
    % 层
    \node[layer] (input) at (0,0) {};
    \node[layer] (hidden) at (3,0) {};
    \node[layer] (output) at (6,0) {};

    % 输入层节点
    \foreach \i in {1,...,3}
        \node[neuron] (i\i) at (0, {-1.5 + \i}) {$x_\i$};

    % 隐藏层节点
    \foreach \i in {1,...,4}
        \node[neuron] (h\i) at (3, {-1.8 + \i*1.0}) {};

    % 输出层节点
    \foreach \i in {1,...,2}
        \node[neuron] (o\i) at (6, {-0.5 + \i}) {$y_\i$};

    % 连线
    \foreach \i in {1,...,3}
        \foreach \j in {1,...,4}
            \draw[->, gray] (i\i) -- (h\j);

    \foreach \i in {1,...,4}
        \foreach \j in {1,...,2}
            \draw[->, gray] (h\i) -- (o\j);

    % 标签
    \node[above] at (input.north) {输入层};
    \node[above] at (hidden.north) {隐藏层};
    \node[above] at (output.north) {输出层};
\end{tikzpicture}
```

## Beamer 演示文稿

Beamer 是 LaTeX 中制作学术演示文稿的标准工具。

```latex
\documentclass[aspectratio=169, 12pt]{beamer}

\usetheme{Madrid}           % 主题：Madrid, Berlin, Warsaw, Singapore
\usecolortheme{whale}       % 配色：whale, dolphin, seahorse
\usefonttheme{serif}        % 字体主题

\usepackage[UTF8]{ctex}
\usepackage{graphicx}
\usepackage{amsmath}

% 自定义设置
\setbeamertemplate{navigation symbols}{}  % 隐藏导航按钮
\setbeamertemplate{footline}[frame number] % 显示页码

% 标题信息
\title{深度学习在图像识别中的应用}
\subtitle{研究进展与未来方向}
\author{张三}
\institute{清华大学计算机系}
\date{\today}

\begin{document}

% 标题页
\begin{frame}
    \titlepage
\end{frame}

% 目录页
\begin{frame}{目录}
    \tableofcontents
\end{frame}

\section{研究背景}

\begin{frame}{研究背景}
    \begin{itemize}
        \item 图像识别是计算机视觉的核心问题
        \item 传统方法依赖手工特征提取
        \item 深度学习带来了革命性的突破
    \end{itemize}
\end{frame}

\section{方法}

\begin{frame}{网络架构}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            网络结构包含：
            \begin{enumerate}
                \item 卷积层提取特征
                \item 池化层降低维度
                \item 全连接层分类
            \end{enumerate}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{equation}
                y = \text{softmax}(Wx + b)
            \end{equation}
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{公式示例}
    交叉熵损失函数：
    \[
        \mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
    \]

    \pause    % 点击后显示下面的内容
    其中 $C$ 为类别数，$y_i$ 为真实标签，$\hat{y}_i$ 为预测概率。
\end{frame}

\end{document}
```

### Beamer 覆盖说明

```latex
% \pause - 暂停，逐段显示
\begin{frame}{逐段显示}
    第一步 \pause
    第二步 \pause
    第三步
\end{frame}

% \onslide, \only, \uncover - 精细控制
\begin{frame}{精细控制}
    \onslide<1->{第一张幻灯片出现}
    \only<2>{仅在第二张出现，不占空间}
    \uncover<3->{始终占空间，但第三张才可见}
\end{frame}
```

## 算法排版

### algorithm + algorithmic

```latex
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{algorithm}
\caption{梯度下降算法}
\label{alg:gd}
\begin{algorithmic}[1]
    \REQUIRE 学习率 $\eta$, 训练数据 $\mathcal{D}$, 最大迭代次数 $T$
    \ENSURE 模型参数 $\theta$
    \STATE 初始化参数 $\theta_0$
    \FOR{$t = 1$ \TO $T$}
        \STATE 从 $\mathcal{D}$ 中采样小批量数据 $(x, y)$
        \STATE 计算梯度 $g_t \leftarrow \nabla_\theta \mathcal{L}(\theta_{t-1}; x, y)$
        \STATE 更新参数 $\theta_t \leftarrow \theta_{t-1} - \eta \cdot g_t$
        \IF{$\|\theta_t - \theta_{t-1}\| < \epsilon$}
            \STATE \textbf{break}
        \ENDIF
    \ENDFOR
    \RETURN $\theta_T$
\end{algorithmic}
\end{algorithm}
```

### algorithm2e（更现代的选择）

```latex
\usepackage[ruled, vlined, linesnumbered]{algorithm2e}

\begin{algorithm}
\caption{K-Means 聚类算法}
\label{alg:kmeans}
\SetAlgoLined
\KwIn{数据集 $X = \{x_1, \ldots, x_n\}$, 聚类数 $K$}
\KwOut{聚类中心 $\mu_1, \ldots, \mu_K$, 聚类分配 $C_1, \ldots, C_K$}

随机初始化 $K$ 个聚类中心 $\mu_1, \ldots, \mu_K$ \;
\Repeat{收敛}{
    % 分配步骤
    \ForEach{$x_i \in X$}{
        $c_i \leftarrow \arg\min_k \|x_i - \mu_k\|^2$ \;
    }
    % 更新步骤
    \ForEach{$k = 1, \ldots, K$}{
        $\mu_k \leftarrow \frac{1}{|C_k|} \sum_{x \in C_k} x$ \;
    }
}
\end{algorithm}
```

## 代码排版（listings）

```latex
\usepackage{listings}
\usepackage{xcolor}

% Python 代码样式
\lstdefinestyle{python}{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{green!60!black}\itshape,
    stringstyle=\color{red},
    numbers=left,
    numberstyle=\tiny\color{gray},
    numbersep=8pt,
    frame=single,
    rulecolor=\color{gray},
    breaklines=true,
    captionpos=b,
    tabsize=4,
    showstringspaces=false
}

\lstset{style=python}

% 插入代码
\begin{lstlisting}[caption={梯度下降实现}, label={lst:gd}]
import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    """使用梯度下降法进行线性回归"""
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= lr * gradient

    return theta
\end{lstlisting}

% 从文件导入代码
\lstinputlisting[style=python, caption={从文件导入}]
{src/model.py}
```

### 其他语言样式

```latex
% C++ 代码样式
\lstdefinestyle{cpp}{
    language=C++,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue},
    commentstyle=\color{green!60!black},
    stringstyle=\color{red},
    morekeywords={nullptr, constexpr, auto}
}

% Java 代码样式
\lstdefinestyle{java}{
    language=Java,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{gray}\itshape
}

% 使用 minted 宏包（需安装 Python Pygments，效果更好）
\usepackage{minted}
\begin{listing}
\begin{minted}{python}
def hello():
    print("Hello, LaTeX!")
\end{minted}
\caption{使用 minted 排版的代码}
\end{listing}
```

## hyperref 超链接

```latex
\usepackage[
    colorlinks=true,
    linkcolor=blue,
    citecolor=green!60!black,
    urlcolor=blue!80!black,
    bookmarks=true,
    bookmarksnumbered=true,
    pdfauthor={作者姓名},
    pdftitle={论文标题},
    pdfkeywords={关键词1, 关键词2}
]{hyperref}

% 超链接命令
\href{https://www.example.com}{链接文字}
\url{https://www.example.com}
\nolinkurl{https://www.example.com}    % 不生成链接

% 书签深度
\hypersetup{bookmarksdepth=3}

% 自定义 PDF 元数据
\usepackage{hyperref}
\hypersetup{
    pdftitle={LaTeX 学术写作指南},
    pdfauthor={张三},
    pdfsubject={学术写作},
    pdfcreator={XeLaTeX}
}
```
