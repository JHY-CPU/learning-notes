# 参考文献与 BibTeX

## 参考文献基础

LaTeX 提供了强大的参考文献管理系统，核心工具包括 BibTeX 和 biblatex 两种方案。学术写作中，正确管理参考文献至关重要。

## .bib 文件格式

BibTeX 使用 `.bib` 文件存储文献数据库，每条记录有特定的类型和字段。

```bibtex
% references.bib

@article{zhang2023transformer,
    author  = {Zhang, Wei and Li, Ming and Wang, Jun},
    title   = {Transformer-based Models for Natural Language Processing},
    journal = {IEEE Transactions on Neural Networks},
    year    = {2023},
    volume  = {34},
    number  = {5},
    pages   = {1023--1035},
    doi     = {10.1109/TNN.2023.1234567}
}

@inproceedings{li2022attention,
    author    = {Li, Xiao and Chen, Yu},
    title     = {Attention Mechanisms in Deep Learning: A Survey},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    year      = {2022},
    pages     = {1567--1575},
    address   = {Virtual},
    publisher = {AAAI Press}
}

@book{bishop2006pattern,
    author    = {Bishop, Christopher M.},
    title     = {Pattern Recognition and Machine Learning},
    publisher = {Springer},
    year      = {2006},
    address   = {New York},
    isbn      = {978-0-387-31073-2}
}

@phdthesis{wang2021thesis,
    author = {Wang, Lei},
    title  = {Deep Learning Methods for Image Recognition},
    school = {Tsinghua University},
    year   = {2021}
}

@misc{tensorflow2015,
    author       = {{Google Brain Team}},
    title        = {TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems},
    year         = {2015},
    url          = {https://www.tensorflow.org/},
    note         = {Accessed: 2024-01-15}
}

@online{github2024,
    author = {{GitHub Inc.}},
    title  = {GitHub: Let's Build from Here},
    year   = {2024},
    url    = {https://github.com}
}
```

### 常用文献类型

| 类型             | 说明               |
|------------------|--------------------|
| `@article`       | 期刊论文           |
| `@inproceedings` | 会议论文           |
| `@book`          | 书籍               |
| `@phdthesis`     | 博士学位论文       |
| `@mastersthesis` | 硕士学位论文       |
| `@techreport`    | 技术报告           |
| `@misc`          | 其他类型           |
| `@online`        | 在线资源           |

## 使用 natbib（推荐）

natbib 是最常用的参考文献管理宏包，支持多种引用格式。

```latex
\usepackage[numbers, sort&compress]{natbib}
% 选项：
% numbers  - 数字引用 [1], [2,3]
% authoryear - 作者-年份 (Zhang et al., 2023)
% sort&compress - 排序并合并 [1-3,5]

% 在文档末尾
\bibliographystyle{plainnat}    % 样式
\bibliography{references}       % .bib 文件名（不加扩展名）
```

### 引用命令

```latex
% 数字引用模式
\citet{zhang2023transformer}
% 输出：Zhang et al. [1]

\citep{zhang2023transformer}
% 输出：[1]

\citep{zhang2023transformer, li2022attention}
% 输出：[1, 2]

\citealp{zhang2023transformer}
% 输出：1（无括号）

% 作者-年份引用模式
\citet{zhang2023transformer}
% 输出：Zhang et al. (2023)

\citep{zhang2023transformer}
% 输出：(Zhang et al., 2023)

\citep[参见][第 3 章]{zhang2023transformer}
% 输出：(参见 Zhang et al., 2023, 第 3 章)

\citeauthor{zhang2023transformer}
% 输出：Zhang et al.

\citeyear{zhang2023transformer}
% 输出：2023
```

### 常用参考文献样式

```latex
\bibliographystyle{plain}       % 按引用顺序编号
\bibliographystyle{plainnat}    % plain 的 natbib 兼容版
\bibliographystyle{unsrt}       % 按引用顺序，格式不排序
\bibliographystyle{abbrv}       % 缩写作者名
\bibliographystyle{IEEEtran}    % IEEE 格式
\bibliographystyle{acm}         % ACM 格式
\bibliographystyle{siam}        % SIAM 格式
```

## 使用 biblatex + Biber（现代方案）

biblatex 是更现代的参考文献管理方案，提供更强的自定义能力。

```latex
\usepackage[
    backend=biber,
    style=numeric,       % 或 authoryear, ieee, apa
    sorting=none,        % 按引用顺序排序
    maxbibnames=3,       % 参考文献列表中最多显示 3 个作者
    maxcitenames=2,      % 引用中最多显示 2 个作者
]{biblatex}
\addbibresource{references.bib}

% 在文档中引用
\textcite{zhang2023transformer}     % 文本引用：Author [1]
\parencite{zhang2023transformer}    % 括号引用：[1]
\autocite{zhang2023transformer}     % 自动格式
\citeauthor{zhang2023transformer}   % 仅作者
\citeyear{zhang2023transformer}     % 仅年份

% 在文档末尾
\printbibliography[title={参考文献}]
% 或使用中文标题
\printbibliography[heading=bibintoc, title={参考文献}]
```

编译命令变为：

```bash
xelatex document.tex
biber document
xelatex document.tex
xelatex document.tex
```

## IEEE 格式

```latex
% 方案一：使用 natbib + IEEEtran
\usepackage[numbers]{natbib}
\bibliographystyle{IEEEtran}

% 方案二：使用 biblatex-ieee
\usepackage[
    backend=biber,
    style=ieee,
]{biblatex}
\addbibresource{references.bib}

% IEEE 引用示例
% 正文中：...as shown in [1].
% 参考文献列表：
% [1] W. Zhang, M. Li, and J. Wang, "Transformer-based models
%     for NLP," IEEE Trans. Neural Netw., vol. 34, no. 5,
%     pp. 1023-1035, May 2023.
```

## APA 格式

```latex
\usepackage[
    backend=biber,
    style=apa,
]{biblatex}
\addbibresource{references.bib}

% APA 引用示例
\parencite{zhang2023transformer}
% 输出：(Zhang et al., 2023)

\textcite{zhang2023transformer}
% 输出：Zhang et al. (2023)
```

## 手动参考文献（无 BibTeX）

对于简短文档，可以直接在 LaTeX 中手动编写参考文献列表。

```latex
\begin{thebibliography}{99}    % 99 表示最大编号宽度
\bibitem{zhang2023}
W.~Zhang, M.~Li, and J.~Wang,
``Transformer-based models for natural language processing,''
\textit{IEEE Trans.\ Neural Netw.}, vol.~34, no.~5, pp.~1023--1035, 2023.

\bibitem{bishop2006}
C.~M.~Bishop,
\textit{Pattern Recognition and Machine Learning}.
New York: Springer, 2006.
\end{thebibliography}

% 正文中引用
如文献~\cite{zhang2023} 所述。
```

## 引用技巧

```latex
% 多文献引用
\citep{ref1, ref2, ref3}
% 输出：[1-3]（使用 sort&compress）

% 带注释的引用
\citep[见第 5 节]{ref1}
% 输出：[1, 见第 5 节]

% 仅引用页码
\citep[p.~42]{ref1}

% 压制作者名（在作者-年份模式中）
\citeyearpar{ref1}
% 输出：(2023)

% 引用但不影响编号
\nocite{ref1}           % 单条
\nocite{*}              % .bib 文件中的所有文献
```

## 引用键命名规范

推荐使用统一的命名规则，便于管理和查找：

```
<姓氏><年份><关键词缩写>
zhang2023transformer
li2022attention
bishop2006pattern

或
<作者首字母><年份><简短描述>
WZ2023transformer
LM2022attention
```
