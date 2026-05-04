# -*- coding: utf-8 -*-
"""
生成「408 统考—操作系统」扩展笔记 HTML（编号 208–799，共 592 篇）。
运行前会删除本目录下除 200–207 精读页以外的全部 .html，再写入新文件。
用法：在 20-操作系统 目录执行  py -3 generate_os_notes.py
"""
from __future__ import annotations

import html
import os
import re

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_NUM = 208
# 592 篇扩展 + 8 篇手写 = 600；扩展编号 208..799
TOTAL_EXT = 799 - START_NUM + 1
assert TOTAL_EXT == 592

KEEP_HTML = {
    "200-操作系统导览与大纲.html",
    "201-操作系统角色与系统调用.html",
    "202-进程与线程.html",
    "203-CPU调度算法.html",
    "204-同步互斥与信号量.html",
    "205-死锁检测与避免.html",
    "206-虚拟内存分页与TLB.html",
    "207-文件系统与磁盘IO.html",
}

STYLE = """        body { font-family: sans-serif; padding: 20px; background: #f5f5f5; }
        .tag408 { display: inline-block; background: #c0392b; color: #fff; padding: 2px 10px; border-radius: 4px; font-size: 12px; margin-left: 8px; vertical-align: middle; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { margin: 0 0 10px 0; color: #2c3e50; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 14px; border-left: 3px solid #3498db; margin: 10px 0; white-space: pre-wrap; }
        .note { background: #fef9e7; border-left: 4px solid #f39c12; padding: 10px 15px; font-size: 14px; color: #7f8c8d; margin-top: 10px; }
        .syllabus { color: #16a085; font-weight: 600; }"""

# 8 种角度 × 74 个知识点 = 592；知识点按 408 OS 常见考纲粒度拆分
ANGLES = ("概念速记", "408选择考点", "典型简答", "计算题模板", "易错辨析", "跨章综合", "真题风格设问", "自测要点")

# (章名, [知识点...]) 知识点总数须为 74
CHAPTER_TOPICS: list[tuple[str, list[str]]] = [
    (
        "操作系统概述",
        [
            "OS定义与作用",
            "并发共享虚拟异步四特征",
            "OS功能五大管理",
            "手工批处理多道程序分时实时",
            "OS运行内核态用户态",
            "中断异常系统调用",
            "大内核与微内核对比",
            "OS结构分层模块化外核",
        ],
    ),
    (
        "进程与线程",
        [
            "进程定义与特征",
            "进程三态五态七态图",
            "PCB内容与作用",
            "进程创建终止阻塞唤醒",
            "进程通信共享存储",
            "管道PIPE与FIFO",
            "消息传递与信箱",
            "线程属性与实现",
            "用户级内核级组合多线程模型",
            "多对一一对一多对多模型对比",
            "进程与线程对比",
            "内核级线程优点",
        ],
    ),
    (
        "处理机调度与死锁",
        [
            "作业调度中级调度进程调度",
            "调度准则吞吐周转响应等待",
            "FCFS先来先服务",
            "SJF短作业优先SRTN",
            "高响应比优先HRRN",
            "时间片轮转RR",
            "优先级调度与饥饿老化",
            "多级队列与多级反馈队列",
            "实时调度EDF速率单调",
            "死锁定义与四个必要条件",
            "死锁预防破坏条件",
            "死锁避免银行家算法",
            "死锁检测与解除",
            "资源分配图化简",
        ],
    ),
    (
        "内存管理",
        [
            "程序链接装入绝对可重定位动态运行",
            "逻辑地址物理地址重定位",
            "连续分配单一固定动态分区",
            "动态分区首次适应邻近适应最佳最坏",
            "基本分页地址结构页表",
            "基本分段段表与二维逻辑地址",
            "段页式访存次数",
            "虚拟内存特征与理论依据",
            "请求分页页表项状态位修改位引用位",
            "缺页中断处理流程",
            "页面分配策略固定局部可变全局",
            "置换范围局部全局置换",
            "OPT最佳置换",
            "FIFO与Belady异常",
            "LRU最近最久未使用",
            "CLOCK时钟置换与改进型CLOCK",
            "工作集与抖动",
            "内存映射文件mmap思想",
        ],
    ),
    (
        "文件管理",
        [
            "文件逻辑结构顺序索引直接",
            "目录结构单级两级树形无环图",
            "FCB与索引节点inode",
            "文件共享硬链接软链接",
            "文件保护访问类型访问控制",
            "文件系统层次结构",
            "连续分配链接分配索引分配",
            "混合索引与寻址次数",
            "空闲表空闲链位示图成组链接",
        ],
    ),
    (
        "输入输出管理",
        [
            "IO设备分类字符块人机通信",
            "IO控制程序轮询中断DMA通道",
            "IO软件层次中断处理驱动设备无关用户层",
            "缓冲区单缓冲双缓冲循环缓冲缓冲池",
            "设备分配数据结构设备独立性逻辑设备表",
            "SPOOLing假脱机技术",
            "磁盘结构磁道扇区柱面",
            "磁盘访问时间寻道旋转传输",
            "磁盘调度FCFS SSTF SCAN CSCAN",
        ],
    ),
    (
        "综合与408答题策略",
        [
            "PV操作模板与互斥同步分离",
            "银行家算法安全性检查手写步骤",
            "页面置换缺页率计算",
            "磁盘调度平均寻道长度计算",
        ],
    ),
]


def verify_topic_count() -> int:
    n = 0
    for _, subs in CHAPTER_TOPICS:
        n += len(subs)
    return n


def clean_generated_html() -> int:
    removed = 0
    for name in os.listdir(OUT_DIR):
        if not name.endswith(".html"):
            continue
        if name in KEEP_HTML:
            continue
        path = os.path.join(OUT_DIR, name)
        os.remove(path)
        removed += 1
    return removed


def safe_stem(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "-", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if len(s) > 55:
        s = s[:55].rstrip("-")
    return s


def angle_block(chapter: str, topic: str, angle: str) -> tuple[str, str, str]:
    """返回 (小节标题, 段落说明, code块文本)"""
    if angle == "概念速记":
        body = (
            f"// 408《操作系统》— {chapter}\n"
            f"// 知识点: {topic}\n"
            "// 作答提示: 先写定义/组成/对象，再写特点或与其他概念边界。\n"
            "// 常搭配: 与进程状态、页表项位、调度指标一起记关键词。"
        )
        hint = "用 3～6 条短句背熟定义与判定条件，适合名解与选择题题干识别。"
    elif angle == "408选择考点":
        body = (
            f"// 典型选择题角度: {topic}\n"
            "// 关注: 「是否/一定/可能」、单位、缺页后行为、调度是否可抢占。\n"
            "// 练习: 把易混项写成「A对B错」对照表。"
        )
        hint = "408 选择常考概念辨析与边界条件，注意题目中的前提（单CPU、局部置换等）。"
    elif angle == "典型简答":
        body = (
            f"// 简答题框架: {topic}\n"
            "// 结构建议: (1)定义 (2)要点分条 (3)对比/举例 (4)小结。\n"
            "// 控制篇幅: 一般 0.5～1 页答题纸，先骨架后细节。"
        )
        hint = "简答按得分点列条，宁可多写层次标题，少写长段叙述。"
    elif angle == "计算题模板":
        body = (
            f"// 计算题切入点: {topic}\n"
            "// 步骤: 列已知 → 选公式/画表 → 迭代过程写清楚 → 结论带单位。\n"
            "// 408 常要求展示中间态（如银行家表、缺页序列、磁头移动）。"
        )
        hint = "计算题过程分大于结果分，表格与序号对齐阅卷。"
    elif angle == "易错辨析":
        body = (
            f"// 易错点: {topic}\n"
            "// 常见坑: 混淆「中断与异常」「并发与并行」「饥饿与死锁」等。\n"
            "// 建议: 各写一句反例或反证提醒自己。"
        )
        hint = "把自己错过的选项整理成「错因→正确表述」两行一组。"
    elif angle == "跨章综合":
        body = (
            f"// 综合: {topic} × 内存/文件/IO\n"
            "// 408 综合题常把调度+同步 或 分页+文件 组合，先拆子问题。\n"
            "// 画图: 资源分配图、页表、磁盘磁头移动示意。"
        )
        hint = "先标出子考点分值倾向，再按时间分配，避免在单一小问上超时。"
    elif angle == "真题风格设问":
        body = (
            f"// 设问示例（风格仿408）: {topic}\n"
            "// 「若…则是否可能发生缺页？」「给出调度序列计算平均等待时间」\n"
            "// 「说明为何满足四个条件仍可能不死锁」等。"
        )
        hint = "用真题句式自问自答，训练审题中的隐藏条件。"
    else:  # 自测要点
        body = (
            f"// 自测: {topic}\n"
            "// 闭卷能否默写: 定义 + 关键性质 + 1个例子？\n"
            "// 能否在空白纸上画出相关示意图或表格头？"
        )
        hint = "每章结束用 10 分钟过一遍本组自测，不会的回到王道对应小节。"
    return angle, hint, body


def render_html(num: int, chapter: str, topic: str, angle: str) -> str:
    title = f"408OS-{topic}-{angle}"
    safe_title = html.escape(title)
    h_ch = html.escape(chapter)
    h_topic = html.escape(topic)
    h_angle = html.escape(angle)
    a, hint, code_text = angle_block(chapter, topic, angle)
    h_a = html.escape(a)
    h_hint = html.escape(hint)
    h_code = html.escape(code_text)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
{STYLE}
    </style>
</head>
<body>
    <div class="card">
        <h2>📘 {safe_title}<span class="tag408">408操作系统</span></h2>
        <p>全国硕士研究生招生考试计算机学科专业基础综合（408）— <strong>操作系统</strong>专项笔记。</p>
        <p><span class="syllabus">考纲章节：</span>{h_ch}　｜　<strong>知识点：</strong>{h_topic}</p>
    </div>
    <div class="card">
        <h2>📌 {h_a}</h2>
        <p>{h_hint}</p>
        <div class="code">{h_code}</div>
        <div class="note">本篇编号 <strong>{num}</strong>，与手写精读 <code>201–207</code>、导览 <code>200</code> 配套使用；教材建议以王道《操作系统考研复习指导》章节为主对照补充。</div>
    </div>
    <div class="card">
        <h2>🔗 408 四门协同提示</h2>
        <div class="code">// 操作系统 × 计组：Cache、TLB、缺页开销、DMA
// 操作系统 × 数据结构：B树、文件索引、目录检索
// 操作系统 × 计算机网络：套接字缓冲、IO模型（408网络部分单独复习）</div>
    </div>
</body>
</html>
"""


def main() -> None:
    n_topics = verify_topic_count()
    if n_topics * len(ANGLES) != TOTAL_EXT:
        raise SystemExit(f"知识点数×角度应等于{TOTAL_EXT}，当前为 {n_topics}×{len(ANGLES)}={n_topics*len(ANGLES)}")

    removed = clean_generated_html()
    print(f"Removed {removed} old html (kept {len(KEEP_HTML)} manual pages).")

    idx = 0
    for chapter, subs in CHAPTER_TOPICS:
        for topic in subs:
            for angle in ANGLES:
                num = START_NUM + idx
                if num > 799:
                    raise SystemExit("index overflow")
                stem = safe_stem(f"{topic}-{angle}")
                fname = f"{num}-{stem}.html"
                path = os.path.join(OUT_DIR, fname)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(render_html(num, chapter, topic, angle))
                idx += 1

    if idx != TOTAL_EXT:
        raise SystemExit(f"expected {TOTAL_EXT} files, wrote {idx}")

    final = sum(1 for n in os.listdir(OUT_DIR) if n.endswith(".html"))
    print(f"Wrote {idx} 408 OS pages (numbers {START_NUM}-799). Total html in folder: {final}")


if __name__ == "__main__":
    main()
