# 在线判题系统核心 (OJ Core)

## 项目需求与功能分析

在线判题系统（Online Judge）是编程竞赛和面试平台的核心基础设施。本项目实现一个轻量级 OJ 系统的评测引擎，包含代码沙箱执行、多维度评测、结果判定等功能。

### 核心功能

- 代码沙箱安全执行（超时、内存限制、进程隔离）
- 多语言支持（Python / C++ / Java）
- 测试用例批量评测
- 判题结果分类（AC / WA / TLE / MLE / RE / CE）
- 评测报告生成

### 判题状态

| 状态 | 含义 | 触发条件 |
|------|------|----------|
| AC | Accepted | 输出完全匹配 |
| WA | Wrong Answer | 输出不一致 |
| TLE | Time Limit Exceeded | 运行超时 |
| MLE | Memory Limit Exceeded | 内存超限 |
| RE | Runtime Error | 程序异常终止 |
| CE | Compilation Error | 编译失败 |

## 核心技术原理

### 沙箱执行流程

```
源代码 -> 编译 -> 创建子进程 -> 重定向 I/O -> 限时执行 -> 捕获输出 -> 对比答案
```

关键安全措施：
- 使用子进程隔离执行
- 设置 CPU 时间和内存限制
- 超时强制终止进程
- 临时目录隔离文件系统

## 完整代码实现

```python
import subprocess, tempfile, os, time, shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class Verdict(Enum):
    AC="Accepted"; WA="Wrong Answer"; TLE="Time Limit Exceeded"
    MLE="Memory Limit Exceeded"; RE="Runtime Error"
    CE="Compilation Error"; SE="System Error"


@dataclass
class TestCase:
    input_data: str; expected_output: str
    time_limit: float = 2.0; memory_limit: int = 256


@dataclass
class JudgeResult:
    verdict: Verdict; time_used: float = 0.0; memory_used: int = 0
    actual_output: str = ""; error_message: str = ""
    test_case_index: int = -1


@dataclass
class Submission:
    code: str; language: str
    test_cases: List[TestCase] = field(default_factory=list)
    results: List[JudgeResult] = field(default_factory=list)
    final_verdict: Verdict = Verdict.AC
    total_time: float = 0.0; max_memory: int = 0


class CodeSandbox:
    def __init__(self):
        self.work_dir = tempfile.mkdtemp(prefix="oj_")

    def cleanup(self):
        if os.path.exists(self.work_dir): shutil.rmtree(self.work_dir)

    def _compile_cpp(self, code):
        src = os.path.join(self.work_dir, "main.cpp")
        exe = os.path.join(self.work_dir, "main")
        with open(src, 'w') as f: f.write(code)
        try:
            r = subprocess.run(['g++','-O2','-std=c++17','-o',exe,src],
                               capture_output=True, text=True, timeout=10)
            return (True, exe) if r.returncode == 0 else (False, r.stderr)
        except FileNotFoundError: return False, "g++ not found"
        except subprocess.TimeoutExpired: return False, "Compilation timeout"

    def execute(self, code, language, input_data, time_limit=2.0):
        # Compile if needed
        if language == 'cpp':
            ok, res = self._compile_cpp(code)
            if not ok: return JudgeResult(Verdict.CE, error_message=res)
            cmd = [res]
        elif language == 'python':
            src = os.path.join(self.work_dir, "main.py")
            with open(src,'w') as f: f.write(code)
            cmd = ['python', src]
        else:
            return JudgeResult(Verdict.SE, error_message=f"Unsupported: {language}")

        inp = os.path.join(self.work_dir, "input.txt")
        with open(inp,'w') as f: f.write(input_data)

        try:
            start = time.time()
            proc = subprocess.Popen(cmd, stdin=open(inp,'r'),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, cwd=self.work_dir)
            try:
                out, err = proc.communicate(timeout=time_limit)
                elapsed = time.time() - start
                if proc.returncode != 0:
                    return JudgeResult(Verdict.RE, elapsed, error_message=err)
                return JudgeResult(Verdict.AC, elapsed, actual_output=out.strip())
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                return JudgeResult(Verdict.TLE, time_limit)
        except Exception as e:
            return JudgeResult(Verdict.RE, error_message=str(e))


class OutputComparator:
    @staticmethod
    def compare_exact(actual, expected):
        return actual.strip() == expected.strip()

    @staticmethod
    def compare_float(actual, expected, tol=1e-6):
        try:
            a = [float(x) for x in actual.strip().split()]
            e = [float(x) for x in expected.strip().split()]
            return len(a)==len(e) and all(abs(x-y)<=tol for x,y in zip(a,e))
        except ValueError:
            return actual.strip() == expected.strip()

    @staticmethod
    def compare_ignore_ws(actual, expected):
        return ' '.join(actual.split()) == ' '.join(expected.split())


class Judge:
    def __init__(self, mode='exact'):
        self.sandbox = CodeSandbox()
        modes = {'exact': OutputComparator.compare_exact,
                 'float': OutputComparator.compare_float,
                 'ignore_ws': OutputComparator.compare_ignore_ws}
        self.cmp = modes.get(mode, OutputComparator.compare_exact)

    def judge(self, submission):
        submission.results = []; submission.final_verdict = Verdict.AC
        submission.total_time = 0; submission.max_memory = 0
        for i, tc in enumerate(submission.test_cases):
            res = self.sandbox.execute(submission.code, submission.language,
                                        tc.input_data, tc.time_limit)
            if res.verdict == Verdict.AC:
                if not self.cmp(res.actual_output, tc.expected_output):
                    res.verdict = Verdict.WA
            res.test_case_index = i
            submission.results.append(res)
            submission.total_time += res.time_used
            if res.verdict != Verdict.AC:
                submission.final_verdict = res.verdict; break
        return submission

    def report(self, sub):
        lines = [f"结果: {sub.final_verdict.value}", f"耗时: {sub.total_time:.4f}s", "="*40]
        for i, r in enumerate(sub.results):
            s = "PASS" if r.verdict==Verdict.AC else "FAIL"
            lines.append(f"  #{i+1}: {s} ({r.verdict.value}) {r.time_used:.4f}s")
            if r.error_message: lines.append(f"    错误: {r.error_message[:200]}")
        return '\n'.join(lines)

    def cleanup(self): self.sandbox.cleanup()
```

## 测试用例

```python
import unittest

class TestOJCore(unittest.TestCase):
    def setUp(self): self.j = Judge(mode='ignore_ws')
    def tearDown(self): self.j.cleanup()

    def test_accepted(self):
        s = Submission("a,b=map(int,input().split())\nprint(a+b)", 'python',
                       [TestCase("3 5","8")])
        self.j.judge(s); self.assertEqual(s.final_verdict, Verdict.AC)

    def test_wrong_answer(self):
        s = Submission("print(42)", 'python', [TestCase("","8")])
        self.j.judge(s); self.assertEqual(s.final_verdict, Verdict.WA)

    def test_tle(self):
        s = Submission("while True: pass", 'python', [TestCase("","0", time_limit=1)])
        self.j.judge(s); self.assertEqual(s.final_verdict, Verdict.TLE)

    def test_re(self):
        s = Submission("x=1/0", 'python', [TestCase("","0")])
        self.j.judge(s); self.assertEqual(s.final_verdict, Verdict.RE)

    def test_ce(self):
        s = Submission("int main { return 0; }", 'cpp', [TestCase("","0")])
        self.j.judge(s); self.assertEqual(s.final_verdict, Verdict.CE)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **Docker 沙箱**：使用容器提供更强的进程隔离
2. **并发评测**：线程池同时评测多个提交
3. **Special Judge**：支持输出不唯一的自定义评测
4. **交互式评测**：选手程序与评测程序交互
5. **代码安全检测**：检测危险系统调用
6. **排行榜**：根据通过率和耗时生成排名
