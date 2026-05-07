# 11_Code Agent 与代码生成执行

## 1. Code Agent 概述

Code Agent 是专注于**代码生成、理解和执行**的 AI Agent，代表产品包括 GitHub Copilot Workspace、Cursor、Devin 等。

```
Code Agent 能力谱系：

代码补全 ─→ 代码生成 ─→ 代码理解 ─→ 自主编程
(Ghost)    (Copilot)   (Chat)     (Devin)
  │           │          │          │
  └─── 被动 ──┴── 交互 ──┴── 自主 ──┘
```

## 2. 代码解释器 (Code Interpreter)

```python
import subprocess
import tempfile
import os

class CodeInterpreter:
    """安全执行代码的沙箱环境"""

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory = max_memory_mb

    def execute_python(self, code: str) -> dict:
        """在沙箱中执行 Python 代码"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.mkdtemp()  # 隔离工作目录
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"执行超时 ({self.timeout}s)",
                "return_code": -1,
                "success": False
            }
        finally:
            os.unlink(temp_path)

class DockerSandbox:
    """Docker 容器沙箱（更强隔离）"""

    def execute(self, code: str, language: str = "python") -> dict:
        image_map = {
            "python": "python:3.11-slim",
            "node": "node:20-slim",
            "java": "openjdk:17-slim",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{language}", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--memory", "512m",
                "--cpus", "1",
                "--network", "none",  # 禁用网络
                "-v", f"{temp_path}:/code/main.{language}:ro",
                image_map[language],
                "sh", "-c", f"cd /tmp && cp /code/main.{language} . && {language} main.{language}"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        return {"stdout": result.stdout, "stderr": result.stderr}
```

## 3. 代码生成与调试循环

```python
class CodingAgent:
    """生成-执行-调试循环"""

    def __init__(self, llm, interpreter: CodeInterpreter):
        self.llm = llm
        self.interpreter = interpreter
        self.max_retries = 3

    def generate_and_run(self, task: str) -> dict:
        """生成代码并自动调试"""

        for attempt in range(self.max_retries):
            # 1. 生成代码
            if attempt == 0:
                code = self.generate_code(task)
            else:
                code = self.fix_code(task, code, result)

            print(f"[尝试 {attempt + 1}] 生成代码:\n{code}\n")

            # 2. 安全检查
            safety_check = self.check_safety(code)
            if not safety_check["safe"]:
                return {"error": f"安全检查失败: {safety_check['reason']}"}

            # 3. 执行
            result = self.interpreter.execute_python(code)

            # 4. 检查是否成功
            if result["success"]:
                return {
                    "code": code,
                    "output": result["stdout"],
                    "attempts": attempt + 1
                }

            # 5. 分析错误
            print(f"[错误] {result['stderr']}")

        return {
            "code": code,
            "error": result["stderr"],
            "attempts": self.max_retries,
            "failed": True
        }

    def generate_code(self, task: str) -> str:
        return self.llm.generate(f"""
根据以下需求生成 Python 代码。只输出代码，不要解释。

需求: {task}

要求：
1. 代码可直接运行
2. 包含必要的 import
3. 包含测试/验证代码
4. 有适当的错误处理
""")

    def fix_code(self, task: str, code: str, error: dict) -> str:
        return self.llm.generate(f"""
代码执行出错，请修复。

原始需求: {task}
代码:
```python
{code}
```

错误信息:
stdout: {error['stdout']}
stderr: {error['stderr']}

请输出修复后的完整代码。
""")

    def check_safety(self, code: str) -> dict:
        """检查代码安全性"""
        dangerous_patterns = [
            r"os\.system",
            r"subprocess\.call.*shell=True",
            r"exec\(",
            r"eval\(",
            r"__import__",
            r"open\(.+[\'\"]w[\'\"]",  # 写文件
            r"shutil\.rmtree",
            r"rm\s+-rf",
            r"import\s+socket",
            r"requests\.(get|post)",  # 网络请求
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return {"safe": False, "reason": f"检测到危险模式: {pattern}"}

        return {"safe": True}
```

## 4. 代码理解 Agent

```python
class CodeUnderstandingAgent:
    """理解和分析代码库"""

    def analyze_file(self, file_path: str) -> dict:
        with open(file_path) as f:
            code = f.read()

        return {
            "structure": self.extract_structure(code),
            "dependencies": self.extract_dependencies(code),
            "complexity": self.calculate_complexity(code),
            "summary": self.summarize_code(code),
        }

    def extract_structure(self, code: str) -> dict:
        """提取代码结构（类、函数、变量）"""
        import ast
        tree = ast.parse(code)

        structure = {"classes": [], "functions": [], "imports": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "name": node.name,
                    "methods": [
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ],
                    "line": node.lineno
                })
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "name": node.name,
                    "args": [a.arg for a in node.args.args],
                    "line": node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                structure["imports"].append(
                    ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                )

        return structure

    def explain_code(self, code_snippet: str, context: str = "") -> str:
        """解释代码片段的功能"""
        return self.llm.generate(f"""
请详细解释以下代码的功能、逻辑和设计思路：

{f"上下文: {context}" if context else ""}

```python
{code_snippet}
```

请包括：
1. 整体功能概述
2. 关键逻辑解释
3. 输入输出说明
4. 潜在问题或改进点
""")
```

## 5. 自动测试生成

```python
class TestGeneratorAgent:
    """自动生成单元测试"""

    def generate_tests(self, code: str, framework: str = "pytest") -> str:
        structure = self.analyze_code_structure(code)

        tests = []

        for func in structure["functions"]:
            test_code = self.llm.generate(f"""
为以下函数生成 {framework} 单元测试：

函数: {func['name']}
参数: {func['args']}
代码:
```python
{self.get_function_code(code, func['name'])}
```

要求：
1. 覆盖正常输入
2. 覆盖边界条件
3. 覆盖异常情况
4. 使用 assert 验证结果
5. 使用 {framework} 的 fixture 和 parametrize
""")
            tests.append(test_code)

        return "\n\n".join(tests)

    def improve_coverage(self, code: str, existing_tests: str,
                         coverage_report: str) -> str:
        """根据覆盖率报告补充测试"""
        return self.llm.generate(f"""
代码:
```python
{code}
```

现有测试:
```python
{existing_tests}
```

覆盖率报告:
{coverage_report}

请补充测试，覆盖未覆盖的代码路径。
""")
```

## 6. 代码重构 Agent

```python
class RefactoringAgent:
    """智能代码重构"""

    def refactor(self, code: str, goal: str = "improve_readability") -> dict:
        """执行代码重构"""
        # 1. 分析代码问题
        analysis = self.analyze_issues(code)

        # 2. 生成重构方案
        plan = self.generate_refactoring_plan(code, analysis, goal)

        # 3. 逐步执行重构
        refactored = code
        changes = []

        for step in plan["steps"]:
            new_code = self.apply_refactoring(refactored, step)

            # 4. 验证重构正确性
            if self.verify_refactoring(refactored, new_code):
                refactored = new_code
                changes.append(step)
            else:
                print(f"重构步骤失败: {step['description']}")

        return {
            "original": code,
            "refactored": refactored,
            "changes": changes,
            "improvement_score": self.score_improvement(code, refactored)
        }

    def analyze_issues(self, code: str) -> list[dict]:
        """分析代码质量问题"""
        return self.llm.generate(f"""
分析以下代码的质量问题：

```python
{code}
```

检查以下方面：
1. 代码重复
2. 过长函数
3. 复杂嵌套
4. 命名不佳
5. 缺少类型注解
6. 缺少文档

输出 JSON 数组，每个问题包含: category, description, line, severity
""")
```

## 7. 安全考量

```
Code Agent 安全风险矩阵：

┌──────────────┬─────────────┬──────────────────────────┐
│ 风险类型     │ 严重程度     │ 防护措施                  │
├──────────────┼─────────────┼──────────────────────────┤
│ 代码注入     │ 高          │ 沙箱执行 + 输入过滤       │
│ 数据泄露     │ 高          │ 网络隔离 + 输出审查       │
│ 资源耗尽     │ 中          │ 超时 + 内存限制           │
│ 文件系统访问 │ 中          │ 只读挂载 + 白名单路径     │
│ 依赖攻击     │ 中          │ 锁定依赖 + 安全扫描       │
│ 权限提升     │ 高          │ 非 root 运行 + seccomp   │
└──────────────┴─────────────┴──────────────────────────┘
```

## 总结

Code Agent 将 LLM 的代码能力从**辅助补全**提升到**自主生成-执行-调试循环**。核心技术包括安全沙箱执行、自动错误修复和代码理解。安全是 Code Agent 的第一要务 -- 所有执行必须在隔离环境中进行，且有严格的资源和权限限制。
