# 12_Web Agent 与浏览器自动化

## 1. Web Agent 概述

Web Agent 是能够在**浏览器环境中自主导航、理解和操作网页**的 AI Agent，执行搜索、数据提取、表单填写等任务。

```
Web Agent 技术栈：

自然语言指令 ─→ 任务规划 ─→ 网页理解 ─→ 操作执行 ─→ 结果提取
                    │           │           │
                 ReAct      DOM/VLM      Click/Type
                             分析         Scroll
```

## 2. 网页理解与表示

### 2.1 DOM 树解析

```python
from bs4 import BeautifulSoup
import json

class WebPageParser:
    """解析网页为 Agent 可理解的结构"""

    def parse(self, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        # 提取可交互元素
        interactive_elements = self.extract_interactive(soup)

        # 构建简化 DOM
        simplified = self.simplify_dom(soup)

        return {
            "title": soup.title.string if soup.title else "",
            "text_content": self.extract_text(soup),
            "interactive_elements": interactive_elements,
            "simplified_dom": simplified,
        }

    def extract_interactive(self, soup) -> list[dict]:
        """提取所有可交互元素"""
        elements = []

        # 链接
        for i, a in enumerate(soup.find_all("a", href=True)):
            elements.append({
                "id": f"link_{i}",
                "type": "link",
                "text": a.get_text(strip=True)[:100],
                "href": a["href"],
                "tag": "a"
            })

        # 按钮
        for i, btn in enumerate(soup.find_all(["button", "input"])):
            if btn.name == "input" and btn.get("type") not in ["submit", "button", "reset"]:
                continue
            elements.append({
                "id": f"btn_{i}",
                "type": "button",
                "text": btn.get_text(strip=True) or btn.get("value", ""),
                "tag": btn.name
            })

        # 输入框
        for i, inp in enumerate(soup.find_all("input")):
            if inp.get("type") in ["hidden", "submit", "button"]:
                continue
            elements.append({
                "id": f"input_{i}",
                "type": "input",
                "input_type": inp.get("type", "text"),
                "name": inp.get("name", ""),
                "placeholder": inp.get("placeholder", ""),
                "tag": "input"
            })

        return elements

    def simplify_dom(self, soup, max_depth: int = 5) -> str:
        """生成简化版 DOM 用于 LLM 理解"""
        result = []
        self._traverse(soup.body or soup, result, depth=0, max_depth=max_depth)
        return "\n".join(result)

    def _traverse(self, element, result, depth, max_depth):
        if depth > max_depth:
            return

        if element.name:
            attrs = {}
            for attr in ["id", "class", "href", "type", "name", "placeholder"]:
                val = element.get(attr)
                if val:
                    attrs[attr] = val if isinstance(val, str) else " ".join(val)

            text = element.string
            if text:
                text = text.strip()[:50]

            indent = "  " * depth
            attrs_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            tag_str = f"<{element.name} {attrs_str}>"
            if text:
                tag_str += f" {text}"

            result.append(f"{indent}{tag_str}")

            for child in element.children:
                self._traverse(child, result, depth + 1, max_depth)
```

### 2.2 基于视觉的网页理解 (VLM)

```python
class VisualWebAgent:
    """使用视觉语言模型理解网页截图"""

    def __init__(self, vlm):
        self.vlm = vlm  # 视觉语言模型

    def analyze_screenshot(self, screenshot_path: str,
                           task: str) -> dict:
        """分析网页截图"""
        response = self.vlm.generate(
            image=screenshot_path,
            prompt=f"""
分析这个网页截图。

当前任务: {task}

请描述：
1. 页面主要内容
2. 可交互元素的位置和功能
3. 为完成任务，下一步应该点击哪里或输入什么？

输出格式:
{{"page_description": "...", "action": {{"type": "click/input/scroll", "target": "元素描述", "value": "输入值(可选)"}}}}
"""
        )
        return json.loads(response)
```

## 3. 浏览器操作

### 3.1 使用 Playwright 自动化

```python
from playwright.async_api import async_playwright

class BrowserController:
    """浏览器控制层"""

    def __init__(self):
        self.browser = None
        self.page = None

    async def start(self):
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

    async def navigate(self, url: str) -> str:
        await self.page.goto(url, wait_until="domcontentloaded")
        return await self.page.content()

    async def click(self, selector: str) -> str:
        await self.page.click(selector)
        await self.page.wait_for_load_state("domcontentloaded")
        return await self.page.content()

    async def type_text(self, selector: str, text: str) -> str:
        await self.page.fill(selector, text)
        return await self.page.content()

    async def scroll(self, direction: str = "down", amount: int = 500):
        delta = amount if direction == "down" else -amount
        await self.page.mouse.wheel(0, delta)

    async def get_screenshot(self) -> bytes:
        return await self.page.screenshot()

    async def extract_text(self) -> str:
        return await self.page.inner_text("body")
```

### 3.2 操作序列定义

```python
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SELECT = "select"
    WAIT = "wait"
    EXTRACT = "extract"
    SCREENSHOT = "screenshot"

@dataclass
class WebAction:
    action_type: ActionType
    target: str  # CSS 选择器或描述
    value: str = ""
    description: str = ""

class ActionExecutor:
    """执行网页操作"""

    async def execute(self, action: WebAction, browser: BrowserController) -> dict:
        try:
            match action.action_type:
                case ActionType.NAVIGATE:
                    html = await browser.navigate(action.value)
                    return {"success": True, "html": html[:2000]}

                case ActionType.CLICK:
                    html = await browser.click(action.target)
                    return {"success": True, "html": html[:2000]}

                case ActionType.TYPE:
                    await browser.type_text(action.target, action.value)
                    return {"success": True}

                case ActionType.SCROLL:
                    await browser.scroll(action.value, int(action.target))
                    return {"success": True}

                case ActionType.EXTRACT:
                    text = await browser.extract_text()
                    return {"success": True, "text": text[:5000]}

                case ActionType.SCREENSHOT:
                    img = await browser.get_screenshot()
                    return {"success": True, "screenshot": img}

        except Exception as e:
            return {"success": False, "error": str(e)}
```

## 4. Web Agent 完整实现

```python
class WebAgent:
    """完整的 Web Agent"""

    def __init__(self, llm, browser: BrowserController, max_steps: int = 15):
        self.llm = llm
        self.browser = browser
        self.max_steps = max_steps
        self.history = []

    async def run(self, task: str, start_url: str = None) -> str:
        if start_url:
            await self.browser.navigate(start_url)

        for step in range(self.max_steps):
            # 1. 获取当前页面状态
            page_content = await self.browser.extract_text()
            current_url = self.browser.page.url

            # 2. LLM 决策
            decision = await self.decide(task, current_url, page_content)

            if decision["action"] == "finish":
                return decision["result"]

            # 3. 执行操作
            action = WebAction(
                action_type=ActionType(decision["action"]),
                target=decision.get("selector", ""),
                value=decision.get("value", ""),
                description=decision.get("thought", "")
            )

            result = await ActionExecutor().execute(action, self.browser)
            self.history.append({"step": step, "action": action, "result": result})

            print(f"[步骤 {step + 1}] {action.description}")
            print(f"  操作: {action.action_type.value} {action.target}")
            print(f"  结果: {'成功' if result['success'] else result.get('error')}")

        return "达到最大步数限制"

    async def decide(self, task: str, url: str, page_content: str) -> dict:
        """LLM 决策下一步操作"""
        history_summary = "\n".join(
            f"步骤 {h['step']}: {h['action'].description} -> {'成功' if h['result']['success'] else '失败'}"
            for h in self.history[-5:]
        )

        prompt = f"""
你是一个网页操作助手，通过浏览器完成用户任务。

当前任务: {task}
当前 URL: {url}
页面内容 (前2000字): {page_content[:2000]}

历史操作:
{history_summary}

可选操作：
- navigate: 导航到新 URL (需要 value: URL)
- click: 点击元素 (需要 target: CSS 选择器)
- type: 输入文本 (需要 target: 选择器, value: 文本)
- scroll: 滚动页面 (需要 value: "up"/"down")
- extract: 提取页面信息
- finish: 完成任务 (需要 result: 最终结果)

输出 JSON:
{{"thought": "思考下一步", "action": "操作类型", "target": "选择器", "value": "值", "result": "最终结果(仅finish)"}}
"""
        response = self.llm.generate(prompt)
        return json.loads(response)
```

## 5. 网页数据提取

```python
class WebScraperAgent:
    """智能数据提取 Agent"""

    async def extract_structured_data(self, url: str,
                                       schema: dict) -> dict:
        """从网页提取结构化数据"""
        # 导航到页面
        await self.browser.navigate(url)
        html = await self.browser.page.content()

        # LLM 指导提取
        extraction_plan = self.llm.generate(f"""
从以下 HTML 中提取数据，目标 Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

HTML (简化):
{self.simplify_html(html)[:5000]}

请提供：
1. 数据的 CSS 选择器或 XPath
2. 提取逻辑
输出 JSON。
""")

        # 执行提取
        data = await self.execute_extraction(extraction_plan)

        # 验证和清洗
        return self.validate_and_clean(data, schema)

    def simplify_html(self, html: str) -> str:
        """简化 HTML，移除样式和脚本"""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        return str(soup)
```

## 6. Web Agent 挑战与解决

```
常见挑战与解决方案：

1. 动态内容加载
   → 等待策略: wait_for_selector / wait_for_load_state
   → 重试机制

2. CAPTCHA 验证
   → 验证码识别服务
   → 人工介入断点

3. 反爬虫检测
   → 随机延迟 + 用户代理轮换
   → 使用真实浏览器指纹

4. 页面结构变化
   → 多种选择器策略 (CSS > XPath > 文本匹配)
   → 自愈选择器: 用 LLM 重新定位元素

5. 登录和认证
   → Cookie 持久化
   → OAuth 令牌管理
```

## 总结

Web Agent 将 LLM 的理解能力与浏览器自动化结合，核心挑战是**网页理解和可靠操作**。DOM 解析适合结构化页面，VLM 方法适合复杂视觉布局。实践中需要处理动态内容、反爬虫和认证等问题。
