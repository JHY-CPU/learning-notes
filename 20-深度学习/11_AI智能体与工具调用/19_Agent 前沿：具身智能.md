# 19_Agent 前沿：具身智能

## 1. 具身智能概述

具身智能（Embodied AI）是将 AI Agent 从**数字世界延伸到物理世界**的方向，让 Agent 通过机器人等实体与真实环境交互。

```
数字 Agent vs 具身 Agent：

数字 Agent:                具身 Agent:
  用户指令                   用户指令
     ↓                         ↓
  LLM 推理                  LLM 推理
     ↓                         ↓
  工具调用                   动作序列
  (API/代码)                (移动/抓取/操作)
     ↓                         ↓
  数字结果                   物理世界变化
  (文本/数据)                (位置/状态)

核心区别：
- 数字世界：操作可逆、无物理约束、精确控制
- 物理世界：操作不可逆、受物理规律约束、存在感知噪声
```

## 2. 具身 Agent 架构

```
┌─────────────────────────────────────────────────┐
│              具身 Agent 架构                      │
│                                                  │
│  ┌──────────┐     ┌──────────┐                  │
│  │  任务理解  │────→│  任务规划  │                 │
│  │  (LLM)   │     │ (CoT/ToT)│                  │
│  └──────────┘     └────┬─────┘                  │
│                        │                        │
│                    ┌───▼────┐                    │
│                    │ 动作生成 │                   │
│                    │(Motor   │                   │
│                    │ Control)│                   │
│                    └───┬────┘                    │
│                        │                        │
│  ┌──────────┐     ┌───▼────┐                    │
│  │  感知系统  │←────│ 环境交互 │                   │
│  │(Vision/  │     │(Robot) │                    │
│  │ Sensors) │     └────────┘                    │
│  └────┬─────┘                                   │
│       │                                        │
│  ┌────▼─────┐                                  │
│  │  世界模型  │ ← 环境的内部表示                   │
│  └──────────┘                                  │
└─────────────────────────────────────────────────┘
```

## 3. 视觉-语言-动作模型 (VLA)

```python
class VisionLanguageActionModel:
    """VLA 模型：统一处理视觉、语言和动作"""

    def __init__(self, model):
        self.model = model  # 如 RT-2, PaLM-E 等

    def process(self, instruction: str, image: np.ndarray) -> list[float]:
        """从指令和图像生成动作"""
        # 多模态输入
        inputs = {
            "text": f"Task: {instruction}\nWhat action should I take?",
            "image": self.preprocess(image)
        }

        # 模型推理
        output = self.model.generate(inputs)

        # 输出为 7-DoF 动作向量
        # [x, y, z, roll, pitch, yaw, gripper]
        return output["action"]

class SayCanAgent:
    """SayCan: 用 LLM 进行任务规划，用价值函数评估可行性"""

    def __init__(self, llm, value_functions: dict, skills: dict):
        self.llm = llm
        self.value_funcs = value_functions  # 技能 -> 可行性分数
        self.skills = skills

    def plan(self, instruction: str, observation: dict) -> str:
        # LLM: 哪些技能对任务有帮助（affordance）
        helpfulness = self.llm.generate(f"""
任务: {instruction}
可用技能: {list(self.skills.keys())}

为每个技能评分 (0-1)，表示它对完成任务的帮助程度。
输出 JSON: {{"skill_name": score, ...}}
""")
        helpfulness_scores = json.loads(helpfulness)

        # 价值函数: 哪些技能在当前状态下可行
        affordance_scores = {}
        for skill_name in self.skills:
            affordance_scores[skill_name] = self.value_funcs[skill_name](observation)

        # 综合评分: 帮助性 × 可行性
        combined = {}
        for skill in self.skills:
            combined[skill] = (
                helpfulness_scores.get(skill, 0) *
                affordance_scores.get(skill, 0)
            )

        # 选择最佳技能
        best_skill = max(combined, key=combined.get)
        return best_skill
```

## 4. 机器人操作任务

```python
class RobotManipulationAgent:
    """机器人操作 Agent"""

    def __init__(self, llm, vision_model, controller):
        self.llm = llm
        self.vision = vision_model
        self.controller = controller  # 机器人控制接口

    def pick_and_place(self, instruction: str) -> dict:
        """抓取与放置任务"""
        # 1. 理解任务
        parsed = self.parse_instruction(instruction)
        # "把红色杯子放到桌子上" -> object="红色杯子", target="桌子上"

        # 2. 视觉感知
        rgb, depth = self.controller.get_camera_data()
        objects = self.vision.detect_objects(rgb)
        target_loc = self.vision.locate_target(rgb, parsed["target"])

        # 3. 规划抓取
        obj = self.find_object(objects, parsed["object"])
        grasp_pose = self.plan_grasp(obj, depth)

        # 4. 执行
        self.controller.move_to(grasp_pose)
        self.controller.close_gripper()
        self.controller.move_to(target_loc)
        self.controller.open_gripper()

        return {"success": True, "action": f"将 {parsed['object']} 放到 {parsed['target']}"}

    def plan_grasp(self, object_info: dict, depth: np.ndarray) -> np.ndarray:
        """规划抓取位姿"""
        # 基于物体位置和深度信息计算抓取点
        center = object_info["center_3d"]
        orientation = object_info.get("orientation", 0)

        # 生成抓取位姿 [x, y, z, roll, pitch, yaw]
        grasp_pose = np.array([
            center[0], center[1], center[2] + 0.05,  # 略高于物体
            0, np.pi, orientation  # 从上方抓取
        ])
        return grasp_pose
```

## 5. 导航 Agent

```python
class NavigationAgent:
    """室内导航 Agent"""

    def __init__(self, llm, slam, controller):
        self.llm = llm
        self.slam = slam  # 同步定位与建图
        self.controller = controller

    def navigate(self, instruction: str) -> dict:
        """自然语言导航"""
        # 1. 目标理解
        goal = self.parse_goal(instruction)
        # "去厨房拿一瓶水" -> subgoals=["导航到厨房", "找到水", "拿起水"]

        for subgoal in goal["subgoals"]:
            # 2. 获取当前位置和地图
            current_pose = self.slam.get_pose()
            map_data = self.slam.get_map()

            # 3. 路径规划
            if subgoal["type"] == "navigate_to":
                target = self.locate_room(subgoal["target"], map_data)
                path = self.plan_path(current_pose, target, map_data)

                for waypoint in path:
                    self.controller.move_to(waypoint)

            elif subgoal["type"] == "find_object":
                obj = self.search_object(subgoal["object"])
                if obj:
                    self.controller.move_to(obj["approach_pose"])

        return {"success": True}

    def search_object(self, object_name: str) -> dict | None:
        """在环境中搜索物体"""
        # 360度扫描
        for angle in range(0, 360, 30):
            self.controller.rotate(angle)
            rgb, _ = self.controller.get_camera_data()

            detections = self.vision.detect_objects(rgb)
            for det in detections:
                if self.match_object(det, object_name):
                    return det

        return None
```

## 6. 仿真环境

```python
# Habitat-Sim: 3D 仿真环境
import habitat_sim

class SimulationEnvironment:
    """具身智能仿真环境"""

    def __init__(self, scene_path: str):
        self.config = habitat_sim.Configuration(
            backend_cfg=habitat_sim.SimulatorConfiguration(
                scene_id=scene_path
            ),
            agent_cfg=habitat_sim.AgentConfiguration(
                sensor_specifications=[
                    habitat_sim.CameraSensorSpec(
                        uuid="rgb",
                        sensor_type=habitat_sim.SensorType.COLOR,
                        resolution=[512, 512]
                    ),
                    habitat_sim.CameraSensorSpec(
                        uuid="depth",
                        sensor_type=habitat_sim.SensorType.DEPTH,
                        resolution=[512, 512]
                    )
                ]
            )
        )
        self.sim = habitat_sim.Simulator(self.config)

    def step(self, action: str) -> dict:
        obs = self.sim.step(action)
        return {
            "rgb": obs["rgb"],
            "depth": obs["depth"],
            "position": self.sim.get_agent(0).state.position.tolist(),
        }

# ALFWorld: 文字冒险 + 具身任务
class ALFWorldAgent:
    """ALFWorld 文字具身 Agent"""

    def play(self, task_description: str) -> str:
        obs = self.env.reset()

        for step in range(50):
            action = self.llm.generate(f"""
任务: {task_description}
当前观察: {obs}
可用动作: go to <loc>, take <obj>, put <obj> on <loc>, open <loc>, ...

下一步动作:
""")
            obs, reward, done, info = self.env.step(action)

            if done:
                return "任务完成" if reward > 0 else "任务失败"

        return "超时"
```

## 7. 前沿研究方向

```
具身智能前沿方向：

1. 世界模型 (World Model)
   - 学习环境的内部表示
   - 预测行动的后果
   - 代表: UniSim, GAIA-1

2. 模仿学习 (Imitation Learning)
   - 从人类演示学习技能
   - 代表: RT-2, Octo

3. 强化学习 + LLM
   - LLM 进行高层规划
   - RL 进行低层控制
   - 代表: ELLM, SayCan

4. 多模态基础模型
   - 统一视觉、语言、动作
   - 代表: PaLM-E, Gato

5. 安全与鲁棒性
   - 物理世界中的安全保障
   - 感知噪声下的鲁棒决策
```

## 8. 挑战与展望

```
具身智能核心挑战：

1. Sim-to-Real Gap
   仿真中训练的策略难以直接迁移到真实世界
   → 域随机化、系统辨识、少量真实数据微调

2. 长程任务规划
   物理世界任务往往需要数十步操作
   → 层级规划、技能库、记忆系统

3. 开放世界泛化
   需要处理训练中未见过的物体和场景
   → 大规模预训练、零样本迁移

4. 安全性
   物理世界操作不可逆，错误可能造成伤害
   → 约束优化、安全层、人类监督

展望：
- 2024-2025: 实验室环境下的简单操作任务
- 2025-2027: 特定场景（仓库、厨房）的有限自主
- 2027+: 通用家庭/办公环境的机器人助手
```

## 总结

具身智能是 Agent 从**虚拟到物理**的关键跨越。核心技术包括 VLA 模型、视觉感知、运动规划和仿真训练。虽然面临 Sim-to-Real、安全性等重大挑战，但随着多模态基础模型的进步，具身智能正在加速走向实用化。
