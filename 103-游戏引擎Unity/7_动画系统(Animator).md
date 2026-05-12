# 动画系统(Animator)

## 核心概念

Unity动画系统基于Animator Controller（状态机）驱动Animation Clip（动画片段）。Animator组件挂载到GameObject上，通过状态机控制动画的播放、过渡和混合。底层动画计算在C++线程执行，通过Animator组件暴露的C# API进行控制。

## 核心组件关系

```
Animation Clip (动画数据 - 关键帧信息)
    ↓ 引用
Animator Controller (状态机逻辑 - 状态和过渡)
    ↓ 挂载
Animator (组件 - 驱动GameObject播放动画)
    ↓ 使用
Avatar (骨骼映射 - 用于人形动画重定向)

Animation Clip内容:
├── Transform曲线 (位置/旋转/缩放随时间变化)
├── 材质属性曲线 (颜色/浮点值随时间变化)
├── 动画事件 (在特定帧触发回调)
└── Root Motion曲线 (角色位移数据)
```

## Animation Clip详解

动画片段存储对象属性随时间变化的数据。每个Clip由多条曲线(Curve)组成，每条曲线描述一个属性的动画。

```csharp
// 运行时播放AnimationClip（不经过Animator，使用旧Animation组件）
public class SimpleAnimation : MonoBehaviour
{
    [SerializeField] private AnimationClip clip;
    private Animation anim;

    void Start()
    {
        anim = gameObject.AddComponent<Animation>();
        anim.AddClip(clip, "MyClip");
        anim.Play("MyClip");
    }

    // 动画事件：在Clip中特定时间点调用
    void OnAnimationEvent(string eventName)
    {
        Debug.Log($"动画事件触发: {eventName}");
    }
}

// 运行时创建AnimationClip（程序化动画）
public class ProceduralAnimation : MonoBehaviour
{
    void Start()
    {
        var clip = new AnimationClip();

        // 创建位置X曲线
        AnimationCurve posXCurve = new AnimationCurve();
        posXCurve.AddKey(0f, 0f);      // 0秒时在x=0
        posXCurve.AddKey(1f, 5f);      // 1秒时在x=5
        posXCurve.AddKey(2f, 0f);      // 2秒时回到x=0

        // 设置切线使动画平滑
        for (int i = 0; i < posXCurve.keys.Length; i++)
            posXCurve.SmoothTangent(i, 0f);

        // 将曲线绑定到属性
        clip.SetCurve("", typeof(Transform), "localPosition.x", posXCurve);

        // 添加旋转曲线
        AnimationCurve rotYCurve = AnimationCurve.Linear(0f, 0f, 2f, 360f);
        clip.SetCurve("", typeof(Transform), "localRotation.y", rotYCurve);

        // 添加动画事件
        AnimationEvent evt = new AnimationEvent();
        evt.time = 1f;
        evt.functionName = "OnMiddleOfAnimation";
        evt.stringParameter = "hello";
        clip.AddEvent(evt);

        // 播放
        Animation anim = gameObject.AddComponent<Animation>();
        anim.AddClip(clip, "Procedural");
        anim.Play("Procedural");
    }

    void OnMiddleOfAnimation(string param)
    {
        Debug.Log($"动画中间事件: {param}");
    }
}
```

## Animator Controller深度解析

状态机是Animator的核心，管理动画状态和过渡条件：

```csharp
public class PlayerAnimator : MonoBehaviour
{
    private Animator animator;

    // Animator参数名（对应Controller中的Parameters）
    // 使用StringToHash缓存哈希值，避免每帧字符串查找
    private static readonly int Speed = Animator.StringToHash("Speed");
    private static readonly int SpeedX = Animator.StringToHash("SpeedX");
    private static readonly int SpeedY = Animator.StringToHash("SpeedY");
    private static readonly int IsGrounded = Animator.StringToHash("IsGrounded");
    private static readonly int IsCrouching = Animator.StringToHash("IsCrouching");
    private static readonly int Jump = Animator.StringToHash("Jump");
    private static readonly int Attack = Animator.StringToHash("Attack");
    private static readonly int Die = Animator.StringToHash("Die");

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        Vector2 input = new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
        float speed = input.magnitude;

        // ========== 设置Animator参数 ==========

        // Float参数：用于Blend Tree混合
        animator.SetFloat(Speed, speed);
        animator.SetFloat(SpeedX, input.x);
        animator.SetFloat(SpeedY, input.y);

        // Bool参数：用于状态切换条件
        animator.SetBool(IsGrounded, isGrounded);
        animator.SetBool(IsCrouching, isCrouching);

        // Trigger参数：一次性触发（如跳跃、攻击）
        // Trigger会自动重置，不需要手动清零
        if (Input.GetKeyDown(KeyCode.Space))
            animator.SetTrigger(Jump);
        if (Input.GetMouseButtonDown(0))
            animator.SetTrigger(Attack);

        // ========== 获取当前状态信息 ==========

        // 当前层的状态信息
        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);

        // 检查当前正在播放哪个动画状态
        if (stateInfo.IsName("Base Layer.Attack"))
        {
            Debug.Log("正在播放攻击动画");
        }

        if (stateInfo.IsTag("Locomotion"))
        {
            // 使用标签分组，比逐个检查状态名更灵活
            float normalizedTime = stateInfo.normalizedTime; // 动画进度 0-1
            float clipLength = stateInfo.length;             // 动画片段时长
        }

        // 检查是否在过渡中
        AnimatorTransitionInfo transInfo = animator.GetAnimatorTransitionInfo(0);
        if (transInfo.IsUserName("AttackToIdle"))
        {
            // 正在从攻击过渡到Idle
        }

        // 下一帧的状态信息（过渡后的目标状态）
        AnimatorStateInfo nextState = animator.GetNextAnimatorStateInfo(0);
    }

    // ========== 动画事件回调 ==========
    // 在AnimationClip中设置事件，指定函数名和参数

    public void OnAttackHit()
    {
        // 攻击动画中某一帧触发的事件
        // 用于: 攻击判定、播放音效、生成特效
        Debug.Log("攻击判定生效");
        PerformHitDetection();
    }

    public void OnFootStep()
    {
        // 脚步声事件
        AudioManager.PlayFootstep(transform.position);
    }

    public void OnComboWindow()
    {
        // 连击窗口开启事件
        comboManager.EnableComboInput();
    }

    // ========== 运行时控制 ==========

    void RuntimeControl()
    {
        // 暂停/恢复动画
        animator.speed = 0f; // 暂停
        animator.speed = 1f; // 正常速度
        animator.speed = 2f; // 2倍速

        // 强制跳转到指定状态
        animator.Play("Attack", 0, 0f);     // 从头播放攻击动画
        animator.Play("Attack", 0, 0.5f);   // 从中间开始播放

        // 使用CrossFade平滑过渡
        animator.CrossFade("Attack", 0.2f);  // 0.2秒过渡

        // 检查Animator Controller是否存在
        if (animator.runtimeAnimatorController != null)
        {
            // 安全操作
        }

        // 获取所有参数
        int paramCount = animator.parameterCount;
        for (int i = 0; i < paramCount; i++)
        {
            AnimatorControllerParameter param = animator.GetParameter(i);
            Debug.Log($"参数: {param.name}, 类型: {param.type}");
        }
    }
}
```

## Blend Tree（混合树）详解

Blend Tree用于平滑混合多个动画，是实现角色移动动画的关键技术：

```csharp
// 1D混合树：根据Speed参数混合Idle、Walk、Run
// 在Animator Controller中创建Blend Tree:
// Type: 1D
// Parameter: Speed (float)
// Motion列表:
//   - Idle    | Threshold: 0.0
//   - Walk    | Threshold: 0.3
//   - Run     | Threshold: 1.0
// Blend Type: 1D Linear (线性混合)

// 2D混合树：根据两个参数混合（如前后+左右移动）
// Type: 2D Freeform Directional / 2D Simple Directional
// Parameters: SpeedX, SpeedY
// Motions:
//   - Idle           | Pos: (0, 0)
//   - WalkForward    | Pos: (0, 1)
//   - WalkBack       | Pos: (0, -1)
//   - WalkLeft       | Pos: (-1, 0)
//   - WalkRight      | Pos: (1, 0)
//   - RunForward     | Pos: (0, 2)
//   ...

public class BlendTreeController : MonoBehaviour
{
    private Animator animator;
    private static readonly int SpeedX = Animator.StringToHash("SpeedX");
    private static readonly int SpeedY = Animator.StringToHash("SpeedY");

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");

        // 平滑过渡参数值
        animator.SetFloat(SpeedX, h, 0.1f, Time.deltaTime); // 第三个参数是dampTime
        animator.SetFloat(SpeedY, v, 0.1f, Time.deltaTime);
    }

    // Direction混合 vs Freeform混合
    // Direction: 根据方向角度混合，适合8方向移动
    // Freeform: 根据参数空间中的距离混合，适合任意方向
}
```

## 动画层与遮罩详解

动画层允许同时播放多个动画，通过Avatar Mask控制影响的身体部位：

```csharp
public class LayeredAnimation : MonoBehaviour
{
    private Animator animator;

    void Start()
    {
        animator = GetComponent<Animator>();

        // ========== 动画层结构 ==========
        // Layer 0: Base Layer (全身动画 - 移动、空闲)
        // Layer 1: UpperBody (上半身攻击，叠加在移动上)
        // Layer 2: Additive (叠加层 - 呼吸起伏)
        // Layer 3: IK Layer (IK修正层)

        // 设置上半身层权重
        // 权重为1时完全覆盖，0时完全不生效
        animator.SetLayerWeight(1, 1f);

        // 动态调整叠加层权重
        float breathWeight = isCrouching ? 0f : 0.3f;
        animator.SetLayerWeight(2, breathWeight);
    }

    void Update()
    {
        // 上半身层播放攻击动画
        // 下半身仍可以继续播放移动动画（因为Avatar Mask只勾选上半身骨骼）
        if (Input.GetMouseButtonDown(0))
            animator.SetTrigger("Attack");
    }

    // ========== Avatar Mask配置 ==========
    // 1. 创建Avatar Mask: Assets -> Create -> Avatar Mask
    // 2. 在Mask的Humanoid选项卡中选择影响的骨骼
    // 3. 绿色=激活(有动画)，红色=禁用(保持原层动画)
    // 4. 也可以使用Transform方式指定具体的骨骼路径
    //
    // 常用Mask:
    // - UpperBody: 勾选脊柱以上骨骼
    // - LowerBody: 勾选髋部以下骨骼
    // - LeftArm/RightArm: 仅手臂

    // ========== 层混合模式 ==========
    // Override: 覆盖原层的同部位动画
    // Additive: 叠加到原层动画上（适合呼吸、晃动等微调）
}
```

## AnimatorOverrideController

基于已有Animator Controller替换其中的动画Clip，适合多角色复用同一状态机：

```csharp
public class CharacterVariants : MonoBehaviour
{
    [SerializeField] private RuntimeAnimatorController baseController;
    [SerializeField] private AnimationClip[] overrideClips;

    void Start()
    {
        // 创建Override Controller
        AnimatorOverrideController aoc = new AnimatorOverrideController(baseController);

        // 替换指定的动画Clip
        // Key是原始Clip的名称，Value是替换的Clip
        aoc["Player_Idle"] = overrideClips[0];    // 替换Idle动画
        aoc["Player_Run"] = overrideClips[1];     // 替换Run动画
        aoc["Player_Attack"] = overrideClips[2];  // 替换Attack动画

        // 应用到Animator
        Animator animator = GetComponent<Animator>();
        animator.runtimeAnimatorController = aoc;
    }

    // ========== 运行时动态替换 ==========
    public void SwitchCharacterSkin(CharacterSkinData skinData)
    {
        var animator = GetComponent<Animator>();
        var aoc = new AnimatorOverrideController(baseController);

        // 通过反射或字典批量替换
        foreach (var mapping in skinData.clipMappings)
        {
            aoc[mapping.originalClip] = mapping.overrideClip;
        }

        animator.runtimeAnimatorController = aoc;
    }

    // 获取当前Override Controller的所有Clip映射
    void DebugOverrides()
    {
        var aoc = animator.runtimeAnimatorController as AnimatorOverrideController;
        if (aoc != null)
        {
            var overrides = new List<KeyValuePair<AnimationClip, AnimationClip>>();
            aoc.GetOverrides(overrides);
            foreach (var pair in overrides)
            {
                Debug.Log($"原始: {pair.Key?.name} -> 替换: {pair.Value?.name}");
            }
        }
    }
}

[CreateAssetMenu(menuName = "Game/Character Skin")]
public class CharacterSkinData : ScriptableObject
{
    [System.Serializable]
    public struct ClipMapping
    {
        public AnimationClip originalClip;
        public AnimationClip overrideClip;
    }

    public ClipMapping[] clipMappings;
}
```

## Root Motion（根运动）

Root Motion允许角色位移由动画驱动而非代码控制：

```csharp
public class RootMotionHandler : MonoBehaviour
{
    private Animator animator;

    void Start()
    {
        animator = GetComponent<Animator>();

        // Root Motion设置:
        // Animator组件的 "Apply Root Motion" 选项:
        // - Checked: 动画驱动位移 (OnAnimatorMove被调用)
        // - Unchecked: 代码驱动位移 (OnAnimatorMove不被调用)

        // 对于人形动画，还需在Animation Clip中:
        // - 设置Root Transform Position (XZ/Y) 为 "Bake Into Pose" 或 "Original"
        // - Root Transform Rotation 同理
    }

    // 自定义Root Motion处理
    void OnAnimatorMove()
    {
        // animator.deltaPosition: 本帧动画产生的位移增量
        // animator.deltaRotation: 本帧动画产生的旋转增量

        Vector3 deltaPos = animator.deltaPosition;
        Quaternion deltaRot = animator.deltaRotation;

        // 处理Root Motion与物理的结合
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // 使用物理移动而非直接设置position
            rb.MovePosition(rb.position + deltaPos);
            rb.MoveRotation(rb.rotation * deltaRot);
        }
        else
        {
            // 无物理时直接应用
            transform.position += deltaPos;
            transform.rotation *= deltaRot;
        }

        // ========== 混合Root Motion与代码移动 ==========
        // 当需要部分由动画控制、部分由代码控制时
        float codeMovementWeight = animator.GetFloat("CodeMovementWeight");
        Vector3 codeMovement = GetInputMovement() * Time.deltaTime * speed;
        Vector3 finalMovement = Vector3.Lerp(deltaPos, codeMovement, codeMovementWeight);
        transform.position += finalMovement;
    }

    // 关闭Root Motion时的替代方案
    void LateUpdate()
    {
        if (!animator.applyRootMotion)
        {
            // 从Animator的Delta获取位移手动应用
            // 但不使用OnAnimatorMove
        }
    }
}
```

## IK（逆向运动学）

IK允许程序化控制角色手脚的位置，使其贴合目标：

```csharp
public class IKController : MonoBehaviour
{
    private Animator animator;

    [Header("IK目标")]
    [SerializeField] private Transform leftHandTarget;
    [SerializeField] private Transform rightHandTarget;
    [SerializeField] private Transform leftFootTarget;
    [SerializeField] private Transform rightFootTarget;
    [SerializeField] private Transform lookAtTarget;

    [Header("IK权重")]
    [SerializeField] private float leftHandWeight = 0f;
    [SerializeField] private float rightHandWeight = 0f;
    [SerializeField] private float lookAtWeight = 0f;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    // 必须在Animator的IK Pass开启后才能使用
    void OnAnimatorIK(int layerIndex)
    {
        // ========== 手部IK ==========
        if (leftHandTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, leftHandWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, leftHandWeight);
            animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
            animator.SetIKRotation(AvatarIKGoal.LeftHand, leftHandTarget.rotation);
        }

        if (rightHandTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.RightHand, rightHandWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.RightHand, rightHandWeight);
            animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandTarget.position);
            animator.SetIKRotation(AvatarIKGoal.RightHand, rightHandTarget.rotation);
        }

        // ========== 脚部IK（脚适应斜坡） ==========
        if (leftFootTarget != null)
        {
            // Raycast检测地面高度
            RaycastHit hit;
            if (Physics.Raycast(animator.GetIKPosition(AvatarIKGoal.LeftFoot) + Vector3.up,
                Vector3.down, out hit, 1.5f, groundLayer))
            {
                animator.SetIKPositionWeight(AvatarIKGoal.LeftFoot, 1f);
                animator.SetIKPosition(AvatarIKGoal.LeftFoot, hit.point);

                // 根据地面法线调整脚的旋转
                Quaternion footRot = Quaternion.FromToRotation(Vector3.up, hit.normal);
                animator.SetIKRotationWeight(AvatarIKGoal.LeftFoot, 1f);
                animator.SetIKRotation(AvatarIKGoal.LeftFoot,
                    footRot * animator.GetIKRotation(AvatarIKGoal.LeftFoot));
            }
        }

        // ========== 头部IK（注视目标） ==========
        if (lookAtTarget != null)
        {
            animator.SetLookAtWeight(lookAtWeight, 0.3f, 0.6f, 1.0f, 0.5f);
            // 参数: 整体权重, 身体权重, 头部权重, 眼睛权重, 钳制权重
            animator.SetLookAtPosition(lookAtTarget.position);
        }
    }

    // IK的实际应用:
    // 1. 攀爬系统 - 手脚贴合攀爬面
    // 2. 持枪瞄准 - 手部贴合武器握把
    // 3. 地形适应 - 脚适应不平整地面
    // 4. NPC注视 - NPC头部跟踪玩家
}
```

## 常见陷阱与最佳实践

1. **Animator.StringToHash**: 必须缓存参数的哈希值，避免每帧字符串查找开销（约快10倍）
2. **Has Exit Time**: 过渡设置中"Has Exit Time"会等当前动画播完才过渡。即时响应用Trigger+取消勾选Has Exit Time
3. **Transition Duration**: 设置合适的过渡时长（0.1-0.25s），太短会突兀，太长会模糊
4. **Root Motion冲突**: 使用Root Motion时，角色位移由动画驱动，代码不应同时控制位移，否则会加倍
5. **IK性能**: IK计算有一定开销，不需要时关闭IK Pass层权重
6. **Controller为空检查**: Animator.runtimeAnimatorController为null时调用SetTrigger会报错
7. **Blend Tree平滑**: 使用dampTime参数平滑过渡，避免参数突变导致动画抖动
8. **动画事件时机**: 动画事件在播放过程中触发，暂停/跳帧可能导致事件丢失或重复

## 性能分析

| 操作 | 开销 | 优化建议 |
|------|------|---------|
| Animator.Update | 中 | 减少活跃Animator数量 |
| StringToHash | 低 | 缓存为静态字段 |
| SetFloat/SetBool | 极低 | 可每帧调用 |
| Blend Tree | 低中 | 减少混合的动画数量 |
| IK计算 | 中 | 不需要时禁用 |
| GetAnimatorStateInfo | 低 | 可每帧调用 |
| Override Controller创建 | 中 | 缓存，不要每帧创建 |

## 实际游戏案例

### 案例: 动作游戏的战斗动画系统

```csharp
public class CombatAnimator : MonoBehaviour
{
    private Animator animator;
    private static readonly int AttackIndex = Animator.StringToHash("AttackIndex");
    private static readonly int AttackTrigger = Animator.StringToHash("AttackTrigger");
    private static readonly int HitTrigger = Animator.StringToHash("HitTrigger");

    private int comboCount = 0;
    private float comboTimer = 0f;
    [SerializeField] private float comboWindow = 0.8f;

    void Update()
    {
        // 连击计时
        if (comboCount > 0)
        {
            comboTimer -= Time.deltaTime;
            if (comboTimer <= 0)
                comboCount = 0; // 超时重置连击
        }

        // 攻击输入
        if (Input.GetMouseButtonDown(0))
        {
            animator.SetInteger(AttackIndex, comboCount);
            animator.SetTrigger(AttackTrigger);
            comboCount = Mathf.Min(comboCount + 1, 3); // 最多3连击
            comboTimer = comboWindow;
        }
    }

    // 动画事件: 攻击结束
    public void OnAttackEnd()
    {
        // 自动进入下一个连击或回到Idle
    }

    // 受击动画
    public void PlayHitReaction()
    {
        animator.SetTrigger(HitTrigger);
        // 中断当前攻击动画
        comboCount = 0;
    }
}
```

## 与其他系统的关联

- **物理系统**: Root Motion与Rigidbody的结合需注意冲突，推荐用MovePosition
- **Timeline**: 可编排多角色的动画序列，用于过场动画
- **Cinemachine**: 相机抖动等效果可与动画系统协同
- **NavMesh**: 移动动画与AI导航的结合需要同步Animator参数
- **Shader**: 材质属性动画通过动画曲线驱动Shader参数
