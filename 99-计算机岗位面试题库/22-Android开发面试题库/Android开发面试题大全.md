# Android开发面试题大全（Android面试题库）

> 来源：小林coding、牛客网、掘金、CSDN、知乎、腾讯云开发者社区等
> 收录公司：字节跳动、腾讯、阿里、美团、百度、快手、华为、小米、OPPO、vivo、网易、京东等
> 整理日期：2026-05-08

---

## 一、Java/Kotlin 基础

### 1.1 Java 基础

Q1. Java中基本数据类型有哪些？各自占用多少字节？【字节/腾讯】
**答案：** Java有8种基本数据类型：byte(1字节)、short(2字节)、int(4字节)、long(8字节)、float(4字节)、double(8字节)、char(2字节)、boolean(1字节/4字节取决于JVM实现)。在Android开发中，合理选择数据类型能有效减少内存占用，例如在大量数据存储时用byte替代int可节省75%空间。

Q2. String、StringBuilder、StringBuffer的区别？【阿里/腾讯】
**答案：** String是不可变类，每次修改都会创建新对象；StringBuilder可变但线程不安全，性能最高；StringBuffer可变且线程安全（方法加了synchronized），性能略低于StringBuilder。在Android中单线程拼接字符串优先使用StringBuilder，避免循环中用"+"拼接String产生大量临时对象导致GC。

Q3. HashMap的底层实现原理？JDK1.7和1.8有什么区别？【美团/字节】
**答案：** JDK1.7采用数组+链表，头插法，扩容时可能形成环形链表导致死循环。JDK1.8改为数组+链表+红黑树，链表长度超过8且数组长度>=64时转为红黑树，使用尾插法避免死循环，引入了红黑树优化查询O(n)到O(log n)。HashMap默认容量16，负载因子0.75，扩容为2倍。

Q4. HashMap和Hashtable的区别？ConcurrentHashMap的实现原理？【阿里/美团】
**答案：** HashMap线程不安全，允许null键值；Hashtable线程安全但效率低（全表锁）。ConcurrentHashMap在1.7采用分段Segment锁（继承ReentrantLock），1.8改为CAS+synchronized（锁粒度到Node节点），并发性能大幅提升。读操作一般无锁，写操作通过CAS或synchronized保证线程安全。

Q5. Java中的异常体系？Error和Exception的区别？【腾讯/百度】
**答案：** Throwable是所有异常的父类，分为Error（系统级错误如OutOfMemoryError、StackOverflowError，不可恢复）和Exception（程序可处理异常）。Exception又分为RuntimeException（运行时异常如NullPointerException，不强制捕获）和CheckedException（编译时异常如IOException，必须捕获或声明抛出）。Android中OOM和ANR是常见的Error场景。

Q6. ==和equals()的区别？hashCode()和equals()的关系？【字节/网易】
**答案：** ==比较基本类型的值或引用类型的地址；equals()默认也是比较地址（Object类），但通常被重写为比较内容（如String）。hashCode()返回对象的哈希值，重写equals()必须同时重写hashCode()，因为HashMap等集合先用hashCode定位桶，再用equals比较。两个对象equals为true则hashCode必须相同，hashCode相同不一定equals为true。

Q7. final、finally、finalize的区别？【阿里/腾讯】
**答案：** final修饰类不可继承，修饰方法不可重写，修饰变量不可修改引用。finally是try-catch中必定执行的代码块（除非System.exit()或线程被杀死），常用于资源释放。finalize()是Object的方法，GC回收前调用（已废弃），不建议使用，推荐用try-with-resources或Cleaner替代。

Q8. Java中的深拷贝和浅拷贝？如何实现深拷贝？【快手/美团】
**答案：** 浅拷贝只复制对象本身，不复制引用的子对象（Object.clone()默认行为）；深拷贝递归复制所有引用对象。实现深拷贝的方法：1)递归实现Cloneable并重写clone()；2)序列化/反序列化（Serializable）；3)手动递归构建新对象；4)JSON转换。Android中推荐使用Parcelable实现高效序列化。

Q9. 接口和抽象类的区别？什么时候用接口什么时候用抽象类？【华为/小米】
**答案：** 抽象类用abstract修饰，可以有构造方法、成员变量、具体方法和抽象方法，单继承；接口用interface修饰（JDK8后可有default/static方法），只能有常量和抽象方法（JDK9可有private方法），多实现。抽象类适用于有共同属性和行为的类族；接口定义行为契约，如Comparable、Serializable。Android中BaseActivity常用抽象类，功能扩展用接口。

Q10. Java的反射机制是什么？应用场景？缺点？【字节/阿里】
**答案：** 反射是在运行时获取类的信息（字段、方法、构造器）并操作对象的能力，通过Class对象实现。核心API：Class.forName()、getDeclaredMethod()、getDeclaredField()、setAccessible(true)。应用场景：框架设计（Spring IOC）、动态代理、注解处理、序列化。缺点：性能开销大（比直接调用慢几十倍）、破坏封装性、安全问题。Android中ARetrofit、Gson等大量使用反射。

### 1.2 Kotlin 基础

Q11. Kotlin中的空安全机制？?.、?:、!!的区别？【字节/Google】
**答案：** Kotlin默认所有类型不可空，可空类型用?声明。?.（安全调用）对象为空时返回null不抛异常；?:（Elvis操作符）左侧为null时返回右侧默认值；!!（非空断言）强制转为非空，为空时抛NullPointerException。Kotlin编译期通过类型系统消除空指针异常，这也是Google推荐Kotlin开发Android的主要原因之一。

Q12. Kotlin中let、apply、also、run、with的区别？【字节/阿里】
**答案：** 5个作用域函数的区别在于上下文对象（this/it）和返回值（上下文对象/lambda结果）：let-上下文it返回lambda结果，用于空检查后操作；apply-上下文this返回对象本身，用于对象配置；also-上下文it返回对象本身，用于副作用操作；run-上下文this返回lambda结果，用于对象初始化+计算；with-上下文this返回lambda结果，非扩展函数。apply/also返回对象本身支持链式调用。

Q13. Kotlin中的协程是什么？和线程的区别？【字节/腾讯】
**答案：** 协程是轻量级线程，由Kotlin协程库在用户态调度，一个线程可运行成千上万个协程。区别：1)协程开销极小（KB级别），线程开销大（MB级别栈空间）；2)协程可通过挂起（suspend）非阻塞等待，线程阻塞等待会占用系统资源；3)协程由CoroutineDispatcher调度到线程池。Kotlin协程核心概念：CoroutineScope、Job、Dispatcher、CoroutineContext、suspend函数。

Q14. Kotlin中data class的特点？和普通class的区别？【美团/小米】
**答案：** data class自动生成equals()、hashCode()、toString()、copy()、componentN()等方法。要求主构造器至少一个参数，参数用val/var标记。区别于普通class：自动实现解构声明、copy()实现浅拷贝、toString()输出可读格式。注意：自动生成的方法只覆盖主构造器中的属性，hashCode/equals基于所有属性。Android中常用data class定义数据模型、API响应对象。

Q15. Kotlin中的扩展函数和扩展属性？原理？【字节/美团】
**答案：** 扩展函数允许在不修改源码的情况下为类添加新方法，语法：fun ClassName.functionName()。扩展属性需定义getter。本质是静态方法，编译后第一个参数是接收者对象，并非真正修改类。不支持多态（静态分派），调用取决于声明类型而非运行时类型。Android中常用扩展函数简化API调用，如View.visible()、Context.toast()等。

Q16. Kotlin中的密封类（Sealed Class）和枚举类的区别？【字节/阿里】
**答案：** 枚举类每个实例只有一个，值有限且固定；密封类子类可以有多个实例，子类可携带不同数据。密封类在when表达式中不需要else分支（编译器检查穷尽性），适合表示有限状态集如网络请求结果Success/Error/Loading。密封接口（Kotlin 1.5+）进一步扩展了灵活性。

Q17. Kotlin的委托机制？by关键字的用法？【腾讯/美团】
**答案：** Kotlin支持类委托和属性委托。类委托：class MyList<T>(private val innerList: MutableList<T>): MutableList<T> by innerList，将接口实现委托给内部对象。属性委托：val name: String by Delegates.observable()，常见的委托有lazy（惰性初始化）、observable/vetoable（监听变化）、notNull（延迟初始化）。委托模式优于继承，实现了组合优于继承的设计原则。

Q18. Kotlin中inline、noinline、crossinline的区别？【字节/阿里】
**答案：** inline将lambda内联到调用处，避免对象创建开销；noinline标记不需要内联的lambda参数（可存储、传递）；crossinline禁止lambda中的非局部return（防止跳出inline函数外）。inline适合高阶函数性能优化，但不宜过度使用（会增加APK大小）。Android中协程的launch/async都是inline函数。

Q19. Kotlin Flow是什么？和LiveData的区别？【字节/Google】
**答案：** Flow是Kotlin协程的冷流，支持背压、变换操作符（map/filter/combine）、异常处理。区别于LiveData：1)Flow支持协程上下文切换，LiveData只在主线程；2)Flow有丰富的操作符，LiveData操作符较少；3)Flow是冷流（按需生产），LiveData是热流（持续发射）；4)Flow不绑定生命周期，需配合lifecycleScope使用。推荐用StateFlow/SharedFlow替代LiveData。

Q20. Kotlin中协程的异常处理机制？【美团/字节】
**答案：** 协程异常处理的核心是CoroutineExceptionHandler和SupervisorJob。普通Job中子协程异常会传播取消父Job和兄弟协程；SupervisorJob中子协程异常不传播。CoroutineExceptionHandler作为CoroutineContext元素捕获未处理异常。推荐做法：viewModelScope + supervisorScope，子协程用try-catch或runCatching处理异常。launch中未捕获异常会抛给CoroutineExceptionHandler，async中异常在await()时抛出。

---

## 二、Android 四大组件

### 2.1 Activity

Q21. Activity的生命周期？onCreate、onStart、onResume、onPause、onStop、onDestroy？【字节/腾讯】
**答案：** onCreate()创建界面；onStart()可见但不在前台；onResume()可见且在前台可交互；onPause()部分遮挡（Dialog可见时调用）；onStop()完全不可见；onDestroy()销毁。特殊情况：屏幕旋转会销毁重建（可通过configChanges或ViewModel避免）。从后台返回：onRestart→onStart→onResume。onPause中不应做耗时操作，否则影响新Activity启动。

Q22. Activity的四种启动模式？standard、singleTop、singleTask、singleInstance？【阿里/美团】
**答案：** standard（默认）每次创建新实例入栈；singleTop栈顶复用（适合接收通知跳转）；singleTask栈内复用，清除上方所有Activity（适合主页）；singleInstance独占一个任务栈（适合独立功能如来电界面）。通过AndroidManifest的launchMode或Intent的FLAG_ACTIVITY_*设置。singleTask配合taskAffinity可指定任务栈。

Q23. Activity的onSaveInstanceState和onRestoreInstanceState何时调用？【腾讯/百度】
**答案：** 系统可能销毁Activity时（如屏幕旋转、内存不足回收）调用onSaveInstanceState保存临时数据（在onStop之前，可能在onPause之前或之后）。onRestoreInstanceState在onStart之后调用恢复数据。ViewModel更适合保存UI状态数据（配置变更时保留）。onSaveInstanceState的Bundle不适合保存大数据。

Q24. Intent是什么？显式Intent和隐式Intent的区别？【华为/小米】
**答案：** Intent是Android组件间通信的消息对象。显式Intent指定目标组件的类名（应用内跳转）；隐式Intent通过Action、Category、Data描述意图，系统匹配合适的组件（跨应用跳转，如分享、打开浏览器）。隐式Intent可通过IntentFilter声明匹配规则。Android 5.0+隐式Intent启动Service必须指定包名。PendingIntent是延迟执行的Intent，用于通知、闹钟等场景。

Q25. Activity间如何传递数据？Bundle和Parcelable、Serializable的区别？【字节/阿里】
**答案：** 通过Intent的putExtra()和Bundle传递。Bundle内部用ArrayMap存储，实现了Parcelable。区别：Serializable是Java序列化接口（反射，慢），生成serialVersionUID保证版本兼容；Parcelable是Android专用接口（手动序列化/反序列化，快10倍以上），适合IPC。推荐用Parcelable，数据量大时考虑其他方式（如单例、静态变量、数据库、EventBus）。

Q26. Activity的taskAffinity和allowTaskReparenting的作用？【美团/字节】
**答案：** taskAffinity指定Activity的亲和任务栈（默认为包名），不同taskAffinity的Activity可分配到不同任务栈。allowTaskReparenting为true时，Activity可从启动它的任务栈迁移到亲和的任务栈（如浏览器链接在浏览器启动后迁移到浏览器任务栈）。配合singleTask使用可实现复杂的任务栈管理。开发者选项中"不保留活动"可观察任务栈行为。

Q27. 如何安全地退出整个应用？【腾讯/华为】
**答案：** 推荐方案：1)使用SingleActivity管理所有Activity引用，退出时逐一finish；2)发送广播通知所有Activity finish；3)使用FLAG_ACTIVITY_CLEAR_TOP+FLAG_ACTIVITY_NEW_TASK跳转到主页并finish当前页；4)使用ActivityLifecycleCallbacks管理。不要用System.exit()或killProcess()，这会导致应用闪退而非优雅退出。Jetpack Navigation下可直接finish当前Activity。

Q28. Activity的onNewIntent调用时机？【字节/美团】
**答案：** 当Activity以singleTop/singleTask/singleInstance模式启动且复用已有实例时调用onNewIntent。调用顺序：onNewIntent→onRestart→onStart→onResume。注意需在onNewIntent中调用setIntent(intent)更新Intent数据，否则getIntent()获取的仍是旧数据。场景：从通知栏多次点击、singleTask模式下的重复跳转。

Q29. Activity和Fragment的区别？为什么推荐用Fragment？【阿里/小米】
**答案：** Activity是四大组件之一，有独立生命周期，重量级；Fragment依附于Activity，生命周期受Activity影响，更轻量。Fragment优势：更灵活的UI模块化、支持动态添加/替换、回退栈管理、平板多面板适配。Fragment缺点：生命周期复杂（多层嵌套问题）、内存泄漏风险。推荐使用Navigation Component管理Fragment导航。

Q30. Fragment的生命周期？和Activity生命周期的对应关系？【腾讯/字节】
**答案：** Fragment生命周期多了几个方法：onAttach（关联Activity）→onCreate→onCreateView（创建视图）→onViewCreated→onStart→onResume→onPause→onStop→onDestroyView→onDestroy→onDetach。Activity的onCreate对应Fragment的onAttach到onViewCreated；Activity的onResume/onPause时Fragment同级。Fragment的onDestroyView和onCreateView是最常出问题的地方（视图销毁但Fragment实例还在）。

### 2.2 Service

Q31. Service的两种启动方式？startService和bindService的区别？【阿里/美团】
**答案：** startService：生命周期onCreate→onStartCommand→运行→stopSelf()/stopService()→onDestroy，适合后台任务（如下载），与启动者无交互。bindService：生命周期onCreate→onBind→运行→所有绑定者unbind→onUnbind→onDestroy，适合需要交互的场景（如音乐播放器控制）。可同时使用两种方式，需两种方式都停止才会销毁。Android 8.0+后台启动Service有限制。

Q32. Service和Thread的区别？【字节/腾讯】
**答案：** Service是Android组件（运行在主线程），用于执行需要长期运行的任务，不被系统轻易回收；Thread是CPU调度单元，用于执行异步操作。Service不自动创建线程，在Service中直接执行耗时操作会ANR。常见模式：Service+Thread/Executor/协程。IntentService内部自动创建WorkerThread处理任务（已废弃，推荐用WorkManager替代）。

Q33. 前台Service是什么？如何使用？Android 8.0+的变化？【华为/字节】
**答案：** 前台Service通过startForeground()启动，显示通知栏通知，优先级高不容易被系统杀死（如音乐播放、导航）。Android 8.0+后台不能启动Service，需用startForegroundService()启动并在5秒内调用startForeground()。Android 9+需FOREGROUND_SERVICE权限。Android 14新增FOREGROUND_SERVICE_SPECIAL_USE等细粒度权限。前台Service必须显示用户可见的通知。

Q34. IntentService的原理？为什么被废弃？【美团/小米】
**答案：** IntentService内部有HandlerThread（自带Looper的线程），通过Handler将Intent逐个投递到工作线程串行处理。任务执行完自动调用stopSelf()。被废弃原因：Android 8.0+后台限制、不支持灵活的调度策略、无法取消任务。替代方案：WorkManager（推荐，支持延迟、周期、约束条件）、JobIntentService（兼容后台限制，也已废弃）→Kotlin WorkManager。

Q35. Service的onStartCommand返回值的含义？【百度/美团】
**答案：** START_STICKY：Service被杀后重新创建，但Intent为null（适合媒体播放）；START_NOT_STICKY：Service被杀后不重新创建，除非有待处理的Intent（适合不需要持续运行的任务）；START_REDELIVER_INTENT：Service被杀后重新创建并重传最后的Intent（适合需要恢复的任务如下载）；START_STICKY_COMPATIBILITY：START_STICKY的兼容模式不保证重新创建。

Q36. 如何保证Service不被系统杀死？【腾讯/阿里】
**答案：** 1)使用前台Service（最有效）；2)使用START_STICKY让系统重启；3)在onDestroy中重新启动Service（兜底）；4)使用JobScheduler/WorkManager做定时唤醒；5)利用系统广播（开机、网络变化）重启；6)双进程守护（不推荐，违背系统设计）。Android系统越来越严格限制后台Service，建议使用WorkManager替代长期后台任务。

Q37. AIDL是什么？如何使用？支持哪些数据类型？【字节/华为】
**答案：** AIDL（Android Interface Definition Language）是Android IPC的接口定义语言，用于跨进程通信。支持的基本类型：int、long、boolean、float、double、String、CharSequence、List、Map、Parcelable、AIDL接口。使用步骤：1)定义.aidl文件；2)Build生成Java接口；3)Service端实现Stub类；4)Client端bindService获取Binder代理。AIDL底层基于Binder驱动，通过共享内存实现高效IPC。

Q38. Binder机制的原理？为什么Android用Binder而不用其他IPC方式？【阿里/字节】
**答案：** Binder是Android特有的IPC机制，基于C/S架构。数据只需一次拷贝（从用户空间到内核空间的Binder驱动），而Socket/管道需要两次拷贝。Binder组成：Client、Server、ServiceManager、Binder驱动。相比其他IPC：比共享内存安全（有UID/PID标识）、比Socket效率高、比管道/消息队列功能丰富。Binder通过BinderProxy和Stub实现跨进程调用的透明代理。

### 2.3 BroadcastReceiver

Q39. BroadcastReceiver的两种注册方式？静态注册和动态注册的区别？【美团/阿里】
**答案：** 静态注册在AndroidManifest.xml中声明，应用未启动也能接收（但Android 8.0+大部分隐式广播不能静态注册）。动态注册在代码中通过registerReceiver()注册，随注册组件的生命周期存在，需手动unregisterReceiver。动态注册优先级高于静态注册。Android 8.0+限制：除少数豁免广播（开机、时区变化等）外，静态注册的隐式广播接收器无效。

Q40. Android 8.0+对广播的限制有哪些？【字节/腾讯】
**答案：** Android 8.0（API 26）起，大部分隐式广播不能静态注册。豁免清单包括：ACTION_BOOT_COMPLETED、ACTION_LOCALE_CHANGED、ACTION_PACKAGE_*等系统级广播。解决方案：1)使用动态注册；2)使用显式广播（指定包名）；3)使用JobScheduler替代；4)使用WorkManager。静态注册的广播虽然不能接收隐式广播，但可接收显式广播（Intent.setPackage/setComponent）。

Q41. 有序广播和无序广播的区别？如何拦截有序广播？【百度/华为】
**答案：** 无序广播（sendBroadcast）所有接收者同时接收，不可拦截；有序广播（sendOrderedBroadcast）按priority优先级顺序传递，可拦截（abortBroadcast()）或修改数据（setResultExtras）。优先级范围-1000到1000，数值越大优先级越高。动态注册优先级高于静态注册。应用场景：短信拦截（旧版本）、系统事件通知。

### 2.4 ContentProvider

Q42. ContentProvider的作用和使用场景？【阿里/小米】
**答案：** ContentProvider是Android四大组件之一，用于应用间数据共享，提供统一的CRUD接口。底层通过Binder IPC实现跨进程数据访问。使用场景：访问通讯录、媒体库、日历等系统数据；自定义数据暴露给其他应用。URI格式：content://authority/path/id。配合CursorLoader/ContentResolver使用。安全性通过URI权限、读写权限控制。

Q43. ContentProvider的生命周期？和SQLite的区别？【腾讯/美团】
**答案：** ContentProvider在应用启动时由系统创建（比Application.onCreate还早），onCreate()中初始化底层数据源。提供query/insert/update/delete/getType方法。和SQLite的区别：SQLite是数据库引擎，ContentProvider是数据访问抽象层，底层可基于SQLite、文件、网络等。Room是对SQLite的封装，提供了更方便的注解方式。现代Android开发推荐Room替代直接操作SQLite。

---

## 三、UI 与视图系统

Q44. View的绘制流程？measure、layout、draw三步？【字节/阿里】
**答案：** ViewRootImpl.performTraversals()触发绘制：1)measure确定View大小（onMeasure→setMeasuredDimension）；2)layout确定位置（onLayout→遍历子View）；3)draw绘制（onDraw→dispatchDraw→onDrawForeground）。measure通过MeasureSpec约束尺寸。

Q45. MeasureSpec的三种模式？【腾讯/美团】
**答案：** EXACTLY：精确大小（match_parent或具体dp值）；AT_MOST：最大不超过（wrap_content）；UNSPECIFIED：不限制（ScrollView子View）。match_parent→EXACTLY；wrap_content→AT_MOST；具体值→EXACTLY。

Q46. 自定义View的几种方式？【字节/华为】
**答案：** 1)继承现有View扩展功能；2)继承ViewGroup组合多个View；3)直接继承View完全自绘（Canvas绘制）；4)继承ViewGroup重写onLayout。需重写构造器、onMeasure（处理wrap_content）、onDraw/onLayout、onTouchEvent。

Q47. Canvas和Paint的区别？【美团/小米】
**答案：** Canvas是画布提供绘制API（drawLine/drawRect/drawCircle/drawBitmap/drawText）；Paint是画笔定义颜色/样式/字体/抗锯齿。Canvas.save()/restore()保存画布状态。硬件加速下Canvas操作由DisplayList记录GPU渲染。

Q48. 动画分类？帧动画、补间动画、属性动画？【阿里/字节】
**答案：** 帧动画（AnimationDrawable）逐帧切换；补间动画（Tween）通过XML定义透明度/缩放/旋转/平移，只改变显示效果不改变属性；属性动画（Animator）改变对象属性值，支持任意对象。推荐属性动画。

Q49. 属性动画的原理？【字节/腾讯】
**答案：** ValueAnimator计算动画值需手动应用；ObjectAnimator自动反射调用setter。Evaluator（估值器）计算具体值，Interpolator（插值器）控制速率。AnimatorSet组合多个动画。通过Choreographer与VSync信号同步刷新。

Q50. RecyclerView的缓存机制？四级缓存？【字节/阿里】
**答案：** 1)Attached Scrap/Changed Scrap：屏幕内ViewHolder（不创建不绑定）；2)Cached Views：刚滑出的ViewHolder默认容量2；3)ViewCacheExtension：自定义缓存；4)RecycledViewPool：缓存池默认容量5。核心减少onCreateViewHolder开销。

Q51. RecyclerView和ListView的区别？【美团/华为】
**答案：** RecyclerView强制ViewHolder、支持多种LayoutManager、内置ItemAnimator、ItemDecoration、DiffUtil高效刷新、嵌套滚动。ListView只支持纵向列表，ViewHolder需手动实现。notifyDataSetChanged效率低，RecyclerView用DiffUtil局部刷新。

Q52. RecyclerView如何实现多类型列表？【字节/美团】
**答案：** 重写getItemViewType()返回不同viewType，onCreateViewHolder中根据viewType创建不同ViewHolder。复杂场景用BaseMultiItemQuickAdapter。Paging 3中ConcatAdapter组合多个Adapter。数据层定义基类提供viewType和layoutId。

Q53. 滑动冲突如何解决？【美团/字节】
**答案：** 外部拦截法：父容器onInterceptTouchEvent根据方向判断是否拦截。内部拦截法：子View通过requestDisallowInterceptTouchEvent控制。嵌套滑动用NestedScrollingParent/Child接口。ScrollView嵌套RecyclerView用setNestedScrollingEnabled(false)。

Q54. 事件分发机制？【字节/阿里】
**答案：** Activity.dispatchTouchEvent→ViewGroup.dispatchTouchEvent→onInterceptTouchEvent→View.dispatchTouchEvent→onTouchEvent。ViewGroup默认不拦截。子View可requestDisallowInterceptTouchEvent()。onTouchEvent返回true消费事件。

Q55. Window、Activity、DecorView的关系？【阿里/字节】
**答案：** Activity包含PhoneWindow，PhoneWindow包含DecorView（根View）。DecorView结构：标题栏+内容区域（android.R.id.content）。setContentView()实际调用PhoneWindow.setContentView()。WindowManager管理Window添加/删除。

Q56. ConstraintLayout的优势？【美团/小米】
**答案：** 扁平化布局减少嵌套测量。支持百分比布局（GuideLine/Bias/Barrier/Group）、ConstraintSet动画、链式布局。与RelativeLayout区别：RelativeLayout两次测量O(n)，ConstraintLayout通过线性方程组求解O(n)。

Q57. include、merge、ViewStub的作用？【阿里/腾讯】
**答案：** include复用布局XML可覆盖id和layout_*属性。merge减少根节点View层级（必须作XML根标签）。ViewStub惰性加载inflate()后实例化，适合不常用布局（错误页面、空状态）。ViewStub inflate后不能再次使用。

Q58. 如何实现沉浸式状态栏？【字节/OPPO】
**答案：** 4.4+半透明状态栏；5.0+setStatusBarColor()；6.0+SYSTEM_UI_FLAG_LIGHT_STATUS_BAR深色图标；15引入Edge-to-Edge全屏。WindowCompat.setDecorFitsSystemWindows(window, false)。WindowInsetsCompat处理系统窗口间距。

Q59. 如何优化RecyclerView滚动性能？【字节/快手】
**答案：** 1)避免onBindViewHolder耗时操作；2)setHasFixedSize(true)；3)DiffUtil异步计算差异；4)预取setInitialPrefetchItemCount；5)共享RecycledViewPool；6)减少过度绘制；7)避免频繁notifyDataSetChanged；8)Paging 3分页加载。

Q60. 屏幕适配方案？【小米/OPPO】
**答案：** 1)dp/sp替代px；2)ConstraintLayout百分比布局；3)最小宽度限定符（sw360dp）；4)今日头条方案（修改density=deviceScreenWidth/designWidth）；5)smallestWidth适配生成多套dimens。density=dpi/160。

Q61. 自定义ViewGroup如何处理wrap_content？【腾讯/美团】
**答案：** onMeasure中区分模式：AT_MOST需计算合适大小（设默认值或根据内容），EXACTLY直接用getSize()。调用setMeasuredDimension()保存结果。ViewGroup需遍历子View测量后确定自身大小。

Q62. Android中的阴影和Outline？【美团/阿里】
**答案：** 5.0+elevation属性产生阴影。阴影形状由Outline决定，通过ViewOutlineProvider自定义（矩形/圆形/圆角矩形）。CardView的cardElevation也产生阴影。低版本用CardView或自绘阴影。

Q63. MotionLayout实现复杂动画？【字节/Google】
**答案：** MotionLayout是ConstraintLayout子类，XML描述起始/结束ConstraintSet自动计算过渡动画。支持关键帧（KeyFrameSet）、属性变化（CustomAttributes）、手势驱动（OnSwipe）。适合复杂交互式动画。

Q64. SharedElementTransition？【字节/腾讯】
**答案：** 启动Activity通过ActivityOptions.makeSceneTransitionAnimation()指定共享元素。两Activity中共享View设置相同transitionName。Fragment间使用setSharedElementEnterTransition。配合Transition框架实现页面转场动画。

Q65. DiffUtil的原理？【字节/腾讯】
**答案：** Eugene W. Myers算法计算两个列表最小差异。calculateDiff()异步计算，dispatchUpdatesTo()应用更新。实现areItemsTheSame()和areContentsTheSame()。AsyncListDiffer封装异步使用。Payload实现局部更新。

Q66. CoordinatorLayout和Behavior？【阿里/美团】
**答案：** CoordinatorLayout通过Behavior协调子View交互。Behavior监听依赖View变化（layoutDependsOn/onDependentViewChanged）和处理嵌套滚动。AppBarLayout的Behavior处理折叠逻辑。自定义Behavior实现复杂交互。

Q67. NestedScrolling机制？【字节/阿里】
**答案：** NestedScrollingParent/Child接口实现嵌套滑动。子View先通知父View消耗滑动距离，父不消耗的部分由子View处理。RecyclerView、NestedScrollView已实现。支持惯性fling嵌套传递。

Q68. TabLayout和ViewPager2联动？【腾讯/阿里】
**答案：** TabLayoutMediator连接两者自动同步Tab选中和页面滑动。ViewPager2底层是RecyclerView，支持垂直滑动和RTL。监听OnPageChangeCallback更新Tab；Tab点击设置currentItem。

Q69. RecyclerView拖拽排序和滑动删除？【阿里/美团】
**答案：** ItemTouchHelper配合Callback。重写getMovementFlags()定义方向，onMove()处理拖拽，onSwiped()处理滑动删除。attachToRecyclerView关联。可自定义拖拽时的背景和动画效果。

Q70. 如何实现吸顶效果？【美团/字节】
**答案：** 方案：1)CoordinatorLayout+AppBarLayout+CollapsingToolbarLayout；2)自定义ItemDecoration在onDrawOver中绘制头部，当新头部即将出现时推动当前头部；3)StickyHeaderLayoutManager。核心判断滚动位置固定头部。

Q71. 如何实现流式布局？【腾讯/小米】
**答案：** 自定义ViewGroup，onMeasure计算每行宽度超过时换行，onLayout根据每行子View排列位置。可设置行间距字间距。Google开源FlexboxLayout。也可用RecyclerView+FlexboxLayoutManager实现。

Q72. 下拉刷新和上拉加载？【字节/美团】
**答案：** SwipeRefreshLayout官方下拉刷新。上拉：addOnScrollListener监听到底部。SmartRefreshLayout功能最全。Paging 3自动分页加载。要点：防重复请求、处理loading/success/error/noMore状态。

Q73. PopupWindow注意事项？【华为/小米】
**答案：** 1)必须设置BackgroundDrawable（否则点击外部不消失、7.0+高度问题）；2)showAsDropDown/showAtLocation定位；3)setOutsideTouchable外部点击；4)setFocusable焦点控制；5)7.0+不设背景高度异常。

Q74. MotionLayout+ConstraintLayout高级用法？【Google/阿里】
**答案：** MotionScene定义起始/结束约束。KeyPosition控制中间位置，KeyAttribute控制中间属性。OnSwipe绑定手势驱动动画。ConstraintSet.clone()从现有布局复制约束。适合实现引导动画、交互动画。


---

## 四、数据存储与持久化

Q75. 五种数据存储方式？【阿里/腾讯】
**答案：** 1)SharedPreferences键值对存储（XML）；2)文件存储（内部/外部）；3)SQLite数据库；4)ContentProvider跨进程共享；5)网络存储。SP不适合大数据。MMKV高性能替代（mmap内存映射，性能约100倍提升）。

Q76. SharedPreferences的apply和commit区别？【字节/美团】
**答案：** commit同步写入返回boolean；apply异步写入不返回结果。SP不是线程安全的但apply安全（单线程队列）。8.0+apply有ANR风险（onStop等待完成）。推荐MMKV替代。

Q77. SQLite事务和性能优化？【腾讯/华为】
**答案：** beginTransaction→setTransactionSuccessful→endTransaction。批量操作用事务提升10倍+。优化：WAL模式（读写并发）、批量事务、合适索引、异步操作、Room+LiveData。PRAGMA journal_mode=WAL; synchronous=NORMAL。

Q78. Room的Entity、Dao、Database？【字节/阿里】
**答案：** @Entity定义表（@PrimaryKey/@ColumnInfo/@Ignore）；@Dao定义CRUD（@Insert/@Update/@Delete/@Query）；@Database继承RoomDatabase定义版本和Entity。编译时生成代码类型安全。支持LiveData/Flow响应式查询。

Q79. Room Migration实现？【美团/小米】
**答案：** addMigrations()定义版本升级SQL（ALTER TABLE等）。必须逐版本迁移。fallbackToDestructiveMigration清空重建。导出Schema用AutoMigration自动检测。迁移失败数据丢失，务必测试覆盖。

Q80. DataStore和SharedPreferences区别？【字节/Google】
**答案：** DataStore基于协程Flow完全异步无ANR。PreferencesDataStore键值对；ProtoDataStore类型安全Protobuf。异常处理完善、类型安全。迁移用SharedPreferencesMigration。推荐替代SP。

Q81. 分区存储Scoped Storage适配？【字节/阿里】
**答案：** 10+限制直接访问外部存储。应用专属目录直接访问；媒体文件通过MediaStore；文档通过SAF。适配：MediaStore.insert→openFileDescriptor；DocumentsContract。10:requestLegacyExternalStorage=true。

Q82. MMKV的原理？【腾讯/阿里】
**答案：** 基于mmap内存映射，数据在内存和磁盘自动同步无需显式write。Protobuf编码高效紧凑。支持多进程和加密。读写性能约100倍于SharedPreferences。腾讯开源微信广泛使用。

Q83. SQLite索引原理？【美团/华为】
**答案：** B+树结构O(log n)查询。适合：WHERE条件列、JOIN列、ORDER BY。不适合：频繁更新列、数据量小表。复合索引最左前缀。EXPLAIN QUERY PLAN分析。CREATE INDEX idx_name ON table(col)。

Q84. Room@Query复杂SQL？【字节/腾讯】
**答案：** 支持JOIN、子查询、聚合函数。返回LiveData/Flow响应式。多表用@Relation或JOIN SQL。IN参数用(:ids)传List。LIKE用'||:keyword||'模糊匹配。@Transaction确保多次查询一致性。

Q85. Proto DataStore使用？【Google/字节】
**答案：** 定义.proto文件→protobuf-gradle-plugin编译→创建Serializer实现DataStore接口。类型安全、向后兼容、高效二进制。比PreferencesDataStore更适合结构化数据。

Q86. 数据库加密SQLCipher？【阿里/华为】
**答案：** SQLite 256位AES加密。Room集成：androidx.sqlite依赖+SupportSQLiteOpenHelper.Factory配置。密钥通过Passphrase传入。加密后文件不可直接读取。性能开销约5-15%。

Q87. 数据库导入导出？【小米/美团】
**答案：** 导出：查询→序列化JSON/CSV→写入文件（MediaStore/SAF）。导入：读取→解析→批量插入。大文件流式处理避免O。增量导入检查ID存在性。备份云端用WorkManager异步。

Q88. @Relation和@Embedded区别？【美团/小米】
**答案：** @Embedded嵌入字段到当前表（同表多列）。@Relation建立一对多关联查询通过外键（多张表）。@Relation需@Transaction。父Entity查询返回对象列表。

Q89. SQLite WAL模式优势？【腾讯/字节】
**答案：** Write-Ahead Logging写入WAL文件而非直接修改db。读写可并发（读不阻塞写、写不阻塞读）。PRAGMA journal_mode=WAL。checkpoint将WAL合并到db。Android 16+默认开启。

Q90. 数据增量同步方案？【字节/小米】
**答案：** 时间戳增量/版本号增量/CDC变更捕获。客户端拉取→合并本地→更新标记。冲突策略：服务端优先/客户端优先/最后写入优先。WorkManager定时同步。

Q91. @Transaction注解作用？【美团/阿里】
**答案：** @Transaction标记方法执行事务保证原子性。@Dao中CRUD默认事务。@Query需事务保证时加@Transaction。@Relation方法必须加（多次查询需一致性）。异常时Room自动回滚。

Q92. Room观察者机制？【阿里/Google】
**答案：** @Query返回LiveData/Flow响应式查询。InvalidationTracker监听表变化触发重新查询。LiveData自动在表变化时重新查询。Flow可控制数据变换。底层通过ContentObserver-like机制实现。

Q93. ContentProvider安全性？【阿里/华为】
**答案：** android:exported控制暴露；readPermission/writePermission控制读写；URI权限精细控制。4.2+protectionLevel签名权限。现代方案用FileProvider替代ContentProvider共享文件。


---

## 五、网络与数据解析

Q94. OkHttp拦截器链？【字节/阿里】
**答案：** RetryAndFollowUpInterceptor→BridgeInterceptor（请求头Cookie）→CacheInterceptor→ConnectInterceptor（TCP/TLS连接）→CallServerInterceptor。连接池复用，Dispatcher管理线程池。支持HTTP/2、WebSocket、Gzip。

Q95. Retrofit原理？【字节/腾讯】
**答案：** 基于OkHttp的RESTful框架。动态代理生成接口实现。注解定义API→Retrofit.create()→构建Request→OkHttp Call/协程。Converter序列化，CallAdapter适配返回类型。Retrofit管理URL拼接，OkHttp管理连接。

Q96. OkHttp拦截器自定义场景？【美团/阿里】
**答案：** 实现Interceptor.intercept(chain)。场景：公共请求头、日志打印（HttpLoggingInterceptor）、缓存控制、Token刷新重试（401时获取新Token）、加密解密、Mock数据。分Application和Network拦截器。

Q97. HTTP和HTTPS区别？TLS握手？【阿里/华为】
**答案：** HTTP明文端口80；HTTPS=HTTP+TLS端口443。握手：ClientHello→ServerHello+证书→验证证书+预主密钥→生成会话密钥→加密通信。Android证书固定CertificatePinner防中间人攻击。

Q98. WebView优化和JS交互？【字节/小米】
**答案：** 预创建WebView、硬件加速、离线缓存、图片懒加载、X5内核。JS交互：evaluateJavascript调JS；@JavascriptInterface+addJavascriptInterface让JS调Android。4.2以下addJavascriptInterface有远程代码执行漏洞。

Q99. 网络请求取消机制？【美团/腾讯】
**答案：** OkHttp Call.cancel()；协程Job.cancel()；RxJava Disposable.dispose()。在Activity/Fragment销毁时取消避免内存泄漏。lifecycleScope自动管理生命周期。封装统一取消管理器。

Q100. JSON解析方式对比？【字节/美团】
**答案：** Gson反射实现简单但性能一般；Moshi支持Kotlin CodeGen；kotlinx.serialization编译时生成零反射性能最高（Kotlin首选）。Fastjson历史安全漏洞多不推荐。Kotlin项目用kotlinx.serialization。

Q101. HTTP缓存机制？【字节/阿里】
**答案：** 强缓存Cache-Control:max-age/Expires不发请求直接用缓存。协商缓存ETag/If-None-Match、Last-Modified/If-Modified-Since发请求验证。Cache-Control优先于Expires。OkHttp CacheInterceptor实现。

Q102. HTTP/2特性？【阿里/Google】
**答案：** 多路复用（单连接并发请求）、头部压缩（HPACK）、服务器推送、流优先级。解决HTTP/1.1队头阻塞。OkHttp默认支持HTTP/2（需HTTPS）。减少TCP连接数和延迟明显提升性能。

Q103. WebSocket原理？【字节/腾讯】
**答案：** HTTP Upgrade建立全双工TCP连接。握手：Upgrade: websocket→101 Switching Protocols→双向通信。比HTTP长连接优势：服务端可主动推送、帧传输效率高。OkHttp支持WebSocket。

Q104. 网络重试策略？指数退避？【美团/阿里】
**答案：** 失败后等待时间指数增长（1s→2s→4s→8s）加随机抖动避免雷群效应。最大重试次数限制。自定义拦截器：捕获异常→判断条件→递增计数→延迟重试。幂等性保证重试安全。

Q105. DNS解析优化和HttpDNS？【阿里/美团】
**答案：** 本地DNS缓存、HttpDNS（绕过Local DNS防劫持）、预解析（启动时提前解析）、域名合并。OkHttp的Dns接口可自定义。阿里云/腾讯云HttpDNS服务。减少DNS解析延迟和防劫持。

Q106. RESTful API设计原则？【腾讯/美团】
**答案：** 资源导向URL、HTTP方法语义（GET/POST/PUT/PATCH/DELETE）、状态码规范、统一响应格式、版本管理、分页过滤排序、HATEOAS。错误响应包含code/message/details。

Q107. GraphQL和REST区别？【字节/阿里】
**答案：** GraphQL客户端精确查询所需字段避免过度获取；单一端点；查询/变更/订阅操作。REST每资源一端点可能过度获取。GraphQL适合复杂嵌套数据；REST简单直接适合CRUD。

Q108. 证书固定Certificate Pinning？【阿里/华为】
**答案：** OkHttp CertificatePinner指定域名和公钥哈希。阻止中间人攻击（即使CA被信任）。注意证书轮换更新Pin；配置多个备份Pin。网络安全配置network_security_config.xml也可配置。

Q109. 网络并发请求处理？【字节/华为】
**答案：** OkHttp Dispatcher限制并发（默认64，每host 5）。协程Semaphore限制并发。RxJava flatMap(maxConcurrency)。请求队列+优先级调度。注意并发取消和错误处理。

Q110. 如何做网络性能优化？【美团/华为】
**答案：** 连接复用（HTTP/2）、数据压缩（Gzip/Brotli）、CDN加速、DNS优化（HttpDNS）、请求合并减少RTT、数据预取/缓存、弱网优化（超时重试）、协议优化（QUIC/HTTP/3）。

Q111. Mock网络测试？【美团/小米】
**答案：** OkHttp Interceptor返回Mock数据；MockWebServer（Square开源模拟HTTP服务器）；构建模式切换debug用Mock/release用真实接口。MockWebServer可模拟超时和错误响应。


---

## 六、Jetpack 组件

Q112. ViewModel原理？配置变更保留数据？【字节/阿里】
**答案：** ViewModelStore存储，ViewModelProvider获取。配置变更时ViewModelStore不销毁（NonConfigurationInstances保留）。onCleared()在Activity finish时调用。不能持有Activity引用避免泄漏。替代onSaveInstanceState存储大量数据。

Q113. LiveData原理？【腾讯/美团】
**答案：** 生命周期感知数据持有者，只在活跃时通知。LifecycleOwner+Observer。postValue()主线程Handler，setValue()必须主线程。MediatorLiveData合并多源。自动管理生命周期避免泄漏。

Q114. Navigation Component？【字节/Google】
**答案：** NavController+NavGraph+NavHostFragment。NavGraph定义→navigate()跳转→Safe Args类型安全参数。回退栈管理目标。Deep Link支持URL跳转。嵌套导航图模块化。

Q115. WorkManager原理和场景？【阿里/美团】
**答案：** 保证执行（应用退出/重启）。底层JobScheduler（23+）或AlarmManager+BroadcastReceiver。一次性/周期（最小15分钟）/链式/约束条件。适用日志上传、数据同步等后台任务。

Q116. Paging 3原理？【字节/腾讯】
**答案：** PagingSource（load返回LoadResult）→PagingData→PagingAdapter。Pager创建Flow→cachedIn(viewModelScope)→collect到Adapter。RemoteMediator网络+本地联合分页。DiffUtil自动差异更新。

Q117. Hilt/Dagger依赖注入？【阿里/字节】
**答案：** 编译时生成代码。@Inject构造器；@Module+@Provides提供依赖；@Component连接。Hilt预定义Component@HiltAndroidApp初始化。Scope控制生命周期。KSP替代KAPT加速编译。

Q118. Jetpack Compose基本概念？【Google/字节】
**答案：** 声明式UI框架，@Composable标记函数。无XML布局、无findViewById、自动响应状态。Column/Row/Box替代LinearLayout。Modifier链式修饰。编译时优化生成高效代码。

Q119. Compose重组机制？【阿里/Google】
**答案：** State变化时读取该State的Composable重新执行。通过Snapshot追踪状态变化，智能跳过未变化范围。关键：避免Composable中做副作用。稳定key减少不必要重组。

Q120. Compose State和remember？【字节/腾讯】
**答案：** mutableStateOf创建可观察状态。remember在重组时保留值。rememberSaveable配置变更保留。derivedStateOf派生状态减少重组。collectAsState将Flow转为State。

Q121. Compose SideEffect API？【美团/Google】
**答案：** LaunchedEffect协程副作用（key变化重启）。DisposableEffect有清理的副作用。SideEffect每次重组执行。rememberCoroutineScope获取作用域。produceState将非Compose数据转State。

Q122. Compose动画？【腾讯/Google】
**答案：** animateXAsState（简单属性）、Animatable（灵活）、AnimatedVisibility（显隐）、AnimatedContent（内容切换）、Crossfade（交叉淡入）。spring物理动画、tween缓动、keyframes关键帧。

Q123. Compose Navigation？【阿里/Google】
**答案：** NavHost+NavController+composable()定义导航图。navArgument传参。deepLink URL跳转。LaunchedEffect配合ViewModel导航。AnimatedContent页面转场。

Q124. Lifecycle原理？【阿里/腾讯】
**答案：** LifecycleOwner提供Lifecycle，Observer通过@OnLifecycleEvent接收事件。LifecycleRegistry管理状态流转。Fragment/Activity默认实现。生命周期事件与Activity/Fragment对应。

Q125. SavedStateHandle？【字节/Google】
**答案：** ViewModel中存储进程被杀后可恢复的数据。asLiveData()/getStateFlow()暴露。区别于ViewModel普通数据（进程杀后丢失）。类似onSaveInstanceState但更方便类型安全。

Q126. App Startup？【美团/阿里】
**答案：** ContentProvider中初始化组件替代多个ContentProvider各自初始化。Initializer<T>定义逻辑。dependencies控制初始化顺序。减少ContentProvider数量提升启动速度。

Q127. Benchmark性能测试？【字节/Google】
**答案：** BenchmarkRule测代码执行时间。Microbenchmark测试小函数（进程内热循环高精度）。Macrobenchmark测试完整场景（启动/滚动独立进程）。CI集成自动化性能测试。

Q128. Room @Embedded和@Relation？【美团/小米】
**答案：** @Embedded嵌入对象字段到当前表（同表）。@Relation一对多关联查询通过外键（多表）。@Relation需@Transaction确保一致性。区别：@Embedded同表，@Relation多表关联。

Q129. Compose LazyColumn优化？【字节/阿里】
**答案：** items(key={})稳定key避免重组。contentType按类型优化回收。beyondBoundsItemCount预加载。避免在item中使用weight触发重新测量。derivedStateOf减少重组范围。

Q130. Compose和传统View混合？【字节/小米】
**答案：** AndroidView在Compose中嵌入传统View。ComposeView在传统View中嵌入Compose。互操作注意生命周期管理和状态同步。渐进迁移：新功能Compose旧页面逐步替换。

Q131. Jetpack Macrobenchmark？【阿里/Google】
**答案：** 测试完整用户场景（Activity启动/列表滚动/动画）。独立进程运行更接近真实环境。与Microbenchmark区别：Microbenchmark测试小代码块进程内运行精度高。自动化性能回归检测。


---

## 七、性能优化

Q132. 内存泄漏常见场景和检测？【字节/阿里】
**答案：** 1)Handler非静态内部类；2)单例持有Context（应传Application）；3)匿名内部类；4)未取消监听器；5)WebView泄漏。检测：LeakCanary实时、Android Profiler、MAT、StrictMode。

Q133. Bitmap内存优化？【腾讯/美团】
**答案：** 1)采样压缩inSampleSize；2)RGB_565替代ARGB_8888；3)inBitmap复用；4)三级缓存LruCache→DiskLruCache→网络；5)WebP格式；6)BitmapRegionDecoder局部加载。8.0+像素数据在Native堆。

Q134. LruCache原理？【字节/阿里】
**答案：** LinkedHashMap（accessOrder=true）get/put移到尾部，超maxSize驱逐头部。sizeOf自定义大小计算。maxSize建议maxMemory/8。synchronized线程安全。

Q135. SparseArray和ArrayMap？【字节/小米】
**答案：** SparseArray用int[] keys避免Integer装箱，二分查找。ArrayMap两个数组存hash和键值对。优势：无额外Entry减少GC、内存紧凑。缺点：大数据O(log n)。int→Object用SparseArray，Object→Object用ArrayMap。

Q136. ART和Dalvik GC区别？【阿里/字节】
**答案：** Dalvik标记-清除全暂停碎片化。ART并发标记减少STW、移动式压缩减少碎片、AOT+JIT混合。7.0+并发GC大幅减少卡顿。关注减少分配频率。

Q137. APK体积优化？【字节/美团】
**答案：** 1)ProGuard/R8代码混淆；2)shrinkResources资源压缩；3)移除未用资源；4)WebP图片；5)So动态下发；6)资源分包；7)精简三方库。APK Analyzer分析各部分占比。

Q138. 应用启动优化？【阿里/字节】
**答案：** 冷启动：进程→Application→Activity。热启动直接前台。温启动Activity重建。优化：Application.onCreate<100ms、异步初始化SDK、App Startup管理顺序、Theme背景避免白屏、预加载关键数据。

Q139. UI卡顿检测和优化？【腾讯/华为】
**答案：** Choreographer.FrameCallback监控帧率、BlockCanary检测卡顿。Systrace/Perfetto可视化。优化：减少主线程耗时、降低过度绘制、优化布局层级、RecyclerView优化、硬件加速。

Q140. ANR原因和排查？【阿里/美团】
**答案：** 主线程5秒无响应。排查：traces.txt堆栈、logcat搜索ANR、分析主线程阻塞原因（IO/锁/死循环）、Systrace查看CPU状态。避免异步处理耗时操作。

Q141. 线程优化和线程池？【字节/阿里】
**答案：** 避免频繁创建线程。ThreadPoolExecutor管理：corePoolSize、maxPoolSize、workQueue、handler。IO密集型core=CPU*2，CPU密集型core=CPU+1。AsyncTask废弃用协程替代。

Q142. I/O优化？【字节/腾讯】
**答案：** Buffer减少系统调用、批量事务、异步IO、mmap（MMKV）、SQLite WAL、MMKV替代SP、Okio高效IO。StrictMode检测主线程I/O。

Q143. 线上Crash监控？【阿里/字节】
**答案：** Firebase Crashlytics/自研SDK捕获上报。分析：堆栈定位、设备/系统分布。Java异常和Native崩溃分别处理。ANR用WatchDog监控。目标崩溃率<0.1%。

Q144. 电量优化？【华为/小米】
**答案：** JobScheduler/WorkManager批量唤醒CPU。避免WakeLock（必须用尽快释放）。网络请求合并、传感器合理采样、Doze适配。Battery Historian分析。AlarmManager用setExactAndAllowWhileIdle。

Q145. 包体积分析？【字节/腾讯】
**答案：** APK Analyzer查看dex/arsc/so/res大小。R8混淆+资源压缩。移除未用代码和资源。WebP图片。So split动态下发。精简三方库。主dex方法数<65535。

Q146. 渲染管线和16ms？【阿里/字节】
**答案：** CPU处理→GPU渲染→Display显示。VSync每16.67ms（60Hz）。渲染管线：测量→布局→绘制→DisplayList→栅格化→合成。掉帧原因：主线程超16ms或GPU过度绘制。

Q147. Systrace使用和分析？【腾讯/美团】
**答案：** 记录CPU调度/线程状态/系统调用/View绘制。分析：查看每帧是否超时、主线程状态（Running/Runnable/Sleeping）。Alert自动检测。Perfetto是新版Systrace。

Q148. 列表滑动流畅度优化？【字节/阿里】
**答案：** setHasFixedSize、DiffUtil异步差异、onBindViewHolder异步、图片异步+缩略图、预取、减少过度绘制、Hardware Layer、稳定ID避免全量刷新。

Q149. 多dex和类加载优化？【阿里/华为】
**答案：** 65536方法数限制。ProGuard移除未用代码。dex分包mainDexClassList。5.0+ART原生多dex。预加载常用类。dexmerge合并优化。

Q150. StrictMode使用场景？【字节/小米】
**答案：** 线程策略检测主线程IO/网络；VM策略检测Activity泄漏/未关闭Cursor。penaltyLog日志。开发阶段启用发布关闭。debug构建自动开启。

Q151. 如何监控和优化内存？【美团/华为】
**答案：** Debug.getNativeHeapSize跟踪Native内存。Runtime.totalMemory/freeMemory跟踪Java堆。ComponentCallbacks2监听onTrimMemory。优化：减少分配、泄漏检测、响应trimMemory释放资源。


---

## 八、Android Framework 深入

Q152. Android系统启动流程？【阿里/字节】
**答案：** BootLoader→Linux Kernel→init（解析init.rc）→Zygote（预加载+fork）→SystemServer（启动AMS/WMS/PMS等服务）→Launcher。Zygote预加载资源，fork时Copy-on-Write共享。

Q153. Zygote进程的作用？【字节/华为】
**答案：** 所有应用进程的父进程。预加载类和资源，fork新进程时Copy-on-Write共享内存减少启动时间。通过Socket接收创建进程请求。64位系统有Zygote和Zygote64。

Q154. SystemServer核心服务？【阿里/腾讯】
**答案：** AMS（Activity管理）、WMS（窗口管理）、PMS（包管理）、PowerManagerService、DisplayManagerService等。分三批启动：引导→核心→其他。运行在System进程。

Q155. AMS和Activity启动流程？【字节/阿里】
**答案：** startActivity→ActivityStarter→ActivityStack→ActivityThread（Binder调用）→performLaunchActivity→onCreate。AMS管理生命周期、任务栈、进程调度。客户端服务端通过Binder通信。

Q156. Handler机制？Looper/MessageQueue/Handler？【字节/腾讯】
**答案：** Handler发Message到MessageQueue（单链表），Looper.loop()取出Message分发。每线程一Looper（ThreadLocal），主线程自动prepareMainLooper。next()通过nativePollOnce（epoll）阻塞等待。

Q157. Handler内存泄漏问题？【阿里/美团】
**答案：** 非静态内部类隐式持有Activity引用。Message持有Handler→Handler持有Activity→无法GC。解决：静态内部类+WeakReference；onDestroy中removeCallbacksAndMessages(null)。

Q158. Looper.loop()为什么不会ANR？【腾讯/字节】
**答案：** loop()阻塞在MessageQueue.next()（nativePollOnce），线程处于epoll_wait不消耗CPU。ANR是处理消息超时不是loop阻塞。消息处理超5秒/10秒才会ANR。

Q159. 消息屏障SyncBarrier？【阿里/字节】
**答案：** Message.target为null的消息屏障。插入后异步消息优先处理（Choreographer的VSync消息）。postSyncBarrier→异步消息优先→removeSyncBarrier移除。保证渲染消息优先处理。

Q160. IdleHandler使用场景？【美团/腾讯】
**答案：** MessageQueue.IdleHandler在队列空闲时执行。queueIdle返回true则保留。场景：延迟初始化非关键组件、空闲时GC优化、空闲时数据库清理。Activity启动优化中延迟初始化。

Q161. 进程优先级？【阿里/华为】
**答案：** 前台>可见>服务>后台>空进程。oom_adj越小优先级越高。系统根据优先级回收。低内存时从低优先级杀。前台Service优先级最高（仅次于前台Activity）。

Q162. 多进程架构的优缺点？【字节/阿里】
**答案：** 优势：突破单进程内存限制、独立崩溃不影响主进程、独立GC减少卡顿。缺点：IPC开销、Application多次创建、静态变量不共享、SQLite多进程锁、SharedPreferences不可靠。

Q163. ClassLoader体系？【字节/阿里】
**答案：** BootClassLoader加载Framework类。PathClassLoader加载已安装应用DEX。DexClassLoader加载外部DEX（插件化）。双亲委派：先委托父加载器加载，找不到再自己加载。

Q164. Binder驱动原理？【阿里/字节】
**答案：** 一次拷贝（用户空间→内核Binder驱动）。C/S架构+ServiceManager+Binder驱动。比Socket高效（减少拷贝）比共享内存安全（UID/PID标识）。通过BinderProxy和Stub代理透明调用。

Q165. Application.onCreate优化？【美团/阿里】
**答案：** 异步初始化非关键SDK、按需延迟初始化、App Startup管理顺序、预加载。目标<100ms。Trace追踪耗时操作。按功能分批初始化。

Q166. 插件化原理？【阿里/字节】
**答案：** Hook ClassLoader加载插件DEX、Hook AMS拦截组件启动、Hook Resource加载资源、Proxy代理模式。代表框架：VirtualAPK、RePlugin、Shadow。核心拦截系统调用加载未安装APK。

Q167. 热修复原理？Tinker/Sophix？【字节/阿里】
**答案：** 类加载方案（Tinker）：合成修复DEX→插入DexPathList前面→优先加载。底层替换（Sophix）：替换ArtMethod结构体即时生效。类加载兼容性好但不能即时生效；底层替换即时但兼容性差。

Q168. Hook技术？【阿里/字节】
**答案：** 反射替换系统对象拦截调用。Hook AMS拦截Activity启动、Hook Instrumentation拦截创建、Hook ClassLoader加载插件、Hook Handler处理消息。核心：动态代理+反射修改静态字段。

Q169. Android中的Context体系？【字节/阿里】
**答案：** Context抽象类。Activity Context关联Window可弹出Dialog；Application Context全局唯一不可弹出Dialog。Service Context也不能弹出Dialog。单例应使用Application Context避免泄漏。

Q170. APK文件结构？【阿里/华为】
**答案：** META-INF/（签名）、lib/（Native库）、res/（资源）、assets/（原始文件）、AndroidManifest.xml、classes.dex（字节码）、resources.arsc（资源映射）。APK本质ZIP文件。V2签名后不能直接修改条目。


---

## 九、架构设计模式

Q171. MVP和MVVM的区别？【阿里/字节】
**答案：** MVP：View-Presenter-Model，通过接口交互，接口爆炸、生命周期管理复杂。MVVM：数据驱动UI，ViewModel通过LiveData暴露数据，View自动更新。减少模板代码、生命周期安全、可测试性好。Jetpack天然支持MVVM。

Q172. Clean Architecture分层？【阿里/Google】
**答案：** Presentation（UI+ViewModel）→Domain（UseCase+Entity+Repository接口）→Data（Repository实现+数据源）。外层依赖内层不反向。可测试性高各层独立替换。

Q173. Repository模式？【字节/腾讯】
**答案：** 统一数据访问抽象层屏蔽数据来源差异。ViewModel只依赖Repository接口。实现：先缓存→再数据库→最后网络→更新各层。单一数据源、可测试、数据获取策略集中管理。

Q174. 组件化和模块化区别？【阿里/字节】
**答案：** 模块化按功能拆分编译仍在一起。组件化每模块可独立编译运行（isModule开关）。组件化需解决路由（ARouter）、数据传递、资源冲突、Application初始化。

Q175. ARouter路由原理？【美团/阿里】
**答案：** 注解@Route→编译时APT生成路由表→运行时通过路径查找目标类跳转。支持拦截器（登录检查）、依赖注入（Autowired）、降级策略。跨模块解耦无需直接依赖。

Q176. EventBus/RxBus原理？【字节/腾讯】
**答案：** EventBus：注解+反射注册/注销，@Subscribe接收事件，支持线程模式。RxBus：RxJava PublishSubject。都是观察者模式。RxBus更灵活但需管理订阅生命周期。

Q177. 依赖注入和IoC？【阿里/Google】
**答案：** IoC外部容器管理对象创建和依赖。DI是IoC实现：构造器注入、Setter注入、字段注入。Dagger/Hilt编译时生成注入代码避免运行时反射。解耦利于测试。

Q178. 单例模式注意事项？【美团/字节】
**答案：** Application Context避免内存泄漏。DCL+volatile或静态内部类线程安全。枚举单例最安全。考虑Dagger @Singleton管理。可测试性差（全局状态）。

Q179. 观察者模式在Android中？【阿里/腾讯】
**答案：** LiveData、RxJava、EventBus、View.OnClickListener、ContentObserver都是观察者模式。被观察者维护列表状态变化时通知。LiveData生命周期感知是增强版。避免onDestroy后收到通知。

Q180. Builder模式应用？【字节/小米】
**答案：** AlertDialog.Builder、OkHttpClient.Builder、NotificationCompat.Builder。链式调用设置可选参数，最后build()创建对象。适用于参数多有默认值的场景。比构造器参数过多更清晰。

Q181. 图片加载框架设计？【字节/阿里】
**答案：** 请求队列（优先级调度）→三级缓存（LruMemory→DiskCache→Network）→图片解码（采样压缩）→生命周期管理（自动取消）→图片变换（圆角模糊裁剪）→占位图错误图。参考Glide架构。

Q182. 网络框架封装设计？【美团/阿里】
**答案：** 分层：API接口层（Retrofit注解）→Repository层（数据源策略）→统一响应Result<T>（Success/Error/Loading）→错误处理（统一异常类+重试）→拦截器（Token/日志/缓存）→生命周期管理。

Q183. 组件间通信方案？【字节/阿里】
**答案：** 路由框架（ARouter页面跳转）、事件总线（EventBus/RxBus）、接口下沉（公共模块定义接口）、AIDL跨进程、ContentProvider数据共享。推荐接口下沉+路由组合。

Q184. 埋点框架设计？【阿里/腾讯】
**答案：** 代码埋点/声明式埋点（注解+APT）/全埋点（AOP拦截View事件）。事件收集→本地存储→批量上报。数据格式统一、支持实时/批量。大厂方案：无痕埋点+可视化圈选。

Q185. 日志系统设计？【字节/华为】
**答案：** 多级别（V/D/I/W/E）、多输出目标（Logcat/文件/远程）、文件滚动（按大小时间切分）、异步写入Buffer、线上日志采样上报。Timber库简化。开关控制不同环境。

Q186. MVI架构模式？【阿里/Google】
**答案：** Model-View-Intent单向数据流。Intent（用户意图）→ViewModel处理→State（不可变）→View渲染。比MVVM更可预测，State不可变避免竞态。适合复杂状态管理。Compose天然支持MVI。

Q187. 多渠道打包方案？【阿里/美团】
**答案：** Gradle Flavor（慢每渠道独立构建）。美团Walle（快APK文件中插入渠道信息）。360渠道打包。AndroidManifest meta-data注入渠道号。V2签名后修改渠道需特殊处理。

Q188. 配置中心设计？【美团/阿里】
**答案：** 远程配置→本地缓存→默认值兜底→LiveData通知变化。按用户/设备灰度、A/B测试集成。Apollo/自研方案。版本管理（配置版本+应用版本）。实时监听配置变更。


---

## 十、安全与逆向

Q189. Android应用签名v1/v2/v3？【阿里/华为】
**答案：** v1 JAR签名（META-INF目录）；v2 APK整体签名更快更安全；v3支持密钥轮换；v4增量签名。7.0+推荐v2，11+强制v2+。签名确保APK完整性和来源验证。

Q190. 运行时权限请求流程？【字节/腾讯】
**答案：** checkSelfPermission→requestPermissions→onRequestPermissionsResult。普通权限自动授予。危险权限（相机/位置/存储）运行时请求。权限组概念。11+存储权限改为READ_MEDIA_*。

Q191. 数据加密方式？【阿里/华为】
**答案：** 对称加密AES（256位推荐）；非对称加密RSA；哈希SHA-256（MD5不安全）。AES加密数据、RSA交换密钥、SHA-256校验完整性。Android KeyStore系统保护密钥。HMAC消息认证码。

Q192. ProGuard/R8代码混淆？【美团/阿里】
**答案：** 代码混淆（重命名）、优化（移除未用代码）、压缩（移除未用资源）、预验证。-keep保留、-dontoptimize关闭优化。R8是ProGuard替代更快。反射类需keep。混淆增加逆向成本。

Q193. Root检测和反调试？【华为/阿里】
**答案：** 检查su文件、Build.TAGS test-keys、Superuser应用、SafetyNet API。反调试：ptrace防止附加、检测调试标志、时间检测（调试时变慢）。代码混淆+加壳增加逆向难度。

Q194. KeyStore安全存储？【阿里/字节】
**答案：** 硬件级安全存储密钥不可导出。支持AES/RSA/EC。KeyChain系统级证书管理。EncryptedSharedPreferences用KeyStore加密SP。密钥绑定认证setUserAuthenticationRequired。

Q195. 防止APK被反编译？【阿里/美团】
**答案：** ProGuard/R8混淆、资源混淆（AndResGuard）、加壳（360加固/梆梆）、NDK Native代码、代码加密+动态解密、反调试检测。没有绝对安全只能增加逆向成本。

Q196. 应用加固原理？【腾讯/阿里】
**答案：** 壳DEX解密加载原始DEX。壳Application先执行→解密原始DEX→DexClassLoader加载→反射调用原始Application。360加固、腾讯乐固、梆梆。脱壳：运行时内存dump。

Q197. OWASP Mobile Top 10？【阿里/华为】
**答案：** 1)不安全平台使用；2)数据存储；3)通信；4)认证；5)密码学；6)授权；7)代码质量；8)代码篡改；9)逆向工程；10)多余功能。防御：加密存储、HTTPS、输入验证、混淆。

Q198. 网络安全配置？【字节/华为】
**答案：** network_security_config.xml：证书固定（pin-set）、自定义CA（trust-anchors）、禁止明文（cleartextTrafficPermitted=false）。HTTPS全站+CertificatePinner防中间人。usesCleartextTraffic=false。


---

## 十一、测试与调试

Q199. 单元测试JUnit/Mockito？【字节/阿里】
**答案：** JUnit测试普通类。Mockito mock依赖。@Test标记、assertEquals断言、@Before/@After设置清理。协程用runTest。ViewModel用InstantTaskExecutorRule。

Q200. UI测试Espresso？【腾讯/美团】
**答案：** ViewMatchers查找、ViewActions操作、ViewAssertions断言。onView(withId()).perform(click()).check(matches(isDisplayed()))。IdlingResource处理异步。DataBinding需executePendingBindings。

Q201. Robolectric原理？【阿里/Google】
**答案：** JVM上模拟Android环境不需要真机。@RunWith(RobolectricTestRunner.class)。支持Context/Resources/SharedPreferences。测试速度快（秒级）。Shadow类模拟系统行为。

Q202. ViewModel和LiveData测试？【Google/阿里】
**答案：** 直接实例化传Mock Repository。InstantTaskExecutorRule同步LiveData。TestObserver观察变化。协程ViewModel用runTest。StateFlow用Turbine测试。

Q203. 性能测试工具？【字节/腾讯】
**答案：** Android Profiler（CPU/Memory/Network）、Systrace/Perfetto、LeakCanary、BlockCanary、Jetpack Benchmark、Firebase Test Lab。多维度监控定位性能瓶颈。

Q204. Memory Profiler使用？【阿里/华为】
**答案：** 查看Java/Native/Stack/Graphics内存。Dump Heap查看对象分配。Allocation Tracker追踪分配位置。分析泄漏：Activity/Fragment数量异常增长。实时内存曲线监控。

Q205. ADB常用命令？【华为/阿里】
**答案：** logcat日志、shell进入设备、install/uninstall安装卸载、push/pull传输、forward端口转发、dumpsys系统服务状态、am start启动Activity、pm list packages。

Q206. Monkey测试？【百度/美团】
**答案：** adb shell monkey -p package -v eventCount随机事件测试稳定性。可指定事件比例/种子值复现。发现崩溃和ANR。MonkeyRunner可编写脚本。

Q207. 兼容性测试策略？【阿里/腾讯】
**答案：** 覆盖主流机型和系统版本。云测平台：Firebase Test Lab、阿里云测、腾讯WeTest。自动化多设备并行。关注权限差异、API差异、ROM定制行为。

Q208. APK分析工具？【字节/华为】
**答案：** APK Analyzer查看DEX/arsc/so大小。jadx反编译。apktool反编译资源。classyshark查看类。dexdump分析DEX。Proguard mapping.txt混淆映射。


---

## 十二、新技术与趋势

Q209. Compose相比传统View优势？【Google/字节】
**答案：** 声明式UI、无XML、更少代码、实时预览、强类型安全、易测试、Kotlin深度集成。Animation/Theme现代化。缺点：学习曲线、编译时长。

Q210. Compose重组优化技巧？【Google/阿里】
**答案：** key参数稳定列表项、derivedStateOf减少重组、remember缓存、避免Composable中创建新对象、@Stable标记稳定类型、contentType优化LazyList回收。

Q211. Kotlin Multiplatform KMP？【JetBrains/字节】
**答案：** 共享业务逻辑代码（非UI）。expect/actual平台特定实现。Compose Multiplatform共享UI。适合共享网络层/数据层/业务逻辑。支持Android/iOS/JVM/JS。

Q212. Flutter和原生区别？【Google/阿里】
**答案：** Flutter用Dart自绘引擎Skia一套代码多平台。优势：UI一致、热重载、开发效率。劣势：包体积大、平台特性支持滞后、复杂动画性能不如原生。

Q213. React Native原理？【字节/腾讯】
**答案：** JS通过Bridge与Native通信。新架构Fabric（同步渲染）、TurboModules（延迟加载）、Codegen（类型安全）。Hermes引擎提升JS执行。优势：热更新、JS生态。

Q214. KSP和KAPT区别？【Google/字节】
**答案：** KAPT转Java Stub再用APT处理慢且冗余。KSP直接处理Kotlin符号速度2倍。支持Kotlin完整特性。Room/Hilt已支持KSP。推荐新项目使用KSP。

Q215. Android 14/15新特性？【Google/小米】
**答案：** 14：预测性返回手势、照片选择器、前台服务类型限制、Credential Manager。15：Edge-to-Edge强制全屏、16KB页面支持、Passkeys改进、卫星通信。

Q216. Material Design 3？【Google/阿里】
**答案：** Material You动态颜色从壁纸提取主题色。组件更新：可变形状、更大触摸目标。排版系统（Display/Headline/Title/Body/Label）。Motion规范。

Q217. ML Kit机器学习？【Google/字节】
**答案：** 设备端：文字识别、条码扫描、人脸检测、图像标注、物体检测、姿态估计。云端API更强。TFLite支持自定义模型。移动端ML开箱即用。

Q218. Wear OS/TV/Automotive开发？【Google/小米】
**答案：** Wear OS：低功耗、圆形屏幕适配、Tiles。TV：Leanback库、D-pad焦点管理、大屏布局。Automotive：Car App Library、安全限制、语音控制。各平台特定适配。


---

## 十三、项目经验与综合

Q219. 最有挑战的Android项目？【综合】
**答案：** STAR法则：情境→任务→行动→结果。描述技术难点（性能/架构/兼容）、解决方案、效果指标。示例：内存泄漏导致OOM→LeakCanary定位→重构Handler→崩溃率下降90%。

Q220. Android应用性能调优思路？【阿里/字节】
**答案：** 1)定义性能指标（启动/帧率/内存/崩溃率）；2)监控APM；3)定位瓶颈（Profiler/Systrace）；4)针对性优化；5)回归测试；6)持续监控。建立基线和SLA。

Q221. 版本兼容处理？【腾讯/华为】
**答案：** minSdkVersion最低版本、@RequiresApi标注高版本API、Support库/AndroidX兼容、运行时Build.VERSION.SDK_INT检查、多版本资源values-v21。关注21/23/26/29/31/33。

Q222. 保证应用质量？【阿里/Google】
**答案：** 代码规范ktlint/Checkstyle、Code Review、单元测试>70%覆盖、UI测试关键路径、集成测试、静态分析Lint/SonarQube、CI/CD自动化、灰度发布。

Q223. 灰度发布策略？【字节/美团】
**答案：** Remote Config控制开关、配置中心按设备灰度、应用市场分阶段发布（1%→5%→20%→100%）、Feature Flag。监控关键指标发现问题立即回滚。

Q224. 崩溃率监控优化？【阿里/腾讯】
**答案：** Crashlytics监控崩溃率和ANR率。分类分析Top崩溃逐个修复。防御性编程、异常兜底。目标崩溃率<0.1%。建立崩溃分级响应机制。

Q225. 高质量SDK设计原则？【阿里/字节】
**答案：** 接口简洁稳定（语义化版本）、最小依赖、不侵入业务、可配置Builder、完善错误处理、线程安全、文档示例、向后兼容。Facade模式对外提供简单API。

Q226. Kotlin vs Java选型？【字节/阿里】
**答案：** 推荐Kotlin：空安全、协程、扩展函数、data class、DSL、Google官方推荐。新代码用Kotlin逐步迁移。性能编译后字节码相似。Compose必须Kotlin。

Q227. 多语言国际化？【阿里/华为】
**答案：** strings.xml外部化、values-xx多语言目录、复数plurals、格式化本地化、RTL适配（start/end）、AppCompat LocaleList。翻译管理Weblate/Crowdin。避免拼接字符串。

Q228. 暗黑模式适配？【Google/小米】
**答案：** values/colors.xml和values-night/colors.xml两套颜色。使用?attr/colorPrimary替代硬编码。图片提供暗色版本或tint。WebView prefers-color-scheme。AppCompatDelegate.setDefaultNightMode()。

Q229. A/B测试实现？【字节/阿里】
**答案：** 定义实验指标→随机分组→配置不同行为→收集数据→统计分析。Firebase A/B Testing/自研。功能开关控制。注意样本量、实验时长、避免辛普森悖论。

Q230. 推送系统设计？【阿里/小米】
**答案：** 长连接（WebSocket/MQTT保活）+系统级推送（FCM/厂商通道）。推送协议（通知/透传）。到达率统计、消息优先级、分群推送。组合FCM+厂商通道保活。

Q231. 动态化方案选择？【阿里/字节】
**答案：** React Native：JS Bridge热更新强。Flutter：自绘引擎性能好。小程序容器：微信/支付宝。插件化：动态加载DEX。选择：团队技术栈、性能要求、热更新频率。

Q232. 安全审计要点？【华为/阿里】
**答案：** 静态分析（敏感信息/硬编码密钥）、动态分析（抓包加密/证书固定）、权限审计、存储审计、组件暴露审计、代码混淆检查、三方库漏洞扫描。

Q233. 崩溃分析思路？【字节/阿里】
**答案：** Java：查看堆栈→定位代码→分析原因（空指针/越界/并发）。Native：tombstone→backtrace→信号类型→maps查看库→addr2line解析源码行号。常见野指针/内存越界/多线程竞争。

Q234. 多渠道打包？【阿里/美团】
**答案：** Gradle Flavor每渠道独立构建。美团Walle在APK中插入渠道信息（快）。V2签名后需特殊处理。AndroidManifest meta-data注入渠道号。选择Walle效率最高。

Q235. 埋点数据分析？【字节/阿里】
**答案：** 代码埋点/声明式埋点（注解+APT）/全埋点（AOP）。事件定义→收集→本地存储→批量上报→分析。关注准确性、完整性、上报性能。大厂方案：无痕埋点+可视化圈选。


---

## 十四、高频面试真题

Q236. 为什么主线程可以直接new Handler？【腾讯/字节】
**答案：** 主线程ActivityThread.main()已自动调用Looper.prepareMainLooper()和loop()。子线程默认无Looper需先prepare()→new Handler→loop()。每线程只能一个Looper。

Q237. 子线程如何使用Handler？【美团/阿里】
**答案：** 方式1：Looper.prepare()→new Handler→Looper.loop()手动管理。方式2：HandlerThread自带Looper。方式3：协程替代。HandlerThread内部有Looper可getLooper()创建Handler。

Q238. View.post和Handler.post区别？【字节/腾讯】
**答案：** View.post投递到ViewRootImpl消息队列，View attached后执行，适合onCreate获取View宽高（已measure完）。Handler.post投递到指定Handler。View.post在未attach时缓存。

Q239. AsyncTask原理和缺陷？【阿里/腾讯】
**答案：** ThreadPoolExecutor+Handler。doInBackground工作线程，onPostExecute主线程。缺陷：内存泄漏、串行执行默认、配置变更丢失结果、异常处理弱。替代：协程/ExecutorService。

Q240. 为什么AsyncTask被废弃？【字节/Google】
**答案：** 生命周期管理差致内存泄漏、配置变更数据丢失、异常处理弱、默认串行效率低、API不灵活。替代：Kotlin协程（推荐）、RxJava、ExecutorService+Handler、WorkManager。

Q241. Android进程间通信方式？【阿里/华为】
**答案：** Binder（最常用AIDL）、Socket/管道、文件共享、ContentProvider（基于Binder）、Messenger（基于Binder的Handler）、BroadcastReceiver。Binder最高效一次拷贝。

Q242. 多进程导致的问题？【字节/阿里】
**答案：** Application多次创建、静态变量不共享、线程同步失效、SQLite并发锁、SharedPreferences不可靠、单例失效。解决方案：MMKV替代SP、ContentProvider共享数据。

Q243. 64K方法数限制？【美团/阿里】
**答案：** DEX的short类型方法引用索引限制65536。MultiDex拆分多个DEX。反射修改PathList添加附加DEX。5.0+ART原生支持。优化：ProGuard移除未用方法、按需依赖。

Q244. AsyncTask替代方案？【字节/Google】
**答案：** Kotlin协程（最推荐）：launch/async+withContext。RxJava：Observable调度线程。ExecutorService+Handler。WorkManager后台任务。协程最简洁生命周期安全。

Q245. Android中如何在子线程更新UI？【字节/美团】
**答案：** runOnUiThread()、View.post()/postDelayed()、Handler.sendMessage()、协程withContext(Dispatchers.Main)。View.checkThread()检查主线程否则抛CalledFromWrongThreadException。

Q246. Context体系？Activity和Application区别？【字节/阿里】
**答案：** Activity Context关联Window可弹出Dialog；Application Context全局唯一不可弹出Dialog。Service Context也不能弹出Dialog。单例应使用Application Context避免泄漏。getApplicationContext()获取。

Q247. DEX文件结构？【字节/阿里】
**答案：** header文件头、string_ids/type_ids/proto_ids/field_ids/method_ids/class_defs各种ID表、data数据区。每DEX最多65536个方法引用。Dalvik可执行文件格式。

Q248. ODEX/AOT/JIT编译模式？【阿里/字节】
**答案：** ODEX安装时预编译为OAT（5.0-6.0 ART）。AOT安装或空闲时编译热点代码（7.0+ profile-based）。JIT运行时编译热点。7.0+混合模式JIT+AOT最佳平衡启动速度和运行性能。

Q249. Application为什么要轻量？【美团/阿里】
**答案：** onCreate在主线程执行影响冷启动速度。优化：异步初始化、按需延迟、App Startup管理顺序。目标<100ms。初始化时间直接影响白屏时长。

Q250. SharedUserId作用？【华为/阿里】
**答案：** 多应用共享同一Linux用户ID（相同签名前提）运行在同一进程共享数据和权限。安全风险需相同签名。现代不推荐，应用间共享用ContentProvider替代。

Q251. StrictMode是什么？【小米/字节】
**答案：** 检测违规操作。线程策略检测主线程IO/网络；VM策略检测Activity泄漏/未关闭Cursor。penaltyLog输出。开发阶段启用发布关闭。debug构建自动开启。

Q252. Android中的插桩Instrumentation？【字节/腾讯】
**答案：** 拦截Activity/Fragment生命周期。AndroidJUnitRunner继承它。用于测试Mock系统行为、自动化测试控制应用、性能监控。AndroidManifest注册自定义Instrumentation。

Q253. AppCompatActivity为什么要用？【阿里/小米】
**答案：** 向后兼容Activity实现，AppCompatDelegate代理版本差异。功能：Toolbar替代ActionBar、矢量图兼容、主题颜色系统、夜间模式。核心代理类Delegate。

Q254. MessageQueue.next()阻塞原理？【字节/阿里】
**答案：** nativePollOnce调用Linux epoll阻塞等待。有消息计算等待时间→epoll_wait超时→唤醒取出。无消息timeout=-1永久阻塞。IdleHandler在空闲时执行。epoll高效I/O多路复用。


---

## 十五、Android 算法与数据结构

Q255. 手写LRU Cache？【字节/阿里】
**答案：** LinkedHashMap（accessOrder=true）实现。put超maxSize移除头部。get自动移到尾部。sizeOf自定义大小。线程安全加synchronized。核心：最近最少使用淘汰策略。

Q256. 手写单例模式？【美团/字节】
**答案：** DCL：volatile+synchronized双重检查。静态内部类：首次使用加载JVM保证线程安全。枚举：最简洁JVM保证线程安全防反射。推荐枚举单例。

Q257. 手写生产者消费者？【阿里/腾讯】
**答案：** BlockingQueue：put阻塞入队take阻塞出队。或wait/notify：synchronized+while条件wait+notifyAll。协程Channel。注意虚假唤醒用while而非if。

Q258. 手写事件总线？【腾讯/美团】
**答案：** Map<Class, List<Observer>>存储。register/unregister管理。post遍历通知。支持线程切换。反射调用订阅方法。可参考EventBus源码实现。

Q259. 手写图片加载器三级缓存？【字节/阿里】
**答案：** LruMemoryCache→DiskLruCache→Network。异步线程池。采样压缩inSampleSize。生命周期绑定Activity。占位图和错误图。核心流程：查内存→查磁盘→下载→缓存。

Q260. 手写RecyclerView Adapter？【美团/小米】
**答案：** 继承Adapter<VH>实现getItemCount/onCreateViewHolder/onBindViewHolder。ViewHolder缓存findViewById。多类型：getItemViewType。DiffUtil高效刷新。点击事件接口回调。

Q261. 简单依赖注入容器？【阿里/字节】
**答案：** Map<Class, Object>存储实例。register注册。get通过反射创建。构造器注入解析参数递归。单例缓存。可扩展注解扫描@Inject/@Singleton。

Q262. 简单路由框架？【阿里/字节】
**答案：** @Route(path)注解→编译时生成路由表Map<path, Class>。初始化扫描注册。跳转查找Class→startActivity。支持拦截器、参数Bundle、降级。参考ARouter。

Q263. 简单日志框架？【美团/阿里】
**答案：** 多级别V/D/I/W/E。格式化输出时间戳/Tag/消息/线程。多输出Logcat/文件/远程。文件滚动切分。异步写入Buffer。开关控制。参考Timber。

Q264. 简单网络缓存策略？【字节/美团】
**答案：** Cache-Control强缓存/ETag协商缓存/自定义时间策略。拦截器检查缓存→有效返回→无效请求网络→更新缓存。OkHttp Cache实现HTTP缓存。LruCache内存+DiskCache磁盘。

Q265. 简单数据库ORM？【阿里/腾讯】
**答案：** 注解标记Entity/字段→反射获取表结构→自动生成CRUD SQL。建表SQL生成、Cursor互转、类型适配TypeConverter。Room是完整ORM编译时生成代码。

Q266. 简单权限请求封装？【字节/华为】
**答案：** checkSelfPermission→请求未授权→回调结果。处理永久拒绝引导设置。多权限批量请求。权限组管理。Kotlin DSL风格。RxPermission/PermissionsDispatcher简化。

Q267. 图片压缩算法？【美团/阿里】
**答案：** 尺寸压缩inSampleSize采样→质量压缩Bitmap.compress(JPEG,quality)→JNI libjpeg高效压缩→鲁班算法（采样+质量联合）。先采样到目标尺寸再质量压缩到目标大小。

Q268. 简单热更新方案？【阿里/字节】
**答案：** 合成修复DEX→插入DexPathList.dexElements前面→优先加载修复类。服务端下发差分包→客户端合并→反射修改PathClassLoader。需考虑已加载类和多DEX。

Q269. 简单下拉刷新实现？【字节/美团】
**答案：** 自定义ViewGroup包裹Header+RecyclerView。触摸事件拦截下拉显示Header。松手触发刷新回调。ValueAnimator回弹动画。状态机：正常→下拉→释放刷新→刷新中→完成。

Q270. 简单埋点SDK？【字节/腾讯】
**答案：** Event数据结构（event_id/params/timestamp）→本地SQLite存储→定时定量批量HTTP上报。支持实时/批量。WiFi环境/定时器/积攒阈值策略。异常重试和去重。

Q271. 简单加密工具类？【华为/阿里】
**答案：** AES：SecretKeySpec+IvParameterSpec+Cipher。RSA：KeyPairGenerator+Cipher。SHA-256：MessageDigest。Base64编码。统一接口encrypt/decrypt/hash。密钥用KeyStore保护。

Q272. 简单配置管理？【阿里/小米】
**答案：** Map存储配置→远程拉取→SP本地缓存→默认值兜底→LiveData通知变化。按条件灰度（用户ID取模）。版本管理。实时监听变更。

Q273. 简单数据加密？【阿里/华为】
**答案：** 对称AES加密大量数据；非对称RSA交换密钥；哈希SHA-256完整性校验；Base64编码传输。KeyStore保护密钥。分层加密：RSA加密AES密钥+AES加密数据。


---

## 十六、Gradle 与构建系统

Q274. Gradle的构建生命周期？配置→执行→任务？【阿里/字节】
**答案：** 关于Gradle的构建生命周期？配置→执行→任务的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q275. Gradle Plugin的编写方法？【美团/阿里】
**答案：** 关于Gradle Plugin的编写方法的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q276. Gradle依赖管理？implementation/api的区别？【字节/腾讯】
**答案：** 关于Gradle依赖管理？implementation/api的区别的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q277. Gradle多模块项目配置？【阿里/小米】
**答案：** 关于Gradle多模块项目配置的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q278. Android Gradle Plugin（AGP）的作用？【Google/阿里】
**答案：** 关于Android Gradle Plugin（AGP）的作用的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q279. Gradle Transform API的作用？【阿里/字节】
**答案：** 关于Gradle Transform API的作用的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q280. Gradle构建优化？并行构建/缓存/守护进程？【美团/阿里】
**答案：** 关于Gradle构建优化？并行构建/缓存/守护进程的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q281. Gradle中如何管理版本号？【小米/字节】
**答案：** 关于Gradle中如何管理版本号的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q282. Gradle中如何处理依赖冲突？【阿里/美团】
**答案：** 关于Gradle中如何处理依赖冲突的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q283. Gradle中的Task依赖和增量构建？【字节/阿里】
**答案：** 关于Gradle中的Task依赖和增量构建的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q284. AAPT2资源编译流程？【阿里/Google】
**答案：** 关于AAPT2资源编译流程的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q285. D8/R8编译器的作用？DEX文件生成？【Google/阿里】
**答案：** 关于D8/R8编译器的作用？DEX文件生成的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q286. Gradle中的Flavor和BuildType区别？【美团/字节】
**答案：** 关于Gradle中的Flavor和BuildType区别的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q287. Gradle中如何自定义资源处理？【阿里/小米】
**答案：** 关于Gradle中如何自定义资源处理的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q288. Gradle中的签名配置？【字节/华为】
**答案：** 关于Gradle中的签名配置的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q289. Gradle缓存机制？离线模式？【阿里/腾讯】
**答案：** 关于Gradle缓存机制？离线模式的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q290. Kotlin DSL vs Groovy DSL构建脚本？【JetBrains/阿里】
**答案：** 关于Kotlin DSL vs Groovy DSL构建脚本的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q291. Gradle中的BuildConfig生成原理？【字节/小米】
**答案：** 关于Gradle中的BuildConfig生成原理的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q292. Gradle中的Lint检查配置？【阿里/腾讯】
**答案：** 关于Gradle中的Lint检查配置的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。

Q293. 如何做Gradle插件的单元测试？【美团/阿里】
**答案：** 关于如何做Gradle插件的单元测试的核心要点需结合Android构建系统的实际场景深入理解。Gradle是基于Groovy/Kotlin DSL的构建工具，AGP是Android专用插件，负责APK打包、资源编译、代码混淆等构建流程。


---

## 十七、NDK与JNI开发

Q294. JNI的使用方法？Java调C/C++？【字节/阿里】
**答案：** 关于JNI的使用方法？Java调C/C++的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q295. NDK开发的基本流程？【华为/阿里】
**答案：** 关于NDK开发的基本流程的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q296. JNI中的数据类型映射？【字节/腾讯】
**答案：** 关于JNI中的数据类型映射的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q297. JNI中的字符串传递？UTF-8/UTF-16？【阿里/华为】
**答案：** 关于JNI中的字符串传递？UTF-8/UTF-16的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q298. JNI中的异常处理？【字节/美团】
**答案：** 关于JNI中的异常处理的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q299. JNI中的引用管理？局部引用/全局引用？【阿里/字节】
**答案：** 关于JNI中的引用管理？局部引用/全局引用的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q300. CMake和ndk-build的区别？【Google/华为】
**答案：** 关于CMake和ndk-build的区别的核心方案需结合华为的业务场景和技术栈进行系统设计。涉及系统级优化、深度定制、性能调优等关键技术点。

Q301. Android中的Native内存管理？【字节/阿里】
**答案：** 关于Android中的Native内存管理的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。

Q302. 如何做Native崩溃分析？【阿里/腾讯】
**答案：** 关于如何做Native崩溃分析的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q303. NDK中的性能优化技巧？【字节/华为】
**答案：** 关于NDK中的性能优化技巧的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q304. FFmpeg在Android中的集成？【阿里/字节】
**答案：** 关于FFmpeg在Android中的集成的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q305. OpenCV在Android中的使用？【华为/阿里】
**答案：** 关于OpenCV在Android中的使用的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q306. Vulkan图形API在Android中的应用？【Google/小米】
**答案：** 关于Vulkan图形API在Android中的应用的核心方案需结合小米的业务场景和技术栈进行系统设计。涉及MIUI定制、系统优化、用户体验等关键技术点。

Q307. Android中的OpenGL ES开发？【阿里/华为】
**答案：** 关于Android中的OpenGL ES开发的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q308. JNI线程和Java线程的关系？【字节/阿里】
**答案：** 关于JNI线程和Java线程的关系的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q309. NDK中的日志输出？__android_log_print？【华为/小米】
**答案：** 关于NDK中的日志输出？__android_log_print的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q310. Android中的动态链接库加载？System.loadLibrary？【阿里/字节】
**答案：** 关于Android中的动态链接库加载？System.loadLibrary的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。

Q311. JNI中的方法签名规则？【字节/腾讯】
**答案：** 关于JNI中的方法签名规则的核心要点需结合JNI/NDK开发的实际场景深入理解。JNI是Java与Native代码互操作的标准接口，NDK提供C/C++开发Android应用的工具链。

Q312. Android中的ABI管理？armeabi-v7a/arm64-v8a/x86？【阿里/华为】
**答案：** 关于Android中的ABI管理？armeabi-v7a/arm64-v8a/x86的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q313. 如何减少APK中Native库的大小？【字节/美团】
**答案：** 关于如何减少APK中Native库的大小的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。


---

## 十八、音视频开发

Q314. Android中的MediaPlayer和MediaCodec区别？【阿里/字节】
**答案：** 关于Android中的MediaPlayer和MediaCodec区别的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q315. Camera2 API的使用？【字节/华为】
**答案：** 关于Camera2 API的使用的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q316. Android中的音频采集AudioRecord？【阿里/腾讯】
**答案：** 关于Android中的音频采集AudioRecord的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q317. 音视频编解码的基本概念？H.264/H.265/AAC？【字节/阿里】
**答案：** 关于音视频编解码的基本概念？H.264/H.265/AAC的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q318. Android中的视频播放器如何实现？【阿里/腾讯】
**答案：** 关于Android中的视频播放器如何实现的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q319. MediaExtractor和MediaMuxer的作用？【字节/华为】
**答案：** 关于MediaExtractor和MediaMuxer的作用的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q320. OpenGL ES在视频渲染中的应用？【阿里/小米】
**答案：** 关于OpenGL ES在视频渲染中的应用的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q321. Android中的音视频同步策略？【字节/阿里】
**答案：** 关于Android中的音视频同步策略的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q322. 直播推流和拉流的基本原理？【阿里/腾讯】
**答案：** 关于直播推流和拉流的基本原理的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q323. WebRTC在Android中的应用？【字节/阿里】
**答案：** 关于WebRTC在Android中的应用的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q324. Android中的音频焦点管理？【小米/华为】
**答案：** 关于Android中的音频焦点管理的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q325. ExoPlayer和MediaPlayer的区别？【Google/阿里】
**答案：** 关于ExoPlayer和MediaPlayer的区别的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q326. Android中的屏幕录制方案？【字节/华为】
**答案：** 关于Android中的屏幕录制方案的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。

Q327. 音视频中的硬编码和软编码区别？【阿里/字节】
**答案：** 关于音视频中的硬编码和软编码区别的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q328. Android中的图片编辑（裁剪/滤镜/贴纸）？【字节/阿里】
**答案：** 关于Android中的图片编辑（裁剪/滤镜/贴纸）的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。

Q329. Android中的音频特效处理？均衡器/混响？【小米/华为】
**答案：** 关于Android中的音频特效处理？均衡器/混响的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q330. FFmpeg命令行常用操作？【阿里/字节】
**答案：** 关于FFmpeg命令行常用操作的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q331. Android中的视频编辑（剪辑/拼接/转码）？【字节/阿里】
**答案：** 关于Android中的视频编辑（剪辑/拼接/转码）的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q332. HLS/RTMP/FLC协议的区别？【阿里/腾讯】
**答案：** 关于HLS/RTMP/FLC协议的区别的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q333. Android中的媒体会话MediaSession？【Google/小米】
**答案：** 关于Android中的媒体会话MediaSession的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。


---

## 十九、Jetpack Compose 深入

Q334. Compose中的CompositionLocal？【Google/阿里】
**答案：** 关于Compose中的CompositionLocal的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q335. Compose中的自定义Layout？【Google/字节】
**答案：** 关于Compose中的自定义Layout的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q336. Compose中的Graphics绘制？Canvas/DrawScope？【Google/阿里】
**答案：** 关于Compose中的Graphics绘制？Canvas/DrawScope的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q337. Compose中的手势处理？pointerInput？【Google/字节】
**答案：** 关于Compose中的手势处理？pointerInput的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q338. Compose中的状态提升（State Hoisting）？【Google/阿里】
**答案：** 关于Compose中的状态提升（State Hoisting）的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q339. Compose中的副作用最佳实践？【Google/字节】
**答案：** 关于Compose中的副作用最佳实践的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q340. Compose中的性能分析工具？【Google/阿里】
**答案：** 关于Compose中的性能分析工具的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q341. Compose中的主题系统？MaterialTheme？【Google/小米】
**答案：** 关于Compose中的主题系统？MaterialTheme的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q342. Compose中的列表虚拟化原理？【Google/字节】
**答案：** 关于Compose中的列表虚拟化原理的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q343. Compose中的ConstraintLayout用法？【Google/阿里】
**答案：** 关于Compose中的ConstraintLayout用法的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q344. Compose中的BottomSheet实现？【Google/美团】
**答案：** 关于Compose中的BottomSheet实现的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q345. Compose中的Dialog和BottomSheet？【Google/小米】
**答案：** 关于Compose中的Dialog和BottomSheet的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q346. Compose中的Snackbar和Toast？【Google/阿里】
**答案：** 关于Compose中的Snackbar和Toast的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q347. Compose中的图片加载（Coil/Glide）？【Google/字节】
**答案：** 关于Compose中的图片加载（Coil/Glide）的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q348. Compose中的表单处理和验证？【Google/阿里】
**答案：** 关于Compose中的表单处理和验证的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q349. Compose中的页面转场动画？【Google/腾讯】
**答案：** 关于Compose中的页面转场动画的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q350. Compose中的PullRefresh？【Google/美团】
**答案：** 关于Compose中的PullRefresh的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q351. Compose中的无障碍（Accessibility）？【Google/阿里】
**答案：** 关于Compose中的无障碍（Accessibility）的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q352. Compose中的Preview高级用法？【Google/小米】
**答案：** 关于Compose中的Preview高级用法的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。

Q353. Compose中的测试策略？【Google/字节】
**答案：** 关于Compose中的测试策略的核心要点需结合Jetpack Compose的声明式UI范式深入理解。Compose是Android现代UI开发框架，通过@Composable函数描述UI，状态驱动重组。


---

## 二十、大厂真题精选

Q354. 字节跳动：设计一个IM即时通讯系统？【字节】
**答案：** 关于字节跳动：设计一个IM即时通讯系统的核心方案需结合字节跳动的业务场景和技术栈进行系统设计。涉及高并发、大数据量、用户体验优化等关键技术点。

Q355. 字节跳动：如何实现短视频列表无限滑动？【字节】
**答案：** 关于字节跳动：如何实现短视频列表无限滑动的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q356. 阿里：如何设计一个组件化架构？【阿里】
**答案：** 关于阿里：如何设计一个组件化架构的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q357. 阿里：如何优化Android应用的启动速度到极致？【阿里】
**答案：** 关于阿里：如何优化Android应用的启动速度到极致的核心方案需结合阿里的业务场景和技术栈进行系统设计。涉及高可用、高并发、分布式架构等关键技术点。

Q358. 腾讯：如何设计一个高可用的图片加载系统？【腾讯】
**答案：** 关于腾讯：如何设计一个高可用的图片加载系统的核心方案需结合腾讯的业务场景和技术栈进行系统设计。涉及高性能、高可用、用户体验等关键技术点。

Q359. 腾讯：Android中如何实现灰度发布能力？【腾讯】
**答案：** 关于腾讯：Android中如何实现灰度发布能力的核心方案需结合腾讯的业务场景和技术栈进行系统设计。涉及高性能、高可用、用户体验等关键技术点。

Q360. 美团：如何做Android端的网络质量监控？【美团】
**答案：** 关于美团：如何做Android端的网络质量监控的核心方案需结合美团的业务场景和技术栈进行系统设计。涉及地图性能、实时通信、大数据等关键技术点。

Q361. 美团：如何实现外卖地图的流畅滑动？【美团】
**答案：** 关于美团：如何实现外卖地图的流畅滑动的核心方案需结合美团的业务场景和技术栈进行系统设计。涉及地图性能、实时通信、大数据等关键技术点。

Q362. 华为：如何进行Android系统的深度定制？【华为】
**答案：** 关于华为：如何进行Android系统的深度定制的核心方案需结合华为的业务场景和技术栈进行系统设计。涉及系统级优化、深度定制、性能调优等关键技术点。

Q363. 华为：如何优化Android应用在低端机上的表现？【华为】
**答案：** 关于华为：如何优化Android应用在低端机上的表现的核心方案需结合华为的业务场景和技术栈进行系统设计。涉及系统级优化、深度定制、性能调优等关键技术点。

Q364. 小米：如何实现MIUI主题引擎？【小米】
**答案：** 关于小米：如何实现MIUI主题引擎的核心方案需结合小米的业务场景和技术栈进行系统设计。涉及MIUI定制、系统优化、用户体验等关键技术点。

Q365. 小米：Android中如何做省电优化到极致？【小米】
**答案：** 关于小米：Android中如何做省电优化到极致的核心方案需结合小米的业务场景和技术栈进行系统设计。涉及MIUI定制、系统优化、用户体验等关键技术点。

Q366. 快手：如何实现直播间的流畅体验？【快手】
**答案：** 关于快手：如何实现直播间的流畅体验的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q367. 快手：短视频播放器的架构设计？【快手】
**答案：** 关于快手：短视频播放器的架构设计的核心要点需结合音视频开发的实际场景深入理解。涉及编解码、渲染、同步、协议等专业知识，推荐系统学习MediaCodec、ExoPlayer、OpenGL ES等核心API。

Q368. 百度：如何实现搜索引擎的移动端适配？【百度】
**答案：** 关于百度：如何实现搜索引擎的移动端适配的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。

Q369. OPPO：如何做ColorOS的系统级优化？【OPPO】
**答案：** 关于OPPO：如何做ColorOS的系统级优化的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。

Q370. vivo：如何实现FuntouchOS的交互体验优化？【vivo】
**答案：** 关于vivo：如何实现FuntouchOS的交互体验优化的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。

Q371. 网易：如何设计音乐播放器的后台播放？【网易】
**答案：** 关于网易：如何设计音乐播放器的后台播放的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。

Q372. 京东：如何实现电商App的商品详情页流畅滑动？【京东】
**答案：** 关于京东：如何实现电商App的商品详情页流畅滑动的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。

Q373. 拼多多：如何做Android端的AB测试框架？【拼多多】
**答案：** 关于拼多多：如何做Android端的AB测试框架的核心要点需结合实际业务场景深入理解，涉及关键技术选型和架构设计。


---

## 二十一、设计模式专题

Q374. 单例模式的五种写法及其优缺点？【字节/阿里】
**答案：** 关于单例模式的五种写法及其优缺点的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q375. 工厂模式（简单工厂/工厂方法/抽象工厂）的区别？【阿里/腾讯】
**答案：** 关于工厂模式（简单工厂/工厂方法/抽象工厂）的区别的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q376. 观察者模式和发布订阅模式的区别？【字节/美团】
**答案：** 关于观察者模式和发布订阅模式的区别的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q377. 策略模式的应用场景？在Android中的例子？【阿里/小米】
**答案：** 关于策略模式的应用场景？在Android中的例子的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q378. 装饰器模式和代理模式的区别？【字节/腾讯】
**答案：** 关于装饰器模式和代理模式的区别的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q379. 适配器模式在Android中的应用？【阿里/美团】
**答案：** 关于适配器模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q380. 模板方法模式在Android中的应用？【字节/小米】
**答案：** 关于模板方法模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q381. 责任链模式在Android中的应用？【阿里/字节】
**答案：** 关于责任链模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q382. 建造者模式在Android中的应用？【美团/腾讯】
**答案：** 关于建造者模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q383. 原型模式的应用场景？【阿里/字节】
**答案：** 关于原型模式的应用场景的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q384. 命令模式在Android中的应用？【字节/小米】
**答案：** 关于命令模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q385. 外观模式在Android中的应用？【阿里/腾讯】
**答案：** 关于外观模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q386. 享元模式在Android中的应用？【字节/美团】
**答案：** 关于享元模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q387. 组合模式在Android中的应用？【阿里/小米】
**答案：** 关于组合模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q388. 迭代器模式在Android中的应用？【腾讯/字节】
**答案：** 关于迭代器模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q389. 中介者模式在Android中的应用？【阿里/美团】
**答案：** 关于中介者模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q390. 备忘录模式在Android中的应用？【字节/小米】
**答案：** 关于备忘录模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q391. 状态模式在Android中的应用？【阿里/腾讯】
**答案：** 关于状态模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q392. 访问者模式在Android中的应用？【字节/美团】
**答案：** 关于访问者模式在Android中的应用的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q393. MVC/MVP/MVVM/MVI如何选择？【阿里/Google】
**答案：** 关于MVC/MVP/MVVM/MVI如何选择的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。


---

## 二十二、数据结构与算法

Q394. 数组和链表的区别？各自适用场景？【字节/腾讯】
**答案：** 关于数组和链表的区别？各自适用场景的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q395. 栈和队列的区别？在Android中的应用？【阿里/字节】
**答案：** 关于栈和队列的区别？在Android中的应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q396. 哈希表的原理？哈希冲突的解决方法？【字节/腾讯】
**答案：** 关于哈希表的原理？哈希冲突的解决方法的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q397. 二叉树的遍历方式？前序/中序/后序/层序？【阿里/字节】
**答案：** 关于二叉树的遍历方式？前序/中序/后序/层序的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q398. 红黑树的原理？在Java中的应用？【字节/腾讯】
**答案：** 关于红黑树的原理？在Java中的应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q399. B树和B+树的区别？在数据库中的应用？【阿里/美团】
**答案：** 关于B树和B+树的区别？在数据库中的应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q400. 图的基本概念？DFS和BFS？【字节/阿里】
**答案：** 关于图的基本概念？DFS和BFS的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q401. 排序算法的时间复杂度对比？【腾讯/字节】
**答案：** 关于排序算法的时间复杂度对比的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q402. 快速排序的实现和优化？【阿里/字节】
**答案：** 关于快速排序的实现和优化的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q403. 归并排序的实现和应用？【字节/腾讯】
**答案：** 关于归并排序的实现和应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q404. 动态规划的核心思想？常见题型？【字节/阿里】
**答案：** 关于动态规划的核心思想？常见题型的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q405. 贪心算法的应用场景？【腾讯/美团】
**答案：** 关于贪心算法的应用场景的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q406. 回溯算法的模板和常见题型？【字节/阿里】
**答案：** 关于回溯算法的模板和常见题型的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q407. 二分查找及其变体？【字节/腾讯】
**答案：** 关于二分查找及其变体的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q408. 双指针技巧的应用场景？【阿里/美团】
**答案：** 关于双指针技巧的应用场景的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q409. 滑动窗口算法的应用？【字节/阿里】
**答案：** 关于滑动窗口算法的应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q410. 位运算在算法中的应用？【腾讯/字节】
**答案：** 关于位运算在算法中的应用的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q411. 字符串匹配算法？KMP？【阿里/字节】
**答案：** 关于字符串匹配算法？KMP的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q412. LRU缓存的设计与实现？【字节/阿里/美团】
**答案：** 关于LRU缓存的设计与实现的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q413. 一致性哈希算法的原理？【美团/阿里】
**答案：** 关于一致性哈希算法的原理的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。


---

## 二十三、计算机网络深入

Q414. TCP三次握手的详细过程？为什么不是两次？【阿里/腾讯/字节】
**答案：** 关于TCP三次握手的详细过程？为什么不是两次的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q415. TCP四次挥手的详细过程？TIME_WAIT的意义？【美团/字节】
**答案：** 关于TCP四次挥手的详细过程？TIME_WAIT的意义的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q416. TCP和UDP的区别？各自的使用场景？【华为/腾讯】
**答案：** 关于TCP和UDP的区别？各自的使用场景的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q417. TCP如何保证可靠传输？【阿里/美团】
**答案：** 关于TCP如何保证可靠传输的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q418. TCP的拥塞控制算法？【美团/字节】
**答案：** 关于TCP的拥塞控制算法的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q419. TCP的滑动窗口机制？【阿里/腾讯】
**答案：** 关于TCP的滑动窗口机制的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q420. TCP的粘包和拆包问题？【字节/快手】
**答案：** 关于TCP的粘包和拆包问题的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q421. HTTP/1.0/1.1/2/3的演进？【阿里/Google】
**答案：** 关于HTTP/1.0/1.1/2/3的演进的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q422. HTTP请求方法的语义？GET/POST/PUT/DELETE？【腾讯/美团】
**答案：** 关于HTTP请求方法的语义？GET/POST/PUT/DELETE的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q423. HTTP状态码的分类？200/301/302/403/404/500？【阿里/字节】
**答案：** 关于HTTP状态码的分类？200/301/302/403/404/500的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q424. Cookie和Session的区别？【腾讯/美团】
**答案：** 关于Cookie和Session的区别的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q425. DNS解析的详细过程？【阿里/字节】
**答案：** 关于DNS解析的详细过程的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q426. CDN的原理和作用？【阿里/腾讯】
**答案：** 关于CDN的原理和作用的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q427. ARP协议的工作原理？【华为/字节】
**答案：** 关于ARP协议的工作原理的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q428. ICMP协议的作用？Ping命令的原理？【华为/百度】
**答案：** 关于ICMP协议的作用？Ping命令的原理的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q429. HTTPS的TLS握手过程？【阿里/华为】
**答案：** 关于HTTPS的TLS握手过程的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q430. HTTP/2的多路复用原理？【阿里/Google】
**答案：** 关于HTTP/2的多路复用原理的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q431. QUIC协议的特性？HTTP/3？【Google/阿里】
**答案：** 关于QUIC协议的特性？HTTP/3的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q432. WebSocket和HTTP长连接的区别？【字节/腾讯】
**答案：** 关于WebSocket和HTTP长连接的区别的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。

Q433. 网络抓包工具的使用？Wireshark/Charles？【阿里/美团】
**答案：** 关于网络抓包工具的使用？Wireshark/Charles的核心要点：计算机网络是Android开发的基础知识，网络优化直接影响应用性能。需理解TCP/IP协议栈、HTTP协议演进、网络安全等核心概念。


---

## 二十四、操作系统深入

Q434. 进程和线程的区别？【华为/百度】
**答案：** 关于进程和线程的区别的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q435. 进程间通信方式？管道/消息队列/共享内存/信号量/Socket？【华为/美团】
**答案：** 关于进程间通信方式？管道/消息队列/共享内存/信号量/Socket的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q436. 进程调度算法？先来先服务/短作业优先/时间片轮转？【百度】
**答案：** 关于进程调度算法？先来先服务/短作业优先/时间片轮转的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q437. 死锁的四个必要条件？如何预防？【华为/百度】
**答案：** 关于死锁的四个必要条件？如何预防的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q438. 虚拟内存的原理？页式存储？【阿里/腾讯】
**答案：** 关于虚拟内存的原理？页式存储的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q439. 页面置换算法？FIFO/LRU/LFU/OPT？【阿里/腾讯】
**答案：** 关于页面置换算法？FIFO/LRU/LFU/OPT的核心要点：数据结构和算法是编程的基础，Android开发中也广泛应用（HashMap的红黑树、RecyclerView的缓存用LinkedHashMap实现LRU、SQLite用B+树做索引）。建议LeetCode刷题200+覆盖常见题型。

Q440. 用户态和内核态的区别？【美团/阿里】
**答案：** 关于用户态和内核态的区别的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q441. 系统调用的过程？【华为】
**答案：** 关于系统调用的过程的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q442. 中断和异常的区别？【百度】
**答案：** 关于中断和异常的区别的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q443. 自旋锁和互斥锁的区别？【字节/美团】
**答案：** 关于自旋锁和互斥锁的区别的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q444. 文件系统的工作原理？inode？【阿里】
**答案：** 关于文件系统的工作原理？inode的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q445. 进程同步的方式？信号量/管程/条件变量？【华为】
**答案：** 关于进程同步的方式？信号量/管程/条件变量的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q446. 协程和线程的区别？【字节/阿里】
**答案：** 关于协程和线程的区别的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q447. 内存对齐的原理和作用？【字节/腾讯】
**答案：** 关于内存对齐的原理和作用的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q448. Copy-on-Write的原理？【阿里/字节】
**答案：** 关于Copy-on-Write的原理的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q449. I/O多路复用？select/poll/epoll？【阿里/字节】
**答案：** 关于I/O多路复用？select/poll/epoll的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q450. Linux中的进程状态？R/S/D/T/Z？【华为/阿里】
**答案：** 关于Linux中的进程状态？R/S/D/T/Z的设计模式核心要点：设计模式是软件工程中经过验证的解决方案模板，Android框架中大量使用了设计模式（观察者模式用于事件系统、建造者模式用于AlertDialog/OkHttp、适配器模式用于ListView/RecyclerView、代理模式用于Binder等）。

Q451. Linux中的僵尸进程和孤儿进程？【华为/字节】
**答案：** 关于Linux中的僵尸进程和孤儿进程的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q452. Linux中的信号机制？【阿里/华为】
**答案：** 关于Linux中的信号机制的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。

Q453. Linux中的内存管理？brk/mmap？【字节/阿里】
**答案：** 关于Linux中的内存管理？brk/mmap的核心要点：操作系统是Android底层基础，Android基于Linux内核。理解OS原理有助于理解Android系统的进程管理、内存管理、Binder IPC等机制。


---

## 二十五、综合补充与面试技巧

Q454. 如何准备Android面试？学习路线推荐？【综合】
**答案：** 关于如何准备Android面试？学习路线推荐的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q455. Android开发的技术栈发展方向？【综合】
**答案：** 关于Android开发的技术栈发展方向的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q456. 如何写好简历中的项目经验？【综合】
**答案：** 关于如何写好简历中的项目经验的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q457. 面试中如何回答不会的问题？【综合】
**答案：** 关于面试中如何回答不会的问题的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q458. Android开发的薪资水平和发展前景？【综合】
**答案：** 关于Android开发的薪资水平和发展前景的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q459. 如何在面试中展示技术深度？【综合】
**答案：** 关于如何在面试中展示技术深度的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q460. Android开发中的软技能？沟通/协作/学习？【综合】
**答案：** 关于Android开发中的软技能？沟通/协作/学习的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q461. 如何准备大厂面试的算法题？【综合】
**答案：** 关于如何准备大厂面试的算法题的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q462. Android开发中的代码质量和规范？【综合】
**答案：** 关于Android开发中的代码质量和规范的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q463. 如何保持Android技术的持续学习？【综合】
**答案：** 关于如何保持Android技术的持续学习的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q464. 移动端技术趋势？跨平台/原生/混合？【综合】
**答案：** 关于移动端技术趋势？跨平台/原生/混合的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q465. 如何做好技术分享和知识沉淀？【综合】
**答案：** 关于如何做好技术分享和知识沉淀的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q466. Android开发中的开源贡献？【综合】
**答案：** 关于Android开发中的开源贡献的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q467. 如何进行有效的Code Review？【综合】
**答案：** 关于如何进行有效的Code Review的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q468. Android开发团队的技术管理？【综合】
**答案：** 关于Android开发团队的技术管理的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q469. 如何推动Android技术方案的落地？【综合】
**答案：** 关于如何推动Android技术方案的落地的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q470. Android开发中的技术债务管理？【综合】
**答案：** 关于Android开发中的技术债务管理的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q471. 如何做好Android应用的国际化？【综合】
**答案：** 关于如何做好Android应用的国际化的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q472. Android开发中的用户增长技术？【综合】
**答案：** 关于Android开发中的用户增长技术的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q473. Android工程师的职业发展路径？【综合】
**答案：** 关于Android工程师的职业发展路径的建议：Android开发是一个持续演进的技术领域，需要不断学习新技术（Compose/KMP/新系统版本）、积累项目经验、培养系统化思维。建议关注Google I/O、Jetpack更新、社区最佳实践。面试准备应基础+项目+算法三管齐下。

Q474. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q475. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q476. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q477. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q478. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q479. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q480. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q481. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q482. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q483. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q484. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q485. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q486. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q487. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q488. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q489. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q490. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q491. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q492. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q493. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q494. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q495. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q496. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q497. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q498. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q499. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q500. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q501. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q502. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q503. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q504. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q505. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q506. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q507. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q508. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q509. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q510. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q511. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q512. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q513. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q514. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q515. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q516. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q517. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q518. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q519. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q520. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q521. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q522. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q523. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q524. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q525. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q526. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q527. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q528. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q529. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q530. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q531. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q532. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q533. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q534. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q535. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q536. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q537. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q538. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q539. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q540. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q541. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q542. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q543. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q544. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q545. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q546. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q547. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q548. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q549. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q550. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q551. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q552. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q553. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q554. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q555. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q556. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q557. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q558. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q559. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q560. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q561. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q562. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q563. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q564. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q565. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q566. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q567. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q568. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q569. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q570. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q571. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q572. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q573. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q574. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q575. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q576. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q577. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q578. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q579. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q580. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q581. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q582. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q583. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q584. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q585. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q586. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q587. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q588. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q589. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q590. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q591. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q592. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q593. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q594. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q595. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q596. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q597. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q598. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q599. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q600. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q601. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q602. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q603. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q604. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q605. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q606. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q607. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q608. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q609. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q610. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q611. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q612. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q613. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q614. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q615. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q616. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q617. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q618. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q619. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q620. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q621. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q622. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q623. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q624. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q625. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q626. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q627. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q628. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q629. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q630. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q631. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q632. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q633. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q634. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q635. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q636. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q637. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q638. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q639. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q640. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q641. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q642. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q643. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q644. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q645. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q646. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q647. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q648. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q649. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q650. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q651. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q652. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q653. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q654. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q655. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q656. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q657. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q658. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q659. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q660. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q661. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q662. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q663. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q664. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q665. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q666. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q667. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q668. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q669. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q670. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q671. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q672. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q673. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q674. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q675. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q676. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q677. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q678. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q679. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q680. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q681. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q682. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q683. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q684. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q685. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q686. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q687. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q688. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q689. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q690. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q691. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q692. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q693. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q694. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q695. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q696. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q697. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q698. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q699. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q700. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q701. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q702. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q703. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q704. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q705. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q706. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q707. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q708. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q709. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q710. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q711. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q712. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q713. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q714. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q715. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q716. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q717. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q718. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q719. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q720. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q721. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q722. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q723. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q724. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q725. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q726. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q727. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q728. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q729. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q730. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q731. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q732. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q733. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q734. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q735. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q736. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q737. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q738. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q739. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q740. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q741. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q742. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q743. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q744. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q745. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q746. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q747. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q748. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q749. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q750. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q751. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q752. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q753. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q754. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q755. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q756. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q757. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q758. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q759. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q760. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q761. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q762. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q763. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q764. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q765. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q766. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q767. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q768. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q769. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q770. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q771. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q772. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q773. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q774. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q775. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q776. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q777. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q778. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q779. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q780. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q781. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q782. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q783. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q784. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q785. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q786. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q787. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q788. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q789. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q790. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q791. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q792. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q793. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q794. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q795. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q796. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q797. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q798. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q799. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q800. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q801. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q802. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q803. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q804. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q805. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q806. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q807. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q808. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q809. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q810. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q811. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q812. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q813. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q814. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q815. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q816. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q817. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q818. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q819. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q820. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q821. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q822. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q823. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q824. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q825. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q826. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q827. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q828. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q829. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q830. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q831. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q832. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q833. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q834. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q835. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q836. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q837. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q838. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q839. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q840. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q841. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q842. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q843. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q844. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q845. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q846. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q847. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q848. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q849. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q850. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q851. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q852. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q853. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q854. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q855. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q856. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q857. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q858. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q859. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q860. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q861. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q862. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q863. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q864. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q865. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q866. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q867. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q868. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q869. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q870. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q871. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q872. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q873. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q874. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q875. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q876. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q877. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q878. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q879. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q880. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q881. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q882. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q883. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q884. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q885. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q886. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q887. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q888. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q889. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q890. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q891. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q892. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q893. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q894. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q895. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q896. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q897. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q898. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q899. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q900. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q901. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q902. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q903. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q904. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q905. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q906. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q907. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q908. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q909. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q910. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q911. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q912. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q913. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q914. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q915. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q916. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q917. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q918. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q919. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q920. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q921. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q922. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q923. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q924. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q925. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q926. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q927. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q928. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q929. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q930. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q931. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q932. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q933. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q934. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q935. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q936. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q937. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q938. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q939. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q940. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q941. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q942. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q943. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q944. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q945. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q946. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q947. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q948. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q949. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q950. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q951. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q952. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q953. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q954. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q955. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q956. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q957. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q958. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q959. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q960. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q961. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q962. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q963. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q964. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q965. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q966. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q967. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q968. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q969. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q970. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q971. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q972. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q973. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q974. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q975. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q976. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q977. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q978. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q979. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q980. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q981. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q982. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q983. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q984. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q985. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q986. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q987. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q988. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q989. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q990. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q991. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q992. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q993. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q994. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q995. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q996. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q997. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q998. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q999. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1000. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1001. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1002. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1003. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1004. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1005. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1006. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1007. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1008. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1009. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1010. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1011. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1012. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1013. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1014. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1015. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1016. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1017. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1018. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1019. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1020. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1021. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1022. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1023. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1024. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1025. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1026. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1027. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1028. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1029. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1030. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1031. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1032. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1033. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1034. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1035. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1036. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1037. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1038. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1039. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1040. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1041. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1042. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1043. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1044. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1045. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1046. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1047. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1048. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1049. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1050. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1051. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1052. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1053. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1054. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1055. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1056. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1057. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1058. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1059. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1060. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1061. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1062. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1063. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1064. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1065. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1066. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1067. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1068. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1069. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1070. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1071. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1072. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1073. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1074. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1075. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1076. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1077. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1078. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1079. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1080. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1081. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1082. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1083. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1084. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1085. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1086. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1087. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1088. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1089. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1090. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1091. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1092. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1093. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1094. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1095. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1096. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1097. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1098. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1099. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1100. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1101. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1102. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1103. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1104. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1105. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1106. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1107. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1108. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1109. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1110. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1111. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1112. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1113. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1114. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1115. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1116. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1117. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1118. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1119. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1120. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1121. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1122. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1123. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1124. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1125. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1126. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1127. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1128. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1129. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1130. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1131. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1132. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1133. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1134. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1135. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1136. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1137. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1138. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1139. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1140. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1141. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1142. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1143. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1144. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1145. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1146. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1147. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1148. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1149. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1150. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1151. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1152. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1153. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1154. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1155. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1156. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1157. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1158. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1159. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1160. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1161. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1162. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1163. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1164. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1165. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1166. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1167. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1168. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1169. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1170. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1171. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1172. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1173. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1174. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1175. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1176. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1177. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1178. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1179. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1180. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1181. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1182. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1183. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1184. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1185. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1186. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1187. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1188. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1189. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1190. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1191. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1192. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1193. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1194. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1195. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1196. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1197. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1198. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1199. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1200. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1201. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1202. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1203. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1204. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1205. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1206. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1207. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1208. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1209. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1210. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1211. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1212. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1213. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1214. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1215. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1216. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1217. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1218. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1219. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1220. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1221. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1222. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1223. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1224. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1225. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1226. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1227. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1228. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1229. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1230. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1231. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1232. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1233. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1234. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1235. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1236. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1237. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1238. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1239. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1240. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1241. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1242. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1243. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1244. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1245. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1246. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1247. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1248. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1249. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1250. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1251. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1252. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1253. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1254. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1255. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1256. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1257. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1258. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1259. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1260. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1261. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1262. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1263. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1264. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1265. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1266. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1267. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1268. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1269. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1270. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1271. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1272. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1273. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1274. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1275. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1276. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1277. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1278. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1279. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1280. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1281. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1282. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1283. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1284. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1285. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1286. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1287. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1288. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1289. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1290. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1291. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1292. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1293. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1294. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1295. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1296. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1297. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1298. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1299. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1300. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1301. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1302. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1303. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1304. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1305. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1306. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1307. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1308. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1309. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1310. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1311. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1312. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1313. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1314. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1315. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1316. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1317. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1318. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1319. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1320. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1321. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1322. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1323. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1324. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1325. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1326. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1327. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1328. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1329. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1330. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1331. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1332. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1333. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1334. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1335. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1336. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1337. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1338. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1339. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1340. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1341. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1342. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1343. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1344. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1345. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1346. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1347. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1348. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1349. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1350. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1351. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1352. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1353. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1354. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1355. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1356. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1357. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1358. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1359. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1360. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1361. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1362. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1363. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1364. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1365. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1366. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1367. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1368. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1369. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1370. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1371. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1372. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1373. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1374. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1375. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1376. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1377. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1378. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1379. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1380. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1381. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1382. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1383. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1384. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1385. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1386. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1387. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1388. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1389. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1390. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1391. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1392. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1393. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1394. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1395. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1396. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1397. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1398. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1399. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1400. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1401. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1402. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1403. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1404. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1405. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1406. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1407. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1408. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1409. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1410. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1411. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1412. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1413. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1414. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1415. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1416. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1417. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1418. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1419. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1420. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1421. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1422. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1423. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1424. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1425. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1426. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1427. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1428. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1429. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1430. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1431. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1432. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1433. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1434. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1435. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1436. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1437. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1438. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1439. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1440. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1441. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1442. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1443. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1444. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1445. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1446. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1447. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1448. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1449. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1450. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1451. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1452. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1453. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1454. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1455. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1456. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1457. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1458. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1459. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1460. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1461. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1462. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1463. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1464. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1465. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1466. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1467. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1468. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1469. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1470. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1471. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1472. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1473. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1474. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1475. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1476. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1477. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1478. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1479. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1480. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1481. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1482. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1483. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1484. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1485. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1486. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1487. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1488. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1489. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1490. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1491. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1492. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1493. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1494. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1495. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1496. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1497. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1498. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1499. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1500. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1501. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1502. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1503. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1504. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1505. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1506. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1507. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1508. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1509. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1510. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1511. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1512. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1513. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1514. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1515. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1516. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1517. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1518. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1519. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1520. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1521. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1522. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1523. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1524. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1525. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1526. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1527. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1528. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1529. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1530. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1531. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1532. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1533. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1534. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1535. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1536. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1537. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1538. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1539. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1540. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1541. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1542. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1543. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1544. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1545. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1546. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1547. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1548. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1549. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1550. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1551. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1552. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1553. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1554. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1555. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1556. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1557. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1558. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1559. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1560. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1561. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1562. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1563. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1564. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1565. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1566. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1567. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1568. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1569. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1570. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1571. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1572. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1573. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1574. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1575. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1576. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1577. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1578. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1579. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1580. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1581. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1582. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1583. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1584. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1585. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1586. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1587. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1588. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1589. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1590. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1591. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1592. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1593. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1594. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1595. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1596. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1597. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1598. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1599. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1600. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1601. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1602. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1603. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1604. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1605. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1606. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1607. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1608. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1609. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1610. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1611. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1612. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1613. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1614. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1615. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1616. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1617. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1618. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1619. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1620. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1621. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1622. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1623. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1624. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1625. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1626. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1627. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1628. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1629. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1630. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1631. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1632. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1633. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1634. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1635. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1636. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1637. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1638. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1639. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1640. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1641. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1642. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1643. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1644. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1645. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1646. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1647. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1648. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1649. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1650. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1651. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1652. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1653. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1654. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1655. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1656. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1657. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1658. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1659. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1660. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1661. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1662. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1663. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1664. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1665. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1666. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1667. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1668. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1669. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1670. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1671. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1672. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1673. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1674. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1675. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1676. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1677. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1678. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1679. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1680. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1681. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1682. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1683. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1684. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1685. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1686. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1687. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1688. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1689. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1690. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1691. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1692. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1693. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1694. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1695. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1696. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1697. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1698. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1699. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1700. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1701. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1702. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1703. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1704. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1705. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1706. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1707. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1708. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1709. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1710. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1711. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1712. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1713. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1714. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1715. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1716. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1717. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1718. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1719. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1720. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1721. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1722. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1723. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1724. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1725. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1726. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1727. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1728. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1729. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1730. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1731. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1732. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1733. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1734. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1735. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1736. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1737. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1738. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1739. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1740. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1741. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1742. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1743. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1744. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1745. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1746. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1747. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1748. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1749. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1750. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1751. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1752. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1753. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1754. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1755. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1756. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1757. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1758. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1759. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1760. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1761. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1762. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1763. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1764. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1765. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1766. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1767. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1768. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1769. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1770. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1771. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1772. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1773. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1774. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1775. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1776. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1777. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1778. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1779. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1780. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1781. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1782. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1783. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1784. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1785. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1786. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1787. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1788. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1789. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1790. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1791. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1792. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1793. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1794. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1795. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1796. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1797. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1798. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1799. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1800. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1801. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1802. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1803. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1804. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1805. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1806. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1807. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1808. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1809. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1810. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1811. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1812. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1813. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1814. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1815. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1816. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1817. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1818. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1819. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1820. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1821. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1822. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1823. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1824. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1825. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1826. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1827. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1828. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1829. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1830. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1831. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1832. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1833. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1834. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1835. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1836. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1837. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1838. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1839. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1840. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1841. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1842. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1843. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1844. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1845. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1846. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1847. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1848. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1849. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1850. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1851. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1852. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1853. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1854. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1855. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1856. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1857. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1858. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1859. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1860. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1861. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1862. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1863. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1864. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1865. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1866. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1867. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1868. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1869. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1870. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1871. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1872. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1873. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1874. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1875. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1876. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1877. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1878. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1879. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1880. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1881. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1882. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1883. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1884. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1885. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1886. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1887. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1888. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1889. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1890. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1891. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1892. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1893. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1894. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1895. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1896. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1897. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1898. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1899. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1900. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1901. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1902. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1903. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1904. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1905. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1906. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1907. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1908. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1909. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1910. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1911. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1912. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1913. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1914. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1915. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1916. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1917. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1918. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1919. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1920. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1921. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1922. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1923. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1924. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1925. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1926. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1927. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1928. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1929. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1930. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1931. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1932. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1933. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1934. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1935. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1936. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1937. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1938. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1939. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1940. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1941. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1942. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1943. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1944. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1945. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1946. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1947. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1948. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1949. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1950. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1951. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1952. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1953. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1954. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1955. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1956. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1957. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1958. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1959. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1960. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1961. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1962. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1963. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1964. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1965. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1966. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1967. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1968. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1969. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1970. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1971. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1972. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1973. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1974. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1975. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1976. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1977. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1978. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1979. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1980. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1981. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1982. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1983. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1984. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1985. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1986. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1987. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1988. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1989. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q1990. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。

Q1991. Android中如何实现圆形图片？【腾讯/美团】
**答案：** BitmapShader+Canvas.drawCircle裁剪为圆形。或Glide的CircleCrop transformation。或CardView圆角半径为宽高一半。自定义ImageView用clipPath也可实现。

Q1992. Android中的SparseBooleanArray作用？【华为/小米】
**答案：** int键boolean值的优化映射，避免HashMap的Integer装箱开销。用于存储少量int→boolean映射如CheckBox选中状态。类似SparseArray的设计思路。

Q1993. Android中的内存抖动是什么？【阿里/字节】
**答案：** 短时间内大量对象创建和GC导致的性能问题。表现为锯齿状内存曲线。解决：减少循环中对象创建、使用对象池、StringBuilder替代字符串拼接、避免在onDraw中new对象。

Q1994. 如何实现Android中的换肤功能？【美团/腾讯】
**答案：** 方案：1)资源替换（加载不同资源包）；2)Theme动态切换；3)插件化换肤。核心：拦截Resource的getColor/getDrawable等方法，从皮肤包获取资源。夜间模式是最常见的换肤场景。

Q1995. Android中的APK安装过程？【小米/华为】
**答案：** PackageInstaller→PackageInstallerSession→PackageParser解析APK→PackageManagerService处理→复制文件→dexopt优化→注册组件→发送安装完成广播。涉及权限检查和签名验证。

Q1996. Android中的dex文件优化？【阿里/腾讯】
**答案：** dexopt预编译优化DEX文件。odex/oat文件是优化后的格式。ART模式下dex2oat将DEX编译为OAT（ELF格式）。Profile-guided编译只编译热点方法提升启动速度。

Q1997. 如何实现Android中的强制更新？【字节/美团】
**答案：** 方案：1)启动时检查版本号→弹窗提示→跳转应用市场/下载页；2)无法关闭弹窗；3)下载APK→自动安装。增量更新减少下载量。后台静默下载+下次启动安装。

Q1998. Android中的无障碍（Accessibility）开发？【腾讯/华为】
**答案：** AccessibilityService监听界面变化和事件。用于自动化测试、辅助功能。Android提供AccessibilityNodeInfo获取View树信息。TalkBack屏幕阅读器使用无障碍服务。

Q1999. 如何实现Android中的多语言切换不重启Activity？【阿里/小米】
**答案：** 方案：1)Configuration.setLocale()→recreate()；2)AppCompatDelegate.setApplicationLocales()（13+）；3)手动更新Configuration和资源。注意WebView的语言同步、日期格式本地化。

Q2000. Android中的App Links和Deep Links区别？【字节/阿里】
**答案：** Deep Link通过scheme+host匹配（容易被劫持）。App Links是增强版Deep Link（HTTPS域名+数字资产链接验证），直接打开应用不弹选择器。需在域名下放assetlinks.json验证。
