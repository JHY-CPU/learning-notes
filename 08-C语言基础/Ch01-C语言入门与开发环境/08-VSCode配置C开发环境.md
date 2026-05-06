# VSCode配置C开发环境

## 一、环境准备

### 1.1 安装VSCode

1. 访问 https://code.visualstudio.com/ 下载安装
2. 安装完成后启动

### 1.2 安装C编译器

**Windows**：
- 方案A：安装 MinGW-w64（通过MSYS2）
- 方案B：安装 WSL（Windows Subsystem for Linux）

**Linux**：
```bash
sudo apt install gcc gdb build-essential   # Ubuntu/Debian
sudo yum install gcc gdb make              # CentOS/RHEL
```

**macOS**：
```bash
xcode-select --install
```

### 1.3 验证安装

```bash
gcc --version
gdb --version    # Linux/macOS
```

## 二、安装VSCode插件

### 2.1 必装插件

| 插件名 | 说明 |
|--------|------|
| C/C++ (ms-vscode.cpptools) | 微软官方C/C++插件 |
| C/C++ Extension Pack | C/C++扩展包合集 |

### 2.2 推荐插件

| 插件名 | 说明 |
|--------|------|
| C/C++ Themes | C/C++代码主题 |
| Code Runner | 一键运行代码 |
| GitLens | Git增强 |
| Error Lens | 行内错误显示 |
| Chinese (Simplified) | 中文语言包 |

### 2.3 安装方法

1. 点击左侧扩展图标（或 `Ctrl + Shift + X`）
2. 搜索插件名称
3. 点击 **Install**

## 三、项目结构

```
my-c-project/
├── .vscode/            （VSCode配置文件夹）
│   ├── tasks.json      （编译任务配置）
│   ├── launch.json     （调试配置）
│   └── settings.json   （工作区设置）
├── src/                （源代码目录）
│   ├── main.c
│   ├── utils.c
│   └── utils.h
├── include/            （头文件目录）
│   └── utils.h
├── build/              （编译输出目录）
└── Makefile            （可选）
```

## 四、配置tasks.json

### 4.1 创建编译任务

1. 按 `Ctrl + Shift + P` 打开命令面板
2. 输入 **Tasks: Configure Task**
3. 选择 **Create tasks.json file from template**
4. 选择 **Others**

### 4.2 单文件编译配置

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "gcc编译",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-Wall",
                "-Wextra",
                "-g",
                "-std=c11",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"],
            "detail": "编译当前C文件"
        }
    ]
}
```

### 4.3 多文件编译配置

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "gcc编译所有文件",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-Wall",
                "-Wextra",
                "-g",
                "-std=c11",
                "-I${workspaceFolder}/include",
                "${workspaceFolder}/src/*.c",
                "-o",
                "${workspaceFolder}/build/program",
                "-lm"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"],
            "detail": "编译所有C源文件"
        },
        {
            "label": "gcc编译当前文件",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-Wall",
                "-Wextra",
                "-g",
                "-std=c11",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": "build",
            "problemMatcher": ["$gcc"],
            "detail": "仅编译当前打开的文件"
        }
    ]
}
```

### 4.4 运行编译任务

- **快捷键**：`Ctrl + Shift + B`（运行默认构建任务）
- **命令面板**：`Ctrl + Shift + P` → **Tasks: Run Task**

## 五、配置launch.json

### 5.1 创建调试配置

1. 按 `F5`（或点击运行和调试）
2. 选择 **C++ (GDB/LLDB)**（Linux/macOS）或 **C++ (Windows)**
3. VSCode会自动生成 `launch.json`

### 5.2 GDB调试配置（Linux/macOS）

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "gcc调试",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为gdb启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "gcc编译",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

### 5.3 Windows调试配置

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "gcc调试 (Windows)",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": true,
            "MIMode": "gdb",
            "miDebuggerPath": "C:/msys64/mingw64/bin/gdb.exe",
            "setupCommands": [
                {
                    "description": "为gdb启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "gcc编译"
        }
    ]
}
```

### 5.4 多配置调试

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "调试主程序",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/program",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为gdb启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "gcc编译所有文件"
        },
        {
            "name": "附加到进程",
            "type": "cppdbg",
            "request": "attach",
            "program": "${workspaceFolder}/build/program",
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

## 六、settings.json配置

### 6.1 工作区设置

```json
// .vscode/settings.json
{
    // 文件关联
    "files.associations": {
        "*.h": "c",
        "*.c": "c"
    },

    // 编辑器设置
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "editor.insertSpaces": true,

    // C/C++ 插件设置
    "C_Cpp.default.cStandard": "c11",
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/include",
        "/usr/include"
    ],
    "C_Cpp.clang_format_style": "file",

    // 文件排除
    "files.exclude": {
        "**/*.o": true,
        "**/*.exe": true,
        "**/build/": true
    },

    // 搜索排除
    "search.exclude": {
        "**/build": true
    }
}
```

### 6.2 Clang-Format配置

创建 `.clang-format` 文件：

```yaml
# .clang-format
Language: C
BasedOnStyle: Google
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: None
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
BreakBeforeBraces: Linux
IndentCaseLabels: true
```

## 七、使用Code Runner

### 7.1 配置

```json
// settings.json
{
    "code-runner.runInTerminal": true,
    "code-runner.executorMap": {
        "c": "cd $dir && gcc -Wall -Wextra -g -std=c11 $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt"
    },
    "code-runner.saveFileBeforeRun": true,
    "code-runner.clearPreviousOutput": true
}
```

### 7.2 使用

- **运行**：`Ctrl + Alt + N`
- **停止**：`Ctrl + Alt + M`
- 右键 → **Run Code**

## 八、调试操作

### 8.1 快捷键

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 开始/继续 | `F5` | 启动或继续调试 |
| 停止 | `Shift + F5` | 终止调试 |
| 重启 | `Ctrl + Shift + F5` | 重新启动调试 |
| 切换断点 | `F9` | 在当前行设置/删除断点 |
| 逐过程 | `F10` | Step Over |
| 逐语句 | `F11` | Step Into |
| 跳出 | `Shift + F11` | Step Out |

### 8.2 调试面板

- **变量**：查看局部变量和全局变量
- **监视**：添加自定义监视表达式
- **调用堆栈**：查看函数调用链
- **断点**：管理所有断点

### 8.3 调试控制台

在调试模式下，可以在 **调试控制台** 中输入表达式：

```
-print x          # 打印变量x的值
-print *arr@5     # 打印数组arr的前5个元素
-exec info locals  # 执行GDB命令
-exec bt           # 查看调用堆栈
```

## 九、多文件项目完整配置

### 9.1 项目结构

```
project/
├── .vscode/
│   ├── tasks.json
│   ├── launch.json
│   └── settings.json
├── src/
│   ├── main.c
│   ├── math_ops.c
│   └── utils.c
├── include/
│   ├── math_ops.h
│   └── utils.h
├── build/          （编译输出）
└── Makefile        （可选）
```

### 9.2 完整tasks.json

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "创建build目录",
            "type": "shell",
            "command": "mkdir",
            "args": ["-p", "${workspaceFolder}/build"],
            "windows": {
                "command": "if not exist \"${workspaceFolder}\\build\" mkdir \"${workspaceFolder}\\build\""
            },
            "problemMatcher": []
        },
        {
            "label": "编译项目",
            "type": "shell",
            "command": "gcc",
            "args": [
                "-Wall", "-Wextra", "-g", "-std=c11",
                "-I${workspaceFolder}/include",
                "${workspaceFolder}/src/*.c",
                "-o", "${workspaceFolder}/build/program",
                "-lm"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "dependsOn": "创建build目录",
            "problemMatcher": ["$gcc"]
        },
        {
            "label": "清理",
            "type": "shell",
            "command": "rm",
            "args": ["-rf", "${workspaceFolder}/build"],
            "windows": {
                "command": "rmdir",
                "args": ["/s", "/q", "${workspaceFolder}\\build"]
            },
            "problemMatcher": []
        }
    ]
}
```

### 9.3 完整launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "调试项目",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/program",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为gdb启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "编译项目"
        }
    ]
}
```

## 十、常见问题

### 10.1 找不到gcc

```bash
# Windows: 检查PATH环境变量
echo $PATH   # 或在cmd中: echo %PATH%

# 添加MinGW路径到PATH
# 例如：C:\msys64\mingw64\bin
```

在VSCode中检查终端配置：
```json
// settings.json
{
    "terminal.integrated.profiles.windows": {
        "Git Bash": {
            "path": "C:\\Program Files\\Git\\bin\\bash.exe"
        }
    },
    "terminal.integrated.defaultProfile.windows": "Git Bash"
}
```

### 10.2 调试无法启动

1. 确保使用 `-g` 编译选项生成调试信息
2. 检查 `launch.json` 中的 `program` 路径是否正确
3. 检查 `miDebuggerPath` 是否指向正确的gdb路径

### 10.3 IntelliSense不工作

1. 确保安装了C/C++插件
2. 检查 `settings.json` 中的 `compilerPath` 设置
3. 按 `Ctrl + Shift + P` → **C/C++: Edit Configurations** 重新配置

## 十一、关键要点

> **重要提示**：
> 1. VSCode需要配合编译器（GCC/Clang）使用，本身不自带编译器
> 2. `tasks.json` 配置编译命令，`launch.json` 配置调试命令
> 3. `preLaunchTask` 将调试与编译关联起来
> 4. 必须用 `-g` 编译选项才能调试
> 5. `${file}` 是当前文件，`${workspaceFolder}` 是项目根目录
> 6. 安装C/C++插件是必须的，它提供IntelliSense和调试支持
> 7. 推荐安装Code Runner插件快速运行单文件程序
