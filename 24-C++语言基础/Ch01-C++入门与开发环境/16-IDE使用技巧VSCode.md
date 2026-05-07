# IDE使用技巧 - VS Code

## 一、概念说明

VS Code（Visual Studio Code）是微软开发的免费、开源的轻量级代码编辑器。通过扩展可以成为功能强大的C++开发环境，支持智能补全、调试、代码导航等功能。

## 二、具体用法

### 2.1 必装扩展

```
C/C++          - Microsoft官方扩展，提供智能感知和调试
C/C++ Extension Pack - C/C++扩展合集
CMake Tools    - CMake项目支持
Code Runner    - 一键运行代码
Better Comments - 彩色注释
Error Lens     - 行内错误提示
```

### 2.2 settings.json配置

```json
{
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.compilerPath": "/usr/bin/g++",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/**",
        "${workspaceFolder}/include"
    ],
    "C_Cpp.clang_format_fallbackStyle": "Google",
    "editor.formatOnSave": true,
    "files.associations": {
        "*.hpp": "cpp",
        "*.h": "cpp"
    },
    "C_Cpp.errorSquiggles": "enabled"
}
```

### 2.3 tasks.json构建任务

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "编译当前文件",
            "type": "shell",
            "command": "g++",
            "args": [
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"]
        },
        {
            "label": "CMake构建",
            "type": "shell",
            "command": "cmake",
            "args": ["--build", "${workspaceFolder}/build"],
            "group": "build",
            "problemMatcher": ["$gcc"]
        }
    ]
}
```

### 2.4 launch.json调试配置

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "调试当前文件",
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
                    "text": "-enable-pretty-printing"
                }
            ],
            "preLaunchTask": "编译当前文件",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

### 2.5 常用快捷键

| 功能 | Windows/Linux | macOS |
|------|--------------|-------|
| 编译运行 | `Ctrl+Shift+B` | `Cmd+Shift+B` |
| 开始调试 | `F5` | `F5` |
| 切换断点 | `F9` | `F9` |
| 单步跳过 | `F10` | `F10` |
| 单步进入 | `F11` | `F11` |
| 跳出函数 | `Shift+F11` | `Shift+F11` |
| 转到定义 | `F12` | `F12` |
| 查找引用 | `Shift+F12` | `Shift+F12` |
| 重命名符号 | `F2` | `F2` |
| 格式化文档 | `Shift+Alt+F` | `Shift+Option+F` |
| 命令面板 | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| 打开终端 | `` Ctrl+` `` | `` Cmd+` `` |

### 2.6 多文件编译配置

```json
{
    "label": "编译所有源文件",
    "type": "shell",
    "command": "g++",
    "args": [
        "-std=c++17", "-Wall", "-Wextra", "-g",
        "${workspaceFolder}/src/*.cpp",
        "-I", "${workspaceFolder}/include",
        "-o", "${workspaceFolder}/build/app"
    ],
    "group": { "kind": "build", "isDefault": true }
}
```

## 三、注意事项与常见陷阱

1. **IntelliSense配置**：如果代码补全不工作，检查`c_cpp_properties.json`中的编译器路径
2. **编码问题**：Windows下确保终端编码为UTF-8，否则中文输出乱码
3. **调试器路径**：Windows下需要指定MinGW的gdb路径，如`C:/mingw64/bin/gdb.exe`
4. **多根工作区**：大型项目可以用多根工作区管理不同模块
5. **扩展冲突**：安装过多C++扩展可能导致冲突，按需安装即可
