# Python logging模块


## 📋 Python logging 模块


logging 模块的日志级别、Logger/Handler/Formatter 组件、文件与控制台输出、日志配置 dictConfig、模块化日志最佳实践、日志轮转。


## 日志级别


```
// ========== 日志级别 ==========
import logging

# 5 个内置级别 (数字越小越严重):
# DEBUG    10  调试信息
# INFO     20  常规信息
# WARNING  30  警告 (默认级别)
# ERROR    40  错误
# CRITICAL 50  严重错误

# 基本使用:
logging.debug("调试信息")      # 默认不输出
logging.info("常规信息")       # 默认不输出
logging.warning("警告!")       # 输出到 stderr
logging.error("错误!")         # 输出到 stderr
logging.critical("严重错误!")  # 输出到 stderr

# 输出: WARNING:root:警告!

# 设置级别:
logging.basicConfig(level=logging.DEBUG)
logging.debug("现在会显示了")  # 会显示
```


## basicConfig 基本配置


```
// ========== basicConfig ==========
import logging

# 只能在第一次调用前配置!
logging.basicConfig(
    level=logging.DEBUG,           # 日志级别
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",   # 时间格式
    filename="app.log",            # 输出到文件 (不设置则输出到控制台)
    filemode="a",                  # 'a' 追加, 'w' 覆盖
    encoding="utf-8",
)

logging.info("应用启动")
logging.error("发生错误")

# 控制台输出格式:
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler(),      # 控制台
#         logging.FileHandler("app.log") # 文件
#     ]
# )

# ========== 格式字符串字段 ==========
# %(name)s      — logger 名称
# %(levelname)s — 级别名 (INFO, ERROR)
# %(levelno)s   — 级别数字 (20, 40)
# %(message)s   — 日志消息
# %(asctime)s   — 时间
# %(filename)s  — 文件名
# %(lineno)d    — 行号
# %(funcName)s  — 函数名
# %(process)d   — 进程 ID
# %(thread)d    — 线程 ID
```


## Logger 对象


```
// ========== 获取 Logger ==========
import logging

# 获取 module 级别的 logger (推荐方式)
logger = logging.getLogger(__name__)

logger.info("使用模块名作为 logger 名")
# 输出: INFO:__main__:使用模块名作为 logger 名

# Logger 层级: 点分隔
parent = logging.getLogger("app")
child = logging.getLogger("app.database")
# child 会继承 parent 的配置

# ========== 配置 Logger ==========
logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)

# 处理器:
console = logging.StreamHandler()
console.setLevel(logging.INFO)

file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# 格式化器:
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器:
logger.addHandler(console)
logger.addHandler(file_handler)

# 使用:
logger.debug("仅在文件中的调试信息")
logger.info("控制台和文件都显示")
logger.error("错误信息")
```


## Handler 类型


```
// ========== 常用 Handler ==========
import logging

# StreamHandler — 输出到控制台/流
logging.StreamHandler(sys.stdout)   # 可指定输出流

# FileHandler — 输出到文件
logging.FileHandler("app.log", encoding="utf-8")

# RotatingFileHandler — 按大小轮转
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "app.log",
    maxBytes=1024*1024,      # 1 MB
    backupCount=5,            # 保留 5 个备份
    encoding="utf-8"
)

# TimedRotatingFileHandler — 按时间轮转
from logging.handlers import TimedRotatingFileHandler

handler = TimedRotatingFileHandler(
    "app.log",
    when="midnight",          # 每天午夜
    interval=1,
    backupCount=7,            # 保留 7 天
    encoding="utf-8"
)

# HTTPHandler — HTTP POST 到远程
# SMTPHandler — 邮件发送
# QueueHandler — 队列 (异步)

// ========== Filter 过滤器 ==========
class SensitiveFilter(logging.Filter):
    """过滤敏感信息"""
    def filter(self, record):
        msg = record.getMessage()
        if "password" in msg.lower():
            return False        # 不记录
        return True

logger.addFilter(SensitiveFilter())
```


## dictConfig 配置


```
// ========== logging.config.dictConfig ==========
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s [%(levelname)s] %(message)s",
            "datefmt": "%H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "app.log",
            "maxBytes": 1048576,  # 1MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "app": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "app.database": {
            "level": "DEBUG",
            "handlers": ["file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("app")
logger.info("dictConfig 配置完成")
```


## 最佳实践


```
// ========== 模块级 Logger ==========
# 在每个模块中:
import logging
logger = logging.getLogger(__name__)

def process():
    logger.info("处理开始")
    try:
        result = do_something()
        logger.debug("结果: %s", result)
        return result
    except Exception:
        logger.exception("处理失败")  # 自动记录 traceback!
        raise

// ========== 异常记录 ==========
try:
    1 / 0
except ZeroDivisionError:
    logging.exception("除零错误")
    # 自动包含完整 traceback

// ========== 日志性能 ==========
# ✅ 惰性格式化 (推荐):
logger.debug("用户 %s 登录", username)

# ❌ 先格式化 (即使日志不输出也浪费):
logger.debug(f"用户 {username} 登录")

# ✅ 检查级别 (复杂对象):
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("复杂数据: %s", expensive_function())

// ========== 项目结构 ==========
# config/logging.yaml:
# version: 1
# formatters:
#   standard:
#     format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# handlers:
#   console:
#     class: logging.StreamHandler
#     level: INFO
#     formatter: standard
#   file:
#     class: logging.handlers.RotatingFileHandler
#     level: DEBUG
#     formatter: standard
#     filename: logs/app.log
# root:
#   level: INFO
#   handlers: [console, file]

# 加载 YAML 配置:
# import yaml
# with open("config/logging.yaml") as f:
#     config = yaml.safe_load(f)
#     logging.config.dictConfig(config)
```


> **Note:** 💡 logging 要点: (1) 5 级别: DEBUG < INFO < WARNING < ERROR < CRITICAL; (2) 核心组件: Logger → Handler → Formatter; (3) 每个模块用 logging.getLogger(__name__) 获取 logger; (4) logging.exception() 自动记录 traceback; (5) dictConfig 统一配置; (6) RotatingFileHandler 日志轮转防磁盘爆满。


## 练习


<!-- Converted from: 82_Python logging日志模块.html -->
