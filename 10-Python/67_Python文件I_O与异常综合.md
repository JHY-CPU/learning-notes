# Python文件I/O与异常综合


## 🔧 Python 文件 I/O 与异常综合


文件 I/O 知识全景、异常处理全览、综合实战: 数据导入器 (CSV 读取/JSON 导出/异常处理/日志记录)、上下文管理器文件处理、最佳实践总结。


## 文件 I/O 知识全景


```
// ========== 文件 I/O 全景图 ==========
// 打开文件:
//   open(filename, mode, encoding)
//
// 读取模式:
//   f.read()          — 全部 (小心大文件)
//   f.read(n)         — n 个字符/字节
//   f.readline()      — 一行
//   f.readlines()     — 所有行到列表
//   for line in f     — 逐行迭代 (推荐)
//
// 写入模式:
//   f.write(str)      — 写入字符串
//   f.writelines(lst) — 写入多行
//
// 文件指针:
//   f.tell()          — 当前位置
//   f.seek(offset)    — 移动指针
//
// 文件模式:
//   r, w, a, x        — 文本模式
//   rb, wb, ab        — 二进制模式
//   r+, w+, a+        — 读写模式
//
// 模块支持:
//   csv — CSV 文件
//   json — JSON 文件
//   pickle — Python 对象序列化
//   os — 文件系统操作
//   pathlib — 面向对象路径

// ========== 异常处理全览 ==========
// try/except/else/finally — 完整异常处理
// raise — 抛出异常
// raise ... from — 异常链
// assert — 调试断言
//
// 自定义异常:
//   class MyError(Exception):
//       def __init__(self, msg, data=None):
//           ...
//
// 日志记录:
//   logging.exception("消息")  # 记录异常+堆栈
//
// 上下文管理器:
//   with open(...) as f:  # 自动关闭
//   @contextmanager       # 生成器版上下文
//
// EAFP 原则:
//   先尝试,出错再处理
```


## 综合实战: 数据导入器


```
// ========== CSV → JSON 导入器 ==========
// 综合运用: 文件 I/O + CSV + JSON + 异常 + 日志 + 上下文管理器

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("importer.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("importer")

# ---------- 异常体系 ----------
class ImportError(Exception):
    """导入基础异常"""
    pass

class FileReadError(ImportError):
    """文件读取错误"""
    pass

class DataValidationError(ImportError):
    """数据验证错误"""
    def __init__(self, message, row=None, errors=None):
        super().__init__(message)
        self.row = row
        self.errors = errors or []

class SchemaError(ImportError):
    """数据结构错误"""
    pass

# ---------- 数据模型 ----------
@dataclass
class User:
    """用户数据模型"""
    name: str
    email: str
    age: int
    city: str = ""
    active: bool = True

    def validate(self):
        """验证数据"""
        errors = []
        if not self.name or not self.name.strip():
            errors.append("名字不能为空")
        if not self.email or "@" not in self.email:
            errors.append("邮箱格式无效")
        if not (0 < self.age < 150):
            errors.append("年龄超出范围")
        return errors

# ---------- 导入器 ----------
class CSVImporter:
    """CSV 数据导入器"""
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "errors": [],
        }

    def validate_schema(self, headers: List[str]):
        """验证 CSV 表头"""
        required = {"name", "email", "age"}
        headers_set = {h.strip().lower() for h in headers}
        missing = required - headers_set
        if missing:
            raise SchemaError(f"缺少必要列: {missing}")

    def parse_row(self, row: Dict) -> User:
        """解析单行数据"""
        try:
            return User(
                name=row.get("name", "").strip(),
                email=row.get("email", "").strip(),
                age=int(row.get("age", 0)),
                city=row.get("city", "").strip(),
                active=row.get("active", "true").lower() == "true",
            )
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"解析行失败", row=row, errors=[str(e)])

    def import_data(self) -> List[User]:
        """执行导入"""
        logger.info(f"开始导入: {self.filename}")
        users = []

        try:
            with open(self.filename, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self.validate_schema(reader.fieldnames or [])

                for row_num, row in enumerate(reader, start=2):
                    self.stats["total"] += 1
                    try:
                        user = self.parse_row(row)
                        errors = user.validate()
                        if errors:
                            raise DataValidationError(
                                f"第{row_num}行验证失败",
                                row=row,
                                errors=errors
                            )
                        users.append(user)
                        self.stats["success"] += 1
                    except DataValidationError as e:
                        self.stats["failed"] += 1
                        self.stats["errors"].append({
                            "row": row_num,
                            "message": str(e),
                            "errors": e.errors,
                        })
                        logger.warning(f"第{row_num}行跳过: {e}")

        except FileNotFoundError:
            raise FileReadError(f"文件不存在: {self.filename}")
        except csv.Error as e:
            raise FileReadError(f"CSV 解析错误: {e}")

        logger.info(f"导入完成: 成功{self.stats['success']}/总计{self.stats['total']}")
        return users

# ---------- 使用 ----------
def export_to_json(users: List[User], output: str):
    """导出到 JSON"""
    data = [asdict(u) for u in users]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"导出 {len(users)} 条到 {output}")

# 执行导入
try:
    importer = CSVImporter("users.csv")
    users = importer.import_data()
    export_to_json(users, "users_export.json")
    print(f"导入报告: {importer.stats}")
except ImportError as e:
    logger.error(f"导入失败: {e}")
```


## 上下文管理文件模式总结


```
// ========== with 管理资源模式 ==========
// 模式 1: 文件自动关闭
with open("file.txt", encoding="utf-8") as f:
    data = f.read()
# 自动 close

// 模式 2: 多个文件
with open("src.txt") as src, open("dst.txt", "w") as dst:
    dst.write(src.read())

// 模式 3: 异常安全写入
@contextmanager
def safe_write(filename):
    """写入文件: 异常时保留原文件"""
    import tempfile, os, shutil
    backup = filename + ".bak"
    if os.path.exists(filename):
        shutil.copy2(filename, backup)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            yield f
    except Exception:
        # 恢复备份
        if os.path.exists(backup):
            shutil.copy2(backup, filename)
        raise
    finally:
        if os.path.exists(backup):
            os.remove(backup)

with safe_write("important.json") as f:
    f.write('{"key": "value"}')

// ========== 文件 I/O 最佳实践 ==========
// 1. 始终用 with 管理文件
// 2. 大文件用 for line in f 逐行迭代
// 3. 编码永远指定 encoding="utf-8"
// 4. CSV 用 newline="" 避免空行
// 5. 未知编码用 errors="replace" 或 utf-8-sig
// 6. 用 pathlib.Path 而非 os.path
// 7. 写日志用 logging 而非 print
// 8. 自定义异常体系,用 raise ... from e
```


> **Note:** 💡 综合要点: (1) 文件 + 异常 + 日志 是数据处理的铁三角; (2) CSV DictReader 简化字段访问; (3) 自定义异常 + logging 让错误可追踪; (4) dataclass + asdict() 简化 JSON 导出; (5) with 始终确保资源正确释放。


## 练习


<!-- Converted from: 67_Python文件I_O与异常综合.html -->
