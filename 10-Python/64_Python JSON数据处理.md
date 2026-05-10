# Python JSON数据处理


## 📋 Python JSON 数据处理


json 模块: dumps/loads (字符串)、dump/load (文件)、自定义编码/解码、JSON 类型与 Python 类型映射、格式化输出、异常处理。


## JSON 基础: dump/dumps/load/loads


```
// ========== JSON 基本操作 ==========
import json

// ---------- Python → JSON ----------

// json.dumps(): 对象 → JSON 字符串
data = {
    "name": "Alice",
    "age": 30,
    "scores": [85, 92, 78],
    "active": True,
    "address": None,
}

json_str = json.dumps(data)
print(json_str)
# {"name": "Alice", "age": 30, "scores": [85, 92, 78], "active": true, "address": null}

print(type(json_str))          #

// json.dump(): 对象 → JSON 文件
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

// ---------- JSON → Python ----------

// json.loads(): JSON 字符串 → 对象
json_str = '{"name": "Bob", "age": 25, "active": true}'
parsed = json.loads(json_str)
print(parsed)                  # {'name': 'Bob', 'age': 25, 'active': True}
print(parsed["name"])          # Bob

// json.load(): JSON 文件 → 对象
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
    print(loaded)

// ========== 类型映射 ==========
// JSON            Python
// object          dict
// array           list
// string          str
// number (int)    int
// number (float)  float
// true            True
// false           False
// null            None

// JSON 只有这些类型!
// tuple, set, datetime, Decimal 不能直接序列化
// 需要自定义编码器 (见下文)
```


## 格式化输出


```
// ========== 美化输出 ==========
import json

data = {
    "name": "Alice",
    "age": 30,
    "address": {
        "city": "Beijing",
        "zip": "100000"
    },
    "hobbies": ["reading", "coding", "swimming"]
}

// 默认: 紧凑输出
print(json.dumps(data))
# {"name": "Alice", "age": 30, "address": {"city": "Beijing", ...}}

// indent: 缩进
print(json.dumps(data, indent=2))
# {
#   "name": "Alice",
#   "age": 30,
#   "address": {
#     "city": "Beijing",
#     "zip": "100000"
#   },
#   ...
# }

// ========== 其他常用参数 ==========
// sort_keys: 按键排序
print(json.dumps(data, indent=2, sort_keys=True))

// ensure_ascii: 是否转义非 ASCII (默认 True)
print(json.dumps({"name": "中文"}))        # {"name": "中文"}
print(json.dumps({"name": "中文"}, ensure_ascii=False))  # {"name": "中文"}

// separators: 自定义分隔符 (压缩用)
print(json.dumps(data, separators=(",", ":")))
# 最紧凑格式,无多余空格

// skipkeys: 跳过非字符串键
data2 = {(1, 2): "tuple key"}  # 元组键不能 JSON 序列化
# json.dumps(data2)             # TypeError!
print(json.dumps(data2, skipkeys=True))  # {} (跳过)

// default: 处理不可序列化类型
from datetime import datetime

def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"不能序列化 {type(obj)}")

now = datetime.now()
print(json.dumps({"time": now}, default=datetime_serializer))
# {"time": "2026-04-29T10:30:00"}
```


## 自定义编码/解码


```
// ========== 自定义 JSONEncoder ==========
import json
from datetime import datetime

class CustomEncoder(json.JSONEncoder):
    """自定义 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super().default(obj)

data = {
    "name": "Alice",
    "created_at": datetime.now(),
    "tags": {"python", "json", "coding"},
}

json_str = json.dumps(data, cls=CustomEncoder, indent=2)
print(json_str)

// ========== 自定义解码 (JSON → 对象) ==========
def custom_decoder(obj):
    """自定义 JSON 解码"""
    if "__type__" in obj and obj["__type__"] == "datetime":
        return datetime.fromisoformat(obj["value"])
    return obj

json_str = '{"time": {"__type__": "datetime", "value": "2026-04-29T10:30:00"}}'
parsed = json.loads(json_str, object_hook=custom_decoder)
print(parsed["time"])          # 2026-04-29 10:30:00 (datetime 对象!)
print(type(parsed["time"]))    #

// ========== 序列化 dataclass ==========
from dataclasses import dataclass, asdict

@dataclass
class Person:
    name: str
    age: int
    email: str = ""

p = Person("Alice", 30)
json_str = json.dumps(asdict(p), indent=2)
print(json_str)
# {"name": "Alice", "age": 30, "email": ""}

// 或者用 __dict__:
json_str = json.dumps(p.__dict__)
```


## JSON 实战与异常处理


```
// ========== 异常处理 ==========
import json

def safe_load_json(filename):
    """安全加载 JSON 文件"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

// ========== 大文件流式处理 ==========
// 使用 ijson 或分批处理

def process_large_json(filename):
    """流式处理 JSON 数组"""
    import json
    with open(filename, "r", encoding="utf-8") as f:
        # 如果 JSON 是数组,可以逐行读取
        # 但标准 json.load 会一次加载全部
        # 大文件建议用 ijson 库
        pass

// 对于超大 JSON,每行一个对象 (JSON Lines):
// {"id": 1, "name": "Alice"}
// {"id": 2, "name": "Bob"}
def read_json_lines(filename):
    """读取 JSON Lines 格式"""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

// ========== 实战: 配置管理 ==========
class Config:
    """基于 JSON 的配置管理器"""
    def __init__(self, filename="config.json"):
        self.filename = filename
        self._data = self._load()

    def _load(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self._save()

    def _save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def __repr__(self):
        return json.dumps(self._data, indent=2, ensure_ascii=False)

config = Config()
config.set("debug", True)
config.set("db.host", "localhost")
```


> **Note:** 💡 JSON 要点: (1) dumps/dump 序列化,loads/load 反序列化; (2) indent 美化输出,ensure_ascii=False 保留中文; (3) 无法直接序列化 datetime/set,用 default 参数或自定义编码器; (4) object_hook 自定义解码; (5) JSON Lines 格式适合大文件流式处理。


## 练习


<!-- Converted from: 64_Python JSON数据处理.html -->
