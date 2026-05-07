# Flutter 本地存储

## 一、概念说明

Flutter 提供了多种本地存储方案，从简单的键值对存储到完整的数据库。根据数据复杂度和查询需求选择合适的存储方案。

```yaml
# pubspec.yaml 常用存储依赖
dependencies:
  shared_preferences: ^2.2.0   # 简单键值对
  hive: ^2.2.3                 # NoSQL 数据库
  sqflite: ^2.3.0              # SQLite 数据库
  flutter_secure_storage: ^9.0.0 # 安全存储
  path_provider: ^2.1.0        # 文件路径
```

## 二、SharedPreferences

```dart
import 'package:shared_preferences/shared_preferences.dart';

class PrefsStorage {
  // 存储
  static Future<void> saveString(String key, String value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(key, value);
  }

  static Future<void> saveInt(String key, int value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(key, value);
  }

  static Future<void> saveBool(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  static Future<void> saveStringList(String key, List<String> value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(key, value);
  }

  // 读取
  static Future<String?> getString(String key) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(key);
  }

  static Future<int?> getInt(String key) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(key);
  }

  static Future<bool> getBool(String key, {bool defaultValue = false}) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(key) ?? defaultValue;
  }

  // 删除
  static Future<void> remove(String key) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(key);
  }

  // 清空
  static Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
  }
}

// 使用示例
class SettingsService {
  static const _darkModeKey = 'dark_mode';
  static const _languageKey = 'language';
  static const _tokenKey = 'auth_token';

  static Future<bool> isDarkMode() => PrefsStorage.getBool(_darkModeKey);
  static Future<void> setDarkMode(bool value) => PrefsStorage.saveBool(_darkModeKey, value);
  static Future<String?> getToken() => PrefsStorage.getString(_tokenKey);
  static Future<void> setToken(String token) => PrefsStorage.saveString(_tokenKey, token);
}
```

## 三、Hive（NoSQL 数据库）

```dart
import 'package:hive_flutter/hive_flutter.dart';

// 初始化 Hive
Future<void> initHive() async {
  await Hive.initFlutter();
  Hive.registerAdapter(UserAdapter());
  Hive.registerAdapter(TodoAdapter());
}

// 实体类
@HiveType(typeId: 0)
class User extends HiveObject {
  @HiveField(0)
  final int id;

  @HiveField(1)
  final String name;

  @HiveField(2)
  final String email;

  User({required this.id, required this.name, required this.email});
}

// 生成适配器: flutter packages pub run build_runner build

// 数据仓库
class UserRepository {
  late Box<User> _box;

  Future<void> init() async {
    _box = await Hive.openBox<User>('users');
  }

  List<User> getAll() => _box.values.toList();

  User? getById(int id) => _box.get(id);

  Future<void> save(User user) async {
    await _box.put(user.id, user);
  }

  Future<void> delete(int id) async {
    await _box.delete(id);
  }

  // 监听变化
  Stream<BoxEvent> watch() => _box.watch();

  Future<void> close() async {
    await _box.close();
  }
}
```

## 四、SQLite 数据库

```dart
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DatabaseHelper {
  static Database? _database;

  static Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  static Future<Database> _initDatabase() async {
    final path = join(await getDatabasesPath(), 'app.db');
    return openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
          )
        ''');
      },
    );
  }

  // CRUD 操作
  static Future<int> insertTodo(String title) async {
    final db = await database;
    return db.insert('todos', {'title': title});
  }

  static Future<List<Map<String, dynamic>>> getTodos() async {
    final db = await database;
    return db.query('todos', orderBy: 'created_at DESC');
  }

  static Future<int> updateTodo(int id, bool completed) async {
    final db = await database;
    return db.update('todos', {'completed': completed ? 1 : 0},
        where: 'id = ?', whereArgs: [id]);
  }

  static Future<int> deleteTodo(int id) async {
    final db = await database;
    return db.delete('todos', where: 'id = ?', whereArgs: [id]);
  }

  // 批量操作（事务）
  static Future<void> batchInsert(List<String> titles) async {
    final db = await database;
    await db.transaction((txn) async {
      for (final title in titles) {
        await txn.insert('todos', {'title': title});
      }
    });
  }
}
```

## 五、安全存储

```dart
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class SecureStorage {
  static const _storage = FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
  );

  static Future<void> write(String key, String value) async {
    await _storage.write(key: key, value: value);
  }

  static Future<String?> read(String key) async {
    return await _storage.read(key: key);
  }

  static Future<void> delete(String key) async {
    await _storage.delete(key: key);
  }

  static Future<void> deleteAll() async {
    await _storage.deleteAll();
  }
}

// 使用示例 - 存储敏感信息
class AuthService {
  static const _tokenKey = 'auth_token';
  static const _refreshTokenKey = 'refresh_token';

  static Future<void> saveTokens(String token, String refreshToken) async {
    await SecureStorage.write(_tokenKey, token);
    await SecureStorage.write(_refreshTokenKey, refreshToken);
  }

  static Future<String?> getToken() => SecureStorage.read(_tokenKey);
  static Future<void> clearTokens() async {
    await SecureStorage.delete(_tokenKey);
    await SecureStorage.delete(_refreshTokenKey);
  }
}
```

## 六、存储方案对比

| 方案 | 数据类型 | 查询能力 | 性能 | 安全性 |
|------|----------|----------|------|--------|
| SharedPreferences | 键值对 | 无 | 高 | 低 |
| Hive | 对象 | 基础 | 极高 | 低 |
| SQLite | 关系型 | 完整 SQL | 高 | 中 |
| Secure Storage | 键值对 | 无 | 中 | 高 |

## 七、注意事项与常见陷阱

1. **SharedPreferences 容量**：不适合存储大量数据，建议只存配置项
2. **Hive 适配器**：修改实体类后需要重新生成适配器
3. **SQLite 版本迁移**：升级数据库版本时需要在 onUpgrade 中处理迁移
4. **异步操作**：所有存储操作都是异步的，注意 await 使用
5. **数据备份**：考虑 iOS iCloud 备份对敏感数据的影响
