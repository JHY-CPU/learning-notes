# React Native 本地存储

## 一、概念说明

本地存储是移动应用在设备上持久化数据的关键技术。React Native 提供了多种存储方案，适用于不同场景：简单键值对存储、结构化数据库、安全存储等。

```javascript
// AsyncStorage 基础使用
import AsyncStorage from '@react-native-async-storage/async-storage';

// 存储数据
const storeData = async (key, value) => {
  try {
    const jsonValue = JSON.stringify(value);
    await AsyncStorage.setItem(key, jsonValue);
  } catch (error) {
    console.error('存储失败:', error);
  }
};

// 读取数据
const getData = async (key) => {
  try {
    const jsonValue = await AsyncStorage.getItem(key);
    return jsonValue != null ? JSON.parse(jsonValue) : null;
  } catch (error) {
    console.error('读取失败:', error);
    return null;
  }
};
```

## 二、存储方案对比

### 2.1 AsyncStorage（键值对存储）

```javascript
// AsyncStorage 完整封装
import AsyncStorage from '@react-native-async-storage/async-storage';

const Storage = {
  // 设置值
  set: async (key, value, expireMinutes = null) => {
    const item = {
      value,
      timestamp: Date.now(),
      expire: expireMinutes ? Date.now() + expireMinutes * 60000 : null,
    };
    await AsyncStorage.setItem(key, JSON.stringify(item));
  },

  // 获取值
  get: async (key) => {
    const json = await AsyncStorage.getItem(key);
    if (!json) return null;

    const item = JSON.parse(json);
    // 检查过期
    if (item.expire && Date.now() > item.expire) {
      await AsyncStorage.removeItem(key);
      return null;
    }
    return item.value;
  },

  // 删除值
  remove: async (key) => {
    await AsyncStorage.removeItem(key);
  },

  // 批量操作
  multiSet: async (keyValuePairs) => {
    const pairs = keyValuePairs.map(([key, value]) => [
      key,
      JSON.stringify({ value, timestamp: Date.now() }),
    ]);
    await AsyncStorage.multiSet(pairs);
  },

  // 清空所有
  clear: async () => {
    await AsyncStorage.clear();
  },

  // 获取所有键
  getAllKeys: async () => {
    return await AsyncStorage.getAllKeys();
  },
};

// 使用示例
const saveUserSession = async (user, token) => {
  await Storage.set('user', user);
  await Storage.set('token', token, 60 * 24); // 24小时过期
};

const getUserSession = async () => {
  const user = await Storage.get('user');
  const token = await Storage.get('token');
  return { user, token };
};
```

### 2.2 MMKV（高性能存储）

```javascript
// react-native-mmkv - 高性能键值存储
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV();

// 基础操作
storage.set('user.name', '张三');
storage.set('user.age', 25);
storage.set('settings.darkMode', true);

const name = storage.getString('user.name');
const age = storage.getNumber('user.age');
const darkMode = storage.getBoolean('settings.darkMode');

// 监听变化
const unsubscribe = storage.addOnValueChangedListener((key) => {
  console.log(`${key} 的值发生了变化`);
});

// 使用 MMKV 的 Hook
import { useMMKVString, useMMKVNumber, useMMKVBoolean } from 'react-native-mmkv';

const SettingsScreen = () => {
  const [darkMode, setDarkMode] = useMMKVBoolean('settings.darkMode');
  const [fontSize, setFontSize] = useMMKVNumber('settings.fontSize', 16);

  return (
    <View>
      <Switch value={darkMode} onValueChange={setDarkMode} />
      <Slider value={fontSize} onValueChange={setFontSize} />
    </View>
  );
};
```

### 2.3 SQLite 数据库

```javascript
// react-native-sqlite-storage
import SQLite from 'react-native-sqlite-storage';

SQLite.enablePromise(true);

const openDatabase = async () => {
  return SQLite.openDatabase({
    name: 'app.db',
    location: 'default',
  });
};

// 初始化数据库
const initDatabase = async () => {
  const db = await openDatabase();

  await db.executeSql(`
    CREATE TABLE IF NOT EXISTS todos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      completed INTEGER DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  return db;
};

// CRUD 操作
const TodoDB = {
  getAll: async () => {
    const db = await openDatabase();
    const [results] = await db.executeSql('SELECT * FROM todos ORDER BY created_at DESC');
    const todos = [];
    for (let i = 0; i < results.rows.length; i++) {
      todos.push(results.rows.item(i));
    }
    return todos;
  },

  add: async (title) => {
    const db = await openDatabase();
    const [result] = await db.executeSql(
      'INSERT INTO todos (title) VALUES (?)',
      [title]
    );
    return result.insertId;
  },

  update: async (id, completed) => {
    const db = await openDatabase();
    await db.executeSql(
      'UPDATE todos SET completed = ? WHERE id = ?',
      [completed ? 1 : 0, id]
    );
  },

  delete: async (id) => {
    const db = await openDatabase();
    await db.executeSql('DELETE FROM todos WHERE id = ?', [id]);
  },
};
```

### 2.4 安全存储

```javascript
// Keychain/Keystore 安全存储
import * as Keychain from 'react-native-keychain';

// 存储敏感数据
const saveSecureData = async (username, password) => {
  await Keychain.setGenericPassword(username, password, {
    service: 'com.myapp.login',
    accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY,
  });
};

// 读取敏感数据
const getSecureData = async () => {
  const credentials = await Keychain.getGenericPassword({
    service: 'com.myapp.login',
  });
  if (credentials) {
    return { username: credentials.username, password: credentials.password };
  }
  return null;
};

// 删除敏感数据
const removeSecureData = async () => {
  await Keychain.resetGenericPassword({ service: 'com.myapp.login' });
};
```

## 三、存储方案选择指南

| 方案 | 数据类型 | 容量限制 | 性能 | 安全性 | 适用场景 |
|------|----------|----------|------|--------|----------|
| AsyncStorage | 键值对 | ~6MB | 一般 | 低 | 配置、缓存 |
| MMKV | 键值对 | 无限制 | 极高 | 低 | 高频读写 |
| SQLite | 结构化 | 无限制 | 高 | 中 | 复杂查询 |
| Keychain | 键值对 | 小 | 一般 | 高 | 敏感信息 |

## 四、注意事项与常见陷阱

1. **存储容量**：AsyncStorage 容量有限，大量数据应使用 SQLite
2. **数据迁移**：版本更新时可能需要数据迁移，提前设计好迁移策略
3. **线程安全**：数据库操作应使用事务，避免并发写入冲突
4. **加密存储**：敏感数据（密码、Token）必须使用加密存储
5. **备份策略**：考虑 iCloud/Google Drive 备份对数据的影响
