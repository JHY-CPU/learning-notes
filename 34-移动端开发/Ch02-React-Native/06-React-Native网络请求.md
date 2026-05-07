# React Native 网络请求

## 一、概念说明

网络请求是移动应用与后端服务器通信的基础。React Native 内置了 Fetch API，同时也支持 Axios 等第三方库。合理封装网络请求层能提高代码可维护性和错误处理能力。

```javascript
// Fetch API 基础用法
const fetchExample = async () => {
  try {
    const response = await fetch('https://api.example.com/data', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer token123',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP Error: ${response.status}`);
    }

    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('请求失败:', error);
  }
};
```

## 二、网络请求封装

### 2.1 Axios 封装

```javascript
// services/api.js
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const api = axios.create({
  baseURL: 'https://api.example.com/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  async (config) => {
    const token = await AsyncStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  async (error) => {
    if (error.response?.status === 401) {
      // Token 过期，尝试刷新
      await AsyncStorage.removeItem('auth_token');
      // 跳转到登录页
    }
    return Promise.reject(error);
  }
);

export default api;
```

### 2.2 请求方法封装

```javascript
// services/http.js
import api from './api';

export const http = {
  get: (url, params) => api.get(url, { params }),
  post: (url, data) => api.post(url, data),
  put: (url, data) => api.put(url, data),
  patch: (url, data) => api.patch(url, data),
  delete: (url) => api.delete(url),

  // 文件上传
  upload: (url, formData, onProgress) => {
    return api.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        const progress = Math.round((e.loaded * 100) / e.total);
        onProgress?.(progress);
      },
    });
  },

  // 下载文件
  download: (url, filePath, onProgress) => {
    return api.get(url, {
      responseType: 'blob',
      onDownloadProgress: (e) => {
        onProgress?.(e.loaded / e.total);
      },
    });
  },
};
```

### 2.3 使用示例

```javascript
// services/userService.js
import { http } from './http';

export const userService = {
  // 获取用户列表
  getUsers: (page = 1, size = 20) =>
    http.get('/users', { page, size }),

  // 获取用户详情
  getUser: (id) =>
    http.get(`/users/${id}`),

  // 创建用户
  createUser: (data) =>
    http.post('/users', data),

  // 更新用户
  updateUser: (id, data) =>
    http.put(`/users/${id}`, data),

  // 删除用户
  deleteUser: (id) =>
    http.delete(`/users/${id}`),

  // 上传头像
  uploadAvatar: (userId, imageUri, onProgress) => {
    const formData = new FormData();
    formData.append('avatar', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'avatar.jpg',
    });
    return http.upload(`/users/${userId}/avatar`, formData, onProgress);
  },
};

// 在组件中使用
const UserScreen = ({ userId }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadUser();
  }, [userId]);

  const loadUser = async () => {
    try {
      setLoading(true);
      const data = await userService.getUser(userId);
      setUser(data);
    } catch (error) {
      Alert.alert('错误', '加载用户信息失败');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <ActivityIndicator />;
  return <UserProfile user={user} />;
};
```

## 三、错误处理与重试

```javascript
// 带重试的请求
const fetchWithRetry = async (url, options = {}, retries = 3) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      // 指数退避
      await new Promise(r => setTimeout(r, Math.pow(2, i) * 1000));
    }
  }
};

// 网络状态检测
import NetInfo from '@react-native-community/netinfo';

const checkNetwork = async () => {
  const state = await NetInfo.fetch();
  if (!state.isConnected) {
    Alert.alert('网络错误', '请检查网络连接');
    return false;
  }
  return true;
};
```

## 四、注意事项与常见陷阱

1. **HTTPS 要求**：iOS 默认要求 HTTPS，需要在 Info.plist 中配置允许 HTTP
2. **超时设置**：始终设置合理的超时时间，避免请求无限等待
3. **取消请求**：组件卸载时取消未完成的请求，避免内存泄漏和状态更新错误
4. **并发控制**：使用请求队列或并发限制，避免同时发起过多请求
5. **缓存策略**：对不常变化的数据实施缓存，减少网络请求次数
