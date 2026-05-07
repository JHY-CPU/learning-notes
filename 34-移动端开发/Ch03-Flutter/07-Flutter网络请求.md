# Flutter 网络请求

## 一、概念说明

Flutter 中最常用的 HTTP 客户端库是 Dio，它提供了拦截器、全局配置、FormData 上传等强大功能。当然也可以使用内置的 http 包。

```dart
// 安装依赖
// pubspec.yaml
dependencies:
  dio: ^5.3.0
  retrofit: ^4.0.0
  json_annotation: ^4.8.0
dev_dependencies:
  json_serializable: ^6.7.0
  build_runner: ^2.4.0
  retrofit_generator: ^8.0.0
```

## 二、Dio 封装

### 2.1 基础配置

```dart
// services/http_client.dart
import 'package:dio/dio.dart';

class HttpClient {
  static final HttpClient _instance = HttpClient._internal();
  late Dio _dio;

  factory HttpClient() => _instance;

  HttpClient._internal() {
    _dio = Dio(BaseOptions(
      baseUrl: 'https://api.example.com/v1',
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 10),
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ));

    // 请求拦截器
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) async {
        // 添加 Token
        final token = await Storage.get('token');
        if (token != null) {
          options.headers['Authorization'] = 'Bearer $token';
        }
        print('请求: ${options.method} ${options.path}');
        return handler.next(options);
      },
      onResponse: (response, handler) {
        print('响应: ${response.statusCode}');
        return handler.next(response);
      },
      onError: (error, handler) async {
        if (error.response?.statusCode == 401) {
          // Token 过期，尝试刷新
          final refreshed = await _refreshToken();
          if (refreshed) {
            return handler.resolve(await _retry(error.requestOptions));
          }
        }
        return handler.next(error);
      },
    ));

    // 日志拦截器（调试模式）
    _dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
    ));
  }

  Dio get dio => _dio;

  Future<bool> _refreshToken() async {
    // 刷新 Token 逻辑
    return false;
  }

  Future<Response> _retry(RequestOptions options) async {
    return _dio.fetch(options);
  }
}
```

### 2.2 API 封装

```dart
// services/api_service.dart
import 'package:dio/dio.dart';

class ApiService {
  final Dio _dio = HttpClient().dio;

  // GET 请求
  Future<T> get<T>(
    String path, {
    Map<String, dynamic>? queryParameters,
    required T Function(dynamic data) fromJson,
  }) async {
    final response = await _dio.get(path, queryParameters: queryParameters);
    return fromJson(response.data);
  }

  // POST 请求
  Future<T> post<T>(
    String path, {
    dynamic data,
    required T Function(dynamic data) fromJson,
  }) async {
    final response = await _dio.post(path, data: data);
    return fromJson(response.data);
  }

  // 文件上传
  Future<T> upload<T>(
    String path, {
    required String filePath,
    required String fileName,
    ProgressCallback? onSendProgress,
    required T Function(dynamic data) fromJson,
  }) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(filePath, filename: fileName),
    });

    final response = await _dio.post(
      path,
      data: formData,
      onSendProgress: onSendProgress,
    );
    return fromJson(response.data);
  }

  // 分页列表
  Future<PaginatedData<T>> getList<T>(
    String path, {
    int page = 1,
    int size = 20,
    Map<String, dynamic>? queryParameters,
    required T Function(Map<String, dynamic> json) fromJson,
  }) async {
    final params = {...?queryParameters, 'page': page, 'size': size};
    final response = await _dio.get(path, queryParameters: params);
    return PaginatedData.fromJson(response.data, fromJson);
  }
}

// 分页数据模型
class PaginatedData<T> {
  final List<T> items;
  final int total;
  final int page;
  final int totalPages;

  PaginatedData({
    required this.items,
    required this.total,
    required this.page,
    required this.totalPages,
  });

  factory PaginatedData.fromJson(
    Map<String, dynamic> json,
    T Function(Map<String, dynamic>) fromJson,
  ) {
    return PaginatedData(
      items: (json['data'] as List).map((e) => fromJson(e)).toList(),
      total: json['total'],
      page: json['page'],
      totalPages: json['totalPages'],
    );
  }
}
```

### 2.3 使用示例

```dart
// models/user.dart
class User {
  final int id;
  final String name;
  final String email;
  final String avatar;

  User({
    required this.id,
    required this.name,
    required this.email,
    required this.avatar,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      name: json['name'],
      email: json['email'],
      avatar: json['avatar'],
    );
  }
}

// services/user_api.dart
class UserApi {
  final _api = ApiService();

  Future<User> getUser(int id) {
    return _api.get('/users/$id', fromJson: (data) => User.fromJson(data));
  }

  Future<PaginatedData<User>> getUsers({int page = 1}) {
    return _api.getList('/users', page: page, fromJson: User.fromJson);
  }

  Future<User> updateUser(int id, Map<String, dynamic> data) {
    return _api.put('/users/$id', data: data, fromJson: (d) => User.fromJson(d));
  }
}
```

## 三、数据序列化

```dart
// 使用 json_serializable 自动生成
import 'package:json_annotation/json_annotation.dart';
part 'product.g.dart';

@JsonSerializable()
class Product {
  final int id;
  final String name;
  final double price;

  @JsonKey(name: 'image_url')
  final String imageUrl;

  @JsonKey(name: 'created_at')
  final DateTime createdAt;

  Product({
    required this.id,
    required this.name,
    required this.price,
    required this.imageUrl,
    required this.createdAt,
  });

  factory Product.fromJson(Map<String, dynamic> json) => _$ProductFromJson(json);
  Map<String, dynamic> toJson() => _$ProductToJson(this);
}

// 运行代码生成: flutter pub run build_runner build
```

## 四、注意事项与常见陷阱

1. **超时配置**：务必设置合理的连接和接收超时时间
2. **错误处理**：统一处理网络错误，给用户友好的提示
3. **请求取消**：页面销毁时取消未完成的请求
4. **并发限制**：控制同时进行的请求数量，避免过多并发
5. **HTTPS 证书**：生产环境必须使用 HTTPS，避免中间人攻击
