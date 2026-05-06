# 综合项目 - 简易 HTTP 服务器

## 1. 项目概述

```c
/*
 * 实现一个简易的 HTTP 服务器，功能包括:
 * - 处理 GET 请求
 * - 静态文件服务
 * - MIME 类型识别
 * - 多线程并发处理
 * - 错误页面
 * - 日志记录
 *
 * 文件结构:
 *   webserver.c     - 主程序
 *   http_parser.c   - HTTP 解析
 *   file_handler.c  - 文件处理
 *   Makefile
 */
```

## 2. HTTP 请求解析

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * HTTP 请求格式:
 *   GET /path HTTP/1.1\r\n
 *   Host: example.com\r\n
 *   User-Agent: curl/7.0\r\n
 *   \r\n
 */

#define MAX_REQUEST_SIZE 8192
#define MAX_URI_SIZE 1024
#define MAX_HEADERS 32

typedef struct {
    char method[16];       // GET, POST, etc.
    char uri[MAX_URI_SIZE]; // /path/to/file
    char version[16];      // HTTP/1.1
    struct {
        char name[128];
        char value[512];
    } headers[MAX_HEADERS];
    int num_headers;
} HttpRequest;

// 解析请求行
int parse_request_line(const char *line, HttpRequest *req)
{
    char method[16], uri[MAX_URI_SIZE], version[16];

    if (sscanf(line, "%15s %1023s %15s", method, uri, version) != 3) {
        return -1;
    }

    strncpy(req->method, method, sizeof(req->method) - 1);
    strncpy(req->uri, uri, sizeof(req->uri) - 1);
    strncpy(req->version, version, sizeof(req->version) - 1);

    return 0;
}

// 解析 HTTP 请求
int parse_request(int fd, HttpRequest *req)
{
    char buf[MAX_REQUEST_SIZE];
    int total = 0;

    // 读取请求
    ssize_t n;
    while (total < MAX_REQUEST_SIZE - 1) {
        n = read(fd, buf + total, MAX_REQUEST_SIZE - 1 - total);
        if (n <= 0) break;
        total += n;

        // 检查是否读到了完整请求头
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n")) break;
    }

    if (total == 0) return -1;

    // 解析请求行
    char *line_end = strstr(buf, "\r\n");
    if (!line_end) return -1;

    *line_end = '\0';
    if (parse_request_line(buf, req) != 0) return -1;

    // 解析头部
    req->num_headers = 0;
    char *line = line_end + 2;

    while (*line && req->num_headers < MAX_HEADERS) {
        line_end = strstr(line, "\r\n");
        if (!line_end) break;

        // 空行表示头部结束
        if (line_end == line) break;

        *line_end = '\0';

        char *colon = strchr(line, ':');
        if (colon) {
            *colon = '\0';
            strncpy(req->headers[req->num_headers].name, line, 127);
            char *value = colon + 1;
            while (*value == ' ') value++;
            strncpy(req->headers[req->num_headers].value, value, 511);
            req->num_headers++;
        }

        line = line_end + 2;
    }

    return 0;
}
```

## 3. MIME 类型与响应

```c
#include <stdio.h>
#include <string.h>
#include <time.h>

// MIME 类型映射
typedef struct {
    const char *ext;
    const char *mime;
} MimeEntry;

static const MimeEntry mime_types[] = {
    {".html", "text/html; charset=utf-8"},
    {".htm",  "text/html; charset=utf-8"},
    {".css",  "text/css"},
    {".js",   "application/javascript"},
    {".json", "application/json"},
    {".png",  "image/png"},
    {".jpg",  "image/jpeg"},
    {".jpeg", "image/jpeg"},
    {".gif",  "image/gif"},
    {".ico",  "image/x-icon"},
    {".svg",  "image/svg+xml"},
    {".txt",  "text/plain; charset=utf-8"},
    {".pdf",  "application/pdf"},
    {".zip",  "application/zip"},
    {NULL,    "application/octet-stream"}
};

const char *get_mime_type(const char *path)
{
    const char *dot = strrchr(path, '.');
    if (!dot) return "application/octet-stream";

    for (const MimeEntry *e = mime_types; e->ext; e++) {
        if (strcasecmp(dot, e->ext) == 0) {
            return e->mime;
        }
    }

    return "application/octet-stream";
}

// 发送 HTTP 响应
int send_response(int fd, int status, const char *status_text,
                  const char *content_type, const char *body, size_t body_len)
{
    char header[1024];
    time_t now = time(NULL);
    char time_buf[64];

    strftime(time_buf, sizeof(time_buf), "%a, %d %b %Y %H:%M:%S GMT",
             gmtime(&now));

    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Date: %s\r\n"
        "Server: MiniHTTP/1.0\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, time_buf, content_type, body_len);

    write(fd, header, hlen);
    if (body && body_len > 0) {
        write(fd, body, body_len);
    }

    return 0;
}

// 发送错误页面
void send_error(int fd, int status, const char *status_text)
{
    char body[512];
    int len = snprintf(body, sizeof(body),
        "<html><head><title>%d %s</title></head>"
        "<body><h1>%d %s</h1></body></html>\r\n",
        status, status_text, status, status_text);

    send_response(fd, status, status_text, "text/html", body, len);
}
```

## 4. 文件服务

```c
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define WEB_ROOT "/var/www/html"

// 安全路径检查
int is_safe_path(const char *uri)
{
    // 防止目录遍历攻击
    if (strstr(uri, "..")) return 0;
    if (uri[0] != '/') return 0;
    return 1;
}

// 构建文件路径
void build_path(char *dest, size_t size, const char *uri)
{
    if (strcmp(uri, "/") == 0) {
        snprintf(dest, size, "%s/index.html", WEB_ROOT);
    } else {
        snprintf(dest, size, "%s%s", WEB_ROOT, uri);
    }
}

// 处理 GET 请求
void handle_get(int fd, const char *uri)
{
    if (!is_safe_path(uri)) {
        send_error(fd, 403, "Forbidden");
        return;
    }

    char path[512];
    build_path(path, sizeof(path), uri);

    struct stat st;
    if (stat(path, &st) == -1) {
        if (errno == ENOENT) {
            send_error(fd, 404, "Not Found");
        } else {
            send_error(fd, 500, "Internal Server Error");
        }
        return;
    }

    // 检查是否是目录
    if (S_ISDIR(st.st_mode)) {
        strncat(path, "/index.html", sizeof(path) - strlen(path) - 1);
        if (stat(path, &st) == -1) {
            send_error(fd, 404, "Not Found");
            return;
        }
    }

    // 打开文件
    int file_fd = open(path, O_RDONLY);
    if (file_fd == -1) {
        send_error(fd, 403, "Forbidden");
        return;
    }

    // 发送响应头
    const char *mime = get_mime_type(path);
    char header[1024];
    time_t now = time(NULL);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%a, %d %b %Y %H:%M:%S GMT",
             gmtime(&now));

    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Date: %s\r\n"
        "Server: MiniHTTP/1.0\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %ld\r\n"
        "Connection: close\r\n"
        "\r\n",
        time_buf, mime, (long)st.st_size);
    write(fd, header, hlen);

    // 使用 sendfile 零拷贝发送文件
    #ifdef __linux__
    #include <sys/sendfile.h>
    off_t offset = 0;
    size_t remaining = st.st_size;
    while (remaining > 0) {
        ssize_t n = sendfile(fd, file_fd, &offset, remaining);
        if (n <= 0) break;
        remaining -= n;
    }
    #else
    char buf[8192];
    ssize_t n;
    while ((n = read(file_fd, buf, sizeof(buf))) > 0) {
        write(fd, buf, n);
    }
    #endif

    close(file_fd);
}
```

## 5. 主程序

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <syslog.h>
#include <errno.h>

#define PORT 8080
#define NUM_THREADS 8

// 工作线程
typedef struct {
    int conn_fd;
    struct sockaddr_in client_addr;
} ClientInfo;

void *worker_thread(void *arg)
{
    ClientInfo *ci = (ClientInfo *)arg;
    int fd = ci->conn_fd;

    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &ci->client_addr.sin_addr,
              client_ip, sizeof(client_ip));
    int client_port = ntohs(ci->client_addr.sin_port);
    free(ci);

    pthread_detach(pthread_self());

    syslog(LOG_INFO, "Connection from %s:%d", client_ip, client_port);

    // 解析请求
    HttpRequest req;
    memset(&req, 0, sizeof(req));

    if (parse_request(fd, &req) == 0) {
        syslog(LOG_INFO, "%s %s from %s:%d",
               req.method, req.uri, client_ip, client_port);

        // 只支持 GET
        if (strcmp(req.method, "GET") == 0) {
            handle_get(fd, req.uri);
        } else {
            send_error(fd, 405, "Method Not Allowed");
        }
    } else {
        send_error(fd, 400, "Bad Request");
    }

    close(fd);
    return NULL;
}

static volatile sig_atomic_t running = 1;

void signal_handler(int sig)
{
    if (sig == SIGINT || sig == SIGTERM) running = 0;
}

int main(int argc, char *argv[])
{
    int port = PORT;
    if (argc > 1) port = atoi(argv[1]);

    // 打开日志
    openlog("minihttp", LOG_PID | LOG_NDELAY, LOG_DAEMON);

    // 设置信号处理
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    // 创建监听 socket
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(listen_fd, (struct sockaddr *)&server_addr,
             sizeof(server_addr)) == -1) {
        syslog(LOG_ERR, "bind: %s", strerror(errno));
        closelog();
        return EXIT_FAILURE;
    }

    listen(listen_fd, 128);
    syslog(LOG_INFO, "MiniHTTP server listening on port %d", port);
    printf("MiniHTTP server listening on port %d\n", port);

    // 主循环：接受连接并分发到线程
    while (running) {
        ClientInfo *ci = malloc(sizeof(ClientInfo));
        socklen_t len = sizeof(ci->client_addr);

        ci->conn_fd = accept(listen_fd,
                             (struct sockaddr *)&ci->client_addr, &len);
        if (ci->conn_fd == -1) {
            free(ci);
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, worker_thread, ci) != 0) {
            syslog(LOG_ERR, "pthread_create: %s", strerror(errno));
            close(ci->conn_fd);
            free(ci);
        }
    }

    syslog(LOG_INFO, "Server shutting down");
    printf("\nServer stopped\n");

    close(listen_fd);
    closelog();
    return EXIT_SUCCESS;
}

/*
 * 编译和使用:
 *
 * 编译:
 *   gcc -o minihttp webserver.c -lpthread
 *
 * 运行:
 *   sudo ./minihttp 8080
 *
 * 测试:
 *   curl http://localhost:8080/
 *   curl http://localhost:8080/index.html
 *
 * 目录结构:
 *   /var/www/html/
 *     index.html
 *     style.css
 *     images/
 *
 * 注意:
 *   - 需要创建 WEB_ROOT 目录
 *   - 端口 < 1024 需要 root 权限
 *   - 可以修改 WEB_ROOT 宏改变根目录
 */
```
