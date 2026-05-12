# path路径模块


## path 路径模块


join / resolve / basename / dirname / extname / parse / format / sep / delimiter。


## path 模块 API


```
// ========== path 路径操作 ==========
const path = require('path');

// ========== 路径拼接 ==========
path.join('/base', 'dir', 'file.txt');
// '/base/dir/file.txt' (自动处理分隔符)

path.resolve('dist', 'js', 'app.js');
// '/current/working/dir/dist/js/app.js' (绝对路径)

// ========== 路径解析 ==========
path.basename('/a/b/file.txt');      // 'file.txt'
path.basename('/a/b/file.txt', '.txt'); // 'file'
path.dirname('/a/b/file.txt');       // '/a/b'
path.extname('/a/b/file.txt');       // '.txt'

// ========== 路径对象 ==========
const parsed = path.parse('/home/user/app.js');
// { root: '/', dir: '/home/user', base: 'app.js',
//   ext: '.js', name: 'app' }

const formatted = path.format({
    dir: '/home/user',
    base: 'app.js'
});
// '/home/user/app.js'

// ========== 路径信息 ==========
path.isAbsolute('/a/b');  // true
path.isAbsolute('a/b');   // false
path.relative('/a/b', '/a/c'); // '../c'

// ========== 分隔符 ==========
path.sep;       // '/' (POSIX) 或 '\' (Windows)
path.delimiter; // ':' (POSIX) 或 ';' (Windows)
```


## 演示：路径操作

点击按钮查看


## 跨平台路径处理

```javascript
// ========== Windows vs POSIX ==========
// path 模块根据运行平台自动选择:
// Windows: path === path.win32
// Linux/Mac: path === path.posix

// 强制使用特定平台
path.win32.join('C:\\Users', 'file.txt');  // 'C:\\Users\\file.txt'
path.posix.join('/home', 'file.txt');      // '/home/file.txt'

// ========== 跨平台路径规范化 ==========
function normalizePath(p) {
    // 统一转换为正斜杠 (适用于 URL 和 git 路径)
    return p.replace(/\\/g, '/');
}

// ========== 处理 URL 路径 vs 文件路径 ==========
const { pathToFileURL, fileURLToPath } = require('url');

// 文件路径 → file:// URL
const fileUrl = pathToFileURL('/home/user/file.txt');
// URL { href: 'file:///home/user/file.txt' }

// file:// URL → 文件路径
const filePath = fileURLToPath('file:///C:/Users/file.txt');
// 'C:\\Users\\file.txt' (Windows)
```

## 路径安全：防止路径遍历攻击

```javascript
// ========== 防止路径遍历 (Directory Traversal) ==========
const path = require('path');

// 危险: 用户输入可能包含 ../
// GET /files?name=../../../etc/passwd

function safePath(baseDir, userInput) {
    const resolved = path.resolve(baseDir, userInput);
    // 确保解析后的路径仍在 baseDir 内
    if (!resolved.startsWith(path.resolve(baseDir))) {
        throw new Error('路径越界');
    }
    return resolved;
}

// 使用示例
const publicDir = '/var/www/public';
try {
    const safe = safePath(publicDir, req.query.name);
    // 安全地读取文件
} catch (err) {
    res.statusCode = 403;
    res.end('Forbidden');
}
```

## 实战：项目路径工具

```javascript
// ========== 项目根目录工具 ==========
const path = require('path');

// 查找项目根目录 (通过 package.json)
function findProjectRoot(startDir = __dirname) {
    let dir = startDir;
    while (true) {
        if (fs.existsSync(path.join(dir, 'package.json'))) {
            return dir;
        }
        const parent = path.dirname(dir);
        if (parent === dir) break; // 到达文件系统根目录
        dir = parent;
    }
    throw new Error('未找到项目根目录');
}

// ========== 构建路径别名 ==========
const projectRoot = findProjectRoot();
const paths = {
    src: path.join(projectRoot, 'src'),
    dist: path.join(projectRoot, 'dist'),
    config: path.join(projectRoot, 'config'),
    public: path.join(projectRoot, 'public'),
    nodeModules: path.join(projectRoot, 'node_modules'),
};

// 使用
const entryFile = path.join(paths.src, 'index.js');
const outputPath = path.join(paths.dist, 'bundle.js');
```

## join vs resolve 的区别

```javascript
// ========== join vs resolve ==========
// join: 拼接路径，不解析为绝对路径
path.join('src', 'utils', 'helper.js');
// 'src/utils/helper.js'

path.join('/base', '..', 'other', 'file.txt');
// '/base/../other/file.txt' → 规范化后: '/other/file.txt'

// resolve: 从右到左解析，直到生成绝对路径
path.resolve('src', 'utils');
// '/current/working/dir/src/utils'

path.resolve('/base', 'src', '/absolute', 'file');
// '/absolute/file' (遇到绝对路径会重新开始)

path.resolve('src');  // '/cwd/src'
path.resolve('/src'); // '/src'

// ========== 常见用法 ==========
// Webpack/Vite 配置中的典型用法
const entry = path.resolve(__dirname, 'src/index.js');
const output = path.resolve(__dirname, 'dist');
```

<!-- Converted from: 8_path路径模块.html -->
