# Express 分页过滤排序


## 📑 Express 分页过滤排序


分页实现 (offset/游标)、过滤中间件 (精确/范围/模糊/多字段)、排序实现 (单字段/多字段/黑名单)、高级查询组合、MongoDB/Mongoose 查询构建器、Prisma 查询示例。


## 分页实现


```
// ========== 分页方案 ==========

// ========== 方案 1: Offset 分页 ==========
// GET /api/users?page=1&limit=20
// 简单, 但大偏移量性能差

const page = Math.max(1, parseInt(req.query.page) || 1);
const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));
const skip = (page - 1) * limit;

const [users, total] = await Promise.all([
    User.find().skip(skip).limit(limit).lean(),
    User.countDocuments(),
]);

// ========== 方案 2: 游标分页 ==========
// GET /api/users?cursor=xxx&limit=20
// 性能好, 适合大数据量和实时数据

// 基于 _id 的游标:
app.get('/api/users', async (req, res) => {
    const limit = Math.min(100, parseInt(req.query.limit) || 20);
    const cursor = req.query.cursor;  // 上一页最后一个 _id

    const query = {};
    if (cursor) {
        query._id = { $gt: cursor };
    }

    const users = await User.find(query)
        .sort({ _id: 1 })
        .limit(limit + 1)  // 多取一个判断有无下一页
        .lean();

    const hasNext = users.length > limit;
    if (hasNext) users.pop();

    res.json({
        data: users,
        pagination: {
            cursor: users.length > 0 ? users[users.length - 1]._id : null,
            hasNext,
            limit,
        },
    });
});

// ========== 方案 3: 基于时间游标 ==========
// 适合按时间排序的场景 (如: 消息列表/Feed)
app.get('/api/posts', async (req, res) => {
    const limit = Math.min(100, parseInt(req.query.limit) || 20);
    const before = req.query.before;  // 时间戳

    const query = {};
    if (before) {
        query.createdAt = { $lt: new Date(before) };
    }

    const posts = await Post.find(query)
        .sort({ createdAt: -1 })
        .limit(limit)
        .lean();

    res.json({
        data: posts,
        pagination: {
            before: posts.length > 0 ? posts[posts.length - 1].createdAt : null,
            hasNext: posts.length === limit,
            limit,
        },
    });
});
```


## 过滤实现


```
// ========== 过滤系统 ==========
// 构建可组合的查询过滤

// ========== 过滤中间件 ==========
// 从 req.query 构建 MongoDB 查询

function buildQuery(filters) {
    const query = {};

    // 精确匹配: ?role=admin&status=active
    if (filters.role) query.role = filters.role;
    if (filters.status) query.status = filters.status;
    if (filters.category) query.category = filters.category;

    // 多值匹配: ?status=active,pending
    if (filters.status?.includes(',')) {
        query.status = { $in: filters.status.split(',') };
    }

    // 范围查询:
    // ?priceMin=100&priceMax=500
    if (filters.priceMin || filters.priceMax) {
        query.price = {};
        if (filters.priceMin) query.price.$gte = Number(filters.priceMin);
        if (filters.priceMax) query.price.$lte = Number(filters.priceMax);
    }

    // 时间范围: ?createdAfter=2024-01-01&createdBefore=2024-12-31
    if (filters.createdAfter || filters.createdBefore) {
        query.createdAt = {};
        if (filters.createdAfter) query.createdAt.$gte = new Date(filters.createdAfter);
        if (filters.createdBefore) query.createdAt.$lte = new Date(filters.createdBefore);
    }

    // 模糊搜索: ?search=keyword
    if (filters.search) {
        query.$or = [
            { name: { $regex: filters.search, $options: 'i' } },
            { email: { $regex: filters.search, $options: 'i' } },
            { description: { $regex: filters.search, $options: 'i' } },
        ];
    }

    // ID 列表: ?ids=id1,id2,id3
    if (filters.ids) {
        query._id = { $in: filters.ids.split(',') };
    }

    // 布尔/存在判断: ?hasAvatar=true
    if (filters.hasAvatar === 'true') query.avatarUrl = { $ne: null };
    if (filters.hasAvatar === 'false') query.avatarUrl = null;

    return query;
}

// ========== 中间件集成 ==========
// 分页/过滤/排序中间件:
function paginate(model, options = {}) {
    return async (req, res, next) => {
        const page = Math.max(1, parseInt(req.query.page) || 1);
        const limit = Math.min(options.maxLimit || 100, parseInt(req.query.limit) || 20);
        const skip = (page - 1) * limit;

        const query = buildQuery(req.query);

        try {
            const [data, total] = await Promise.all([
                model.find(query)
                    .sort(buildSort(req.query.sort, options.sortableFields))
                    .skip(skip)
                    .limit(limit)
                    .lean(),
                model.countDocuments(query),
            ]);

            res.paginated(data, {
                page, limit, total,
                totalPages: Math.ceil(total / limit),
                hasNext: page * limit < total,
                hasPrev: page > 1,
            });
        } catch (err) {
            next(err);
        }
    };
}

// 使用:
app.get('/api/users', paginate(User, {
    maxLimit: 50,
    sortableFields: ['name', 'email', 'createdAt'],
}));
```


## 排序实现


```
// ========== 排序系统 ==========

// ========== 排序构建函数 ==========
function buildSort(sortParam, allowedFields = []) {
    const sortObj = {};

    if (!sortParam) {
        return { createdAt: -1 };  // 默认排序
    }

    // ?sort=name → { name: 1 }
    // ?sort=-name → { name: -1 }
    // ?sort=name,-createdAt → { name: 1, createdAt: -1 }

    const fields = sortParam.split(',');
    for (const field of fields) {
        let order = 1;
        let fieldName = field;

        if (field.startsWith('-')) {
            order = -1;
            fieldName = field.slice(1);
        }

        // 安全检查: 只允许排序白名单字段
        if (allowedFields.length > 0 && !allowedFields.includes(fieldName)) {
            continue;  // 跳过不允许的字段
        }

        // 防止 MongoDB 注入 (只允许字段名)
        if (!/^[a-zA-Z][a-zA-Z0-9_.]*$/.test(fieldName)) {
            continue;
        }

        sortObj[fieldName] = order;
    }

    return sortObj;
}

// ========== 高级排序 ==========
// 支持嵌套字段: ?sort=-user.profile.age
function buildSortNested(sortParam) {
    const sortObj = {};
    if (!sortParam) return { createdAt: -1 };

    sortParam.split(',').forEach(field => {
        let order = 1;
        let fieldName = field;
        if (field.startsWith('-')) {
            order = -1;
            fieldName = field.slice(1);
        }
        sortObj[fieldName] = order;
    });

    return sortObj;
}

// ========== 完整控制器 ==========
app.get('/api/products', asyncHandler(async (req, res) => {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, parseInt(req.query.limit) || 20);

    const filter = buildQuery(req.query);
    const sort = buildSort(req.query.sort, ['name', 'price', 'createdAt', 'sales']);

    // 计算总匹配数 (加缓存优化)
    const cacheKey = `count:${JSON.stringify(filter)}`;
    const total = await cacheGetOrCompute(cacheKey, () =>
        Product.countDocuments(filter), 60);

    const products = await Product.find(filter)
        .sort(sort)
        .skip((page - 1) * limit)
        .limit(limit)
        .lean();

    res.json({
        success: true,
        data: products,
        pagination: {
            page, limit, total,
            totalPages: Math.ceil(total / limit),
            hasNext: page * limit < total,
        },
    });
}));
```


> **Note:** 💡 分页/过滤/排序要点: Offset 分页简单但大偏移慢; 游标分页性能好 (基于 _id 或时间); 过滤用 $regex 做模糊搜索但要防注入; 排序字段白名单防注入; 分页参数有上限 (maxLimit); countDocuments 是慢操作可缓存; 复杂过滤组合用 $and/$or; 前端传参用标准命名 (page/limit/sort/search)。


## 练习


<!-- Converted from: 15_Express 分页过滤排序.html -->
