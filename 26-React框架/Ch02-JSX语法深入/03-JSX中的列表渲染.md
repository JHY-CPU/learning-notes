# JSX 中的列表渲染

## 基本用法：Array.map()

在 React 中，使用 `Array.map()` 将数组转换为 JSX 元素列表是最常见的模式：

```jsx
function FruitList() {
  const fruits = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']

  return (
    <ul>
      {fruits.map((fruit) => (
        <li key={fruit}>{fruit}</li>
      ))}
    </ul>
  )
}
```

### 渲染对象数组

```jsx
function UserList() {
  const users = [
    { id: 1, name: '张三', age: 25 },
    { id: 2, name: '李四', age: 30 },
    { id: 3, name: '王五', age: 28 },
  ]

  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>
          {user.name} - {user.age} 岁
        </li>
      ))}
    </ul>
  )
}
```

## key 属性

`key` 是 React 用于追踪列表中每个元素的特殊属性，在 reconciliation（调和）过程中起关键作用。

### 为什么需要 key

React 使用 key 来判断列表中哪些元素发生了变化、增加或删除：

```
旧列表: [A, B, C, D]
新列表: [A, C, D, E]

无 key 时：React 逐个对比 → B变成了C，C变成了D，D变成了E，新增E
有 key 时：React 根据 key 对比 → 删除B，新增E，C和D保持不变
```

有正确的 key 时，React 可以：
- 准确识别哪些元素是新增的、删除的、移动的
- 避免不必要的组件重新创建
- 保持组件的内部状态

### key 的使用规则

```jsx
// 正确：使用数据中的唯一标识
function ProductList({ products }) {
  return (
    <ul>
      {products.map((product) => (
        <li key={product.id}>
          <h3>{product.name}</h3>
          <p>¥{product.price}</p>
        </li>
      ))}
    </ul>
  )
}

// 正确：没有 id 时，使用其他唯一值
function TagList({ tags }) {
  return (
    <div>
      {tags.map((tag) => (
        <span key={tag} className="tag">
          {tag}
        </span>
      ))}
    </div>
  )
}
```

### key 的放置位置

`key` 应该放在 `map()` 的直接子元素上，而不是子元素内部：

```jsx
// 错误：key 放在了组件内部
{users.map((user) => (
  <div>
    <UserCard key={user.id} user={user} />
  </div>
))}

// 正确：key 放在 map 的直接子元素上
{users.map((user) => (
  <UserCard key={user.id} user={user} />
))}

// 正确：key 放在最外层元素上
{users.map((user) => (
  <div key={user.id}>
    <UserCard user={user} />
  </div>
))}
```

## 使用 index 作为 key 的问题

在某些情况下可以用 `index` 作为 key，但大多数场景应该避免：

```jsx
// 可以用 index 的场景：列表不会重新排序、不会增删
function StaticList({ items }) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  )
}
```

### 为什么不应该用 index 作为 key

```jsx
function TodoList() {
  const [todos, setTodos] = useState([
    { text: '学习 React', done: false },
    { text: '写代码', done: false },
    { text: '看书', done: false },
  ])

  const removeTodo = (index) => {
    setTodos(todos.filter((_, i) => i !== index))
  }

  return (
    <ul>
      {todos.map((todo, index) => (
        // 使用 index 作为 key 的问题：
        // 删除第一个 todo 后，剩余 todo 的 key 发生变化
        // React 会错误地复用组件，导致输入框等交互组件状态错乱
        <li key={index}>
          <input type="checkbox" checked={todo.done} />
          <span>{todo.text}</span>
          <button onClick={() => removeTodo(index)}>删除</button>
        </li>
      ))}
    </ul>
  )
}
```

### 对比演示

```
删除前（用 index 作为 key）:
  key=0: [复选框A] 学习 React
  key=1: [复选框B] 写代码
  key=2: [复选框C] 看书

删除 "学习 React" 后:
  key=0: [复选框B] 写代码   ← 复选框B被错放到key=0
  key=1: [复选框C] 看书     ← 复选框C被错放到key=1

删除前（用 id 作为 key）:
  key=1: [复选框A] 学习 React
  key=2: [复选框B] 写代码
  key=3: [复选框C] 看书

删除 "学习 React" 后:
  key=2: [复选框B] 写代码   ← 正确，复选框B保持不变
  key=3: [复选框C] 看书     ← 正确，复选框C保持不变
```

## 稳定 ID 策略

确保每个列表项有稳定且唯一的 ID：

```jsx
// 方式一：后端返回的 ID
const users = response.data.map((user) => ({
  ...user,
  key: user.id,
}))

// 方式二：前端生成 UUID
import { v4 as uuidv4 } from 'uuid'

const newTodo = {
  id: uuidv4(),
  text: '新任务',
  done: false,
}

// 方式三：自增 ID（简单场景）
let nextId = 0
const newItem = {
  id: nextId++,
  text: '新任务',
}
```

## 列表过滤

```jsx
function SearchableList({ items }) {
  const [search, setSearch] = useState('')

  const filteredItems = items.filter((item) =>
    item.name.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div>
      <input
        type="text"
        placeholder="搜索..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <ul>
        {filteredItems.length > 0 ? (
          filteredItems.map((item) => (
            <li key={item.id}>{item.name}</li>
          ))
        ) : (
          <li className="empty">未找到匹配项</li>
        )}
      </ul>
    </div>
  )
}
```

## 列表排序

```jsx
function SortableList({ items }) {
  const [sortBy, setSortBy] = useState('name')

  const sortedItems = [...items].sort((a, b) => {
    if (sortBy === 'name') return a.name.localeCompare(b.name)
    if (sortBy === 'price') return a.price - b.price
    if (sortBy === 'date') return new Date(b.createdAt) - new Date(a.createdAt)
    return 0
  })

  return (
    <div>
      <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
        <option value="name">按名称</option>
        <option value="price">按价格</option>
        <option value="date">按日期</option>
      </select>

      <ul>
        {sortedItems.map((item) => (
          <li key={item.id}>
            {item.name} - ¥{item.price}
          </li>
        ))}
      </ul>
    </div>
  )
}
```

## 嵌套列表

```jsx
function CourseList({ courses }) {
  return (
    <div>
      {courses.map((course) => (
        <div key={course.id} className="course">
          <h3>{course.name}</h3>
          <ul>
            {course.chapters.map((chapter) => (
              <li key={chapter.id}>
                <h4>{chapter.title}</h4>
                <ul>
                  {chapter.lessons.map((lesson) => (
                    <li key={lesson.id}>{lesson.title}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
```

## 提取列表项组件

当列表项结构复杂时，提取为独立组件：

```jsx
function TodoItem({ todo, onToggle, onDelete }) {
  return (
    <li className={todo.done ? 'done' : ''}>
      <input
        type="checkbox"
        checked={todo.done}
        onChange={() => onToggle(todo.id)}
      />
      <span>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)}>删除</button>
    </li>
  )
}

function TodoList({ todos, onToggle, onDelete }) {
  return (
    <ul>
      {todos.map((todo) => (
        <TodoItem
          key={todo.id}
          todo={todo}
          onToggle={onToggle}
          onDelete={onDelete}
        />
      ))}
    </ul>
  )
}
```

## 性能考虑：虚拟列表

当列表数据量很大（数百上千条）时，渲染所有 DOM 节点会影响性能。虚拟列表只渲染可视区域内的元素：

```jsx
// 概念：只渲染可见区域的元素
// 推荐库：react-window、react-virtuoso

import { FixedSizeList } from 'react-window'

function VirtualList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      {items[index].name}
    </div>
  )

  return (
    <FixedSizeList
      height={400}        // 容器高度
      itemCount={items.length}
      itemSize={35}       // 每行高度
      width={300}
    >
      {Row}
    </FixedSizeList>
  )
}
```

## Fragment 在列表中的应用

当列表项需要返回多个元素时，使用 Fragment：

```jsx
function DefinitionList({ items }) {
  return (
    <dl>
      {items.map((item) => (
        <React.Fragment key={item.id}>
          <dt>{item.term}</dt>
          <dd>{item.definition}</dd>
        </React.Fragment>
      ))}
    </dl>
  )
}
```

## 小结

- 使用 `Array.map()` 渲染列表，每个元素必须有唯一的 `key`
- `key` 帮助 React 高效追踪元素变化，避免不必要的重新渲染
- 优先使用数据中的稳定唯一标识作为 key，避免使用 `index`
- 复杂列表项应提取为独立组件
- 大数据量列表考虑使用虚拟列表优化性能
- 列表过滤、排序等操作应创建新数组，而非修改原数组
