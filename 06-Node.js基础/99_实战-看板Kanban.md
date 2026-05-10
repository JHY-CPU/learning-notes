# 实战 - 看板 Kanban (Task Board)

## 项目需求与功能分析

看板（Kanban）是项目管理中广泛使用的任务可视化工具。本项目实现一个支持拖拽、多列管理的交互式看板。

### 核心功能

- 多列看板（待办、进行中、已完成）
- 卡片拖拽在列之间移动
- 添加 / 删除卡片
- 卡片编辑
- 拖拽排序
- localStorage 持久化
- 列标题可编辑

## 完整代码实现

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>看板 Kanban</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; padding: 30px 20px; }
  h1 { text-align: center; color: #fff; margin-bottom: 25px; font-size: 24px; }
  .board { display: flex; gap: 20px; overflow-x: auto; padding-bottom: 20px; min-height: calc(100vh - 130px); }
  .column { background: #ebecf0; border-radius: 12px; min-width: 300px; max-width: 300px; display: flex; flex-direction: column; max-height: calc(100vh - 130px); }
  .column-header { display: flex; justify-content: space-between; align-items: center; padding: 14px 16px; font-weight: 600; font-size: 15px; color: #333; }
  .column-header .count { background: #ddd; border-radius: 10px; padding: 2px 8px; font-size: 12px; color: #666; }
  .column-header .title { cursor: pointer; padding: 2px 4px; border-radius: 4px; }
  .column-header .title:hover { background: rgba(0,0,0,0.05); }
  .column-header .title:focus { background: #fff; outline: 2px solid #667eea; }
  .card-list { flex: 1; overflow-y: auto; padding: 0 10px 10px; min-height: 50px; }
  .card { background: #fff; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); cursor: grab; transition: all 0.15s; position: relative; }
  .card:hover { box-shadow: 0 3px 8px rgba(0,0,0,0.12); }
  .card.dragging { opacity: 0.4; transform: rotate(3deg); }
  .card.drag-over { border-top: 3px solid #667eea; }
  .card-text { font-size: 14px; color: #333; line-height: 1.4; word-break: break-word; }
  .card-text.editing { border: 1px solid #667eea; padding: 4px 8px; border-radius: 4px; outline: none; }
  .card-actions { display: none; position: absolute; top: 8px; right: 8px; gap: 4px; }
  .card:hover .card-actions { display: flex; }
  .card-actions button { width: 24px; height: 24px; border: none; background: #f0f0f0; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center; }
  .card-actions button:hover { background: #ddd; }
  .card-actions .delete-btn:hover { background: #ff6b6b; color: #fff; }
  .card-tag { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-top: 8px; }
  .tag-red { background: #ffe0e0; color: #e74c3c; }
  .tag-blue { background: #e0e8ff; color: #3498db; }
  .tag-green { background: #e0ffe0; color: #27ae60; }
  .tag-yellow { background: #fff8e0; color: #f39c12; }
  .add-card { padding: 10px; }
  .add-card-btn { width: 100%; padding: 10px; background: none; border: 2px dashed #ccc; border-radius: 8px; color: #999; cursor: pointer; font-size: 14px; transition: all 0.2s; }
  .add-card-btn:hover { border-color: #667eea; color: #667eea; background: rgba(102,126,234,0.05); }
  .add-card-form { display: none; }
  .add-card-form.show { display: block; }
  .add-card-form textarea { width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 14px; resize: none; outline: none; font-family: inherit; }
  .add-card-form .form-actions { display: flex; gap: 8px; margin-top: 8px; }
  .add-card-form .form-actions button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .add-card-form .btn-add { background: #667eea; color: #fff; }
  .add-card-form .btn-cancel { background: #eee; color: #666; }
  .drop-indicator { height: 4px; background: #667eea; border-radius: 2px; margin: 4px 0; }
</style>
</head>
<body>

<h1>看板 Kanban</h1>
<div class="board" id="board"></div>

<script>
const DEFAULT_DATA = {
  columns: [
    { id: 'todo', title: '待办', cards: [
      { id: 'c1', text: '设计数据库表结构', tag: 'red' },
      { id: 'c2', text: '编写 API 接口文档', tag: 'blue' },
      { id: 'c3', text: '搭建项目基础框架', tag: 'green' },
    ]},
    { id: 'doing', title: '进行中', cards: [
      { id: 'c4', text: '实现用户认证模块', tag: 'blue' },
      { id: 'c5', text: '前端页面开发', tag: 'yellow' },
    ]},
    { id: 'done', title: '已完成', cards: [
      { id: 'c6', text: '项目需求分析', tag: 'green' },
      { id: 'c7', text: '技术选型评审', tag: 'green' },
    ]},
  ],
};

class KanbanBoard {
  constructor(container) {
    this.container = container;
    this.data = JSON.parse(localStorage.getItem('kanban')) || JSON.parse(JSON.stringify(DEFAULT_DATA));
    this.draggedCard = null;
    this.draggedFromColumn = null;
    this.render();
  }

  save() {
    localStorage.setItem('kanban', JSON.stringify(this.data));
  }

  render() {
    this.container.innerHTML = '';
    this.data.columns.forEach(col => {
      const colEl = document.createElement('div');
      colEl.className = 'column';
      colEl.dataset.colId = col.id;
      colEl.innerHTML = `
        <div class="column-header">
          <span class="title" contenteditable="false">${col.title}</span>
          <span class="count">${col.cards.length}</span>
        </div>
        <div class="card-list" data-col-id="${col.id}">
          ${col.cards.map(card => this.renderCard(card)).join('')}
        </div>
        <div class="add-card">
          <button class="add-card-btn" data-col="${col.id}">+ 添加卡片</button>
          <div class="add-card-form" data-col="${col.id}">
            <textarea rows="3" placeholder="输入卡片内容..." data-col="${col.id}"></textarea>
            <div class="form-actions">
              <button class="btn-add" data-col="${col.id}">添加</button>
              <button class="btn-cancel" data-col="${col.id}">取消</button>
            </div>
          </div>
        </div>`;
      this.container.appendChild(colEl);

      // 列标题编辑
      const titleEl = colEl.querySelector('.title');
      titleEl.addEventListener('dblclick', () => {
        titleEl.contentEditable = true;
        titleEl.focus();
      });
      titleEl.addEventListener('blur', () => {
        titleEl.contentEditable = false;
        col.title = titleEl.textContent.trim() || col.title;
        this.save();
      });
      titleEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); titleEl.blur(); }
      });
    });

    this.bindCardEvents();
    this.bindAddEvents();
  }

  renderCard(card) {
    const tags = { red: '紧急', blue: '开发', green: '设计', yellow: '测试' };
    return `
      <div class="card" draggable="true" data-card-id="${card.id}">
        <div class="card-actions">
          <button class="edit-btn" title="编辑">✏</button>
          <button class="delete-btn" title="删除">✕</button>
        </div>
        <div class="card-text">${this.escapeHtml(card.text)}</div>
        ${card.tag ? `<span class="card-tag tag-${card.tag}">${tags[card.tag]||card.tag}</span>` : ''}
      </div>`;
  }

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  bindCardEvents() {
    this.container.querySelectorAll('.card').forEach(card => {
      card.addEventListener('dragstart', e => {
        this.draggedCard = card;
        this.draggedFromColumn = card.closest('.card-list').dataset.colId;
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
      });
      card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        document.querySelectorAll('.card').forEach(c => c.classList.remove('drag-over'));
      });

      // 删除
      card.querySelector('.delete-btn').addEventListener('click', () => {
        const colId = card.closest('.card-list').dataset.colId;
        const cardId = card.dataset.cardId;
        const col = this.data.columns.find(c => c.id === colId);
        col.cards = col.cards.filter(c => c.id !== cardId);
        this.save();
        this.render();
      });

      // 编辑
      card.querySelector('.edit-btn').addEventListener('click', () => {
        const textEl = card.querySelector('.card-text');
        textEl.contentEditable = true;
        textEl.classList.add('editing');
        textEl.focus();
        const finish = () => {
          textEl.contentEditable = false;
          textEl.classList.remove('editing');
          const colId = card.closest('.card-list').dataset.colId;
          const cardId = card.dataset.cardId;
          const col = this.data.columns.find(c => c.id === colId);
          const c = col.cards.find(cc => cc.id === cardId);
          c.text = textEl.textContent.trim() || c.text;
          this.save();
        };
        textEl.addEventListener('blur', finish, { once: true });
        textEl.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); textEl.blur(); } });
      });
    });

    // 拖拽目标
    this.container.querySelectorAll('.card-list').forEach(list => {
      list.addEventListener('dragover', e => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const afterEl = this.getDragAfterElement(list, e.clientY);
        const dragging = document.querySelector('.dragging');
        if (afterEl == null) list.appendChild(dragging);
        else list.insertBefore(dragging, afterEl);
      });
      list.addEventListener('drop', e => {
        e.preventDefault();
        const toColId = list.dataset.colId;
        const cardId = this.draggedCard.dataset.cardId;
        // 从源列移除
        const fromCol = this.data.columns.find(c => c.id === this.draggedFromColumn);
        const cardIndex = fromCol.cards.findIndex(c => c.id === cardId);
        const [card] = fromCol.cards.splice(cardIndex, 1);
        // 添加到目标列
        const toCol = this.data.columns.find(c => c.id === toColId);
        const children = [...list.children];
        const newIndex = children.indexOf(this.draggedCard);
        toCol.cards.splice(newIndex, 0, card);
        this.save();
        this.render();
      });
    });
  }

  getDragAfterElement(list, y) {
    const elements = [...list.querySelectorAll('.card:not(.dragging)')];
    return elements.reduce((closest, el) => {
      const box = el.getBoundingClientRect();
      const offset = y - box.top - box.height / 2;
      if (offset < 0 && offset > closest.offset) return { offset, element: el };
      return closest;
    }, { offset: Number.NEGATIVE_INFINITY }).element;
  }

  bindAddEvents() {
    this.container.querySelectorAll('.add-card-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const col = btn.dataset.col;
        btn.style.display = 'none';
        const form = this.container.querySelector(`.add-card-form[data-col="${col}"]`);
        form.classList.add('show');
        form.querySelector('textarea').focus();
      });
    });

    this.container.querySelectorAll('.btn-cancel').forEach(btn => {
      btn.addEventListener('click', () => {
        const col = btn.dataset.col;
        const form = this.container.querySelector(`.add-card-form[data-col="${col}"]`);
        form.classList.remove('show');
        form.querySelector('textarea').value = '';
        this.container.querySelector(`.add-card-btn[data-col="${col}"]`).style.display = '';
      });
    });

    this.container.querySelectorAll('.btn-add').forEach(btn => {
      btn.addEventListener('click', () => {
        const col = btn.dataset.col;
        const form = this.container.querySelector(`.add-card-form[data-col="${col}"]`);
        const text = form.querySelector('textarea').value.trim();
        if (!text) return;
        const column = this.data.columns.find(c => c.id === col);
        column.cards.push({ id: 'c' + Date.now(), text, tag: '' });
        this.save();
        this.render();
      });
    });
  }
}

new KanbanBoard(document.getElementById('board'));
</script>
</body>
</html>
```

## 核心技术详解

### 拖拽排序

使用 HTML5 Drag API 的 `dragover` 事件计算卡片应该插入的位置，`getDragAfterElement` 方法通过比较鼠标 Y 坐标和卡片位置来确定插入点。

### 数据驱动渲染

所有操作先修改 `data` 对象，然后调用 `save()` 持久化，再调用 `render()` 重新渲染。

### 列标题编辑

使用 `contenteditable` 实现双击编辑列标题，失焦时保存。

## 测试用例

```javascript
describe('Kanban Logic', () => {
  test('添加卡片', () => {
    const col = { id: 'todo', title: '待办', cards: [] };
    col.cards.push({ id: 'c1', text: '新任务', tag: '' });
    expect(col.cards.length).toBe(1);
  });

  test('移动卡片', () => {
    const from = { cards: [{ id: 'c1', text: '任务', tag: '' }] };
    const to = { cards: [] };
    const [card] = from.cards.splice(0, 1);
    to.cards.push(card);
    expect(from.cards.length).toBe(0);
    expect(to.cards.length).toBe(1);
  });

  test('删除卡片', () => {
    const col = { cards: [{ id: 'c1', text: '删我', tag: '' }] };
    col.cards = col.cards.filter(c => c.id !== 'c1');
    expect(col.cards.length).toBe(0);
  });
});
```

## 扩展方向

1. **添加列**：支持动态添加新的列
2. **卡片详情**：点击卡片打开详情面板（描述、截止日期、标签）
3. **过滤和搜索**：按标签或关键词过滤卡片
4. **多人协作**：WebSocket 实时同步多人操作
5. **导出功能**：导出看板为 JSON / CSV
6. **泳道**：按负责人或优先级分组显示
7. **WIP 限制**：限制每列最大卡片数
