# 问题数据结构与状态机

本文件定义审阅系统中问题（Issue）和修订任务（RevisionTask）的数据结构及生命周期规则。

---

## Issue 核心字段

| 字段 | 说明 |
|---|---|
| id | 唯一标识符（如 ISS-001） |
| position | 位置锚点：章节 + 段序号 |
| type | 问题类型（如：问题不清、证据不足、口号化建议） |
| severity | S / A / B / C |
| status | OPEN / IN_PROGRESS / REVISED_PENDING / RESOLVED / CLOSED / ESCALATED / DOWNGRADED |
| description | 问题描述 |
| impact | 对论文的影响 |
| action | 推荐修改动作 |
| target | 修改目标（改到什么程度） |
| source | 来源规则（intro / literature / method 等） |
| round | 首次发现的审阅轮次 |

---

## RevisionTask 核心字段

| 字段 | 说明 |
|---|---|
| id | 唯一标识符（如 T1） |
| title | 任务标题 |
| severity | 继承关联 Issue 的最高级别 |
| issues | 关联的 Issue ID 列表 |
| action_plan | 修改方案 |
| target | 完成标准 |
| dependency | 前置依赖任务 |
| status | PENDING / IN_PROGRESS / DONE / BLOCKED |

---

## Issue 状态机

```
OPEN ─────────── → IN_PROGRESS ─── → REVISED_PENDING_REVIEW
                                          │
                                   ┌──────┴──────────────┐
                                   ↓                      ↓
                              RESOLVED              PARTIALLY_RESOLVED
                                   │                      │
                                   ↓                      ↓
                                CLOSED               REOPENED → OPEN
```

### 特殊状态转换

- **ESCALATED**（升级）：B级问题反复出现在核心章节 → A级；A级导致研究问题无法建立 → S级
- **DOWNGRADED**（降级）：结构性问题解决后残留表达问题 → B/C级

---

## 关闭条件

Issue 可关闭当且仅当：
1. 问题本身已被处理
2. 未引入同类新问题
3. 上下文未被破坏
4. 复审确认不再影响后文

---

## 聚合规则

### 合并规则
- 同一章节中的同类问题（如多个"证据不足"）→ 合并为一个 RevisionTask
- 同一根因导致的连锁问题 → 合并为一个 RevisionTask 并标注根因

### 拆分规则
- 一个问题涉及多个不同章节的修改 → 拆分为多个 RevisionTask
- 一个问题同时涉及结构调整和内容补充 → 拆分为"结构任务"和"内容任务"

### 依赖规则
- 结构问题（S/A级）是内容问题的前置依赖
- 研究问题不清 → 综述/方法/分析修改都依赖它先解决

---

## 复审匹配逻辑

当收到修订稿时，按以下逻辑匹配：

1. **位置匹配**：按章节+段序号定位到对应位置
2. **内容匹配**：检查修改是否针对上一轮指出的具体问题
3. **状态判定**：
   - 问题消失且修改合理 → RESOLVED
   - 问题部分改善 → PARTIALLY_RESOLVED
   - 问题未变化 → UNRESOLVED（保持 OPEN）
   - 出现新问题 → 新建 Issue
4. **升级检查**：未解决的问题如果影响扩大 → ESCALATED
