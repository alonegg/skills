# 问题数据结构与状态机

本文件定义审阅系统中问题（Issue）、修订任务（RevisionTask）和审阅轮次（ReviewRound）的数据结构及生命周期规则。

---

## Issue 核心字段

| 字段 | 说明 | 示例 |
|---|---|---|
| id | 唯一标识符 | ISS-R1-001 |
| position | 位置锚点：章节 + 段序号 | 引言§3 |
| type | 问题类型 | 问题不清、证据不足、口号化建议 |
| subtype | 更细粒度子类型 | 背景过长、缺少过渡、无局限说明 |
| severity | S / A / B / C | A |
| status | 问题当前状态（见状态机） | OPEN |
| description | 问题描述 | 核心研究问题在引言中未明确提出 |
| impact | 对论文的影响 | 后续章节无法围绕稳定问题展开论证 |
| action | 推荐修改动作 | 在引言末尾补写一句明确的问题陈述 |
| target | 修改目标 | 读者读完引言能一句话说出研究什么 |
| source | 来源规则 | intro_rule / D1 |
| round | 首次发现的审阅轮次 | R1 |
| dependency_ids | 依赖先解决的问题 | [ISS-R1-003] |
| action_labels | 动作标签 | [REWRITE, ADD_TRANSITION] |

### 常见问题类型枚举

| 类型标签 | 含义 |
|---|---|
| PROBLEM_UNCLEAR | 研究问题不清 |
| OBJECT_UNFOCUSED | 研究对象失焦 |
| BACKGROUND_FLOODING | 背景淹没问题 |
| REVIEW_LISTING | 综述罗列化 |
| REVIEW_NO_POSITION | 综述无研究位置 |
| METHOD_MISMATCH | 方法与问题不匹配 |
| METHOD_OPAQUE | 方法流程不透明 |
| EVIDENCE_WEAK | 证据不足 |
| ANALYSIS_FLAT | 分析扁平/描述冒充分析 |
| THEORY_DISCONNECTED | 理论与分析脱节 |
| CONCLUSION_OVERREACH | 结论超出证据边界 |
| SUGGESTION_SLOGAN | 建议口号化 |
| MISSING_LIMITATION | 缺少局限说明 |
| FORMAT_INCONSISTENT | 格式不统一 |

### 常见动作标签枚举

| 标签 | 含义 |
|---|---|
| REWRITE | 重写 |
| SPLIT | 拆分 |
| MERGE | 合并 |
| MOVE_FORWARD | 前移 |
| MOVE_BACKWARD | 后移 |
| DELETE | 删除 |
| ADD_EVIDENCE | 补证据 |
| ADD_SOURCE | 补来源 |
| ADD_TRANSITION | 补过渡 |
| ADD_CENTER_SENTENCE | 补中心句 |
| NARROW_CLAIM | 收束结论 |
| COMPRESS | 压缩 |
| ADD_THEORY_LINK | 补理论连接 |

---

## RevisionTask 核心字段

| 字段 | 说明 |
|---|---|
| id | 唯一标识符（如 T1） |
| title | 任务标题（如"重构引言中的研究问题提出"） |
| severity | 继承关联 Issue 的最高级别 |
| issues | 关联的 Issue ID 列表 |
| action_plan | 修改方案（有序步骤列表） |
| target | 完成标准 |
| dependency | 前置依赖任务 |
| status | PENDING / IN_PROGRESS / DONE / BLOCKED |
| completion_criteria | 完成判定标准列表 |

### 任务标题示例

- 重构引言中的研究问题提出
- 重组文献综述以建立研究位置
- 补强分析部分的证据链
- 收束结论边界与建议可操作性
- 统一全文术语与引用格式

---

## ReviewRound 核心字段

| 字段 | 说明 |
|---|---|
| round_id | 轮次编号（R1, R2, ...） |
| mode | 运行模式（FULL_REVIEW / CHAPTER_REVIEW / RE_REVIEW 等） |
| issue_ids | 本轮发现的问题列表 |
| task_ids | 本轮生成的任务列表 |
| summary | 本轮总结 |
| next_round_focus | 下一轮重点 |

---

## Issue 状态机

### 主状态流

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

- **ESCALATED**（升级）：问题严重度在后续轮次中被提高
- **DOWNGRADED**（降级）：根因解决后残留问题降低严重度

### 各状态定义

| 状态 | 含义 |
|---|---|
| OPEN | 刚被识别，还未开始修改 |
| IN_PROGRESS | 作者已开始修改 |
| REVISED_PENDING_REVIEW | 文本已修改，等待复审确认 |
| PARTIALLY_RESOLVED | 有所改善但未完全解决 |
| RESOLVED | 问题已被正面处理 |
| REOPENED | 曾被视为已解决，后续复审发现仍存在 |
| ESCALATED | 严重度在后续判断中被升级 |
| DOWNGRADED | 因主因已解除而降低严重度 |
| CLOSED | 复审确认不再影响系统，可归档 |

---

## 升级规则

| 规则 | 条件 | 结果 |
|---|---|---|
| U1 | B级问题反复出现在核心章节 | → A级 |
| U2 | A级综述问题导致研究问题无法建立 | → S级 |
| U3 | B级方法细节问题影响结果可信度 | → S/A级 |
| U4 | B级结论措辞问题明显超出证据边界 | → A级 |
| U5 | 未解决问题在后续轮次导致衍生问题 | 升一级 |

## 降级规则

| 规则 | 条件 | 结果 |
|---|---|---|
| D1 | 结构性问题解决后残留表达问题 | → B/C级 |
| D2 | 核心综述补齐后个别引文冗余 | → C级 |
| D3 | 方法路径已清楚后少量措辞不准 | → C级 |

---

## 关闭条件

Issue 可关闭当且仅当：
1. 问题本身已被处理
2. 未引入同类新问题
3. 上下文未被破坏
4. 复审确认不再影响后文

---

## 问题聚合规则

### 合并规则

多个问题可合并为一个 RevisionTask，若满足：
- **同根因**：由同一个根本原因导致（如"研究问题失焦"引发综述/方法/分析多个问题）
- **同动作族**：修改方式属于同一类（如多个段落都需要"补证据"）
- **位置相邻或逻辑连接**：在同一章节或逻辑上密切相关

典型例子：
- 多个段落都存在"综述罗列化" → 合并为"重组文献综述"任务
- 多个分析段都缺主张 → 合并为"补强分析段中心判断"任务
- 引言多个段落共同构成"背景淹没问题" → 合并为"压缩引言背景"任务

### 拆分规则

一个问题应拆分为多个任务，若满足：
- **跨多个独立章节**：涉及不同章节的修改，彼此无依赖
- **混合结构与内容**：同时涉及结构调整和内容补充 → 拆为"结构任务"和"内容任务"
- **包含多个独立动作**：修改方式彼此无关

典型例子：
- "引言有问题"同时包含背景过长、对象失焦、问题不清 → 拆为三个 Issue，再聚成一个 Task 并设内部优先级

### 依赖规则

- 结构问题（S/A级）是内容问题的前置依赖
- 研究问题不清 → 综述/方法/分析修改都依赖它先解决
- 理论框架缺失 → 分析部分的理论关联修改依赖它

---

## 复审匹配逻辑

当收到修订稿时，按以下逻辑匹配旧问题：

### Step 1：位置匹配
按章节+段序号定位到对应位置（考虑段落插入/删除导致的偏移）

### Step 2：内容匹配
检查修改是否针对上一轮指出的具体问题

### Step 3：状态判定

| 判定 | 条件 |
|---|---|
| **RESOLVED** | 核心触发信号消失 + 影响减弱或消失 + 未引入同类新问题 |
| **PARTIALLY_RESOLVED** | 部分信号改善 + 仍有残留 + 需继续改 |
| **UNRESOLVED** | 原问题基本原样存在 + 修改未触及核心 |
| **新建 Issue** | 与旧问题相似度低 + 来自新位置或新动作 + 由本轮修改引入 |

### Step 4：升级检查
未解决的问题如果影响扩大 → ESCALATED

---

## 最小实现字段

若先做最小可运行实现，建议保留以下字段：

### Issue 最小字段
id / round / position / type / severity / status / description / impact / action

### Task 最小字段
id / issues / severity / description / action_plan / status

### Round 最小字段
round_id / mode / issue_ids / task_ids / next_round_focus
