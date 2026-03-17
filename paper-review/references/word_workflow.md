# Word 文档审阅工作流

本文件描述如何将论文审阅结果写入 Word 文档。内容审阅由 paper-review 技能完成，文档操作由 **docx 技能**完成，两者协作实现"Word 文档进 → 带批注的 Word 文档 + 审核意见文档出"。

---

## 完整工作流

### Step 1：提取内容

使用 docx 技能解压文档，然后提取文本：

```bash
# 解压 docx 为 XML（用于后续批注插入）
python skills/docx/scripts/office/unpack.py 论文.docx /tmp/review_unpacked/

# 提取文本内容（用于审阅）
pandoc 论文.docx -o /tmp/review_content.md

# 查看段落索引（建立段落编号映射）
python skills/paper-review/scripts/annotate_review.py index /tmp/review_unpacked/
```

段落索引输出示例：
```
共 87 个段落：

  §  0 [Title]: 基于XX理论的XX问题研究
  §  1 [Heading1]: 摘要
  §  2: 本文以XX为研究对象，聚焦于XX问题...
  §  5 [Heading1]: 第一章 绪论
  §  6 [Heading2]: 1.1 研究背景
  §  7: 近年来，随着经济的快速发展...
```

### Step 2：执行审阅

按照 paper-review 技能的五阶段流程对提取的文本进行审阅。

**在审阅过程中，同步记录批注数据**：对每个需要在 Word 中标注的问题，记录一条批注：

```json
{
  "paragraph_index": 5,
  "search_text": "近年来，随着经济的快速发展",
  "comment_text": "[A级] 背景铺陈过长，与核心问题连接弱。建议压缩宏观叙事，增加背景到问题的桥梁句。",
  "severity": "A"
}
```

### Step 3：写入批注

将审阅过程中收集的批注数据保存为 JSON 文件，然后批量写入：

```bash
# 将批注数据写入 JSON 文件（实际内容在审阅过程中生成）
# 然后批量写入批注到 document.xml
python skills/paper-review/scripts/annotate_review.py annotate \
  /tmp/review_unpacked/ /tmp/review_annotations.json \
  --author "审阅教授"
```

### Step 4：打包输出

使用 docx 技能将修改后的 XML 重新打包为 docx：

```bash
python skills/docx/scripts/office/pack.py /tmp/review_unpacked/ 论文_已审阅.docx --original 论文.docx
```

### Step 5：生成审核意见文档

使用 docx 技能生成独立的审核意见报告 Word 文档（参见 `output_templates.md` 中的 Template 7），包含：

1. 论文画像
2. 总体评价
3. 全局关键问题清单（含级别/位置/描述/建议）
4. 修订任务单
5. 复审建议

生成方式参照 docx 技能的 "Creating New Documents" 流程。

---

## 批注记录规则

### 字段说明

| 字段 | 说明 |
|---|---|
| `paragraph_index` | 段落在 document.xml 中的 0-based 索引，从 Step 1 的段落索引中确定 |
| `search_text` | 要标注的**原文文本片段**，5-30字为宜，必须在目标段落中唯一 |
| `comment_text` | 格式：`[级别] 问题描述。建议具体修改动作。` |
| `severity` | S / A / B（C 级不写入批注） |

### 批注内容格式

```
[S级] 问题描述。建议具体修改动作。
[A级] 问题描述。建议具体修改动作。
[B级] 问题描述。建议具体修改动作。
```

### 精确定位规则

1. **先建立段落索引**：用 `annotate_review.py index` 查看段落编号与文本映射
2. **选择独特文本**：`search_text` 应在目标段落中唯一，避免过短（< 5 字）或过长（> 50 字）
3. **整段问题**：如果问题涉及整段，选取段首 10-20 字作为 `search_text`
4. **无需标注每段**：只在有 S/A/B 级问题的段落插入批注
5. **回退策略**：脚本会自动在相邻段落搜索并回退到整段标注

### 级别筛选

- **S/A/B 级**：写入 Word 批注（直观、醒目）
- **C 级**：仅出现在审核意见文档中（避免批注过多干扰阅读）
