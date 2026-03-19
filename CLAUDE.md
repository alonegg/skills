# CLAUDE.md

## 仓库信息

本仓库是 Claude Code 的 skills 集合，远程地址：`https://github.com/alonegg/skills`

## 三处同步规则

本仓库的 skills 文件存在于 3 个位置，修改任何一处后**必须同步到其余两处**：

| 编号 | 位置 | 用途 |
|---|---|---|
| 1 | `/Users/alone/Desktop/openai/review/review_rule/skills/` | 本地 git 仓库（编辑源） |
| 2 | `/Users/alone/.claude/skills/` | Claude Code 运行时加载的 skills |
| 3 | `https://github.com/alonegg/skills` (remote: origin) | 远程仓库 |

### 同步流程

每次修改 skills 文件后，按以下顺序执行：

1. **在本地 git 仓库中编辑**（位置 1）
2. **复制到 Claude Code skills 目录**（位置 2）：
   ```bash
   cp -r /Users/alone/Desktop/openai/review/review_rule/skills/<skill-name>/ /Users/alone/.claude/skills/<skill-name>/
   ```
3. **提交并推送到远程**（位置 3）：
   ```bash
   cd /Users/alone/Desktop/openai/review/review_rule/skills
   git add <changed-files>
   git commit -m "描述修改内容"
   git push origin main
   ```

### 注意事项

- 位置 1 是编辑源，所有修改都从这里开始
- 不要直接在位置 2 编辑，否则下次同步会被覆盖
- 每次同步完成后向用户确认 3 处均已更新
