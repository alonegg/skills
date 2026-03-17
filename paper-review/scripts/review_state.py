#!/usr/bin/env python3
"""
review_state.py — 论文审阅状态管理工具

管理 review_state.json，支持跨轮次问题追踪。

用法：
    python review_state.py init <output_dir> --title "题目" --author "姓名" --reviewer "审阅人"
    python review_state.py load <state_json>
    python review_state.py add-round <state_json> --issues <issues.json> [--tasks <tasks.json>]
    python review_state.py update-issue <state_json> --id ISS-R1-001 --status RESOLVED --round R2
    python review_state.py summary <state_json>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

VALID_STATUSES = {
    "OPEN", "IN_PROGRESS", "REVISED_PENDING_REVIEW",
    "RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED",
    "REOPENED", "ESCALATED", "DOWNGRADED", "CLOSED"
}

VALID_SEVERITIES = {"S", "A", "B", "C"}

VALID_MODES = {"FULL_REVIEW", "RE_REVIEW", "CHAPTER_REVIEW", "PARAGRAPH_REVIEW"}


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_state(path):
    p = Path(path)
    if not p.exists():
        print(f"错误：状态文件不存在：{p}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path, state):
    state["thesis"]["last_updated"] = now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── init ──────────────────────────────────────────────────────────────

def cmd_init(args):
    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"错误：目录不存在：{out_dir}", file=sys.stderr)
        sys.exit(1)

    state_path = out_dir / "review_state.json"
    if state_path.exists() and not args.force:
        print(f"错误：{state_path} 已存在。使用 --force 覆盖。", file=sys.stderr)
        sys.exit(1)

    state = {
        "thesis": {
            "title": args.title,
            "author": args.author,
            "type": args.type or "",
            "reviewer": args.reviewer,
            "created_at": now_iso(),
            "last_updated": now_iso(),
        },
        "rounds": [],
        "issues": [],
        "tasks": [],
    }

    save_state(state_path, state)
    print(f"已创建：{state_path}")
    print(f"  论文：{args.title}")
    print(f"  作者：{args.author}")
    print(f"  审阅人：{args.reviewer}")


# ── load ──────────────────────────────────────────────────────────────

def cmd_load(args):
    state = load_state(args.state_json)
    t = state["thesis"]
    rounds = state["rounds"]
    issues = state["issues"]

    print(f"论文：{t['title']}")
    print(f"作者：{t['author']}  类型：{t['type']}  审阅人：{t['reviewer']}")
    print(f"创建：{t['created_at']}  更新：{t['last_updated']}")
    print(f"累计轮次：{len(rounds)}")
    print()

    # 按状态统计
    status_counts = {}
    severity_counts = {}
    for iss in issues:
        s = iss["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
        sv = iss["severity"]
        severity_counts[sv] = severity_counts.get(sv, 0) + 1

    if issues:
        print(f"累计问题：{len(issues)} 个")
        print(f"  按级别：{_fmt_counts(severity_counts, ['S', 'A', 'B', 'C'])}")
        print(f"  按状态：{_fmt_counts(status_counts)}")
    else:
        print("暂无问题记录。")

    print()
    if rounds:
        latest = rounds[-1]
        print(f"最近轮次：{latest['round_id']}（{latest.get('mode', '?')}）")
        if latest.get("summary"):
            print(f"  总结：{latest['summary']}")
        if latest.get("next_round_focus"):
            print(f"  下轮重点：{latest['next_round_focus']}")

    # 列出未解决的 S/A 级问题
    open_critical = [i for i in issues if i["status"] not in ("RESOLVED", "CLOSED") and i["severity"] in ("S", "A")]
    if open_critical:
        print(f"\n未解决的 S/A 级问题（{len(open_critical)} 个）：")
        for i in open_critical:
            print(f"  {i['id']} [{i['severity']}] {i['position']} — {i['description'][:50]}... ({i['status']})")


def _fmt_counts(counts, order=None):
    if order:
        parts = [f"{k}:{counts.get(k, 0)}" for k in order if counts.get(k, 0) > 0]
    else:
        parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return " / ".join(parts)


# ── add-round ─────────────────────────────────────────────────────────

def cmd_add_round(args):
    state = load_state(args.state_json)

    # 确定轮次编号
    round_num = len(state["rounds"]) + 1
    round_id = f"R{round_num}"
    mode = args.mode or ("FULL_REVIEW" if round_num == 1 else "RE_REVIEW")

    # 加载问题
    new_issues = []
    if args.issues:
        with open(args.issues, "r", encoding="utf-8") as f:
            raw_issues = json.load(f)

        # 计算本轮问题序号起始
        existing_round_issues = [i for i in state["issues"] if i["id"].startswith(f"ISS-{round_id}-")]
        seq = len(existing_round_issues) + 1

        for iss in raw_issues:
            issue_id = f"ISS-{round_id}-{seq:03d}"
            issue = {
                "id": issue_id,
                "round": round_id,
                "position": iss.get("position", ""),
                "type": iss.get("type", ""),
                "severity": iss.get("severity", "B"),
                "status": "OPEN",
                "description": iss.get("description", ""),
                "impact": iss.get("impact", ""),
                "action": iss.get("action", ""),
                "target": iss.get("target", ""),
                "status_history": [
                    {"status": "OPEN", "round": round_id, "timestamp": now_iso()}
                ],
            }
            new_issues.append(issue)
            state["issues"].append(issue)
            seq += 1

    # 加载任务
    new_tasks = []
    if args.tasks:
        with open(args.tasks, "r", encoding="utf-8") as f:
            new_tasks = json.load(f)
            state["tasks"].extend(new_tasks)

    # 统计
    stats = {"total": len(new_issues), "by_severity": {}}
    for iss in new_issues:
        sv = iss["severity"]
        stats["by_severity"][sv] = stats["by_severity"].get(sv, 0) + 1

    # 添加轮次记录
    round_record = {
        "round_id": round_id,
        "mode": mode,
        "started_at": now_iso(),
        "completed_at": now_iso(),
        "issue_ids": [i["id"] for i in new_issues],
        "task_ids": [t.get("id", "") for t in new_tasks],
        "summary": args.summary or "",
        "next_round_focus": args.next_focus or "",
        "stats": stats,
    }
    state["rounds"].append(round_record)

    save_state(args.state_json, state)

    print(f"已添加轮次 {round_id}（{mode}）")
    print(f"  新增问题：{len(new_issues)} 个")
    if stats["by_severity"]:
        print(f"  级别分布：{_fmt_counts(stats['by_severity'], ['S', 'A', 'B', 'C'])}")
    if new_tasks:
        print(f"  新增任务：{len(new_tasks)} 个")


# ── update-issue ──────────────────────────────────────────────────────

def cmd_update_issue(args):
    state = load_state(args.state_json)

    # 找到问题
    issue = None
    for iss in state["issues"]:
        if iss["id"] == args.id:
            issue = iss
            break

    if not issue:
        print(f"错误：未找到问题 {args.id}", file=sys.stderr)
        sys.exit(1)

    new_status = args.status
    if new_status not in VALID_STATUSES:
        print(f"错误：无效状态 {new_status}。有效值：{VALID_STATUSES}", file=sys.stderr)
        sys.exit(1)

    old_status = issue["status"]
    issue["status"] = new_status
    issue["status_history"].append({
        "status": new_status,
        "round": args.round or f"R{len(state['rounds'])}",
        "timestamp": now_iso(),
        "note": args.note or "",
    })

    # 升降级
    if args.severity:
        if args.severity not in VALID_SEVERITIES:
            print(f"错误：无效级别 {args.severity}", file=sys.stderr)
            sys.exit(1)
        old_sev = issue["severity"]
        issue["severity"] = args.severity
        print(f"  级别变更：{old_sev} → {args.severity}")

    save_state(args.state_json, state)
    print(f"已更新 {args.id}：{old_status} → {new_status}")
    if args.note:
        print(f"  备注：{args.note}")


# ── summary ───────────────────────────────────────────────────────────

def cmd_summary(args):
    state = load_state(args.state_json)
    t = state["thesis"]
    issues = state["issues"]
    rounds = state["rounds"]

    print("=" * 70)
    print(f"问题汇总表 — {t['title']}")
    print(f"审阅人：{t['reviewer']}  累计轮次：{len(rounds)}")
    print("=" * 70)

    # 总体统计
    total = len(issues)
    resolved = sum(1 for i in issues if i["status"] in ("RESOLVED", "CLOSED"))
    partial = sum(1 for i in issues if i["status"] == "PARTIALLY_RESOLVED")
    unresolved = sum(1 for i in issues if i["status"] in ("OPEN", "IN_PROGRESS", "UNRESOLVED", "REOPENED"))

    print(f"\n累计发现问题：{total} 个")
    print(f"已解决：{resolved} / 部分解决：{partial} / 未解决：{unresolved}")

    sev_stats = {}
    sev_resolved = {}
    for i in issues:
        sv = i["severity"]
        sev_stats[sv] = sev_stats.get(sv, 0) + 1
        if i["status"] in ("RESOLVED", "CLOSED"):
            sev_resolved[sv] = sev_resolved.get(sv, 0) + 1

    for sv in ["S", "A", "B", "C"]:
        if sev_stats.get(sv, 0) > 0:
            print(f"  {sv}级：{sev_stats[sv]} 个（已解决 {sev_resolved.get(sv, 0)} 个）")

    if total > 0:
        print(f"解决率：{resolved / total * 100:.0f}%")

    # 问题跟踪表
    print(f"\n{'编号':<16} {'首发':<4} {'级别':<4} {'位置':<12} {'类型':<16} {'描述':<30} {'状态':<12} {'变更历史'}")
    print("-" * 120)
    for i in issues:
        history = "→".join(h["status"][:4] + f"({h['round']})" for h in i["status_history"])
        desc = i["description"][:28] + ("…" if len(i["description"]) > 28 else "")
        print(f"{i['id']:<16} {i['round']:<4} {i['severity']:<4} {i['position']:<12} {i['type']:<16} {desc:<30} {i['status']:<12} {history}")

    # 轮次趋势
    if len(rounds) > 1:
        print(f"\n轮次趋势：")
        print(f"{'轮次':<6} {'新增':<6} {'解决':<6} {'部分':<6} {'未解决':<8} {'累计Open'}")
        print("-" * 45)
        for r in rounds:
            rid = r["round_id"]
            new_count = r["stats"]["total"]
            # 统计本轮状态变更
            r_resolved = 0
            r_partial = 0
            r_unresolved = 0
            for iss in issues:
                for h in iss["status_history"]:
                    if h["round"] == rid:
                        if h["status"] in ("RESOLVED", "CLOSED"):
                            r_resolved += 1
                        elif h["status"] == "PARTIALLY_RESOLVED":
                            r_partial += 1

            # 累计 Open
            cum_open = sum(1 for iss in issues
                          if iss["status"] not in ("RESOLVED", "CLOSED")
                          and any(h["round"] <= rid for h in iss["status_history"]))
            print(f"{rid:<6} {new_count:<6} {r_resolved:<6} {r_partial:<6} {r_unresolved:<8} {cum_open}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="论文审阅状态管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="初始化审阅状态文件")
    p_init.add_argument("output_dir", help="输出目录")
    p_init.add_argument("--title", required=True, help="论文题目")
    p_init.add_argument("--author", required=True, help="学生姓名")
    p_init.add_argument("--type", default="", help="论文类型")
    p_init.add_argument("--reviewer", required=True, help="审阅人")
    p_init.add_argument("--force", action="store_true", help="覆盖已有文件")

    # load
    p_load = sub.add_parser("load", help="查看当前审阅状态")
    p_load.add_argument("state_json", help="review_state.json 路径")

    # add-round
    p_add = sub.add_parser("add-round", help="添加审阅轮次")
    p_add.add_argument("state_json", help="review_state.json 路径")
    p_add.add_argument("--mode", choices=list(VALID_MODES), help="审阅模式")
    p_add.add_argument("--issues", help="问题 JSON 文件路径")
    p_add.add_argument("--tasks", help="任务 JSON 文件路径")
    p_add.add_argument("--summary", help="本轮审阅总结")
    p_add.add_argument("--next-focus", help="下一轮修改重点")

    # update-issue
    p_upd = sub.add_parser("update-issue", help="更新问题状态")
    p_upd.add_argument("state_json", help="review_state.json 路径")
    p_upd.add_argument("--id", required=True, help="问题编号")
    p_upd.add_argument("--status", required=True, help="新状态")
    p_upd.add_argument("--round", help="轮次编号")
    p_upd.add_argument("--severity", help="新级别（升降级时使用）")
    p_upd.add_argument("--note", help="备注")

    # summary
    p_sum = sub.add_parser("summary", help="输出跨轮次问题汇总")
    p_sum.add_argument("state_json", help="review_state.json 路径")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "init": cmd_init,
        "load": cmd_load,
        "add-round": cmd_add_round,
        "update-issue": cmd_update_issue,
        "summary": cmd_summary,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
