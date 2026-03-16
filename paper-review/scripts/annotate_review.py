"""Annotate a DOCX document with review comments at precise text locations.

Usage:
    python annotate_review.py <unpacked_dir> <annotations_json> [--author "审阅教授"]

The annotations JSON file should contain an array of annotation objects:
[
  {
    "paragraph_index": 5,
    "search_text": "近年来，随着经济的快速发展",
    "comment_text": "[A级] 背景铺陈过长，与核心问题连接弱。建议压缩宏观叙事，增加背景到问题的桥梁句。",
    "severity": "A"
  },
  ...
]

The script will:
1. Parse document.xml and build a paragraph index
2. For each annotation, locate the target text in the specified paragraph
3. Register comments via comment.py
4. Insert commentRangeStart/End markers at precise positions in document.xml
"""

import argparse
import json
import sys
from pathlib import Path

import defusedxml.minidom

# Import comment.py from docx skill
DOCX_SCRIPTS = Path(__file__).parent.parent.parent / "docx" / "scripts"
sys.path.insert(0, str(DOCX_SCRIPTS))
from comment import add_comment  # noqa: E402

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _get_paragraph_text(p_elem):
    """Extract full text from a <w:p> element by concatenating all <w:t> content."""
    texts = []
    for r in p_elem.getElementsByTagName("w:r"):
        # Skip runs that are inside w:del (deleted text)
        parent = r.parentNode
        if parent and parent.tagName == "w:del":
            continue
        for t in r.getElementsByTagName("w:t"):
            if t.firstChild and t.firstChild.nodeValue:
                texts.append(t.firstChild.nodeValue)
    return "".join(texts)


def _get_paragraph_style(p_elem):
    """Extract paragraph style (e.g., Heading1) from <w:pPr><w:pStyle>."""
    for ppr in p_elem.getElementsByTagName("w:pPr"):
        for ps in ppr.getElementsByTagName("w:pStyle"):
            return ps.getAttribute("w:val")
    return None


def build_paragraph_index(doc_xml_path):
    """Build an index of all paragraphs in document.xml.

    Returns list of dicts: [{"index": 0, "text": "...", "style": "...", "element": <w:p>}, ...]
    """
    dom = defusedxml.minidom.parseString(
        Path(doc_xml_path).read_text(encoding="utf-8")
    )
    body = dom.getElementsByTagName("w:body")[0]
    paragraphs = []

    for i, p in enumerate(body.getElementsByTagName("w:p")):
        # Only direct children of body or table cells
        text = _get_paragraph_text(p)
        style = _get_paragraph_style(p)
        paragraphs.append({
            "index": i,
            "text": text,
            "style": style,
            "element": p,
        })

    return paragraphs, dom


def _find_run_range_for_text(p_elem, search_text):
    """Find the <w:r> elements that contain the search_text.

    Returns (start_run, end_run, found) where start_run and end_run are
    the <w:r> elements that contain the start and end of search_text.
    """
    runs = []
    for child in p_elem.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            if child.tagName == "w:r":
                # Check it's not inside a w:del
                runs.append(child)
            elif child.tagName == "w:ins":
                for r in child.getElementsByTagName("w:r"):
                    runs.append(r)

    # Build character-to-run mapping
    char_positions = []  # [(run_element, char_in_run, global_pos)]
    full_text = ""
    for run in runs:
        for t in run.getElementsByTagName("w:t"):
            if t.firstChild and t.firstChild.nodeValue:
                text = t.firstChild.nodeValue
                for j, ch in enumerate(text):
                    char_positions.append((run, j, len(full_text)))
                    full_text += ch

    if not full_text:
        return None, None, False

    # Find search_text in full_text
    pos = full_text.find(search_text)
    if pos == -1:
        # Try fuzzy: strip whitespace differences
        normalized = full_text.replace(" ", "").replace("\u3000", "")
        normalized_search = search_text.replace(" ", "").replace("\u3000", "")
        norm_pos = normalized.find(normalized_search)
        if norm_pos == -1:
            return None, None, False
        # Map back: find original positions
        norm_idx = 0
        original_start = None
        original_end = None
        for idx, ch in enumerate(full_text):
            if ch in (" ", "\u3000"):
                continue
            if norm_idx == norm_pos and original_start is None:
                original_start = idx
            if norm_idx == norm_pos + len(normalized_search) - 1:
                original_end = idx
                break
            norm_idx += 1
        if original_start is not None and original_end is not None:
            pos = original_start
            end_pos = original_end
        else:
            return None, None, False
    else:
        end_pos = pos + len(search_text) - 1

    if pos >= len(char_positions) or end_pos >= len(char_positions):
        return None, None, False

    start_run = char_positions[pos][0]
    end_run = char_positions[end_pos][0]

    return start_run, end_run, True


def insert_comment_markers(dom, p_elem, start_run, end_run, comment_id):
    """Insert commentRangeStart/End markers around the target runs.

    Markers are direct children of <w:p>, never inside <w:r>.
    """
    # Create marker elements
    range_start = dom.createElementNS(NS_W, "w:commentRangeStart")
    range_start.setAttribute("w:id", str(comment_id))

    range_end = dom.createElementNS(NS_W, "w:commentRangeEnd")
    range_end.setAttribute("w:id", str(comment_id))

    # Create comment reference run
    ref_run = dom.createElementNS(NS_W, "w:r")
    ref_rpr = dom.createElementNS(NS_W, "w:rPr")
    ref_style = dom.createElementNS(NS_W, "w:rStyle")
    ref_style.setAttribute("w:val", "CommentReference")
    ref_rpr.appendChild(ref_style)
    ref_run.appendChild(ref_rpr)
    ref_ref = dom.createElementNS(NS_W, "w:commentReference")
    ref_ref.setAttribute("w:id", str(comment_id))
    ref_run.appendChild(ref_ref)

    # Insert commentRangeStart before start_run
    # Navigate up if start_run is inside w:ins
    start_insert_point = start_run
    if start_run.parentNode and start_run.parentNode.tagName != "w:p":
        start_insert_point = start_run.parentNode

    end_insert_point = end_run
    if end_run.parentNode and end_run.parentNode.tagName != "w:p":
        end_insert_point = end_run.parentNode

    p_elem.insertBefore(range_start, start_insert_point)

    # Insert commentRangeEnd + reference after end_run
    next_sibling = end_insert_point.nextSibling
    if next_sibling:
        p_elem.insertBefore(range_end, next_sibling)
        p_elem.insertBefore(ref_run, next_sibling)
    else:
        p_elem.appendChild(range_end)
        p_elem.appendChild(ref_run)


def annotate_document(unpacked_dir, annotations, author="审阅教授"):
    """Apply all annotations to the unpacked document.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        annotations: List of annotation dicts with keys:
            paragraph_index, search_text, comment_text, severity
        author: Author name for comments

    Returns:
        (success_count, fail_count, messages)
    """
    doc_xml_path = Path(unpacked_dir) / "word" / "document.xml"
    if not doc_xml_path.exists():
        return 0, 0, ["Error: document.xml not found"]

    paragraphs, dom = build_paragraph_index(str(doc_xml_path))
    messages = []
    success_count = 0
    fail_count = 0

    for i, ann in enumerate(annotations):
        p_idx = ann["paragraph_index"]
        search_text = ann["search_text"]
        comment_text = ann["comment_text"]

        if p_idx < 0 or p_idx >= len(paragraphs):
            messages.append(
                f"  ✗ 批注 {i}: 段落索引 {p_idx} 超出范围 (共 {len(paragraphs)} 段)"
            )
            fail_count += 1
            continue

        p_elem = paragraphs[p_idx]["element"]
        p_text = paragraphs[p_idx]["text"]

        # Find target runs
        start_run, end_run, found = _find_run_range_for_text(p_elem, search_text)

        if not found:
            # Fallback: try to find in nearby paragraphs (±2)
            fallback_found = False
            for offset in [-1, 1, -2, 2]:
                alt_idx = p_idx + offset
                if 0 <= alt_idx < len(paragraphs):
                    alt_p = paragraphs[alt_idx]["element"]
                    s, e, f = _find_run_range_for_text(alt_p, search_text)
                    if f:
                        start_run, end_run = s, e
                        p_elem = alt_p
                        p_idx = alt_idx
                        fallback_found = True
                        messages.append(
                            f"  ⚠ 批注 {i}: 在相邻段落 {alt_idx} 中找到目标文本"
                        )
                        break

            if not fallback_found:
                # Last resort: mark the entire paragraph
                runs_in_para = []
                for child in p_elem.childNodes:
                    if child.nodeType == child.ELEMENT_NODE and child.tagName == "w:r":
                        runs_in_para.append(child)
                if runs_in_para:
                    start_run = runs_in_para[0]
                    end_run = runs_in_para[-1]
                    messages.append(
                        f"  ⚠ 批注 {i}: 未精确匹配文本，已标注整段 (段落 {p_idx})"
                    )
                else:
                    messages.append(
                        f"  ✗ 批注 {i}: 段落 {p_idx} 无可标注内容"
                    )
                    fail_count += 1
                    continue

        # Register comment via comment.py
        _, msg = add_comment(
            str(unpacked_dir),
            i,  # comment_id
            comment_text,
            author=author,
            initials=author[0] if author else "R",
        )

        if "Error" in msg:
            messages.append(f"  ✗ 批注 {i}: 注册失败 - {msg}")
            fail_count += 1
            continue

        # Insert XML markers
        insert_comment_markers(dom, p_elem, start_run, end_run, i)
        severity = ann.get("severity", "B")
        messages.append(
            f"  ✓ 批注 {i} [{severity}级]: 段落 {p_idx} - 已标注"
        )
        success_count += 1

    # Write back document.xml
    # Encode smart quotes
    SMART_QUOTES = {
        "\u201c": "&#x201C;", "\u201d": "&#x201D;",
        "\u2018": "&#x2018;", "\u2019": "&#x2019;",
    }
    output = dom.toxml(encoding="UTF-8").decode("utf-8")
    for char, entity in SMART_QUOTES.items():
        output = output.replace(char, entity)
    doc_xml_path.write_text(output, encoding="utf-8")

    return success_count, fail_count, messages


def print_paragraph_index(unpacked_dir, max_chars=60):
    """Print paragraph index for reference during review."""
    doc_xml_path = Path(unpacked_dir) / "word" / "document.xml"
    paragraphs, _ = build_paragraph_index(str(doc_xml_path))

    print(f"共 {len(paragraphs)} 个段落：\n")
    for p in paragraphs:
        text = p["text"][:max_chars]
        if len(p["text"]) > max_chars:
            text += "..."
        style = f' [{p["style"]}]' if p["style"] else ""
        if text.strip():
            print(f"  §{p['index']:3d}{style}: {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate DOCX with review comments at precise locations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command: show paragraph index
    idx_parser = subparsers.add_parser("index", help="Show paragraph index")
    idx_parser.add_argument("unpacked_dir", help="Unpacked DOCX directory")
    idx_parser.add_argument(
        "--max-chars", type=int, default=60, help="Max chars per paragraph preview"
    )

    # Annotate command: apply annotations
    ann_parser = subparsers.add_parser("annotate", help="Apply annotations")
    ann_parser.add_argument("unpacked_dir", help="Unpacked DOCX directory")
    ann_parser.add_argument("annotations_json", help="JSON file with annotations")
    ann_parser.add_argument("--author", default="审阅教授", help="Comment author")

    args = parser.parse_args()

    if args.command == "index":
        print_paragraph_index(args.unpacked_dir, args.max_chars)

    elif args.command == "annotate":
        with open(args.annotations_json, encoding="utf-8") as f:
            annotations = json.load(f)

        print(f"正在标注 {len(annotations)} 条批注...\n")
        ok, fail, msgs = annotate_document(
            args.unpacked_dir, annotations, args.author
        )

        for m in msgs:
            print(m)

        print(f"\n完成：{ok} 条成功，{fail} 条失败")
        if fail > 0:
            sys.exit(1)
