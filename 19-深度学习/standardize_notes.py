#!/usr/bin/env python
"""Standardize markdown formatting across DeepLearningNotes.

Usage:
    python standardize_notes.py            # dry run (report only)
    python standardize_notes.py --apply    # apply changes
"""

import os
import re
import sys


def normalize_h1(lines, filepath):
    """Normalize H1 to format: # NN_标题 (number + underscore prefix)."""
    if not lines or not lines[0].startswith('# '):
        return lines

    basename = os.path.basename(filepath)  # e.g. "01_xxx.md"
    m = re.match(r'^(\d+)_', basename)
    if not m:
        return lines
    num_prefix = m.group(1)  # e.g. "01"

    h1 = lines[0]

    # Check if H1 already has number prefix
    if re.match(r'^# \d+_', h1):
        # Pattern C: already has "NN_" — no change
        return lines

    if re.match(r'^# \d+ ', h1):
        # Pattern B: "# NN Title" (number + space) → "# NN_Title"
        lines[0] = re.sub(r'^(# )\d+ ', r'\g<1>' + num_prefix + '_', h1, count=1)
    elif re.match(r'^# [^0-9]', h1):
        # Pattern A: "# Title" (no number) → "# NN_Title"
        lines[0] = f'# {num_prefix}_{h1[2:]}'

    return lines


def normalize_h2_blank_lines(lines):
    """Ensure one blank line after each ## heading."""
    result = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        result.append(line)
        if re.match(r'^## ', line):
            # Look at the next line — if it's non-empty (and not another heading),
            # insert a blank line
            if i + 1 < n:
                nxt = lines[i + 1]
                if nxt.strip() and not re.match(r'^##?\s', nxt):
                    result.append('\n')
        i += 1
    return result


def normalize_code_block_spacing(lines):
    """Ensure blank line before opening ``` and after closing ```."""
    result = []
    n = len(lines)
    in_code = False
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```') and not in_code:
            # Opening fence — ensure blank line before
            in_code = True
            if result and result[-1].strip() and not result[-1].startswith('## '):
                result.append('\n')
            result.append(line)
        elif stripped.startswith('```') and in_code:
            # Closing fence — ensure blank line after
            in_code = False
            result.append(line)
            if i + 1 < n and lines[i + 1].strip():
                result.append('\n')
        else:
            result.append(line)
        i += 1
    return result


def normalize_bullet_spacing(lines):
    """Remove blank lines between consecutive list items."""
    # First, build a set of indices to skip (blank lines between list items)
    to_skip = set()
    n = len(lines)

    for i in range(n):
        if lines[i].strip():
            continue
        # This is a blank line — check if it sits between two list items
        # Look backward for the previous non-empty line
        prev_idx = i - 1
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        # Look forward for the next non-empty line
        next_idx = i + 1
        while next_idx < n and not lines[next_idx].strip():
            next_idx += 1

        if prev_idx < 0 or next_idx >= n:
            continue

        prev_line = lines[prev_idx].strip()
        next_line = lines[next_idx].strip()

        prev_is_item = prev_line.startswith('- ') or re.match(r'^\d+\.\s+', prev_line)
        next_is_item = next_line.startswith('- ') or re.match(r'^\d+\.\s+', next_line)

        if prev_is_item and next_is_item:
            to_skip.add(i)

    return [line for i, line in enumerate(lines) if i not in to_skip]


def normalize_ordered_lists(lines):
    """Convert '1. text' to '- text' outside code/math blocks."""
    in_code = False
    in_math = False
    result = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code = not in_code

        if not in_code and stripped.startswith('$$'):
            in_math = not in_math

        if not in_code and not in_math:
            line = re.sub(r'^\d+\.\s+', '- ', line)

        result.append(line)
    return result


def process_file(filepath, dry_run=True):
    """Process a single markdown file. Returns status string or None."""
    if not filepath.endswith('.md'):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        original = f.readlines()

    if not original:
        return None

    lines = original[:]

    lines = normalize_h1(lines, filepath)
    lines = normalize_h2_blank_lines(lines)
    lines = normalize_code_block_spacing(lines)
    lines = normalize_bullet_spacing(lines)
    lines = normalize_ordered_lists(lines)

    if lines == original:
        return None

    if not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return 'modified'
    return 'would_change'


def verify_h1_pattern(root_dir):
    """Verify all .md files have the correct H1 format."""
    mismatches = []
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path) or not re.match(r'\d{2}_', item):
            continue
        for fname in sorted(os.listdir(item_path)):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(item_path, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            m_file = re.match(r'^(\d+)_', fname)
            m_h1 = re.match(r'^# (\d+)_', first_line)
            if m_file and m_h1 and m_file.group(1) == m_h1.group(1):
                continue
            mismatches.append((fname, first_line.strip()))
    return mismatches


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dry_run = '--apply' not in sys.argv

    stats = {'dirs': 0, 'checked': 0, 'changed': 0, 'unchanged': 0}

    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path) or not re.match(r'\d{2}_', item):
            continue
        stats['dirs'] += 1
        print(f"\n[{item}]")

        for fname in sorted(os.listdir(item_path)):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(item_path, fname)
            stats['checked'] += 1
            status = process_file(fpath, dry_run)
            if status:
                stats['changed'] += 1
                marker = '~' if dry_run else 'M'
                print(f"  {marker} {fname}")
            else:
                stats['unchanged'] += 1

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n{'='*50}")
    print(f"[{mode}] {stats['dirs']} directories, "
          f"{stats['checked']} files checked, "
          f"{stats['changed']} changed, "
          f"{stats['unchanged']} unchanged")

    if dry_run and stats['changed'] > 0:
        print("Run with --apply to write changes.")

    # Show any H1 mismatches
    mismatches = verify_h1_pattern(root_dir)
    if mismatches:
        print(f"\nH1 mismatches after processing: {len(mismatches)}")
        for fname, h1 in mismatches[:5]:
            print(f"  {fname}: {h1}")

    return 0 if not dry_run or stats['changed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
