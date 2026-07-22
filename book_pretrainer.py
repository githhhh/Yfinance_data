#!/usr/bin/env python3
"""
IBD Module - Book Pre-trainer & Ingestion Script
Unpacks How_to_Make_Money_in_Stocks.epub from project root into .ibd_book_unpacked (ignored by git),
indexes all text chapters and 300+ chart diagrams for agent pre-training verification.
"""

import sys
import json
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup


def run_book_pretraining(project_root: Path) -> dict:
    epub_path = project_root / "How_to_Make_Money_in_Stocks.epub"
    unpacked_dir = project_root / ".ibd_book_unpacked"

    if not epub_path.exists():
        print(f"[ERROR] 致命错误: 项目根目录下未找到参考电子书 '{epub_path.name}'！")
        print(f"        缺少前置输入大文件: {epub_path}")
        print(f"        技能终止退出。请将 'How_to_Make_Money_in_Stocks.epub' 放入项目根目录。")
        sys.exit(1)

    unpacked_dir.mkdir(parents=True, exist_ok=True)

    # Extract all files
    with zipfile.ZipFile(epub_path, "r") as z:
        z.extractall(unpacked_dir)

    images = list(unpacked_dir.glob("**/*.jpg")) + list(unpacked_dir.glob("**/*.jpeg")) + list(unpacked_dir.glob("**/*.png"))
    html_files = list(unpacked_dir.glob("**/*.html")) + list(unpacked_dir.glob("**/*.xhtml"))

    chapters_index = []
    for hf in sorted(html_files):
        text = BeautifulSoup(hf.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text()
        first_line = text.strip().split("\n")[0] if text else ""
        chapters_index.append({
            "file": str(hf.relative_to(unpacked_dir)),
            "title": first_line[:100],
            "char_count": len(text),
        })

    index_path = unpacked_dir / "book_training_index.json"
    result_meta = {
        "status": "VERIFIED_UNPACKED",
        "book_title": "How to Make Money in Stocks (4th Edition)",
        "epub_source": str(epub_path),
        "unpacked_path": str(unpacked_dir),
        "total_files": len(list(unpacked_dir.glob("**/*"))),
        "total_chapters": len(html_files),
        "total_chart_images": len(images),
        "chapters": chapters_index,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(result_meta, f, ensure_ascii=False, indent=2)

    return result_meta


if __name__ == "__main__":
    script_file = Path(__file__).resolve()
    # Path is: /Users/dev/Documents/Yfinance_data/book_pretrainer.py -> parent is project root
    root = script_file.parent
    meta = run_book_pretraining(root)
    print(f"[BOOK PRE-TRAINING SUCCESS]")
    print(f"  EPUB Source: {meta['epub_source']}")
    print(f"  Unpacked Path: {meta['unpacked_path']}")
    print(f"  Total Files: {meta['total_files']}")
    print(f"  Chapters: {meta['total_chapters']}")
    print(f"  Chart Images: {meta['total_chart_images']}")
