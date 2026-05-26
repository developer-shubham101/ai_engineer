"""File save tool for agent system."""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def save_text_file(filename: str, content: str) -> Dict[str, Any]:
    """Save text content to a file."""
    try:
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'

        base_dir = "user_uploaded_files"
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, safe_filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return {"filename": safe_filename, "filepath": filepath, "size": len(content), "status": "success"}
    except Exception as e:
        return {"filename": filename, "error": str(e), "status": "error"}


def save_research_report(
    title: str,
    query: str,
    summary: str,
    markdown: str,
    metadata: str,
    sources: str,
) -> Dict[str, Any]:
    """Save a structured research report as markdown with a JSON sidecar.

    Args:
        title:    Report title (used as filename base).
        query:    Original research query.
        summary:  Executive summary (1-3 sentences).
        markdown: Full report body in markdown format.
        metadata: JSON string of extra metadata (tags, topic, date, etc.).
        sources:  Newline-separated list of source URLs or citations.
    """
    try:
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
        safe_title = safe_title[:80] or "research_report"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"{safe_title}_{timestamp}"

        base_dir = os.path.join("user_uploaded_files", "research_reports")
        os.makedirs(base_dir, exist_ok=True)

        # Parse optional fields gracefully
        try:
            meta_dict = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            meta_dict = {"raw": metadata}

        source_list: List[str] = [s.strip() for s in (sources or "").splitlines() if s.strip()]

        # Build full markdown document
        report_md = f"""# {title}

**Query:** {query}  
**Generated:** {datetime.utcnow().isoformat()}Z

---

## Executive Summary

{summary}

---

{markdown}

---

## Sources

{chr(10).join(f'- {s}' for s in source_list) or '_No sources provided._'}
"""

        md_path = os.path.join(base_dir, f"{base_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        # JSON sidecar for programmatic access
        sidecar = {
            "title": title,
            "query": query,
            "summary": summary,
            "sources": source_list,
            "metadata": meta_dict,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "report_path": md_path,
        }
        json_path = os.path.join(base_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        return {
            "status": "success",
            "title": title,
            "report_path": md_path,
            "sidecar_path": json_path,
            "size": len(report_md),
            "sources_count": len(source_list),
        }
    except Exception as e:
        return {"title": title, "error": str(e), "status": "error"}