import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.doc_parser import parse_text, parse_file, RawFormat

def test_markdown_parsing():
    print("Testing Markdown Parsing...")
    md_content = """
# Header 1
## Header 2

This is a **bold** text and *italic* text.

- Item 1
- Item 2

[Link](http://example.com)

```python
print("code block")
```
    """
    
    expected_snippets = [
        "Header 1",
        "Header 2",
        "This is a bold text and italic text.",
        "Item 1",
        "Item 2",
        "Link",
        "print(\"code block\")"
    ]
    
    text = parse_text(md_content, RawFormat.MARKDOWN)
    print(f"--- Extracted Text ---\n{text}\n----------------------")
    
    for snippet in expected_snippets:
        if snippet in text:
            print(f"[PASS] Found: '{snippet}'")
        else:
            print(f"[FAIL] Missing: '{snippet}'")

def test_plain_text_parsing():
    print("\nTesting Plain Text Parsing...")
    raw = "Just some plain text."
    text = parse_text(raw, RawFormat.PLAIN)
    if text == raw:
        print("[PASS] Plain text returned as is.")
    else:
        print(f"[FAIL] Plain text modified: {text}")

def test_file_inference():
    print("\nTesting File Inference...")
    # Create a dummy file
    dummy_path = "test_doc.md"
    with open(dummy_path, "w") as f:
        f.write("# Test File")
    
    try:
        text = parse_file(dummy_path)
        if "Test File" in text:
            print("[PASS] File inference and parsing successful.")
        else:
            print(f"[FAIL] Content mismatch: {text}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

if __name__ == "__main__":
    try:
        test_markdown_parsing()
        test_plain_text_parsing()
        test_file_inference()
        print("\nAll tests completed.")
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
