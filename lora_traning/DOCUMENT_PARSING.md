# 📄 Document Parsing Guide

## Supported Formats

### Currently Supported:
- ✅ **Markdown (.md)** - Company policies, documentation
- ✅ **HTML (.html, .htm)** - Web pages, exported documents  
- ✅ **Plain Text (.txt)** - Simple text files

### Future Support (Coming Soon):
- 🔄 **PDF (.pdf)** - Reports, official documents
- 🔄 **Word (.docx)** - Microsoft Word documents
- 🔄 **RTF (.rtf)** - Rich text format

## Usage

### Option 1: Automatic Conversion (Recommended)
```bash
# Run complete workflow - includes document conversion
scripts\retrain_improved.bat
```

### Option 2: Manual Conversion
```bash
# Convert all documents to text first
python scripts\doc_parser.py

# Then prepare training data
python scripts\prepare_training_data_improved.py
```

## Directory Structure

### Before Conversion:
```
data/
├── company/
│   ├── v1/
│   │   ├── HR_policies_handbook.md
│   │   ├── brand_guidelines.html
│   │   └── meeting_notes.txt
│   └── v2/
│       ├── AI_strategy.md
│       └── security_policy.html
└── other_docs/
    ├── training_manual.md
    └── procedures.html
```

### After Conversion:
```
data/
├── converted/           # ← New directory with text files
│   ├── company/
│   │   ├── v1/
│   │   │   ├── HR_policies_handbook.txt
│   │   │   ├── brand_guidelines.txt
│   │   │   └── meeting_notes.txt
│   │   └── v2/
│   │       ├── AI_strategy.txt
│   │       └── security_policy.txt
│   └── other_docs/
│       ├── training_manual.txt
│       └── procedures.txt
└── [original files remain unchanged]
```

## How It Works

### Document Parser (`doc_parser.py`):
1. **Scans recursively** - Finds files in all subdirectories
2. **Converts formats**:
   - `.md` → Plain text (removes markdown formatting)
   - `.html/.htm` → Plain text (removes HTML tags)
   - `.txt` → Copies as-is
3. **Maintains structure** - Preserves directory hierarchy
4. **Cleans content** - Removes excessive whitespace

### Training Data Preparation:
1. **Checks for converted files** first in `data/converted/`
2. **Falls back to original files** if no converted directory
3. **Creates Q&A pairs** from all content
4. **Handles nested directories** automatically

## Benefits

### ✅ **Flexibility**:
- Add any supported file format to `data/` folder
- Organize in any directory structure
- Mix different file formats

### ✅ **Quality**:
- HTML tags removed cleanly
- Markdown formatting converted to readable text
- Consistent text format for training

### ✅ **Efficiency**:
- Converted files cached for reuse
- Original files preserved
- Faster subsequent training runs

## Examples

### Adding New Documents:
```bash
# Add files anywhere in data/ directory
data/
├── policies/
│   ├── new_policy.md
│   └── exported_doc.html
├── manuals/
│   └── user_guide.html
└── reports/
    └── quarterly_report.md

# Run conversion
python scripts\doc_parser.py

# Files automatically converted to data/converted/
```

### Supported Content Types:
- **Company policies** (HR, IT, Security)
- **Documentation** (User guides, manuals)
- **Reports** (Financial, project reports)
- **Meeting notes** (Minutes, summaries)
- **Web content** (Exported web pages)

## Troubleshooting

### No Files Found:
```bash
❌ No supported files found in data
Supported formats: .md, .txt, .html, .htm
```
**Solution:** Check file extensions and directory structure

### Conversion Errors:
```bash
❌ Error parsing filename.html: [error details]
```
**Solution:** Check file encoding (should be UTF-8) or file corruption

### Missing Dependencies:
```bash
⚠️ Install parsing dependencies: pip install markdown beautifulsoup4
```
**Solution:** Run `pip install -r requirements.txt`

## Advanced Usage

### Custom Conversion:
```python
from scripts.doc_parser import parse_document

# Convert single file
content = parse_document(Path("my_doc.html"))
print(content)
```

### Batch Processing:
```python
from scripts.doc_parser import scan_and_convert

# Convert all files in directory
results = scan_and_convert(Path("data"), Path("output"))
print(f"Converted {len(results)} files")
```

The document parser ensures your training data includes content from all supported formats while maintaining quality and organization!