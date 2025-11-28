# Company Data Format Guide

## Overview
This directory contains sample company data for **Saarthi Infotech Pvt. Ltd.** - a fictional enterprise used for RAG system testing and demonstration.

## Directory Structure
- `company/v1/` - Version 1 company documents and metadata
- Each subdirectory represents different data versions or categories

## File Structure
Each document requires two files:
- `document_name.md` - Content file
- `document_name.meta.json` - Metadata file

## Metadata Format

```json
{
  "document_type": "policy|memo|guide|report|agreement|handbook|etc",
  "department": "HR|IT|Finance|Operations",
  "sensitivity": "public_internal|department_confidential|role_confidential|highly_confidential|super_confidential|personal",
  "tags": ["tag1", "tag2", "tag3"],
  "public_summary": "Brief description of the document",
  "allowed_roles": ["SuperAdmin", "Manager", "HR", "Employee", "Guest"],
  "effective_date": "YYYY-MM-DD or empty string"
}
```

## Valid Values

### Departments
- `HR` - Human Resources
- `IT` - Information Technology  
- `Finance` - Financial operations
- `Operations` - General operations

### Sensitivity Levels
- `public_internal` (0) - Everyone can access
- `department_confidential` (1) - Employee+ in same department
- `role_confidential` (2) - HR+ level required
- `highly_confidential` (3) - Manager+ level required
- `super_confidential` (4) - SuperAdmin only
- `personal` (1) - Owner + HR+ level

### Roles (Hierarchy)
- `SuperAdmin` (4) - Full access
- `Manager` (3) - Management level
- `HR` (2) - HR functions
- `Employee` (1) - Standard access
- `Guest` (0) - Public only

## Example Files

### sample_policy.md
```markdown
# Sample Company Policy

This is the content of your document...
```

### sample_policy.meta.json
```json
{
  "document_type": "policy",
  "department": "HR",
  "sensitivity": "public_internal",
  "tags": ["policy", "company", "guidelines"],
  "public_summary": "General company policy document",
  "allowed_roles": ["Employee", "Manager", "HR"],
  "effective_date": "2024-01-01"
}
```