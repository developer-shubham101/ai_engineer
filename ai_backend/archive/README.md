# Tests Directory

This directory contains all test cases for the multi-provider RAG system.

## Configuration

All tests use a common base URL defined in `constants.py`:
```python
BASE_URL = "http://localhost:8000"
```

To change the test server URL, update the `BASE_URL` value in `constants.py`.

## Test Files

- `test_api_metadata_validation.py` - API metadata validation tests
- `test_conversation_context.py` - Conversation context tests
- `test_embedding_api.py` - Embedding API tests
- `test_embedding_simple.py` - Simple embedding tests
- `test_flexible_rbac.py` - RBAC logic tests
- `test_live_endpoints.py` - Live endpoint integration tests
- `test_model_download.py` - Model download system tests
- `test_optimized_prompt.py` - Prompt optimization tests
- `test_personalized_responses.py` - Personalized response tests
- `test_prompt_debug.py` - Prompt debugging tests
- `test_prompt_optimization.py` - Prompt optimization tests
- `test_prompt_optimization_debug.py` - Prompt optimization debug tests
- `test_rbac_comprehensive.py` - Comprehensive RBAC tests
- `test_rbac_endpoints.py` - RBAC endpoint tests
- `test_rbac_verification.py` - RBAC verification tests
- `test_simple_prompt.py` - Simple prompt tests
- `test_versioning_flow.py` - Document versioning tests

## Running Tests

Run individual tests:
```bash
python -m tests.test_live_endpoints
```

Or run from the tests directory:
```bash
cd tests
python test_live_endpoints.py
```