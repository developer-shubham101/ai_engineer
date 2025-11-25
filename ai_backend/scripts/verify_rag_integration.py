import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api_routes_rag import add_document_file
from app.services.rag_local_service import seed_from_file
from app.utils.doc_parser import RawFormat

async def test_api_upload():
    print("Testing API Upload Integration...")
    
    # Mock dependencies
    mock_file = MagicMock()
    mock_file.filename = "test.md"
    mock_file.read = AsyncMock(return_value=b"# Header\n\n**Bold** text")
    
    mock_requester = {"user_id": "test_user"}
    
    # Mock the RAG service with dependency injection
    mock_rag_service = MagicMock()
    mock_rag_service.add_document_to_rag_local = AsyncMock(return_value=["id1"])

    # Call the endpoint function directly, passing the mock service
    resp = await add_document_file(
        file=mock_file, 
        requester=mock_requester,
        rag_service=mock_rag_service
    )
    
    # Verify result
    print(f"Response: {resp}")
    
    # Verify that add_document_to_rag_local was called with CLEAN text
    call_kwargs = mock_rag_service.add_document_to_rag_local.call_args.kwargs
    text_arg = call_kwargs.get('text')
    print(f"Passed Text: {text_arg!r}")
    
    if "Header" in text_arg and "**" not in text_arg:
        print("[PASS] Markdown formatting stripped.")
    else:
        print("[FAIL] Markdown formatting NOT stripped.")

async def test_seeding():
    print("\nTesting Seeding Integration...")
    
    # Create a dummy markdown file
    dummy_path = "test_seed.md"
    with open(dummy_path, "w") as f:
        f.write("# Seed Header\n\n*Italic* content")
        
    try:
        # Call seed_from_file directly (it doesn't use DI yet)
        from unittest.mock import patch
        with patch('app.services.rag_local_service.add_document_to_rag_local') as mock_add:
            mock_add.return_value = AsyncMock(return_value=["id_seed"])()
            
            # Call seed_from_file
            ids = await seed_from_file(file_path=dummy_path, force_reseed=True)
            
            print(f"Seeded IDs: {ids}")
            
            # Verify text
            if mock_add.called:
                call_kwargs = mock_add.call_args.kwargs
                text_arg = call_kwargs.get('text')
                print(f"Seeded Text: {text_arg!r}")
                
                if "Seed Header" in text_arg and "*" not in text_arg:
                    print("[PASS] Seed file parsed correctly.")
                else:
                    print("[FAIL] Seed file NOT parsed correctly.")
            else:
                print("[INFO] Seed test skipped (collection not empty or other reason)")
                
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

if __name__ == "__main__":
    asyncio.run(test_api_upload())
    asyncio.run(test_seeding())
