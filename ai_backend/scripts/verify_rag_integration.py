import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

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
    mock_file.read = asyncio.coroutine(lambda: b"# Header\n\n**Bold** text")
    
    mock_requester = {"user_id": "test_user"}
    
    with patch('app.api_routes_rag.add_document_to_rag_local', new_callable=MagicMock) as mock_add:
        # Mock the return value to be an awaitable that returns the list
        f = asyncio.Future()
        f.set_result(["id1"])
        mock_add.return_value = f
        
        # Call the endpoint function directly
        resp = await add_document_file(file=mock_file, requester=mock_requester)
        
        # Verify result
        print(f"Response: {resp}")
        
        # Verify that add_document_to_rag_local was called with CLEAN text
        args, kwargs = mock_add.call_args
        text_arg = kwargs.get('text')
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
        with patch('app.services.rag_local_service.add_document_to_rag_local', new_callable=MagicMock) as mock_add:
            f = asyncio.Future()
            f.set_result(["id_seed"])
            mock_add.return_value = f
            
            # Call seed_from_file
            ids = await seed_from_file(file_path=dummy_path, force_reseed=True)
            
            print(f"Seeded IDs: {ids}")
            
            # Verify text
            args, kwargs = mock_add.call_args
            text_arg = kwargs.get('text')
            print(f"Seeded Text: {text_arg!r}")
            
            if "Seed Header" in text_arg and "*" not in text_arg:
                print("[PASS] Seed file parsed correctly.")
            else:
                print("[FAIL] Seed file NOT parsed correctly.")
                
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_api_upload())
    loop.run_until_complete(test_seeding())
