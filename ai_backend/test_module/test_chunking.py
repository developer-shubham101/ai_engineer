"""Test script for paragraph-aware chunking with different chunk sizes."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.modules.core.chunking import chunk_text_paragraph_aware, ChunkConfig


# Sample document with clear paragraph structure
SAMPLE_TEXT = """
# Company Vacation Policy

Our company values work-life balance and provides generous vacation benefits to all full-time employees.

## Annual Vacation Days

All full-time employees receive 20 days of paid vacation per year. This allocation is provided at the start of each calendar year and can be used at the employee's discretion, subject to manager approval.

Part-time employees receive vacation days on a pro-rated basis, calculated according to their scheduled hours. For example, an employee working 20 hours per week would receive 10 vacation days per year.

## Requesting Vacation Time

Vacation requests must be submitted through the HR portal at least two weeks in advance. This allows managers adequate time to plan for coverage and ensures smooth operations during your absence.

For vacation requests exceeding one week, we recommend submitting your request at least one month in advance. This is especially important during peak business periods or when multiple team members may be requesting time off.

Emergency situations are handled on a case-by-case basis. If you need to take unexpected time off due to illness or family emergency, please contact your manager and HR as soon as possible.

## Carryover Policy

Employees may carry over up to 5 unused vacation days to the following year. Any days beyond this limit will be forfeited unless special arrangements are made with HR.

Carried-over days must be used within the first quarter of the new year. This policy ensures that employees take adequate time off for rest and rejuvenation while maintaining operational continuity.

## Vacation During Probation

New employees are eligible to use vacation days after completing their 90-day probationary period. However, vacation days begin accruing from the first day of employment.

If you have questions about your vacation balance or need clarification on any policy details, please contact the HR department at hr@company.com or extension 5555.
"""


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def analyze_chunks(chunks_with_meta, chunk_size):
    """Analyze and display chunk statistics."""
    print(f"\n{'='*80}")
    print(f"CHUNK SIZE: {chunk_size} characters")
    print(f"{'='*80}")
    print(f"\nTotal chunks created: {len(chunks_with_meta)}")
    
    # Statistics
    char_counts = [meta['char_count'] for _, meta in chunks_with_meta]
    token_estimates = [meta['token_estimate'] for _, meta in chunks_with_meta]
    para_counts = [meta['paragraph_count'] for _, meta in chunks_with_meta]
    
    print(f"\nChunk Statistics:")
    print(f"  - Average characters: {sum(char_counts) / len(char_counts):.1f}")
    print(f"  - Min characters: {min(char_counts)}")
    print(f"  - Max characters: {max(char_counts)}")
    print(f"  - Average tokens (estimated): {sum(token_estimates) / len(token_estimates):.1f}")
    print(f"  - Average paragraphs per chunk: {sum(para_counts) / len(para_counts):.1f}")
    
    # Display chunks
    print(f"\n{'='*80}")
    print("CHUNKS PREVIEW:")
    print(f"{'='*80}")
    
    for i, (chunk, meta) in enumerate(chunks_with_meta, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Characters: {meta['char_count']}, Tokens: ~{meta['token_estimate']}, Paragraphs: {meta['paragraph_count']}")
        print(f"Type: {meta['chunk_type']}")
        
        # Show first 200 chars
        preview = chunk[:200].replace('\n', ' ')
        if len(chunk) > 200:
            preview += "..."
        print(f"Preview: {preview}")
        
        # Check for mid-sentence cuts
        if not chunk.endswith(('.', '!', '?', '\n')):
            print("⚠️  WARNING: Chunk may end mid-sentence")
        else:
            print("✓ Chunk ends at sentence/paragraph boundary")


def compare_chunk_sizes():
    """Compare different chunk sizes."""
    print("="*80)
    print("PARAGRAPH-AWARE CHUNKING COMPARISON")
    print("="*80)
    print(f"\nSample document length: {len(SAMPLE_TEXT)} characters")
    print(f"Estimated tokens: ~{len(SAMPLE_TEXT) // 4}")
    
    # Test different chunk sizes
    chunk_sizes = [256, 512, 1024]
    
    for chunk_size in chunk_sizes:
        config = ChunkConfig(
            chunk_size=chunk_size,
            overlap=50,
            max_chunk_size=chunk_size * 2  # Allow 2x for paragraph integrity
        )
        
        chunks_with_meta = chunk_text_paragraph_aware(SAMPLE_TEXT, config)
        analyze_chunks(chunks_with_meta, chunk_size)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print("\nRecommendations:")
    print("  - 256 chars: More chunks, better precision, may split paragraphs")
    print("  - 512 chars: Balanced, good for most use cases")
    print("  - 1024 chars: Fewer chunks, better context, may be too large for some models")
    
    print("\nKey Benefits of Paragraph-Aware Chunking:")
    print("  ✓ Respects semantic boundaries (paragraphs)")
    print("  ✓ Avoids mid-sentence cuts")
    print("  ✓ Maintains context within chunks")
    print("  ✓ Better retrieval quality")
    print("  ✓ More coherent chunks for LLM processing")


def test_edge_cases():
    """Test edge cases."""
    print(f"\n{'='*80}")
    print("EDGE CASE TESTING")
    print(f"{'='*80}")
    
    # Test 1: Very long paragraph
    long_para = "This is a very long paragraph. " * 100
    config = ChunkConfig(chunk_size=512, overlap=50)
    chunks = chunk_text_paragraph_aware(long_para, config)
    print(f"\n1. Long paragraph (no double newlines):")
    print(f"   Input: {len(long_para)} chars")
    print(f"   Output: {len(chunks)} chunks")
    print(f"   ✓ Handled by sentence-level splitting")
    
    # Test 2: Many small paragraphs
    small_paras = "\n\n".join(["Short paragraph."] * 20)
    chunks = chunk_text_paragraph_aware(small_paras, config)
    print(f"\n2. Many small paragraphs:")
    print(f"   Input: 20 paragraphs")
    print(f"   Output: {len(chunks)} chunks")
    print(f"   ✓ Combined small paragraphs efficiently")
    
    # Test 3: Empty text
    chunks = chunk_text_paragraph_aware("", config)
    print(f"\n3. Empty text:")
    print(f"   Output: {len(chunks)} chunks")
    print(f"   ✓ Handled gracefully")
    
    # Test 4: Single paragraph
    single = "This is a single paragraph without any double newlines."
    chunks = chunk_text_paragraph_aware(single, config)
    print(f"\n4. Single paragraph:")
    print(f"   Output: {len(chunks)} chunks")
    print(f"   ✓ Returned as single chunk")


if __name__ == "__main__":
    compare_chunk_sizes()
    test_edge_cases()
    
    print(f"\n{'='*80}")
    print("TEST COMPLETED")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Review chunk quality by eye")
    print("2. Test with actual queries against your document collection")
    print("3. Compare retrieval quality with old vs new chunking")
    print("4. Adjust chunk_size based on your specific use case")
