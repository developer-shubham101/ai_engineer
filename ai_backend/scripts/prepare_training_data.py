import os
import json
import re

# --- CONFIGURATION ---
INPUT_FOLDER = "companyData"  # Folder containing your .md files
OUTPUT_FILE = "training_data.jsonl"  # Resulting file for Colab
MIN_RESPONSE_LENGTH = 15  # Skip lines that are too short (noise)


def clean_instruction(text):
    """
    Removes markdown headers (#, ##) and extra spaces from the instruction.
    Example: "##   Sick Leave  " -> "Sick Leave"
    """
    # Remove leading hash signs and whitespace
    return text.lstrip('#').strip()


def parse_markdown_file(file_path, filename):
    """
    Reads a markdown file.
    - Treats any line starting with '#' as a new Instruction.
    - Everything following it (until the next '#') is the Response.
    - Uses filename as context if no headers are found (fallback).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pairs = []
    current_instruction = None
    current_response_buffer = []

    # Helper to save the current buffer to pairs
    def save_buffer():
        nonlocal current_instruction, current_response_buffer
        if current_instruction and current_response_buffer:
            full_response = "\n".join(current_response_buffer).strip()
            if len(full_response) >= MIN_RESPONSE_LENGTH:
                pairs.append({
                    "instruction": current_instruction,
                    "response": full_response
                })

    for line in lines:
        stripped_line = line.strip()

        # 1. DETECT HEADER (New Instruction)
        # Matches #, ##, ###, etc.
        if stripped_line.startswith("#"):
            # Save the previous section before starting a new one
            save_buffer()

            # Set the new instruction
            current_instruction = clean_instruction(stripped_line)
            current_response_buffer = []  # Reset buffer

        # 2. CAPTURE BODY (Response)
        else:
            # If we found a header, append text to it
            if current_instruction:
                current_response_buffer.append(line.rstrip())

            # EDGE CASE: File has text at the top but NO header (like an intro)
            # We use the filename as the instruction (e.g. "Crisis Response Plan")
            elif not current_instruction and stripped_line:
                # Convert "crisis_response_plan.md" -> "Crisis Response Plan"
                fallback_title = filename.replace(".md", "").replace("_", " ").title()
                current_instruction = fallback_title
                current_response_buffer.append(line.rstrip())

    # 3. END OF FILE: Save the final section
    save_buffer()

    return pairs


def main():
    # 1. Verify input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' not found.")
        return

    all_training_data = []

    # Dynamically get list of all files in the folder
    files_in_folder = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".md")]

    if not files_in_folder:
        print(f"⚠️  No .md files found in {INPUT_FOLDER}.")
        return

    print(f"📂 Found {len(files_in_folder)} documents. Processing...")

    # 2. Process files
    for filename in files_in_folder:
        file_path = os.path.join(INPUT_FOLDER, filename)
        try:
            data = parse_markdown_file(file_path, filename)
            all_training_data.extend(data)
            # Print status for specific key files you mentioned
            print(f"   ├── Parsed: {filename} \t({len(data)} examples)")
        except Exception as e:
            print(f"   ⚠️  Error reading {filename}: {e}")

    # 3. Save to JSONL
    if all_training_data:
        print(f"\n💾 Generating '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in all_training_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ Done! Created {len(all_training_data)} training pairs.")
        print(f"   Ready to upload '{OUTPUT_FILE}' to Colab.")
    else:
        print("\n⚠️  Warning: No valid data extracted. Check file content.")


if __name__ == "__main__":
    main()