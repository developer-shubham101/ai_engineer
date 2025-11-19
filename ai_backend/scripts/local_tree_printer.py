import os

IGNORE_LIST = ['.git', 'local_tree_printer.py', ".venv", "embeddings_models"]

def print_recursive_tree(dir_path, prefix=""):
    """
    A robust recursive function to print a directory tree
    with correct connectors (├── and └──).
    """
    # 1. Get items, excluding .git (and the script file itself, if it's in the repo root)
    # Listdir items are not guaranteed to be sorted, so we sort them.
    items = [item for item in os.listdir(dir_path) if item not in IGNORE_LIST]
    items.sort()

    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)

        # 2. Determine the connector (├── or └──)
        connector = "└── " if is_last else "├── "

        if os.path.isdir(path):
            # It's a directory
            print(f"{prefix}{connector}📁 {item}/")

            # 3. Create the prefix for the next level's indentation
            next_prefix = prefix + ("    " if is_last else "│   ")

            # 4. Recurse into the subdirectory
            print_recursive_tree(path, next_prefix)
        else:
            # It's a file
            print(f"{prefix}{connector}📄 {item}")


# --- Main execution ---
if __name__ == "__main__":

    # --- IMPORTANT: CHANGE THIS PATH ---
    # Set this to the absolute or relative path of your local 'ai_engineer' folder.
    LOCAL_REPO_PATH = '../'
    # Example: LOCAL_REPO_PATH = '/Users/yourname/Documents/ai_engineer'

    # Get the base name (the folder name itself)
    repo_name = os.path.basename(os.path.abspath(LOCAL_REPO_PATH))

    if not os.path.exists(LOCAL_REPO_PATH):
        print(f"Error: The path '{LOCAL_REPO_PATH}' does not exist.")
        print("Please update the 'LOCAL_REPO_PATH' variable in the script.")
    elif not os.path.isdir(LOCAL_REPO_PATH):
        print(f"Error: The path '{LOCAL_REPO_PATH}' is not a directory.")
    else:
        print(f"📂 {repo_name}/")  # Print the root directory
        print_recursive_tree(LOCAL_REPO_PATH)