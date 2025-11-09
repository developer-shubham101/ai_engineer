import os

# Define the directory structure
structure = {
    "ai_backend": {
        "app": {
            "main.py": "",
            "routes": {
                "__init__.py": "",
                "health.py": "",
                "analyze.py": "",
                "fetch.py": ""
            },
            "utils": {
                "__init__.py": "",
                "http_client.py": ""
            }
        },
        "tests": {
            "test_analyze.py": ""
        },
        ".env.example": "",
        "Dockerfile": "",
        "requirements.txt": "",
        "README.md": ""
    }
}

def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(base_path, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)

if __name__ == "__main__":
    create_structure(".", structure)
    print("Directory structure created successfully.")