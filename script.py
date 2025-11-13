import os
from pathlib import Path

# Root folder name
ROOT_DIR = "ai_backend"

# Project structure definition
project_structure = {
    "app": [
        "__init__.py",
        "main.py"
    ],
    ".dockerignore": None,
    ".gitignore": None,
    "docker-compose.yml": None,
    "Dockerfile": None,
    "requirements.txt": None
}

# Content for each file
file_content = {
    "app/main.py": """from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from the FastAPI template!"}
""",
    "Dockerfile": """# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory to the working directory
COPY ./app /code/app

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "docker-compose.yml": """version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./app:/code/app
""",
    "requirements.txt": """fastapi
uvicorn[standard]
""",
    ".gitignore": """__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.idea/
.vscode/
.DS_Store
""",
    ".dockerignore": """__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.idea
.vscode
README.md
venv
"""
}

def create_file(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def setup_project():
    root = Path(ROOT_DIR)
    for name, files in project_structure.items():
        if isinstance(files, list):
            for file in files:
                file_path = root / name / file
                create_file(file_path)
        else:
            file_path = root / name
            content = file_content.get(name, "")
            create_file(file_path, content)

    # Write content for nested files
    for rel_path, content in file_content.items():
        file_path = root / rel_path
        if not file_path.exists():
            create_file(file_path, content)

    print(f"✅ Project structure created under '{ROOT_DIR}'")

if __name__ == "__main__":
    setup_project()