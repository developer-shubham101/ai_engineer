# Dockerfile

# 1. Use an official Python runtime as a parent image.
# The '-slim' variant is smaller and good for production.
FROM python:3.11-slim

# 2. Set the working directory inside the container.
WORKDIR /code

# 3. Set environment variables.
# This prevents Python from writing .pyc files to the container.
ENV PYTHONDONTWRITEBYTECODE 1
# This ensures Python output is sent straight to the terminal without buffering.
ENV PYTHONUNBUFFERED 1

# 4. Install dependencies.
# First, copy ONLY the requirements file. This allows Docker to cache the
# installed packages. This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Copy the rest of your application code into the container.
# This includes your 'app' directory and your 'data' directory for the RAG service.
COPY ./app /code/app
COPY ./data /code/data

# 6. Expose the port the app runs on.
EXPOSE 5444

# 7. Define the command to run your application.
# This command starts the Uvicorn server, making it accessible from outside.
# --host 0.0.0.0 is crucial for Docker networking.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5444"]