# 1. Use a lightweight Python base
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first (Better for caching)
COPY requirements.txt .

# 4. Upgrade pip and install minimized requirements 
# 4. Install minimized requirements (using verified versions)
RUN pip install --no-cache-dir flask==3.1.3 pandas==2.2.3 scikit-learn==1.5.2 joblib==1.4.2
# 5. Copy the rest of the project files
COPY . .

# 6. Expose the port Flask runs on
EXPOSE 5000

# 7. Use the executive form for the start command
CMD ["python", "app.py"]