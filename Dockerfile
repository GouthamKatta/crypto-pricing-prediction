# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy all files from the project directory into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask API runs on
EXPOSE 5003

# Command to run the application
CMD ["python", "api.py"]
