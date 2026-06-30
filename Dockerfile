# Use an official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for compiling batman-package and transitleastsquares)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY --chown=user . .

# Expose the default Streamlit port
EXPOSE 7860

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
