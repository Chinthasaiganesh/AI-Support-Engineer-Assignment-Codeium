# Resume Parser Pro

A Streamlit-based application for parsing and managing resumes with AI-powered chat capabilities.

## Prerequisites

- Docker and Docker Compose installed on your system
- A `.env` file with the required environment variables (copy from `.env.example` if available)

## Quick Start with Docker

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd CS-First
   ```

2. **Build and start the application**:
   ```bash
   docker-compose up --build
   ```
   This will:
   - Build the Docker image with all dependencies
   - Start the Streamlit application
   - Mount the `uploads` directory and SQLite database as volumes for persistence

3. **Access the application**:
   Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

## Managing the Application

- **Stop the application**: Press `Ctrl+C` in the terminal or run:
  ```bash
  docker-compose down
  ```

- **View logs**:
  ```bash
  docker-compose logs -f
  ```

- **Remove containers and volumes** (warning: this will delete all data):
  ```bash
  docker-compose down -v
  ```

## Volumes

The application uses Docker volumes to persist data:
- `resume_uploads`: Stores uploaded resume files
- `resume_data`: Contains the SQLite database

## Environment Variables

Make sure your `.env` file contains all necessary environment variables, including:
- `GROQ_API_KEY`: Your Groq API key for AI capabilities
- Any other required API keys or configuration

## Development

For development, you can mount your local directory into the container:

```bash
docker-compose -f docker-compose.yml up --build
```

This will mount your local directory into the container, allowing for live code changes.
