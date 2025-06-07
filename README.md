# 📄 AI-Powered Resume Parser Pro

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Resume Parser Pro** is an intelligent application that extracts and analyzes resume data using cutting-edge AI. Built with Streamlit and powered by LangChain and Groq's AI, it provides a seamless experience for parsing, searching, and interacting with resume data.

## ✨ Features

- **Smart Resume Parsing**: Automatically extracts key information from PDF resumes
- **AI-Powered Chat**: Interact with resumes using natural language queries
- **Responsive Web Interface**: Clean, modern UI built with Streamlit
- **Persistent Storage**: SQLite database for storing resume data
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Advanced Search**: Filter and search through resumes using various criteria

## 🚀 Quick Start

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)

### Running with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/resume-parser-pro.git
   cd resume-parser-pro
   ```

2. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your Groq API key
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

3. **Build and start the application**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## 🛠️ Features in Detail

### 📤 Upload Resumes
- Drag and drop or select PDF files
- Automatic duplicate detection
- Batch upload support

### 🔍 Browse & Search
- View all parsed resumes
- Search by name, skills, or experience
- Filter by education or date added

### 💬 AI Chat Interface
- Ask questions about resumes
- Get insights about candidate experience
- Compare candidates

## 🧩 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.10
- **AI/ML**: 
  - LangChain
  - Groq AI
  - Hugging Face Transformers
- **Database**: SQLite
- **Vector Store**: FAISS
- **Containerization**: Docker

## 🏗️ Project Structure

```
resume-parser-pro/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── prac.py              # Main application
├── requirements.txt     # Python dependencies
├── uploads/             # Store uploaded resumes
└── resume_database.db   # SQLite database
```

## 🌟 Powered by Windsurf AI

This project was developed using **Windsurf AI**, an advanced AI coding assistant that helped with:
- Code generation and optimization
- Docker containerization
- Documentation
- Debugging and testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ using Streamlit and Python
</div>
