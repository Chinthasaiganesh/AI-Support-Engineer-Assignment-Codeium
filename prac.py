import os
import re
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Generator
from contextlib import contextmanager
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, func, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session as DBSession

# Load environment variables
load_dotenv()

# Database setup
Base = declarative_base()

def auto_upgrade_resume_table():
    """Add missing columns to the resumes table if they don't exist."""
    import sqlite3
    db_path = os.path.join(os.getcwd(), "resume_database.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    expected_columns = [
        ("name", "TEXT"),
        ("email", "TEXT"),
        ("phone", "TEXT"),
        ("skills", "TEXT"),
        ("experience", "TEXT"),
        ("education", "TEXT"),
        ("file_path", "TEXT"),
        ("file_hash", "TEXT"),
        ("uploaded_at", "DATETIME"),
        ("chat_history", "TEXT"),
    ]
    cursor.execute("PRAGMA table_info(resumes)")
    existing = {row[1] for row in cursor.fetchall()}
    for col, coltype in expected_columns:
        if col not in existing:
            try:
                cursor.execute(f"ALTER TABLE resumes ADD COLUMN {col} {coltype}")
            except Exception:
                pass
    conn.commit()
    conn.close()

auto_upgrade_resume_table()

class Resume(Base):
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    skills = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    education = Column(Text, nullable=True)
    file_path = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, unique=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    chat_history = Column(Text, nullable=True)

def init_db():
    engine = create_engine('sqlite:///resume_database.db')
    Base.metadata.create_all(engine)
    return engine

@contextmanager
def get_db_session() -> Generator[DBSession, None, None]:
    engine = init_db()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

# File management - using absolute path to avoid issues
working_dir = os.getcwd()
pdf_folder = os.path.join(working_dir, "uploads")
os.makedirs(pdf_folder, exist_ok=True)

def verify_uploads_folder():
    """Ensure uploads folder exists and is writable"""
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    if not os.access(pdf_folder, os.W_OK):
        st.error(f"Uploads folder is not writable: {pdf_folder}")

verify_uploads_folder()

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file for duplicate detection"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_duplicate_resume(file_hash: str) -> bool:
    """Check if a resume with the same hash already exists"""
    with get_db_session() as db_session:
        return db_session.query(Resume).filter(Resume.file_hash == file_hash).first() is not None

def load_document(file_path: str):
    """Load PDF document using PyMuPDF with enhanced error handling"""
    if not os.path.exists(file_path):
        st.error(f"File not found at path: {file_path}")
        return None
    
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def format_phone_number(phone: str) -> str:
    """Format phone number to a standard format"""
    if not phone or phone.lower() == 'not found':
        return "Not found"
    
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11:
        return f"+{digits[0]} ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return phone

def extract_phone_number(text: str) -> Optional[str]:
    """Extract phone number from text with robust error handling"""
    if not text:
        return None
    try:
        phone_regex = r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4}|\d{10}|\d{3} \d{3} \d{4})'
        matches = re.finditer(phone_regex, text)
        for match in matches:
            phone = match.group()
            if sum(c.isdigit() for c in phone) >= 10:
                return phone
        return None
    except Exception as e:
        st.warning(f"Error extracting phone number: {str(e)}")
        return None

def extract_email(text: str) -> Optional[str]:
    """Extract email from text with robust error handling"""
    if not text:
        return None
    try:
        email_regex = r'[\w\.-]+@[\w\.-]+\.\w+'
        matches = re.findall(email_regex, text)
        return matches[0] if matches else None
    except Exception as e:
        st.warning(f"Error extracting email: {str(e)}")
        return None

def save_resume_to_db(file_path: str, file_hash: str, text: str, **kwargs) -> Optional[int]:
    """Save resume data to database with transaction handling"""
    try:
        if not os.path.exists(file_path):
            st.error("File does not exist at the specified path")
            return None
            
        with get_db_session() as db_session:
            if is_duplicate_resume(file_hash):
                st.warning("This resume already exists in the database.")
                return None
                
            resume = Resume(
                file_path=file_path,
                file_hash=file_hash,
                **kwargs
            )
            db_session.add(resume)
            db_session.commit()
            return resume.id
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return None

def verify_resume_files():
    """Check all database entries have corresponding files"""
    with get_db_session() as db_session:
        resumes = db_session.query(Resume).all()
        for resume in resumes:
            if not os.path.exists(resume.file_path):
                st.warning(f"Missing file for resume ID {resume.id}: {resume.file_path}")
        db_session.commit()

def get_all_resumes() -> List[Dict[str, Any]]:
    """Get all resumes from database with error handling"""
    try:
        with get_db_session() as db_session:
            resumes = db_session.query(Resume).order_by(Resume.uploaded_at.desc()).all()
            return [{
                'id': r.id,
                'name': r.name,
                'email': r.email,
                'phone': format_phone_number(r.phone),
                'skills': r.skills,
                'experience': r.experience,
                'education': r.education,
                'uploaded_at': utc_to_ist(r.uploaded_at).strftime('%Y-%m-%d %H:%M:%S IST'),
                'file_path': r.file_path,
                'file_exists': os.path.exists(r.file_path)
            } for r in resumes]
    except Exception as e:
        st.error(f"Error fetching resumes: {str(e)}")
        return []

def search_resumes(query: str) -> List[Dict[str, Any]]:
    """Search resumes by query with robust error handling"""
    try:
        with get_db_session() as db_session:
            query = f"%{query}%"
            resumes = db_session.query(Resume).filter(
                or_(
                    Resume.name.ilike(query),
                    Resume.skills.ilike(query),
                    Resume.experience.ilike(query),
                    Resume.education.ilike(query)
                )
            ).all()
            
            return [{
                'id': r.id,
                'name': r.name,
                'email': r.email,
                'phone': format_phone_number(r.phone),
                'skills': r.skills,
                'experience': r.experience,
                'education': r.education,
                'uploaded_at': r.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': r.file_path,
                'file_exists': os.path.exists(r.file_path)
            } for r in resumes]
    except Exception as e:
        st.error(f"Error searching resumes: {str(e)}")
        return []

def get_resume_by_id(resume_id: int) -> Optional[Dict[str, Any]]:
    """Get a single resume by ID with error handling"""
    try:
        with get_db_session() as db_session:
            resume = db_session.query(Resume).filter_by(id=resume_id).first()
            if not resume:
                return None
            
            return {
                'id': resume.id,
                'name': resume.name,
                'email': resume.email,
                'phone': format_phone_number(resume.phone),
                'skills': resume.skills,
                'experience': resume.experience,
                'education': resume.education,
                'uploaded_at': resume.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
                'file_path': resume.file_path,
                'file_exists': os.path.exists(resume.file_path),
                'chat_history': resume.chat_history
            }
    except Exception as e:
        st.error(f"Error fetching resume: {str(e)}")
        return None

def update_resume(resume_id: int, **updates) -> bool:
    """Update resume fields with transaction handling"""
    try:
        with get_db_session() as db_session:
            resume = db_session.query(Resume).filter_by(id=resume_id).first()
            if not resume:
                return False
            
            for key, value in updates.items():
                if hasattr(resume, key):
                    setattr(resume, key, value)
            
            db_session.commit()
            return True
    except Exception as e:
        st.error(f"Error updating resume: {str(e)}")
        return False

def delete_resume(resume_id: int) -> bool:
    """Delete a resume from database with proper cleanup"""
    try:
        with get_db_session() as db_session:
            resume = db_session.query(Resume).filter_by(id=resume_id).first()
            if not resume:
                return False
            
            try:
                if os.path.exists(resume.file_path):
                    os.remove(resume.file_path)
            except Exception as e:
                st.error(f"Error deleting file: {str(e)}")
            
            db_session.delete(resume)
            db_session.commit()
            return True
    except Exception as e:
        st.error(f"Error deleting resume: {str(e)}")
        return False

def setup_vectorstore(documents):
    """Set up FAISS vectorstore with error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )
        doc_chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(doc_chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error setting up vectorstore: {str(e)}")
        return None

def create_chain(vectorstore):
    """Create conversation chain with error handling"""
    try:
        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
        )
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(
            llm=llm,
            output_key="answer",
            memory_key="chat_history",
            return_messages=True,
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",
            memory=memory,
            verbose=True,
        )
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def query_resume_info(chain, question: str) -> str:
    """Query resume information with error handling"""
    try:
        response = chain.invoke({"question": question})
        return response["answer"]
    except Exception as e:
        st.error(f"Error querying resume info: {str(e)}")
        return f"Could not retrieve information: {str(e)}"

def utc_to_ist(dt: datetime) -> datetime:
    """Convert UTC datetime to IST."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=5, minutes=30)))

# Streamlit UI Configuration
st.set_page_config(
    page_title="Resume Parser Pro", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; margin: 5px 0; }
    .resume-card { 
        border: 1px solid #333; 
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 15px;
        background-color: #111 !important;
        color: #fff !important;
    }
    .resume-card h3, .resume-card p, .resume-card strong, .resume-card span {
        color: #fff !important;
    }
    .resume-card:hover { 
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .file-missing { color: #ff5555; font-weight: bold; }
    .file-ok { color: #55ff55; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .chat-message { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .chat-message.user { background-color: #e3f2fd; }
    .chat-message.assistant { background-color: #f5f5f5; }
    .error-message { color: red; }
    .phone-number { font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'selected_resume_id' not in st.session_state:
        st.session_state.selected_resume_id = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processing_resume' not in st.session_state:
        st.session_state.processing_resume = False
    if 'view_details_resume_id' not in st.session_state:
        st.session_state.view_details_resume_id = None

initialize_session_state()
verify_resume_files()

# Sidebar navigation
st.sidebar.title("📄 Resume Parser Pro")
page = st.sidebar.radio(
    "Navigation",
    ["📤 Upload Resume", "📋 Browse Resumes", "💬 Chat with Resume"],
    index=0 if st.session_state.page == 'upload' else 1 if st.session_state.page == 'browse' else 2
)

# Main content
if page == "📤 Upload Resume":
    st.session_state.page = 'upload'
    st.title("📤 Upload New Resume")
    
    # Display upload folder status
    st.sidebar.markdown(f"**Upload folder:** `{pdf_folder}`")
    st.sidebar.markdown(f"**Folder exists:** {'✅' if os.path.exists(pdf_folder) else '❌'}")
    st.sidebar.markdown(f"**Folder writable:** {'✅' if os.access(pdf_folder, os.W_OK) else '❌'}")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None and not st.session_state.processing_resume:
        st.session_state.processing_resume = True

        try:
            with st.spinner("Processing resume..."):
                # Create unique filename with timestamp and hash
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_hash = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
                file_ext = os.path.splitext(uploaded_file.name)[1]
                safe_filename = re.sub(r'[^\w.-]', '_', uploaded_file.name)
                unique_filename = f"{timestamp}_{safe_filename}"
                file_path = os.path.join(pdf_folder, unique_filename)

                # Save the file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Verify file saved successfully
                if not os.path.exists(file_path):
                    st.error("Failed to save file to uploads folder")
                    st.session_state.processing_resume = False
                else:
                    # Check for duplicates
                    if is_duplicate_resume(file_hash):
                        st.warning("This resume already exists in the database.")
                        os.remove(file_path)
                        st.session_state.processing_resume = False
                    else:
                        # Load and process the document
                        documents = load_document(file_path)
                        if not documents:
                            st.error("Failed to load document.")
                            os.remove(file_path)
                            st.session_state.processing_resume = False
                        else:
                            text = "\n".join([doc.page_content for doc in documents])

                            # Extract basic info
                            email = extract_email(text) or "Not found"
                            phone = extract_phone_number(text) or "Not found"
                            formatted_phone = format_phone_number(phone)

                            # Use LLM to extract more structured data
                            vectorstore = setup_vectorstore(documents)
                            if not vectorstore:
                                st.error("Failed to create vectorstore.")
                                os.remove(file_path)
                                st.session_state.processing_resume = False
                            else:
                                chain = create_chain(vectorstore)
                                if not chain:
                                    st.error("Failed to create conversation chain.")
                                    os.remove(file_path)
                                    st.session_state.processing_resume = False
                                else:
                                    with st.spinner("Extracting information..."):
                                        name = query_resume_info(chain, "What is the candidate's full name?") or "Unknown"
                                        skills = query_resume_info(chain, "List the candidate's technical skills and programming languages.")
                                        experience = query_resume_info(chain, "Summarize the candidate's work experience.")
                                        education = query_resume_info(chain, "Summarize the candidate's education.")

                                    # Save to database
                                    resume_id = save_resume_to_db(
                                        file_path=file_path,
                                        file_hash=file_hash,
                                        text=text,
                                        name=name,
                                        email=email,
                                        phone=phone,
                                        skills=skills,
                                        experience=experience,
                                        education=education
                                    )

                                    if resume_id:
                                        st.success("✅ Resume processed and saved successfully!")
                                        st.markdown(f"**File saved to:** `{file_path}`")

                                        # Display extracted information
                                        st.subheader("Extracted Information")
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.info(f"**Name:** {name}")
                                            st.info(f"**Email:** {email}")
                                            st.markdown(f"**Phone:** <span class='phone-number'>{formatted_phone}</span>", unsafe_allow_html=True)

                                        with col2:
                                            st.info(f"**Skills:**\n{skills}")
                                            st.info(f"**Experience:**\n{experience}")
                                            st.info(f"**Education:**\n{education}")

                                        # Add option to go to browse or chat
                                        st.markdown("---")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("📋 View All Resumes"):
                                                st.session_state.page = 'browse'
                                                st.session_state.processing_resume = False
                                                st.rerun()
                                        with col2:
                                            if st.button("💬 Chat with this Resume"):
                                                st.session_state.page = 'chat'
                                                st.session_state.selected_resume_id = resume_id
                                                st.session_state.conversation_chain = chain
                                                st.session_state.vectorstore = vectorstore
                                                st.session_state.messages = []
                                                st.session_state.processing_resume = False
                                                st.rerun()
                                    else:
                                        st.error("Failed to save resume to database.")
                                        if os.path.exists(file_path):
                                            os.remove(file_path)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        finally:
            st.session_state.processing_resume = False
elif page == "📋 Browse Resumes":
    st.session_state.page = 'browse'
    st.title("📋 Browse Resumes")
    
    # Display upload folder status
    st.sidebar.markdown(f"**Upload folder:** `{pdf_folder}`")
    st.sidebar.markdown(f"**Resumes in database:** {len(get_all_resumes())}")
    
    # Search bar
    search_query = st.text_input("Search resumes by name, skills, or keywords", 
                               value=st.session_state.search_query,
                               key="search_resumes")
    
    # Filter by date
    st.sidebar.subheader("Filters")
    date_filter = st.sidebar.selectbox(
        "Filter by upload date",
        ["All time", "Last 7 days", "Last 30 days", "Last 90 days"]
    )
    
    # Get and display resumes
    if search_query:
        resumes = search_resumes(search_query)
        st.session_state.search_query = search_query
    else:
        resumes = get_all_resumes()
    
    # Apply date filter
    if date_filter != "All time":
        days = int(date_filter.split()[1])
        cutoff_date = datetime.now() - timedelta(days=days)
        resumes = [r for r in resumes if 
                 datetime.strptime(r['uploaded_at'], '%Y-%m-%d %H:%M:%S') > cutoff_date]
    
    if not resumes:
        st.info("No resumes found. Upload a resume to get started!")
        if st.button("Upload Resume"):
            st.session_state.page = 'upload'
            st.rerun()
    else:
        st.subheader(f"Found {len(resumes)} resumes")
        
        for resume in resumes:
            name = resume.get('name', 'No Name')
            email = resume.get('email', 'No Email')
            phone = resume.get('phone', 'No Phone')
            uploaded_at = resume.get('uploaded_at', 'Unknown Date')
            resume_id = resume.get('id')
            file_exists = resume.get('file_exists', False)
            file_path = resume.get('file_path', '')

            st.markdown(f"""
            <div class='resume-card'>
                <h3>{name}</h3>
                <p><strong>Email:</strong> {email}</p>
                <p><strong>Phone:</strong> <span class='phone-number'>{phone}</span></p>
                <p><strong>Uploaded:</strong> {uploaded_at}</p>
                <p><strong>Status:</strong> 
                    <span class="{'file-ok' if file_exists else 'file-missing'}">
                        {'✅ File available' if file_exists else '❌ File missing'}
                    </span>
                </p>
                {f'<p><small>Path: {file_path}</small></p>' if not file_exists else ''}
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("View Details", key=f"view_{resume_id}"):
                    st.session_state.view_details_resume_id = resume_id
            with col2:
                if file_exists:
                    if st.button("Chat", key=f"chat_{resume_id}"):
                        st.session_state.page = 'chat'
                        st.session_state.selected_resume_id = resume_id
                        st.session_state.messages = []
                        st.session_state.conversation_chain = None
                        st.rerun()
                else:
                    st.button("Chat (Unavailable)", 
                            key=f"chat_disabled_{resume_id}", 
                            disabled=True,
                            help="Cannot chat because resume file is missing")
            with col3:
                if st.button("Delete", key=f"delete_{resume_id}"):
                    if delete_resume(resume_id):
                        st.success("Resume deleted successfully!")
                        if st.session_state.selected_resume_id == resume_id:
                            st.session_state.selected_resume_id = None
                        st.rerun()
                    else:
                        st.error("Failed to delete resume.")

            # Show details expander if this resume is selected
            if st.session_state.get('view_details_resume_id') == resume_id:
                with st.expander("📄 Extracted Resume Details", expanded=True):
                    st.markdown(f"**Name:** {name}")
                    st.markdown(f"**Email:** {email}")
                    st.markdown(f"**Phone:** <span class='phone-number'>{phone}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Uploaded:** {uploaded_at}")
                    st.markdown(f"**Skills:** {resume.get('skills', 'N/A')}")
                    st.markdown(f"**Experience:** {resume.get('experience', 'N/A')}")
                    st.markdown(f"**Education:** {resume.get('education', 'N/A')}")
                    st.markdown(f"**File Path:** `{file_path}`")
                    st.markdown("---")
                    if st.button("Close Details", key=f"close_{resume_id}"):
                        st.session_state.view_details_resume_id = None

elif page == "💬 Chat with Resume":
    st.session_state.page = 'chat'
    
    if st.session_state.get('selected_resume_id') is None:
        st.warning("Please select a resume to chat with from the Browse Resumes page.")
        if st.button("Browse Resumes"):
            st.session_state.page = 'browse'
            st.rerun()
    else:
        resume = get_resume_by_id(st.session_state.selected_resume_id)
        if not resume:
            st.error("Resume not found. It may have been deleted.")
            st.session_state.selected_resume_id = None
            st.rerun()
        
        # Check if file exists
        if not resume.get('file_exists'):
            st.error(f"Resume file not found at: {resume['file_path']}")
            st.session_state.page = 'browse'
            st.rerun()
        
        st.title(f"💬 Chat with {resume['name']}'s Resume")
        
        # Display resume summary
        with st.expander("👤 View Resume Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {resume['name']}")
                st.write(f"**Email:** {resume['email']}")
                st.markdown(f"**Phone:** <span class='phone-number'>{resume['phone']}</span>", unsafe_allow_html=True)
            with col2:
                st.write(f"**Uploaded:** {resume['uploaded_at']}")
                st.write(f"**Skills:** {resume['skills'][:100]}..." if resume['skills'] else "")
        
        # Initialize chat history from database if available
        if resume.get('chat_history'):
            try:
                chat_history = [line.split(": ", 1) for line in resume['chat_history'].split("\n") if line]
                st.session_state.messages = [
                    {"role": role.lower(), "content": content} 
                    for role, content in chat_history
                ]
            except Exception as e:
                st.warning(f"Error loading chat history: {str(e)}")
                st.session_state.messages = []
        elif 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Initialize conversation chain if not already done
        if st.session_state.conversation_chain is None:
            with st.spinner("Loading resume for chatting..."):
                try:
                    documents = load_document(resume['file_path'])
                    if not documents:
                        st.error("Failed to load resume document.")
                        st.session_state.page = 'browse'
                        st.rerun()
                    
                    vectorstore = setup_vectorstore(documents)
                    if not vectorstore:
                        st.error("Failed to create vectorstore.")
                        st.session_state.page = 'browse'
                        st.rerun()
                    
                    chain = create_chain(vectorstore)
                    if not chain:
                        st.error("Failed to create conversation chain.")
                        st.session_state.page = 'browse'
                        st.rerun()
                    
                    st.session_state.conversation_chain = chain
                    st.session_state.vectorstore = vectorstore
                except Exception as e:
                    st.error(f"Failed to load resume for chatting: {str(e)}")
                    st.session_state.page = 'browse'
                    st.rerun()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask something about this resume..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        assistant_response = response["answer"]
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        assistant_response = "Sorry, I couldn't process that request."
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Update chat history in database
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            if not update_resume(st.session_state.selected_resume_id, chat_history=chat_history):
                st.error("Failed to save chat history to database.")
        
        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            if not update_resume(st.session_state.selected_resume_id, chat_history=""):
                st.error("Failed to clear chat history in database.")
            st.rerun()

# Handle direct linking via URL parameters
query_params = st.query_params
if 'resume_id' in query_params:
    try:
        resume_id = int(query_params['resume_id'])
        st.session_state.selected_resume_id = resume_id
        st.session_state.page = 'browse'
        st.rerun()
    except (ValueError, IndexError):
        pass

elif 'chat_id' in query_params:
    try:
        resume_id = int(query_params['chat_id'])
        st.session_state.selected_resume_id = resume_id
        st.session_state.page = 'chat'
        st.rerun()
    except (ValueError, IndexError):
        pass