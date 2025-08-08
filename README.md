# PolicyParse - Document Processing System

A secure document processing system with user authentication, vector database storage, and semantic search capabilities.

## Features

- User Authentication: Secure login/registration with PostgreSQL
- Document Management: Add, list, and remove PDF documents
- Vector Database: Semantic search using LanceDB
- URL Support: Add documents from local files or URLs
- Document Types: Categorize documents (policy, contract, email, claim)
- Access Control: Private/public document access levels

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up PostgreSQL database:
```bash
# Create database
createdb policyparse_users
```

3. Configure environment variables:
```bash
# Copy the environment template
cp env_template.txt .env

# Edit .env file with your PostgreSQL credentials
nano .env
```

The .env file should contain:
```
DB_HOST=localhost
DB_NAME=policyparse_users
DB_USER=postgres
DB_PASSWORD=your_actual_password
DB_PORT=5432
```

## Usage

Run the main application:
```bash
python main.py
```

### User Registration/Login
- Register with a unique User ID and password
- Login with your credentials
- All data is isolated per user

### Document Management
1. Add Document: Upload PDF from local file or URL
2. List Documents: View all your uploaded documents
3. Remove Document: Delete documents you no longer need
4. Query Documents: Search your documents with natural language

### Document Types
- policy: Insurance policies and coverage documents
- contract: Legal contracts and agreements
- email: Email communications
- claim: Insurance claims and related documents

## Security

- Passwords are hashed with PBKDF2 and salt
- Users can only access their own documents
- Document IDs are UUID-based for security
- No cross-user data access

## File Structure

```
PolicyParse/
├── main.py              # Main application
├── user_auth.py         # User authentication system
├── vector_db.py         # Vector database operations
├── docreader.py         # PDF document reader
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── case1.pdf           # Sample document
```

## Database Schema

### Users Table
- id: Primary key
- user_id: Unique user identifier
- password_hash: Hashed password
- salt: Password salt
- created_at: Registration timestamp
- last_login: Last login timestamp
- is_active: Account status

### User Documents Table
- id: Primary key
- user_id: Foreign key to users
- document_id: Vector database document ID
- filename: Original filename
- doc_type: Document type
- access_level: Access level (private/public)
- added_at: Upload timestamp

## Vector Database

- Uses LanceDB for efficient vector storage
- Sentence transformers for semantic embeddings
- Intelligent text chunking for optimal search
- User-isolated document storage

## Environment Variables

Optional PostgreSQL configuration:
- DB_HOST: Database host (default: localhost)
- DB_NAME: Database name (default: policyparse_users)
- DB_USER: Database user (default: postgres)
- DB_PASSWORD: Database password
- DB_PORT: Database port (default: 5432) 