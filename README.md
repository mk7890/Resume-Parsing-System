# Resume-Parsing-System

The system will include text extraction, entity recognition, data cleaning, synthetic data generation, classification, CV rating, and chatbot-based CV improvement.
Project Breakdown & Approach

## 1. Load and Process Resumes (PDF Dataset)
- Load resume PDFs from a dataset
- Extract text using pdfplumber or PyMuPDF (fitz) 
- Save extracted text to a CSV file
  
## 2. Check and Fill Missing Values
- Identify missing values in extracted fields: Applicant Name, Job Role, Email, Phone Number, Companies Worked, Work Experience, Skills, Education, Certifications, Referees 
- Generate synthetic values for missing fields using: Faker (for realistic names, emails, phone numbers) Randomized industry-relevant values for missing experience, companies, skills 
- Save the cleaned data as a new dataset
  
## 4. Named Entity Recognition (NER) Model for Feature Extraction
- Build an NLP-based Resume Parsing Model using: spaCy (NER for extracting applicant details) BERT for advanced entity recognition 
- Extract all key details ('Applicant Name', 'Job Role', 'Phone', 'Email', 'Companies Worked For',
       'Years of Work Experience', 'Skills', 'Referees', 'LinkedIn Profile',
       'Certifications', 'Education Background', 'Education Institutions')
  
  
## 5. Resume Rating (Matching with Job Description)
- Compare extracted resume skills & experience with job descriptions 
- Use TF-IDF + Cosine Similarity to compute a matching score 


