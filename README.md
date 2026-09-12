# 🎯 CareerMatch AI

**CareerMatch AI** is an intelligent career and job-matching web application built with Python and Streamlit. It helps job seekers identify relevant opportunities by comparing their skills, target role, preferred location, and experience with job listings.

The application combines an explainable matching engine with resume analysis, live job-search integration, AI-powered career guidance, learning recommendations, and interview preparation.

---

## ✨ Key Features

### 📄 Resume Intelligence
- Upload a resume in **PDF, DOCX, or TXT** format.
- Extract relevant skills from the resume.
- Use extracted resume skills directly for job matching.

### 💼 Intelligent Job Matching
- Match candidates with jobs using:
  - **Skills — 50%**
  - **Role — 20%**
  - **Location — 15%**
  - **Experience — 15%**
- Display an overall match score.
- Show individual score contributions for explainability.
- Identify matching skills and skill gaps.
- Analyze experience requirements.
- Detect senior/entry-level opportunities.

### 🔎 Job Search
- Search a local job dataset using `jobs.csv`.
- Search live opportunities through the **Adzuna API** when API credentials are configured.
- Rank jobs according to candidate-job compatibility.
- Validate application URLs before presenting an apply action.

### 🧠 Career Intelligence
- Provides career-oriented guidance for the selected job.
- Uses the selected job title and candidate context when generating guidance.

### 🤖 AI Career Assistance
- Integrates the **Gemini API** for AI-powered career advice.
- Includes retry/fallback handling for temporary API availability or rate-limit errors.

### 📚 Learning Plan
- Suggests learning areas based on the candidate's target career direction and skill gaps.

### 🎤 Interview Preparation
- Provides role-relevant preparation topics.
- Includes common technical and project-related interview questions.

### 👤 Candidate Profile
- Target role
- Preferred location
- Experience level
- Candidate skills
- Resume-derived skills

---

## 🧮 Matching Algorithm

CareerMatch AI uses an explainable weighted scoring model:

```text
Overall Match Score =
    0.50 × Skill Score
  + 0.20 × Role Score
  + 0.15 × Location Score
  + 0.15 × Experience Score
```

### Why these weights?

Skills receive the highest weight because technical and job-relevant capabilities are a major indicator of suitability. Role, location, and experience provide additional context.

The application also handles incomplete job information. When a listing contains limited or missing structured information, it reports reduced score reliability and applies an uncertainty adjustment rather than treating missing information as a perfect match.

This makes the result more transparent and realistic for imperfect real-world job data.

---

## 🔄 Application Workflow

```text
                    ┌─────────────────────┐
                    │   Candidate Profile │
                    │ Role / Location /   │
                    │ Experience / Skills │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Resume Analyzer   │
                    │ Extract & Normalize  │
                    │       Skills        │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │       Job Data           │
                 │                          │
                 │ Local jobs.csv / Adzuna │
                 └────────────┬─────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Job Data Processing &  │
                  │ Skill / Role / Location│
                  │ / Experience Analysis  │
                  └────────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Matching Engine    │
                    │ Skills 50%           │
                    │ Role 20%             │
                    │ Location 15%         │
                    │ Experience 15%       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ranked Job Matches   │
                    │ Score + Explanation  │
                    │ Skills + Skill Gaps  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          Career Advice   Learning Plan   Interview Prep
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application and matching logic |
| **Streamlit** | Interactive web application UI |
| **Pandas** | Job-data processing and tabular operations |
| **Requests** | API communication |
| **Google Gemini API** | AI-powered career assistance |
| **Adzuna API** | Live job-search data |
| **Regular Expressions** | Skill and experience text processing |
| **HTML/CSS** | Custom Streamlit interface styling |

---

## 📁 Project Structure

```text
CareerMatch-AI/
│
├── app.py
├── resume_parser.py
├── jobs.csv
├── test_gemini.py
├── test_resume.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File descriptions

- `app.py` — Main Streamlit application, matching engine, UI, job search, scoring, and AI integration.
- `resume_parser.py` — Resume-analysis module used by the application.
- `jobs.csv` — Local job dataset.
- `test_gemini.py` — Gemini/API-related testing.
- `test_resume.py` — Resume-analysis testing.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — Prevents local environments and secret files from being committed.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/babitha-2025/CareerMatch-AI.git
cd CareerMatch-AI
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API secrets

Create:

```text
.streamlit/secrets.toml
```

Add your API credentials:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
ADZUNA_APP_ID = "your_adzuna_app_id"
ADZUNA_APP_KEY = "your_adzuna_app_key"
```

**Never commit `secrets.toml` to GitHub.**

The repository's `.gitignore` should keep secret files out of version control.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Security

API credentials are read through Streamlit secrets rather than being hard-coded in the application.

Do not publish:

```text
.streamlit/secrets.toml
.env
API keys
passwords
private credentials
```

If an API key is accidentally exposed publicly, revoke/rotate it immediately.

---

## 📊 Example Matching

Suppose a candidate has:

```text
Skill Score       = 100%
Role Score        = 0%
Location Score    = 100%
Experience Score  = 100%
```

The base score is:

```text
100 × 0.50 = 50
  0 × 0.20 =  0
100 × 0.15 = 15
100 × 0.15 = 15
              ───
              80%
```

The application presents the component scores so the user can understand **why** a job received its final score.

---

## 🌍 Real-World Considerations

Real job listings are often incomplete or inconsistently formatted. CareerMatch AI therefore includes:

- Skill normalization
- Flexible role matching
- Location normalization
- Experience extraction
- Seniority detection
- Handling for missing job requirements
- Limited-data reliability indicators
- Uncertainty adjustments for incomplete listings
- Application URL validation

These features help the application behave more realistically than a simple keyword-matching system.

---

## 🧪 Testing

The project includes:

```text
test_gemini.py
test_resume.py
```

These files are intended for testing the AI integration and resume-analysis functionality.

For basic syntax validation of the main application:

```bash
python -m py_compile app.py
```

---

## 🚧 Current Scope

CareerMatch AI is currently a **functional real-world application prototype / MVP**.

The current implementation focuses on demonstrating the complete career-matching workflow rather than providing a production-scale recruitment platform.

Potential production enhancements include:

- User authentication
- Persistent database storage
- Persistent saved-job and application tracking
- Scalable backend APIs
- Advanced recommendation models
- Notification services
- Monitoring and logging
- Cloud deployment
- Specialized AI agents/services

---

## 🔮 Future Enhancements

- 🔐 User authentication and profiles
- 🗄️ PostgreSQL or similar persistent database
- ⭐ Persistent saved jobs
- 📌 Application status tracking
- 🔔 Job alerts and notifications
- 🤖 Specialized career AI agents
- 📈 Advanced recommendation/ranking models
- ☁️ Cloud deployment
- 📱 Responsive/mobile-friendly experience
- 📊 Analytics for job-search activity

---

## 🎓 Project Objective

The project demonstrates practical integration of:

**AI + APIs + Data Processing + Explainable Scoring + Web Application Development**

It was designed to solve a practical career-search problem while keeping the matching logic understandable to the user.

---

## 👩‍💻 Author

**Babitha M**

GitHub: [@babitha-2025](https://github.com/babitha-2025)

Project: [CareerMatch-AI](https://github.com/babitha-2025/CareerMatch-AI)

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and sharing feedback.
