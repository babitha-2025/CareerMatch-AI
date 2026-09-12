import streamlit as st
import pandas as pd
import re
import html
import requests
import hashlib
from google import genai
from google.genai import types

from resume_parser import analyze_resume


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-weight: 700;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #172554);
    border: 1px solid #263b67;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #b8c4d9;
    font-size: 17px;
}

.card {
    background-color: #151a23;
    border: 1px solid #283142;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background-color: #151a23;
    border: 1px solid #283142;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    color: #9aa7bb;
    font-size: 14px;
}

.skill-match {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
    background-color: #123524;
    color: #8ef0b4;
    font-size: 13px;
}

.skill-gap {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
    background-color: #402021;
    color: #ff9b9b;
    font-size: 13px;
}

.info-box {
    background-color: #111827;
    border-left: 4px solid #4f8cff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.job-title {
    font-size: 21px;
    font-weight: 700;
}

.job-company {
    color: #aeb9ca;
    font-size: 14px;
}

.job-score {
    font-size: 26px;
    font-weight: 800;
}

.small-text {
    color: #9aa7bb;
    font-size: 13px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 8px;
    background-color: #1e293b;
    color: #cbd5e1;
    font-size: 12px;
    margin-right: 5px;
}

.confidence-limited {
    background-color: #3b2f12;
    border-left: 4px solid #eab308;
    padding: 14px;
    border-radius: 8px;
    margin: 12px 0;
}

.confidence-good {
    background-color: #123524;
    border-left: 4px solid #22c55e;
    padding: 14px;
    border-radius: 8px;
    margin: 12px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODELS = [
    "gemini-3.6-flash"
]

gemini_client = None
gemini_config_error = None

try:

    if "GEMINI_API_KEY" in st.secrets:

        gemini_client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

    else:

        gemini_config_error = (
            "GEMINI_API_KEY was not found in Streamlit secrets."
        )

except Exception as e:

    gemini_config_error = (
        f"Gemini configuration error: {e}"
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def ask_gemini(prompt):

    if gemini_client is None:

        return (
            None,
            gemini_config_error
        )

    errors = []

    for model_name in GEMINI_MODELS:

        for attempt in range(2):

            try:

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.4
                        )
                    )
                )

                if response and response.text:

                    return (
                        response.text.strip(),
                        None
                    )

                errors.append(
                    f"{model_name}: empty response"
                )

            except Exception as e:

                error_text = str(e)

                errors.append(
                    f"{model_name} "
                    f"(attempt {attempt + 1}): "
                    f"{error_text}"
                )

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    import time

                    time.sleep(2)

                    continue

                break

    return (
        None,
        "\n\n".join(errors)
    )


# ============================================================
# MATCHING WEIGHTS
# ============================================================

SKILL_WEIGHT = 0.50
ROLE_WEIGHT = 0.20
LOCATION_WEIGHT = 0.15
EXPERIENCE_WEIGHT = 0.15


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    "python": "python",
    "sql": "sql",

    "git": "git",
    "github": "github",

    "flask": "flask",
    "django": "django",

    "fastapi": "fastapi",
    "fast api": "fastapi",

    "mysql": "mysql",

    "postgres": "postgresql",
    "postgresql": "postgresql",

    "mongodb": "mongodb",
    "mongo db": "mongodb",

    "sqlite": "sqlite",
    "oracle": "oracle",

    "html": "html",
    "css": "css",

    "javascript": "javascript",
    "java script": "javascript",

    "react": "react",
    "react.js": "react",

    "node.js": "node.js",
    "nodejs": "node.js",

    "express": "express",
    "express.js": "express",

    "numpy": "numpy",
    "pandas": "pandas",

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",

    "scikit-learn": "scikit-learn",
    "scikit learn": "scikit-learn",

    "machine learning": "machine learning",
    "machine-learning": "machine learning",

    "deep learning": "deep learning",
    "deep-learning": "deep learning",

    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",

    "generative ai": "generative ai",
    "gen ai": "generative ai",
    "genai": "generative ai",

    "agentic ai": "agentic ai",

    "large language model": "llm",
    "large language models": "llm",
    "llm": "llm",

    "rag": "rag",
    "retrieval augmented generation": "rag",

    "langchain": "langchain",
    "langgraph": "langgraph",

    "streamlit": "streamlit",

    "data science": "data science",

    "computer vision": "computer vision",

    "natural language processing": "nlp",
    "nlp": "nlp",

    "matplotlib": "matplotlib",

    "opencv": "opencv",
    "open cv": "opencv",

    "rest": "rest api",
    "rest api": "rest api",
    "restful api": "rest api",
    "restful": "rest api",

    "json": "json",
    "jwt": "jwt",

    "authentication": "authentication",

    "api development": "api development",

    "microservices": "microservices",

    "bootstrap": "bootstrap",
    "tailwind": "tailwind",

    "pytest": "pytest",
    "py test": "pytest",

    "unit testing": "unit testing",
    "testing": "testing",

    "docker": "docker",
    "kubernetes": "kubernetes",

    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",

    "jenkins": "jenkins",

    "gitlab": "gitlab",
    "bitbucket": "bitbucket",

    "ci/cd": "ci/cd",
    "cicd": "ci/cd",

    "linux": "linux",
    "redis": "redis",
    "celery": "celery",

    "data structures and algorithms":
        "data structures and algorithms",

    "dsa": "data structures and algorithms",

    "oop": "object oriented programming",

    "object oriented programming":
        "object oriented programming",

    "excel": "excel",

    "power bi": "power bi",
    "powerbi": "power bi",

    "tableau": "tableau",

    "statistics": "statistics",

    "pyspark": "pyspark",
    "spark": "spark",
    "apache spark": "spark",

    "data warehousing": "data warehousing",

    "etl": "etl",
    "elt": "elt",

    "problem solving": "problem solving"
}


# ============================================================
# ROLE ALIASES
# ============================================================

ROLE_ALIASES = {

    "software developer": "software developer",
    "software engineer": "software engineer",
    "sde": "software engineer",

    "python developer": "python developer",
    "python engineer": "python developer",

    "backend developer": "backend developer",
    "backend engineer": "backend developer",

    "frontend developer": "frontend developer",
    "front end developer": "frontend developer",
    "frontend engineer": "frontend developer",

    "full stack developer": "full stack developer",
    "fullstack developer": "full stack developer",

    "web developer": "web developer",

    "data analyst": "data analyst",
    "data scientist": "data scientist",

    "machine learning engineer":
        "machine learning engineer",

    "ml engineer":
        "machine learning engineer",

    "ai engineer":
        "ai engineer",

    "artificial intelligence engineer":
        "ai engineer",

    "ai developer":
        "ai developer",

    "generative ai engineer":
        "generative ai engineer",

    "genai engineer":
        "generative ai engineer",

    "devops engineer":
        "devops engineer",

    "cloud engineer":
        "cloud engineer",

    "software tester":
        "software tester",

    "qa engineer":
        "qa engineer",

    "business analyst":
        "business analyst",

    "intern":
        "intern",

    "python intern":
        "python developer",

    "software engineer intern":
        "software engineer",

    "machine learning intern":
        "machine learning engineer",

    "ai intern":
        "ai engineer",

    "data science intern":
        "data scientist"
}


# ============================================================
# ROLE FAMILIES
# ============================================================

ROLE_FAMILIES = {

    "python developer": {
        "python developer",
        "backend developer",
        "software engineer",
        "software developer",
        "web developer"
    },

    "software engineer": {
        "software engineer",
        "software developer",
        "backend developer",
        "python developer"
    },

    "backend developer": {
        "backend developer",
        "python developer",
        "software engineer",
        "software developer"
    },

    "frontend developer": {
        "frontend developer",
        "web developer",
        "full stack developer"
    },

    "full stack developer": {
        "full stack developer",
        "frontend developer",
        "backend developer",
        "web developer"
    },

    "machine learning engineer": {
        "machine learning engineer",
        "ai engineer",
        "data scientist",
        "python developer"
    },

    "ai engineer": {
        "ai engineer",
        "machine learning engineer",
        "generative ai engineer",
        "data scientist"
    },

    "data scientist": {
        "data scientist",
        "machine learning engineer",
        "data analyst"
    },

    "data analyst": {
        "data analyst",
        "data scientist",
        "business analyst"
    },

    "business analyst": {
        "business analyst",
        "data analyst"
    },

    "web developer": {
        "web developer",
        "frontend developer",
        "full stack developer"
    }
}


# ============================================================
# COMMON SKILLS
# ============================================================

COMMON_SKILLS = set(
    SKILL_ALIASES.values()
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


def strip_html(text):

    text = safe_text(text)

    return re.sub(
        r"<[^>]+>",
        " ",
        text
    )


def format_skill_name(skill):

    if not skill:
        return ""

    special_names = {
        "sql": "SQL",
        "python": "Python",
        "pyspark": "PySpark",
        "power bi": "Power BI",
        "rest api": "REST API",
        "nlp": "NLP",
        "llm": "LLM",
        "rag": "RAG",
        "ai": "AI",
        "generative ai": "Generative AI",
        "agentic ai": "Agentic AI",
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "data warehousing": "Data Warehousing",
        "object oriented programming": "Object Oriented Programming",
        "data structures and algorithms":
            "Data Structures and Algorithms",
        "node.js": "Node.js",
        "fastapi": "FastAPI",
        "scikit-learn": "Scikit-learn",
        "streamlit": "Streamlit",
        "langchain": "LangChain",
        "langgraph": "LangGraph",
        "etl": "ETL",
        "elt": "ELT",
        "dsa": "DSA"
    }

    if skill.lower() in special_names:
        return special_names[skill.lower()]

    words = skill.split()

    return " ".join(
        word.capitalize()
        for word in words
    )


# ============================================================
# SKILL NORMALIZATION
# ============================================================

def normalize_skill(skill):

    skill = safe_text(
        skill
    ).lower().strip()

    skill = re.sub(
        r"[\[\]\(\){}]",
        "",
        skill
    )

    skill = re.sub(
        r"\s+",
        " ",
        skill
    ).strip()

    return SKILL_ALIASES.get(
        skill,
        skill
    )


def parse_skills(skills):

    if skills is None:
        return set()

    if isinstance(skills, set):

        values = skills

    elif isinstance(skills, list):

        values = skills

    else:

        skills = safe_text(
            skills
        )

        if not skills:
            return set()

        values = re.split(
            r"[,;|/\n]+",
            skills
        )

    result = set()

    for skill in values:

        normalized = normalize_skill(
            skill
        )

        if normalized:
            result.add(normalized)

    return result


# ============================================================
# EXTRACT SKILLS FROM TEXT
# ============================================================

def extract_skills_from_text(text):

    text = strip_html(
        text
    ).lower()

    if not text:
        return set()

    found_skills = set()

    sorted_skills = sorted(
        COMMON_SKILLS,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        escaped = re.escape(
            skill
        )

        if re.search(
            rf"(?<!\w){escaped}(?!\w)",
            text
        ):

            found_skills.add(
                skill
            )

    return found_skills


# ============================================================
# ROLE NORMALIZATION
# ============================================================

def normalize_role(role):

    role = safe_text(
        role
    ).lower()

    role = re.sub(
        r"[^a-z0-9+#.\s-]",
        " ",
        role
    )

    role = re.sub(
        r"\s+",
        " ",
        role
    ).strip()

    role_without_level = re.sub(
        r"\b("
        r"senior|sr|junior|jr|lead|principal|"
        r"associate|manager|director|head|trainee"
        r")\b",
        "",
        role
    )

    role_without_level = re.sub(
        r"\s+",
        " ",
        role_without_level
    ).strip()

    if role_without_level in ROLE_ALIASES:

        return ROLE_ALIASES[
            role_without_level
        ]

    return role_without_level


# ============================================================
# EXPERIENCE EXTRACTION
# ============================================================

def extract_experience_range(text):

    text = strip_html(
        text
    ).lower()

    if not text:
        return 0, 0

    patterns = [

        (
            r"\b(\d+)\s*[-–]\s*(\d+)\s*"
            r"(?:years?\s*)?yoe\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\b(\d+)\s+to\s+(\d+)\s*"
            r"(?:years?\s*)?yoe\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\b(\d+)\s*\+\s*"
            r"(?:years?\s*)?yoe\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\b(\d+)\s*(?:years?\s*)?yoe\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1))
            )
        ),

        (
            r"\bexperience\s*[:\-]?\s*"
            r"(\d+)\s*[-–]\s*(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\bexperience\s*[:\-]?\s*"
            r"(\d+)\s+to\s+(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\bexperience\s*[:\-]?\s*"
            r"(\d+)\s*\+\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\b(\d+)\s*[-–]\s*(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\b(\d+)\s+to\s+(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(2))
            )
        ),

        (
            r"\b(\d+)\s*\+\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\bminimum\s+(?:of\s+)?"
            r"(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\bat\s+least\s+"
            r"(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\brequires?\s+"
            r"(\d+)\s*\+?\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1)) + 3
            )
        ),

        (
            r"\b(\d+)\s*years?\s+"
            r"(?:of\s+)?experience\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1))
            )
        ),

        (
            r"\bexperience\s*[:\-]?\s*"
            r"(\d+)\s*years?\b",
            lambda m: (
                int(m.group(1)),
                int(m.group(1))
            )
        )
    ]

    for pattern, converter in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            minimum, maximum = converter(
                match
            )

            return (
                minimum,
                max(
                    minimum,
                    maximum
                )
            )

    if (
        "fresher" in text
        or "freshers" in text
    ):

        return 0, 0

    return 0, 0


# ============================================================
# EXPERIENCE TO YEARS
# ============================================================

def experience_to_years(experience):

    experience = safe_text(
        experience
    ).lower()

    if not experience:
        return 0

    if (
        "fresher" in experience
        or "intern" in experience
    ):

        return 0

    match = re.search(
        r"(\d+)",
        experience
    )

    if match:
        return int(
            match.group(1)
        )

    return 0


# ============================================================
# EXPERIENCE DISPLAY
# ============================================================

def format_job_experience(
    job_experience,
    title,
    description
):

    minimum, maximum = (
        extract_experience_range(
            job_experience
        )
    )

    source = "Job Requirement"

    if (
        minimum == 0
        and maximum == 0
        and "fresher" in safe_text(
            job_experience
        ).lower()
    ):

        return (
            "Fresher",
            0,
            source
        )

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                title
            )
        )

        source = "Job Title"

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                description
            )
        )

        source = "Job Description"

    if (
        minimum == 0
        and maximum == 0
    ):

        return (
            "Not specified",
            None,
            "Not specified"
        )

    if minimum == maximum:

        if minimum == 0:
            requirement = "0 years"

        else:

            requirement = (
                f"{minimum} year"
                if minimum == 1
                else f"{minimum} years"
            )

    else:

        requirement = (
            f"{minimum}-{maximum} years"
        )

    return (
        requirement,
        minimum,
        source
    )


# ============================================================
# EXPERIENCE GAP
# ============================================================

def calculate_experience_gap(
    candidate_experience,
    minimum_required
):

    candidate_years = experience_to_years(
        candidate_experience
    )

    if minimum_required is None:

        return "Not available"

    if candidate_years >= minimum_required:

        return "None"

    gap = (
        minimum_required
        - candidate_years
    )

    if gap == 1:

        return "1 year"

    return f"{gap} years"


# ============================================================
# SENIORITY DETECTION
# ============================================================

def is_senior_job(
    title,
    job_experience="",
    description="",
    job_level=""
):

    title = safe_text(
        title
    ).lower()

    job_experience = safe_text(
        job_experience
    ).lower()

    description = strip_html(
        description
    ).lower()

    job_level = safe_text(
        job_level
    ).lower()

    title_level_text = " ".join([
        title,
        job_level
    ])

    senior_patterns = [

        r"\bassociate\s+manager\b",
        r"\bengineering\s+manager\b",
        r"\btechnical\s+manager\b",
        r"\bproject\s+manager\b",
        r"\bproduct\s+manager\b",
        r"\bprogram\s+manager\b",
        r"\bpeople\s+manager\b",
        r"\bmanager\b",
        r"\bsenior\b",
        r"\bsr\.?\b",
        r"\blead\b",
        r"\bprincipal\b",
        r"\barchitect\b",
        r"\bdirector\b",
        r"\bhead\b",
        r"\bvice\s+president\b",
        r"\bvp\b"
    ]

    if any(
        re.search(
            pattern,
            title_level_text
        )
        for pattern in senior_patterns
    ):

        return True

    minimum, maximum = (
        extract_experience_range(
            job_experience
        )
    )

    if minimum >= 4:
        return True

    title_minimum, title_maximum = (
        extract_experience_range(
            title
        )
    )

    if title_minimum >= 4:
        return True

    if description:

        description_minimum, description_maximum = (
            extract_experience_range(
                description
            )
        )

        if description_minimum >= 4:
            return True

    senior_description_patterns = [

        r"\bassociate\s+manager\b",
        r"\bengineering\s+manager\b",
        r"\btechnical\s+manager\b",
        r"\bproject\s+manager\b",
        r"\bproduct\s+manager\b",
        r"\bprogram\s+manager\b",
        r"\bteam\s+lead\b",
        r"\btechnical\s+lead\b",
        r"\bsenior\s+(?:developer|engineer|analyst|scientist)\b",
        r"\blead\s+(?:developer|engineer|analyst|scientist)\b",
        r"\bprincipal\s+(?:developer|engineer|analyst|scientist)\b"
    ]

    if any(
        re.search(
            pattern,
            description
        )
        for pattern in senior_description_patterns
    ):

        return True

    return False


# ============================================================
# ENTRY LEVEL
# ============================================================

def is_entry_level_job(
    title,
    job_experience="",
    description="",
    job_level=""
):

    title = safe_text(
        title
    ).lower()

    job_experience = safe_text(
        job_experience
    ).lower()

    description = strip_html(
        description
    ).lower()

    job_level = safe_text(
        job_level
    ).lower()

    if is_senior_job(
        title,
        job_experience,
        description,
        job_level
    ):

        return False

    combined_text = " ".join([
        title,
        job_experience,
        job_level,
        description
    ])

    entry_patterns = [

        r"\bfresher\b",
        r"\bfreshers\b",
        r"\bentry[- ]level\b",
        r"\bgraduate\b",
        r"\bgraduates\b",
        r"\btrainee\b",
        r"\bintern\b",
        r"\binternship\b",
        r"\bjunior\b",
        r"\b0\s*[-–]\s*1\s*years?\b",
        r"\b0\s+to\s+1\s*years?\b",
        r"\b0\s*\+\s*years?\b",
        r"\b0\s*[-–]\s*2\s*years?\b",
        r"\b0\s+to\s+2\s*years?\b"
    ]

    if any(
        re.search(
            pattern,
            combined_text
        )
        for pattern in entry_patterns
    ):

        return True

    minimum, maximum = (
        extract_experience_range(
            job_experience
        )
    )

    if (
        job_experience
        and minimum == 0
        and maximum <= 2
    ):

        return True

    return False


# ============================================================
# ROLE SCORE
# ============================================================

def calculate_role_score(
    desired_role,
    job_title
):

    target = normalize_role(
        desired_role
    )

    job_role = normalize_role(
        job_title
    )

    if not target or not job_role:
        return 0

    if target == job_role:
        return 100

    related_roles = ROLE_FAMILIES.get(
        target,
        set()
    )

    if job_role in related_roles:
        return 85

    target_words = set(
        target.split()
    )

    job_words = set(
        job_role.split()
    )

    common_words = (
        target_words.intersection(
            job_words
        )
    )

    if len(common_words) >= 2:
        return 70

    if len(common_words) == 1:
        return 45

    return 0


# ============================================================
# SKILL MATCHES
# ============================================================

def get_skill_matches(
    candidate_skills,
    job_skills
):

    candidate = parse_skills(
        candidate_skills
    )

    required = parse_skills(
        job_skills
    )

    matched_skills = sorted(
        candidate.intersection(
            required
        )
    )

    missing_skills = sorted(
        required - candidate
    )

    return (
        matched_skills,
        missing_skills
    )


# ============================================================
# SKILL SCORE
# ============================================================

def calculate_skill_score(
    candidate_skills,
    job_skills
):

    candidate = parse_skills(
        candidate_skills
    )

    required = parse_skills(
        job_skills
    )

    if not required:
        return 50

    matched = candidate.intersection(
        required
    )

    return (
        len(matched)
        / len(required)
    ) * 100


# ============================================================
# JOB SKILL CONFIDENCE
# ============================================================

def get_job_skill_confidence(job_skills):

    required = parse_skills(
        job_skills
    )

    skill_count = len(
        required
    )

    if skill_count == 0:
        return "No Skill Data"

    if skill_count <= 1:
        return "Limited"

    if skill_count <= 3:
        return "Moderate"

    return "Good"


# ============================================================
# DATA PROTECTION
# ============================================================

def apply_job_data_protection(
    overall_score,
    job_skills
):

    """
    Reduce confidence when a listing has too little technical
    skill data, without forcing every limited-data job to the
    same artificial score.

    This keeps ranking differences meaningful while still
    preventing sparse listings from looking overly certain.
    """

    confidence = get_job_skill_confidence(
        job_skills
    )

    if confidence == "Limited":
        # One detected skill is useful, but not enough for a
        # high-confidence comparison. Apply a modest penalty.
        return max(0, overall_score - 5)

    if confidence == "No Skill Data":
        # No technical requirements were detected, so apply a
        # stronger uncertainty penalty rather than a hard cap.
        return max(0, overall_score - 10)

    return overall_score


# ============================================================
# LOCATION
# ============================================================

def normalize_location(location):

    location = safe_text(
        location
    ).lower()

    location = location.replace(
        "bengaluru",
        "bangalore"
    )

    location = location.replace(
        "blr",
        "bangalore"
    )

    location = location.replace(
        "mysore",
        "mysuru"
    )

    location = re.sub(
        r"\bka\b",
        "karnataka",
        location
    )

    location = re.sub(
        r"\s+",
        " ",
        location
    ).strip()

    return location


def calculate_location_score(
    candidate_location,
    job_location
):

    candidate = normalize_location(
        candidate_location
    )

    job = normalize_location(
        job_location
    )

    if not candidate or not job:
        return 0

    if candidate == job:
        return 100

    if (
        candidate in job
        or job in candidate
    ):

        return 100

    candidate_words = set(
        candidate.split()
    )

    job_words = set(
        job.split()
    )

    generic_words = {
        "india",
        "karnataka",
        "state",
        "district",
        "city"
    }

    candidate_words -= generic_words
    job_words -= generic_words

    common = (
        candidate_words.intersection(
            job_words
        )
    )

    if common:
        return 80

    return 0


# ============================================================
# EXPERIENCE SCORE
# ============================================================

def calculate_experience_score(
    candidate_experience,
    job_experience,
    job_title="",
    description="",
    job_level=""
):

    candidate_years = experience_to_years(
        candidate_experience
    )

    job_experience_text = safe_text(
        job_experience
    ).lower()

    if (
        "fresher" in job_experience_text
        or "freshers" in job_experience_text
    ):

        if candidate_years == 0:
            return 100

        return 80

    minimum, maximum = (
        extract_experience_range(
            job_experience
        )
    )

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                job_title
            )
        )

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                description
            )
        )

    if (
        minimum == 0
        and maximum == 0
    ):

        if is_entry_level_job(
            job_title,
            job_experience,
            description,
            job_level
        ):

            if candidate_years <= 1:
                return 100

            return 80

        return 60

    if candidate_years >= minimum:

        if candidate_years <= maximum:
            return 100

        return 90

    gap = minimum - candidate_years

    if gap == 1:
        return 70

    if gap == 2:
        return 45

    if gap == 3:
        return 25

    return 5


# ============================================================
# SENIORITY PENALTY
# ============================================================

def calculate_seniority_penalty(
    candidate_experience,
    job_title,
    job_experience="",
    description="",
    job_level=""
):

    candidate_years = experience_to_years(
        candidate_experience
    )

    if candidate_years > 2:
        return 0

    description = strip_html(
        description
    ).lower()

    title = safe_text(
        job_title
    ).lower()

    job_experience = safe_text(
        job_experience
    ).lower()

    job_level = safe_text(
        job_level
    ).lower()

    penalty = 0

    if is_senior_job(
        title,
        job_experience,
        description,
        job_level
    ):

        penalty += 30

    minimum, maximum = (
        extract_experience_range(
            job_experience
        )
    )

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                title
            )
        )

    if (
        minimum == 0
        and maximum == 0
    ):

        minimum, maximum = (
            extract_experience_range(
                description
            )
        )

    if minimum >= 6:

        penalty += 30

    elif minimum >= 4:

        penalty += 25

    elif minimum >= 3:

        penalty += 15

    gap = minimum - candidate_years

    if gap >= 6:

        penalty += 10

    elif gap >= 4:

        penalty += 5

    return min(
        penalty,
        60
    )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    skill_score,
    role_score,
    location_score,
    experience_score,
    seniority_penalty=0
):

    score = (
        skill_score * SKILL_WEIGHT
        + role_score * ROLE_WEIGHT
        + location_score * LOCATION_WEIGHT
        + experience_score * EXPERIENCE_WEIGHT
    )

    score -= seniority_penalty

    score = max(
        0,
        min(
            100,
            round(score)
        )
    )

    if seniority_penalty >= 60:

        score = min(
            score,
            35
        )

    elif seniority_penalty >= 50:

        score = min(
            score,
            40
        )

    elif seniority_penalty >= 40:

        score = min(
            score,
            45
        )

    elif seniority_penalty >= 30:

        score = min(
            score,
            50
        )

    return score


# ============================================================
# READINESS
# ============================================================

def get_readiness(
    score,
    missing_skills,
    job_skill_confidence="Good"
):

    high_priority = {
        "python",
        "sql",
        "machine learning",
        "data structures and algorithms",
        "javascript",
        "java"
    }

    important_gaps = [
        skill
        for skill in missing_skills
        if skill in high_priority
    ]

    if job_skill_confidence == "No Skill Data":

        if score >= 60:
            return "Potential Match — Limited Data"

        return "Low Match"

    if job_skill_confidence == "Limited":

        if score >= 75:
            return "Strong Match — Limited Data"

        if score >= 60:
            return "Good Potential — Limited Data"

    if (
        score >= 85
        and not important_gaps
    ):

        return "Excellent Match"

    if score >= 75:
        return "Strong Match"

    if score >= 60:
        return "Good Potential"

    if score >= 45:
        return "Needs Improvement"

    return "Low Match"


# ============================================================
# LOAD JOBS
# ============================================================

@st.cache_data
def load_jobs():

    try:

        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parent

        jobs_path = BASE_DIR / "jobs.csv"

        jobs = pd.read_csv(
            jobs_path
        )

        # Debug informatio

    except Exception as e:

        return pd.DataFrame()

    jobs.columns = [
        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )
        for column in jobs.columns
    ]

    rename_map = {}

    if (
        "job_title" in jobs.columns
        and "title" not in jobs.columns
    ):

        rename_map["job_title"] = "title"

    if (
        "position" in jobs.columns
        and "title" not in jobs.columns
    ):

        rename_map["position"] = "title"

    if (
        "company_name" in jobs.columns
        and "company" not in jobs.columns
    ):

        rename_map["company_name"] = "company"

    if (
        "job_location" in jobs.columns
        and "location" not in jobs.columns
    ):

        rename_map["job_location"] = "location"

    jobs = jobs.rename(
        columns=rename_map
    )

    required_columns = [
        "title",
        "company",
        "location",
        "skills",
        "experience",
        "description",
        "apply_url",
        "job_level"
    ]

    for column in required_columns:

        if column not in jobs.columns:

            jobs[column] = ""

    if (
        "url" in jobs.columns
        and jobs["apply_url"].eq("").all()
    ):

        jobs["apply_url"] = jobs["url"]

    jobs = jobs.fillna("")

    return jobs


# ============================================================
# PREPARE JOB SKILLS
# ============================================================

def prepare_job_skills(row):

    existing_skills = safe_text(
        row.get(
            "skills",
            ""
        )
    )

    if existing_skills:

        return ", ".join(
            sorted(
                parse_skills(
                    existing_skills
                )
            )
        )

    description = safe_text(
        row.get(
            "description",
            ""
        )
    )

    title = safe_text(
        row.get(
            "title",
            ""
        )
    )

    detected_skills = (
        extract_skills_from_text(
            description
        )
        |
        extract_skills_from_text(
            title
        )
    )

    return ", ".join(
        sorted(
            detected_skills
        )
    )


# ============================================================
# ADZUNA LIVE SEARCH
# ============================================================

def search_adzuna_jobs(
    role,
    location,
    max_results=20
):

    app_id = st.secrets.get(
        "ADZUNA_APP_ID",
        ""
    )

    app_key = st.secrets.get(
        "ADZUNA_APP_KEY",
        ""
    )

    if not app_id or not app_key:

        return pd.DataFrame()

    url = (
        "https://api.adzuna.com/v1/api/jobs/in/search/1"
    )

    params = {

        "app_id": app_id,

        "app_key": app_key,

        "results_per_page": max_results,

        "what": role,

        "where": location,

        "content-type": "application/json"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code != 200:

            return pd.DataFrame()

        data = response.json()

        results = data.get(
            "results",
            []
        )

        rows = []

        for job in results:

            company = job.get(
                "company",
                {}
            )

            location_data = job.get(
                "location",
                {}
            )

            rows.append({

                "title":
                    job.get(
                        "title",
                        ""
                    ),

                "company":
                    company.get(
                        "display_name",
                        ""
                    ),

                "location":
                    location_data.get(
                        "display_name",
                        ""
                    ),

                "description":
                    job.get(
                        "description",
                        ""
                    ),

                "skills":
                    "",

                "experience":
                    "",

                "job_level":
                    "",

                "apply_url":
                    job.get(
                        "redirect_url",
                        ""
                    )
            })

        return pd.DataFrame(
            rows
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOCAL CAREER ADVICE
# ============================================================

def generate_local_career_advice(
    score,
    job_title,
    matched_skills,
    missing_skills
):

    if score >= 85:

        level = (
            "You are very well aligned with this role."
        )

    elif score >= 75:

        level = (
            "You have a strong foundation for this role."
        )

    elif score >= 60:

        level = (
            "You have good potential, but some targeted "
            "preparation is needed."
        )

    elif score >= 45:

        level = (
            "You have some relevant skills, but should "
            "strengthen your preparation."
        )

    else:

        level = (
            "This job is currently not a strong match."
        )

    advice = [level]

    if matched_skills:

        skills = ", ".join(
            format_skill_name(skill)
            for skill in matched_skills[:6]
        )

        advice.append(
            f"Your matching skills include {skills}."
        )

    if missing_skills:

        skills = ", ".join(
            format_skill_name(skill)
            for skill in missing_skills[:5]
        )

        advice.append(
            f"Prioritize learning {skills}."
        )

    advice.append(
        f"Continue building projects and skills relevant "
        f"to {job_title} positions."
    )

    return " ".join(advice)


# ============================================================
# RESUME SKILLS
# ============================================================

def get_resume_skills(resume_data):

    if not resume_data:
        return set()

    if not isinstance(
        resume_data,
        dict
    ):

        return set()

    skills = resume_data.get(
        "skills",
        []
    )

    return parse_skills(
        skills
    )

# ============================================================
# REAL-WORLD JOB HELPERS
# ============================================================

def get_job_key(row):

    raw = "|".join([
        safe_text(row.get("title", "")),
        safe_text(row.get("company", "")),
        safe_text(row.get("location", "")),
        safe_text(row.get("apply_url", ""))
    ]).lower()

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def html_text(value):

    return html.escape(
        safe_text(value),
        quote=True
    )


def valid_url(url):

    url = safe_text(url)

    return (
        url.startswith("http://")
        or url.startswith("https://")
    )


def get_application_status(job_key):

    return st.session_state.application_status.get(
        job_key,
        "Not Applied"
    )


def set_application_status(
    job_key,
    status
):

    st.session_state.application_status[
        job_key
    ] = status


def toggle_saved_job(job_key):

    if job_key in st.session_state.saved_jobs:

        st.session_state.saved_jobs.remove(
            job_key
        )

    else:

        st.session_state.saved_jobs.add(
            job_key
        )


def calculate_profile_completion(
    desired_role,
    candidate_location,
    candidate_experience,
    candidate_skills,
    resume_data
):

    completed = 0
    total = 5

    if safe_text(desired_role):
        completed += 1

    if safe_text(candidate_location):
        completed += 1

    if safe_text(candidate_experience):
        completed += 1

    if candidate_skills:
        completed += 1

    if resume_data:
        completed += 1

    return round(
        completed / total * 100
    )


def calculate_career_readiness(
    top_match_score,
    candidate_skills,
    resume_data,
    interview_progress,
    profile_completion
):

    skill_count = len(
        candidate_skills
    )

    if skill_count >= 10:
        skill_strength = 100
    elif skill_count >= 7:
        skill_strength = 85
    elif skill_count >= 5:
        skill_strength = 70
    elif skill_count >= 3:
        skill_strength = 55
    else:
        skill_strength = 35

    resume_score = (
        100
        if resume_data
        else 40
    )

    readiness = (
        top_match_score * 0.45
        + skill_strength * 0.20
        + resume_score * 0.15
        + interview_progress * 0.10
        + profile_completion * 0.10
    )

    return round(
        max(
            0,
            min(
                100,
                readiness
            )
        )
    )


# ============================================================
# SESSION STATE
# ============================================================

if "resume_data" not in st.session_state:
    st.session_state.resume_data = None

if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = set()

if "jobs" not in st.session_state:
    st.session_state.jobs = None

if "top_job" not in st.session_state:
    st.session_state.top_job = None

if "saved_jobs" not in st.session_state:
    st.session_state.saved_jobs = set()

if "application_status" not in st.session_state:
    st.session_state.application_status = {}

if "learning_progress" not in st.session_state:
    st.session_state.learning_progress = {}

if "interview_progress" not in st.session_state:
    st.session_state.interview_progress = {}

if "ai_advice_cache" not in st.session_state:
    st.session_state.ai_advice_cache = {}

if "job_source" not in st.session_state:
    st.session_state.job_source = "Local Jobs"

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🎯 CareerMatch AI
</div>

<div class="hero-subtitle">
AI-powered career matching that compares your
skills, target role, location and experience
with relevant job opportunities.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎯 Career Profile"
)

desired_role = st.sidebar.text_input(
    "Target Role",
    value="Python Developer"
)

candidate_location = st.sidebar.text_input(
    "Preferred Location",
    value="Bengaluru"
)

candidate_experience = st.sidebar.selectbox(
    "Experience",
    [
        "Fresher",
        "0-1 years",
        "1-2 years",
        "2-3 years",
        "3-5 years",
        "5+ years"
    ]
)


# ============================================================
# RESUME / MANUAL SKILLS
# ============================================================

resume_skills = st.session_state.resume_skills

if resume_skills:

    st.sidebar.success(
        f"Using {len(resume_skills)} skills from your resume."
    )

    use_resume_skills = st.sidebar.checkbox(
        "Use resume skills",
        value=True
    )

else:

    use_resume_skills = False


manual_skills_default = (
    "Python, DSA, Machine Learning, "
    "TensorFlow, SQL, HTML, CSS, "
    "Streamlit, Git"
)

candidate_skills_text = st.sidebar.text_area(
    "Your Skills",
    value=manual_skills_default
)


if use_resume_skills:

    candidate_skills = resume_skills

else:

    candidate_skills = parse_skills(
        candidate_skills_text
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Dashboard",
    "📄 Resume Analyzer",
    "💼 Job Search",
    "🧠 Career Intelligence",
    "📚 Learning Plan",
    "🎤 Interview Prep",
    "👤 Profile"
])


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    st.subheader(
        "Your Career Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {len(candidate_skills)}
                </div>
                <div class="metric-label">
                    Skills
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    50%
                </div>
                <div class="metric-label">
                    Skill Weight
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {html_text(desired_role)}
                </div>
                <div class="metric-label">
                    Target Role
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {html_text(candidate_location)}
                </div>
                <div class="metric-label">
                    Preferred Location
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader(
        "How CareerMatch AI Scores You"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

        <h3>Matching Formula</h3>

        <p>Skills → <b>50%</b></p>
        <p>Role → <b>20%</b></p>
        <p>Location → <b>15%</b></p>
        <p>Experience → <b>15%</b></p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h3>What You Get</h3>

        <p>✓ Job match score</p>
        <p>✓ Matching skills</p>
        <p>✓ Missing skills</p>
        <p>✓ Experience analysis</p>
        <p>✓ Career readiness</p>
        <p>✓ Personalized learning direction</p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RESUME ANALYZER
# ============================================================

with tabs[1]:

    st.subheader(
        "📄 Resume Intelligence"
    )

    uploaded_resume = st.file_uploader(
        "Upload your resume",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_resume:

        if st.button(
            "Analyze Resume",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Analyzing resume..."
                ):

                    result = analyze_resume(
                        uploaded_resume
                    )

                st.session_state.resume_data = result

                extracted_skills = (
                    get_resume_skills(
                        result
                    )
                )

                st.session_state.resume_skills = (
                    extracted_skills
                )

                st.success(
                    f"Resume analyzed successfully. "
                    f"{len(extracted_skills)} skills detected."
                )

            except Exception as e:

                st.error(
                    f"Resume analysis failed: {e}"
                )

    if st.session_state.resume_data:

        resume_data = (
            st.session_state.resume_data
        )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Resume Analysis"
        )

        if isinstance(
            resume_data,
            dict
        ):

            if resume_data.get(
                "skills"
            ):

                st.write(
                    "**Detected Skills:**"
                )

                skills = parse_skills(
                    resume_data["skills"]
                )

                html_content = ""

                for skill in sorted(
                    skills
                ):

                    html_content += (
                        '<span class="skill-match">'
                        f'{format_skill_name(skill)}'
                        '</span>'
                    )

                st.markdown(
                    html_content,
                    unsafe_allow_html=True
                )

            for key, value in resume_data.items():

                if key == "skills":
                    continue

                st.write(
                    f"**{key.capitalize()}:** {value}"
                )

        else:

            st.write(
                resume_data
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# JOB SEARCH
# ============================================================

with tabs[2]:

    st.subheader(
        "💼 Job Search & Matching"
    )

    local_jobs = load_jobs()

    job_source = st.radio(
        "Choose Job Source",
        [
            "Local Jobs",
            "Live Jobs"
        ],
        horizontal=True
    )


    # ========================================================
    # JOB SOURCE
    # ========================================================

    if job_source == "Live Jobs":

        if st.button(
            "🔎 Search Live Jobs",
            type="primary"
        ):

            # Clear previous live results so a failed/empty search
            # can never leave stale jobs displayed as current results.
            st.session_state.jobs = None

            with st.spinner(
                "Searching for jobs..."
            ):

                live_jobs = (
                    search_adzuna_jobs(
                        desired_role,
                        candidate_location,
                        20
                    )
                )

            if live_jobs.empty:

                st.warning(
                    "No live jobs found. "
                    "Check your Adzuna API keys "
                    "or try another search."
                )

            else:

                st.session_state.jobs = live_jobs

                st.success(
                    f"{len(live_jobs)} jobs found."
                )

    else:

        st.session_state.jobs = local_jobs


    jobs = st.session_state.jobs


    # ========================================================
    # NO JOBS
    # ========================================================

    if jobs is None or jobs.empty:

        st.info(
            "No jobs available yet."
        )

    else:

        jobs = jobs.copy()


        # ====================================================
        # STANDARDIZE COLUMNS
        # ====================================================

        for column in [
            "title",
            "company",
            "location",
            "skills",
            "experience",
            "description",
            "apply_url",
            "job_level"
        ]:

            if column not in jobs.columns:

                jobs[column] = ""

            jobs[column] = (
                jobs[column]
                .fillna("")
                .astype(str)
            )


        # ====================================================
        # PREPARE SKILLS
        # ====================================================

        jobs["skills"] = jobs.apply(
            prepare_job_skills,
            axis=1
        )


        # ====================================================
        # SENIORITY
        # ====================================================

        jobs["senior_job"] = jobs.apply(

            lambda row:
            is_senior_job(
                row["title"],
                row["experience"],
                row["description"],
                row["job_level"]
            ),

            axis=1
        )


        # ====================================================
        # ENTRY LEVEL
        # ====================================================

        jobs["entry_level"] = jobs.apply(

            lambda row:
            is_entry_level_job(
                row["title"],
                row["experience"],
                row["description"],
                row["job_level"]
            ),

            axis=1
        )


        # ====================================================
        # SKILL SCORE
        # ====================================================

        jobs["skill_score"] = jobs.apply(

            lambda row:
            calculate_skill_score(
                candidate_skills,
                row["skills"]
            ),

            axis=1
        )


        # ====================================================
        # ROLE SCORE
        # ====================================================

        jobs["role_score"] = jobs.apply(

            lambda row:
            calculate_role_score(
                desired_role,
                row["title"]
            ),

            axis=1
        )


        # ====================================================
        # LOCATION SCORE
        # ====================================================

        jobs["location_score"] = jobs.apply(

            lambda row:
            calculate_location_score(
                candidate_location,
                row["location"]
            ),

            axis=1
        )


        # ====================================================
        # EXPERIENCE SCORE
        # ====================================================

        jobs["experience_score"] = jobs.apply(

            lambda row:
            calculate_experience_score(
                candidate_experience,
                row["experience"],
                row["title"],
                row["description"],
                row["job_level"]
            ),

            axis=1
        )


        # ====================================================
        # SENIORITY PENALTY
        # ====================================================

        jobs["seniority_penalty"] = jobs.apply(

            lambda row:
            calculate_seniority_penalty(
                candidate_experience,
                row["title"],
                row["experience"],
                row["description"],
                row["job_level"]
            ),

            axis=1
        )


        # ====================================================
        # OVERALL SCORE
        # ====================================================

        jobs["overall_score"] = jobs.apply(

            lambda row:
            calculate_overall_score(
                row["skill_score"],
                row["role_score"],
                row["location_score"],
                row["experience_score"],
                row["seniority_penalty"]
            ),

            axis=1
        )


        # ====================================================
        # JOB DATA CONFIDENCE
        # ====================================================

        jobs["job_skill_confidence"] = jobs[
            "skills"
        ].apply(
            get_job_skill_confidence
        )


        # ====================================================
        # SCORE PROTECTION
        # ====================================================

        jobs["overall_score"] = jobs.apply(

            lambda row:
            apply_job_data_protection(
                row["overall_score"],
                row["skills"]
            ),

            axis=1
        )


        jobs["match_score"] = (
            jobs["overall_score"]
            .astype(int)
        )


        # ====================================================
        # FILTERS
        # ====================================================

        st.markdown("---")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            minimum_score = st.slider(
                "Minimum Match Score",
                0,
                100,
                0
            )

        with filter_col2:

            prefer_entry = st.checkbox(
                "Prefer entry-level jobs",
                value=True
            )


        # ====================================================
        # SCORE FILTER
        # ====================================================

        filtered_jobs = jobs[
            jobs["match_score"]
            >= minimum_score
        ].copy()


        # ====================================================
        # ENTRY PRIORITY
        # ====================================================

        if prefer_entry:

            filtered_jobs["entry_priority"] = (
                filtered_jobs[
                    "entry_level"
                ].astype(int)
            )

        else:

            filtered_jobs["entry_priority"] = 0


        # ====================================================
        # FRESHER PROTECTION
        # ====================================================

        if (
            prefer_entry
            and
            "fresher"
            in candidate_experience.lower()
        ):

            non_senior = filtered_jobs[
                ~filtered_jobs[
                    "senior_job"
                ]
            ]

            if not non_senior.empty:

                filtered_jobs = (
                    non_senior.copy()
                )


        # ====================================================
        # FINAL RANKING
        # ====================================================

        if not filtered_jobs.empty:

            filtered_jobs = (
                filtered_jobs
                .sort_values(
                    [
                        "entry_priority",
                        "senior_job",
                        "match_score",
                        "skill_score",
                        "role_score",
                        "experience_score"
                    ],
                    ascending=[
                        False,
                        True,
                        False,
                        False,
                        False,
                        False
                    ]
                )
            )


        # ====================================================
        # TOP JOB
        # ====================================================

        if filtered_jobs.empty:

            st.warning(
                "No jobs match your current filters."
            )

            st.session_state.top_job = None

        else:

            top_job = filtered_jobs.iloc[0]

            st.session_state.top_job = top_job

            top_score = int(
                top_job["match_score"]
            )


            # =================================================
            # SKILL MATCHES
            # =================================================

            matched_skills, missing_skills = (
                get_skill_matches(
                    candidate_skills,
                    top_job["skills"]
                )
            )


            job_skill_set = parse_skills(
                top_job["skills"]
            )


            job_skill_confidence = (
                get_job_skill_confidence(
                    top_job["skills"]
                )
            )


            readiness = get_readiness(
                top_score,
                missing_skills,
                job_skill_confidence
            )


            # =================================================
            # EXPERIENCE ANALYSIS
            # =================================================

            (
                job_experience_text,
                minimum_required,
                experience_source
            ) = format_job_experience(
                top_job["experience"],
                top_job["title"],
                top_job["description"]
            )

            experience_gap = (
                calculate_experience_gap(
                    candidate_experience,
                    minimum_required
                )
            )


            # =================================================
            # TOP MATCH
            # =================================================

            st.subheader(
                "🏆 Top Career Match"
            )

            st.markdown(
                f"""
                <div class="card">

                <div class="job-title">
                {html_text(top_job["title"])}
                </div>

                <div class="job-company">
                {html_text(top_job["company"])}
                •
                {html_text(top_job["location"])}
                </div>

                <br>

                <div class="job-score">
                {top_score}% Match
                </div>

                <p>
                <b>{readiness}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            if job_skill_confidence == "Limited":

                st.markdown(
                    """
                    <div class="confidence-limited">

                    <b>⚠️ Limited job-skill data</b><br><br>

                    Only a small number of technical
                    requirements could be detected from
                    this listing.

                    A small uncertainty adjustment is applied
                    because only one technical requirement was
                    detected. The score is not artificially capped.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif job_skill_confidence == "No Skill Data":

                st.markdown(
                    """
                    <div class="confidence-limited">

                    <b>⚠️ Limited job information</b><br><br>

                    This listing does not contain enough
                    identifiable technical requirements
                    for a high-confidence skill comparison.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif job_skill_confidence == "Good":

                st.markdown(
                    """
                    <div class="confidence-good">

                    <b>✓ Good skill data</b><br><br>

                    Multiple technical requirements were
                    detected from the job listing.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # MATCH BREAKDOWN
            # =================================================

            st.subheader(
                "Match Breakdown"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Skills",
                    f"{int(top_job['skill_score'])}%"
                )

            with c2:

                st.metric(
                    "Role",
                    f"{int(top_job['role_score'])}%"
                )

            with c3:

                st.metric(
                    "Location",
                    f"{int(top_job['location_score'])}%"
                )

            with c4:

                st.metric(
                    "Experience",
                    f"{int(top_job['experience_score'])}%"
                )


            # =================================================
            # EXPERIENCE ANALYSIS
            # =================================================

            st.subheader(
                "📊 Experience Analysis"
            )

            e1, e2, e3 = st.columns(3)

            with e1:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <div class="metric-label">
                    Your Experience
                    </div>

                    <div class="metric-value">
                    {candidate_experience}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with e2:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <div class="metric-label">
                    Job Requirement
                    </div>

                    <div class="metric-value">
                    {job_experience_text}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with e3:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <div class="metric-label">
                    Experience Gap
                    </div>

                    <div class="metric-value">
                    {experience_gap}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if (
                minimum_required is None
                and experience_source == "Not specified"
            ):

                st.caption(
                    "No experience requirement detected from the listing."
                )

            elif experience_source == "Job Description":

                st.caption(
                    "Experience requirement detected from "
                    "the job description."
                )

            elif experience_source == "Job Title":

                st.caption(
                    "Experience requirement detected from "
                    "the job title."
                )

            if (
                experience_gap == "None"
                and minimum_required is not None
            ):

                st.success(
                    "Your current experience meets the "
                    "detected minimum requirement."
                )

            elif (
                experience_gap != "None"
                and experience_gap != "Not available"
            ):

                st.warning(
                    f"You are approximately {experience_gap} "
                    "below the detected minimum experience."
                )


            # =================================================
            # SENIORITY WARNING
            # =================================================

            if (
                top_job["seniority_penalty"] > 0
            ):

                st.warning(
                    "This position appears more senior than "
                    "your current experience level, so the "
                    "match score was reduced."
                )


            # =================================================
            # REQUIRED SKILLS
            # =================================================

            st.subheader(
                "Required Skills Detected"
            )

            if job_skill_set:

                html_content = ""

                for skill in sorted(
                    job_skill_set
                ):

                    html_content += (
                        '<span class="badge">'
                        f'{format_skill_name(skill)}'
                        '</span>'
                    )

                st.markdown(
                    html_content,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "No reliable technical skills could be "
                    "detected from this listing."
                )


            # =================================================
            # MATCHING SKILLS
            # =================================================

            st.subheader(
                "✅ Matching Skills"
            )

            if matched_skills:

                html_content = ""

                for skill in matched_skills:

                    html_content += (
                        '<span class="skill-match">'
                        f'{format_skill_name(skill)}'
                        '</span>'
                    )

                st.markdown(
                    html_content,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "No matching skills detected."
                )


            # =================================================
            # SKILL GAPS
            # =================================================

            st.subheader(
                "⚠️ Skill Gaps"
            )

            if not job_skill_set:

                st.info(
                    "No reliable job skills could be detected "
                    "from this job listing."
                )

            elif missing_skills:

                html_content = ""

                for skill in missing_skills:

                    html_content += (
                        '<span class="skill-gap">'
                        f'{format_skill_name(skill)}'
                        '</span>'
                    )

                st.markdown(
                    html_content,
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "You match all detected required skills."
                )


            # =================================================
            # WHY MATCH
            # =================================================

            st.subheader(
                "💡 Why This Job Matches"
            )

            skill_contribution = (
                top_job["skill_score"]
                * SKILL_WEIGHT
            )

            role_contribution = (
                top_job["role_score"]
                * ROLE_WEIGHT
            )

            location_contribution = (
                top_job["location_score"]
                * LOCATION_WEIGHT
            )

            experience_contribution = (
                top_job["experience_score"]
                * EXPERIENCE_WEIGHT
            )

            st.markdown(
                f"""
                <div class="info-box">

                <b>Skills contribution:</b>
                {int(top_job["skill_score"])}%
                × 50%
                = {skill_contribution:.0f} points

                <br><br>

                <b>Role contribution:</b>
                {int(top_job["role_score"])}%
                × 20%
                = {role_contribution:.0f} points

                <br><br>

                <b>Location contribution:</b>
                {int(top_job["location_score"])}%
                × 15%
                = {location_contribution:.0f} points

                <br><br>

                <b>Experience contribution:</b>
                {int(top_job["experience_score"])}%
                × 15%
                = {experience_contribution:.0f} points

                <br><br>

                <b>Seniority adjustment:</b>
                -{int(top_job["seniority_penalty"])} points

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # SCORE RELIABILITY
            # =================================================

            if job_skill_confidence == "Limited":

                st.caption(
                    "Score reliability: Limited — "
                    "the job listing contains very few "
                    "detectable technical requirements."
                )

            elif job_skill_confidence == "No Skill Data":

                st.caption(
                    "Score reliability: Low — "
                    "the job listing does not provide "
                    "enough technical skill information."
                )

            elif job_skill_confidence == "Moderate":

                st.caption(
                    "Score reliability: Moderate — "
                    "some technical requirements were detected."
                )

            else:

                st.caption(
                    "Score reliability: Good — "
                    "multiple technical requirements were detected."
                )


            # =================================================
            # JOB DESCRIPTION
            # =================================================

            description = safe_text(
                top_job["description"]
            )

            if description:

                st.subheader(
                    "Job Description"
                )

                st.write(
                    description
                )


            # =================================================
            # APPLY
            # =================================================

            apply_url = safe_text(
                top_job["apply_url"]
            )

            if valid_url(apply_url):

                st.link_button(
                    "Apply for this job",
                    apply_url
                )

            elif apply_url:

                st.caption("Application link is unavailable or invalid.")


            # =================================================
            # LOCAL CAREER ADVISOR
            # =================================================

            st.subheader(
                "🧠 Career Advisor"
            )

            local_advice = (
                generate_local_career_advice(
                    top_score,
                    safe_text(top_job["title"]),
                    matched_skills,
                    missing_skills
                )
            )

            st.markdown(
                f"""
                <div class="info-box">
                {html_text(local_advice)}
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # GEMINI AI CAREER ADVICE
            # =================================================

            st.subheader(
                "✨ AI Career Advice"
            )

            if gemini_config_error:

                st.error(
                    f"Gemini configuration error: "
                    f"{gemini_config_error}"
                )

            elif gemini_client:

                if st.button(
                    "✨ Generate AI Career Advice",
                    key="generate_ai_advice"
                ):

                    prompt = f"""
You are a professional career advisor.

Candidate target role:
{desired_role}

Candidate location:
{candidate_location}

Candidate experience:
{candidate_experience}

Candidate skills:
{", ".join(sorted(candidate_skills))}

Job title:
{html_text(top_job["title"])}

Company:
{html_text(top_job["company"])}

Job location:
{html_text(top_job["location"])}

Match score:
{top_score}%

Job skill data confidence:
{job_skill_confidence}

Detected job skills:
{", ".join(sorted(job_skill_set))}

Matched skills:
{", ".join(matched_skills)}

Missing skills:
{", ".join(missing_skills)}

Detected experience requirement:
{job_experience_text}

Experience gap:
{experience_gap}

Give practical career advice in simple English.

Use these exact sections:

## 1. Reality Check

Explain whether this is a realistic application target.

## 2. Why This Job Matches

Explain the strongest matching factors.

## 3. Confirmed Skill Gaps

Only mention skills that were actually detected from
the job listing and are missing from the candidate.

If there are none, clearly say:
"No confirmed skill gaps based on the available job data."

## 4. Experience Gap

Explain whether the candidate meets the detected
experience requirement.

## 5. What You Should Learn Next

Give general industry recommendations for the target role.

IMPORTANT:
These are general recommendations and must NOT be presented
as confirmed requirements of this particular job.

## 6. Interview Preparation

Give practical interview preparation topics.

## 7. One Practical Project Idea

Suggest one project relevant to the target role.

## 8. Short Action Plan

Give 4 practical next steps.

Important rules:

- If job skill data confidence is Limited or No Skill Data,
  clearly explain that the listing does not provide enough
  technical requirements for a complete comparison.
- Do not treat general industry skills as confirmed job requirements.
- Do not invent company information.
- Do not invent job requirements.
- Do not claim the candidate has skills not listed.
- Keep the language simple and realistic.
- Do not use HTML.
- Do not use code fences.
"""

                    with st.spinner(
                        "Generating AI career advice..."
                    ):

                        ai_advice, ai_error = (
                            ask_gemini(
                                prompt
                            )
                        )

                    if ai_advice:

                        st.markdown(
                            ai_advice
                        )

                    else:

                        st.error(
                            "AI advice could not be generated."
                        )

                        if ai_error:

                            with st.expander(
                                "View Gemini error"
                            ):

                                st.code(
                                    ai_error,
                                    language="text"
                                )

            else:

                st.warning(
                    "Gemini AI is not configured."
                )


            # =================================================
            # OTHER JOBS
            # =================================================

            st.subheader(
                "💼 Other Recommended Jobs"
            )

            other_jobs = (
                filtered_jobs.iloc[1:6]
            )

            if other_jobs.empty:

                st.info(
                    "No additional matching jobs."
                )

            else:

                for _, job in (
                    other_jobs.iterrows()
                ):

                    other_matched, other_missing = (
                        get_skill_matches(
                            candidate_skills,
                            job["skills"]
                        )
                    )

                    other_confidence = (
                        get_job_skill_confidence(
                            job["skills"]
                        )
                    )

                    confidence_text = ""

                    if other_confidence == "Limited":

                        confidence_text = (
                            " • Limited job data"
                        )

                    elif other_confidence == "No Skill Data":

                        confidence_text = (
                            " • Limited skill information"
                        )

                    st.markdown(
                        f"""
                        <div class="card">

                        <div class="job-title">
                        {html_text(job["title"])}
                        </div>

                        <div class="job-company">
                        {html_text(job["company"])}
                        •
                        {html_text(job["location"])}
                        </div>

                        <br>

                        <b>
                        {int(job["match_score"])}% Match
                        </b>

                        <span class="small-text">
                        {confidence_text}
                        </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if other_matched:

                        matched_display = ", ".join(
                            format_skill_name(skill)
                            for skill in other_matched[:5]
                        )

                        st.caption(
                            f"Matching: {matched_display}"
                        )

                    elif other_confidence == "No Skill Data":

                        st.caption(
                            "No reliable job skills detected."
                        )

                    job_apply_url = safe_text(
                        job["apply_url"]
                    )

                    if valid_url(job_apply_url):

                        st.link_button(
                            "View / Apply",
                            job_apply_url
                        )


# ============================================================
# CAREER INTELLIGENCE
# ============================================================

with tabs[3]:

    st.subheader(
        "🧠 Career Intelligence"
    )

    st.markdown(
        f"""
        <div class="card">

        <h3>Your Target</h3>

        <p>
        <b>Role:</b> {html_text(desired_role)}
        </p>

        <p>
        <b>Location:</b> {html_text(candidate_location)}
        </p>

        <p>
        <b>Experience:</b> {html_text(candidate_experience)}
        </p>

        <p>
        <b>Skills:</b> {len(candidate_skills)}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Career Direction"
    )

    role = normalize_role(
        desired_role
    )

    if "machine learning" in role:

        direction = """
        Focus on Python, NumPy, Pandas, SQL,
        machine learning algorithms, TensorFlow
        or PyTorch, model evaluation and deployment.
        """

    elif "data scientist" in role:

        direction = """
        Focus on Python, statistics, SQL,
        Pandas, visualization, machine learning
        and practical data projects.
        """

    elif "data analyst" in role:

        direction = """
        Focus on SQL, Excel, Python, Pandas,
        Power BI/Tableau and data storytelling.
        """

    elif "business analyst" in role:

        direction = """
        Focus on SQL, Excel, Power BI,
        requirements analysis, statistics,
        communication and business problem solving.
        """

    elif "frontend" in role:

        direction = """
        Focus on HTML, CSS, JavaScript,
        React and responsive web development.
        """

    elif "backend" in role:

        direction = """
        Focus on Python or Java, APIs, databases,
        backend frameworks, Git and deployment.
        """

    elif "python" in role:

        direction = """
        Focus on Python, DSA, SQL, REST APIs,
        Flask/Django, Git and practical backend projects.
        """

    else:

        direction = """
        Focus on DSA, programming fundamentals,
        Git, SQL, web development and practical projects.
        """

    st.markdown(
        f"""
        <div class="info-box">
        {direction}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Recommended Skill Priorities"
    )

    priority_skills = [
        "Data Structures and Algorithms",
        "Python",
        "SQL",
        "Git",
        "Problem Solving"
    ]

    if (
        "ai" in role
        or "machine learning" in role
    ):

        priority_skills.extend([
            "Machine Learning",
            "TensorFlow",
            "Generative AI"
        ])

    elif "data" in role:

        priority_skills.extend([
            "Pandas",
            "NumPy",
            "Data Visualization"
        ])

    elif "business analyst" in role:

        priority_skills.extend([
            "Excel",
            "Power BI",
            "Statistics"
        ])

    elif (
        "web" in role
        or "frontend" in role
        or "backend" in role
    ):

        priority_skills.extend([
            "JavaScript",
            "REST API",
            "Databases"
        ])

    for index, skill in enumerate(
        priority_skills,
        1
    ):

        st.write(
            f"**{index}. {skill}**"
        )


# ============================================================
# LEARNING PLAN
# ============================================================

with tabs[4]:

    st.subheader(
        "📚 Personalized Learning Plan"
    )

    st.write(
        "Build your skills progressively based on your target role."
    )

    current_skills = parse_skills(
        candidate_skills
    )


    def learning_skill_is_known(skill):

        skill_lower = safe_text(
            skill
        ).lower()

        normalized = normalize_skill(
            skill_lower
        )

        if "python" in skill_lower:

            return "python" in current_skills

        if "object-oriented" in skill_lower:

            return (
                "object oriented programming"
                in current_skills
            )

        if "data structures" in skill_lower:

            return (
                "data structures and algorithms"
                in current_skills
            )

        if "git" in skill_lower:

            return (
                "git" in current_skills
                or
                "github" in current_skills
            )

        if "sql" in skill_lower:

            return "sql" in current_skills

        if "rest api" in skill_lower:

            return "rest api" in current_skills

        if "database" in skill_lower:

            return (
                "mysql" in current_skills
                or
                "postgresql" in current_skills
                or
                "mongodb" in current_skills
            )

        if "project" in skill_lower:

            return True

        if "framework" in skill_lower:

            return (
                "flask" in current_skills
                or
                "django" in current_skills
                or
                "fastapi" in current_skills
                or
                "react" in current_skills
                or
                "node.js" in current_skills
            )

        return normalized in current_skills


    foundation_skills = [
        "Python fundamentals",
        "Object-oriented programming",
        "Data Structures and Algorithms",
        "Git and GitHub"
    ]

    technical_skills = [
        "SQL",
        "REST APIs",
        "Databases",
        "Role-specific frameworks"
    ]


    st.markdown(
        """
        <div class="card">
        <h3>Phase 1 — Foundation</h3>
        """,
        unsafe_allow_html=True
    )

    for skill in foundation_skills:

        status = (
            "✓"
            if learning_skill_is_known(skill)
            else "→"
        )

        st.write(
            f"{status} {skill}"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">
        <h3>Phase 2 — Technical Skills</h3>
        """,
        unsafe_allow_html=True
    )

    for skill in technical_skills:

        status = (
            "✓"
            if learning_skill_is_known(skill)
            else "→"
        )

        st.write(
            f"{status} {skill}"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <h3>Phase 3 — Projects</h3>

        <p>✓ Build 2–3 practical projects</p>
        <p>✓ Upload projects to GitHub</p>
        <p>✓ Write clear README files</p>
        <p>✓ Deploy at least one project</p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <h3>Phase 4 — Placement Preparation</h3>

        <p>✓ DSA practice</p>
        <p>✓ Aptitude</p>
        <p>✓ Technical interviews</p>
        <p>✓ HR interviews</p>
        <p>✓ Resume preparation</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INTERVIEW PREPARATION
# ============================================================

with tabs[5]:

    st.subheader(
        "🎤 Interview Preparation"
    )

    role = normalize_role(
        desired_role
    )

    st.markdown(
        "### Core Topics"
    )

    topics = [
        "Data Structures",
        "Algorithms",
        "Object-Oriented Programming",
        "DBMS",
        "SQL",
        "Operating Systems",
        "Computer Networks",
        "Git and GitHub"
    ]

    if (
        "python" in role
        or "machine learning" in role
        or "ai" in role
    ):

        topics.extend([
            "Python",
            "Machine Learning",
            "Model Evaluation",
            "NumPy",
            "Pandas",
            "TensorFlow / PyTorch"
        ])

    if (
        "web" in role
        or "frontend" in role
        or "backend" in role
    ):

        topics.extend([
            "HTML",
            "CSS",
            "JavaScript",
            "REST APIs"
        ])

    if "data analyst" in role:

        topics.extend([
            "Excel",
            "Power BI",
            "Statistics",
            "Data Visualization"
        ])

    if "business analyst" in role:

        topics.extend([
            "Excel",
            "Power BI",
            "Requirements Analysis",
            "Statistics"
        ])

    for topic in topics:

        st.checkbox(
            topic,
            key=f"interview_{topic}"
        )

    st.markdown(
        "### Common Questions"
    )

    questions = [
        "Tell me about yourself.",
        "Explain your strongest project.",
        "Why did you choose this technology?",
        "What is your strongest programming language?",
        "Explain OOP concepts.",
        "What is the difference between an array and a linked list?",
        "What is a database?",
        "What is SQL?",
        "Explain time complexity.",
        "What challenges did you face in your project?"
    ]

    for question in questions:

        with st.expander(
            question
        ):

            st.write(
                "Prepare a simple answer using "
                "your own project and experience."
            )


# ============================================================
# PROFILE
# ============================================================

with tabs[6]:

    st.subheader(
        "👤 Candidate Profile"
    )

    st.markdown(
        f"""
        <div class="card">

        <h3>Career Profile</h3>

        <p>
        <b>Target Role:</b>
        {html_text(desired_role)}
        </p>

        <p>
        <b>Preferred Location:</b>
        {html_text(candidate_location)}
        </p>

        <p>
        <b>Experience:</b>
        {html_text(candidate_experience)}
        </p>

        <p>
        <b>Total Skills:</b>
        {len(candidate_skills)}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Your Skills"
    )

    if candidate_skills:

        html_content = ""

        for skill in sorted(
            candidate_skills
        ):

            html_content += (
                '<span class="skill-match">'
                f'{format_skill_name(skill)}'
                '</span>'
            )

        st.markdown(
            html_content,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Add your skills from the sidebar "
            "or upload your resume."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7f8ba3;
        padding:10px;
    ">
    CareerMatch AI • Intelligent Career & Job Matching
    </div>
    """,
    unsafe_allow_html=True
)
