import re
import os
from io import BytesIO

from pypdf import PdfReader
import fitz
import pytesseract
from PIL import Image


# --------------------------------------------------
# TESSERACT CONFIGURATION
# --------------------------------------------------

TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

for path in TESSERACT_CANDIDATES:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break


# --------------------------------------------------
# KNOWN SKILLS
# --------------------------------------------------

SKILLS = [
    "Python",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "HTML",
    "CSS",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",

    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "Data Analytics",
    "Artificial Intelligence",
    "Generative AI",
    "NLP",
    "Computer Vision",

    "TensorFlow",
    "Keras",
    "PyTorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "OpenCV",

    "Power BI",
    "Tableau",
    "Excel",

    "Django",
    "Flask",
    "React",
    "Node.js",
    "Express",
    "REST API",

    "Git",
    "GitHub",
    "Docker",
    "Kubernetes",

    "DSA",
    "OOP",

    "AWS",
    "Azure",
    "Google Cloud",
    "GCP",
]


# --------------------------------------------------
# EXTRACT TEXT FROM NORMAL PDF
# --------------------------------------------------

def extract_pdf_text(pdf_file):
    """
    Extract text from a normal text-based PDF.
    """

    if hasattr(pdf_file, "getvalue"):
        pdf_bytes = pdf_file.getvalue()
    else:
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip(), pdf_bytes


# --------------------------------------------------
# OCR FOR SCANNED / IMAGE PDF
# --------------------------------------------------

def extract_text_with_ocr(pdf_bytes):
    """
    Convert PDF pages into images and use Tesseract OCR.
    """

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        extracted_text = []

        for page in document:

            # Render page at higher resolution
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image_bytes = pix.tobytes("png")

            image = Image.open(
                BytesIO(image_bytes)
            )

            # OCR
            text = pytesseract.image_to_string(image)

            if text.strip():
                extracted_text.append(text)

        document.close()

        return "\n".join(extracted_text).strip()

    except Exception:
        return ""


# --------------------------------------------------
# MAIN TEXT EXTRACTION
# --------------------------------------------------

def extract_resume_text(pdf_file):
    """
    Automatically chooses normal PDF extraction
    or OCR for scanned PDFs.
    """

    text, pdf_bytes = extract_pdf_text(pdf_file)

    # Normal text PDF
    if len(text.strip()) >= 50:
        return text, "PDF Text"

    # Scanned/image PDF
    ocr_text = extract_text_with_ocr(pdf_bytes)

    return ocr_text, "OCR"


# --------------------------------------------------
# SKILL EXTRACTION
# --------------------------------------------------

def extract_skills(text):
    """
    Detect known technical skills from resume text.
    """

    if not text:
        return []

    found_skills = []

    text_lower = text.lower()

    for skill in SKILLS:

        skill_lower = skill.lower()

        # Escape special characters such as C++, C#
        escaped_skill = re.escape(skill_lower)

        # Match complete skill names
        pattern = rf"(?<!\w){escaped_skill}(?!\w)"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


# --------------------------------------------------
# EDUCATION EXTRACTION
# --------------------------------------------------

def extract_education(text):
    """
    Detect highest/first recognizable education qualification.
    """

    if not text:
        return "Not detected"

    text_lower = text.lower()

    education_patterns = [

        # Engineering
        (r"\bb\.?\s*e\.?\b", "BE"),
        (r"\bbachelor\s+of\s+engineering\b", "BE"),

        # Technology
        (r"\bb\.?\s*tech\.?\b", "B.Tech"),
        (r"\bbachelor\s+of\s+technology\b", "B.Tech"),

        # Masters Engineering
        (r"\bm\.?\s*e\.?\b", "ME"),
        (r"\bmaster\s+of\s+engineering\b", "ME"),

        # Masters Technology
        (r"\bm\.?\s*tech\.?\b", "M.Tech"),
        (r"\bmaster\s+of\s+technology\b", "M.Tech"),

        # BCA
        (r"\bb\.?\s*c\.?\s*a\.?\b", "BCA"),
        (r"\bbachelor\s+of\s+computer\s+applications\b", "BCA"),

        # MCA
        (r"\bm\.?\s*c\.?\s*a\.?\b", "MCA"),
        (r"\bmaster\s+of\s+computer\s+applications\b", "MCA"),

        # Other common qualifications
        (r"\bbachelor\s+of\s+science\b", "B.Sc"),
        (r"\bb\.?\s*sc\.?\b", "B.Sc"),

        (r"\bmaster\s+of\s+science\b", "M.Sc"),
        (r"\bm\.?\s*sc\.?\b", "M.Sc"),

        (r"\bdiploma\b", "Diploma"),
    ]

    for pattern, result in education_patterns:

        if re.search(pattern, text_lower):
            return result

    return "Not detected"


# --------------------------------------------------
# EXPERIENCE HELPER
# --------------------------------------------------

def convert_years_to_category(years):
    """
    Convert numeric experience into the categories
    used by CareerMatch AI.
    """

    if years <= 0:
        return "Fresher"

    if years <= 1:
        return "0-1 years"

    if years == 2:
        return "2-3 years"

    if years <= 5:
        return "3-5 years"

    return "5+ years"


# --------------------------------------------------
# EXPERIENCE EXTRACTION
# --------------------------------------------------

def extract_experience(text):
    """
    Detect work experience from resume text.
    """

    if not text:
        return "Fresher"

    text_lower = text.lower()

    # ----------------------------------------------
    # Fresher
    # ----------------------------------------------

    fresher_patterns = [
        "fresher",
        "fresh graduate",
        "recent graduate",
        "no experience",
        "entry level",
        "entry-level",
    ]

    for pattern in fresher_patterns:

        if pattern in text_lower:
            return "Fresher"

    # ----------------------------------------------
    # Explicit 0-1 year experience
    # ----------------------------------------------

    zero_one_patterns = [
        r"\b0\s*[-–]\s*1\s*years?\b",
        r"\bless\s+than\s+1\s+year\b",
        r"\bunder\s+1\s+year\b",
    ]

    for pattern in zero_one_patterns:

        if re.search(pattern, text_lower):
            return "0-1 years"

    # ----------------------------------------------
    # Experience ranges
    # ----------------------------------------------

    range_patterns = [

        # 1-2 years
        r"\b(\d+)\s*[-–]\s*(\d+)\s*years?\b",

        # 1 to 2 years
        r"\b(\d+)\s+to\s+(\d+)\s*years?\b",
    ]

    for pattern in range_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            start = int(match.group(1))
            end = int(match.group(2))

            if start == 0:
                return "0-1 years"

            if start == 1 and end <= 2:
                return "1-2 years"

            if start == 2 and end <= 3:
                return "2-3 years"

            if start >= 3 and end <= 5:
                return "3-5 years"

            if start >= 5:
                return "5+ years"

    # ----------------------------------------------
    # X+ years of experience
    # ----------------------------------------------

    plus_patterns = [

        r"\b(\d+)\s*\+\s*years?\s+of\s+experience\b",

        r"\b(\d+)\s*\+\s*years?\s+experience\b",

        r"\b(\d+)\s*\+\s*years?\b",
    ]

    for pattern in plus_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            years = int(match.group(1))

            return convert_years_to_category(years)

    # ----------------------------------------------
    # "X years of experience"
    # ----------------------------------------------

    experience_patterns = [

        r"\b(\d+)\s+years?\s+of\s+experience\b",

        r"\bexperience\s*[:\-]?\s*(\d+)\s*years?\b",

        r"\btotal\s+experience\s*[:\-]?\s*(\d+)\s*years?\b",
    ]

    for pattern in experience_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            years = int(match.group(1))

            return convert_years_to_category(years)

    # ----------------------------------------------
    # Internship
    # ----------------------------------------------

    if (
        "internship" in text_lower
        or "intern" in text_lower
    ):
        return "0-1 years"

    # ----------------------------------------------
    # Default
    # ----------------------------------------------

    return "Fresher"


# --------------------------------------------------
# ROLE EXTRACTION
# --------------------------------------------------

def extract_role(text):
    """
    Detect the candidate's most likely career role.
    """

    if not text:
        return "Software Engineer"

    text_lower = text.lower()

    # ----------------------------------------------
    # Explicit role names
    # ----------------------------------------------

    explicit_roles = [

        (
            "machine learning engineer",
            "Machine Learning Engineer"
        ),

        (
            "machine learning developer",
            "Machine Learning Engineer"
        ),

        (
            "ml engineer",
            "Machine Learning Engineer"
        ),

        (
            "ml developer",
            "Machine Learning Engineer"
        ),

        (
            "data scientist",
            "Data Scientist"
        ),

        (
            "data analyst",
            "Data Analyst"
        ),

        (
            "business analyst",
            "Business Analyst"
        ),

        (
            "python developer",
            "Python Developer"
        ),

        (
            "python engineer",
            "Python Developer"
        ),

        (
            "backend developer",
            "Backend Developer"
        ),

        (
            "backend engineer",
            "Backend Developer"
        ),

        (
            "frontend developer",
            "Frontend Developer"
        ),

        (
            "frontend engineer",
            "Frontend Developer"
        ),

        (
            "front end developer",
            "Frontend Developer"
        ),

        (
            "full stack developer",
            "Full Stack Developer"
        ),

        (
            "full-stack developer",
            "Full Stack Developer"
        ),

        (
            "fullstack developer",
            "Full Stack Developer"
        ),

        (
            "web developer",
            "Web Developer"
        ),

        (
            "software engineer",
            "Software Engineer"
        ),

        (
            "software developer",
            "Software Engineer"
        ),

        (
            "ai engineer",
            "AI Engineer"
        ),

        (
            "artificial intelligence engineer",
            "AI Engineer"
        ),

        (
            "ai developer",
            "AI Developer"
        ),

        (
            "generative ai engineer",
            "Generative AI Engineer"
        ),

        (
            "genai engineer",
            "Generative AI Engineer"
        ),

        (
            "devops engineer",
            "DevOps Engineer"
        ),

        (
            "cloud engineer",
            "Cloud Engineer"
        ),

        (
            "software tester",
            "Software Tester"
        ),

        (
            "qa engineer",
            "QA Engineer"
        ),

        (
            "data science intern",
            "Data Scientist"
        ),

        (
            "machine learning intern",
            "Machine Learning Engineer"
        ),

        (
            "ml intern",
            "Machine Learning Engineer"
        ),

        (
            "ai intern",
            "AI Engineer"
        ),

        (
            "python intern",
            "Python Developer"
        ),

        (
            "software engineer intern",
            "Software Engineer"
        ),
    ]

    for phrase, role in explicit_roles:

        if phrase in text_lower:
            return role

    # ----------------------------------------------
    # Role from resume headings/objective
    # ----------------------------------------------

    role_patterns = [

        r"(?:target role|desired role|preferred role|career goal|position)\s*[:\-]\s*([a-zA-Z ]+)",

        r"(?:seeking|looking for|applying for)\s+(?:a\s+)?([a-zA-Z ]+(?:engineer|developer|analyst|scientist|intern))",
    ]

    for pattern in role_patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            role_text = match.group(1).strip()

            if "machine learning" in role_text:
                return "Machine Learning Engineer"

            if "data scientist" in role_text:
                return "Data Scientist"

            if "data analyst" in role_text:
                return "Data Analyst"

            if "python developer" in role_text:
                return "Python Developer"

            if "software engineer" in role_text:
                return "Software Engineer"

            if "ai engineer" in role_text:
                return "AI Engineer"

    # ----------------------------------------------
    # Infer role from skills
    # ----------------------------------------------

    ml_score = 0
    data_science_score = 0
    analyst_score = 0
    python_score = 0
    ai_score = 0
    web_score = 0

    # Machine Learning
    if "machine learning" in text_lower:
        ml_score += 3

    if "tensorflow" in text_lower:
        ml_score += 2

    if "pytorch" in text_lower:
        ml_score += 2

    if "deep learning" in text_lower:
        ml_score += 2

    if "scikit-learn" in text_lower:
        ml_score += 1

    # Data Science
    if "data science" in text_lower:
        data_science_score += 3

    if "pandas" in text_lower:
        data_science_score += 1

    if "numpy" in text_lower:
        data_science_score += 1

    if "matplotlib" in text_lower:
        data_science_score += 1

    # Data Analyst
    if "power bi" in text_lower:
        analyst_score += 3

    if "tableau" in text_lower:
        analyst_score += 3

    if "excel" in text_lower:
        analyst_score += 2

    if "data analytics" in text_lower:
        analyst_score += 2

    # Python Developer
    if "python" in text_lower:
        python_score += 2

    if "django" in text_lower:
        python_score += 2

    if "flask" in text_lower:
        python_score += 2

    # AI
    if "artificial intelligence" in text_lower:
        ai_score += 3

    if "generative ai" in text_lower:
        ai_score += 3

    if "computer vision" in text_lower:
        ai_score += 2

    if "nlp" in text_lower:
        ai_score += 2

    # Web Development
    if "html" in text_lower:
        web_score += 1

    if "css" in text_lower:
        web_score += 1

    if "javascript" in text_lower:
        web_score += 2

    if "react" in text_lower:
        web_score += 2

    if "node.js" in text_lower:
        web_score += 2

    # ----------------------------------------------
    # Select highest score
    # ----------------------------------------------

    scores = {
        "Machine Learning Engineer": ml_score,
        "Data Scientist": data_science_score,
        "Data Analyst": analyst_score,
        "Python Developer": python_score,
        "AI Engineer": ai_score,
        "Web Developer": web_score,
    }

    best_role = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_role]

    if best_score > 0:
        return best_role

    # ----------------------------------------------
    # Default
    # ----------------------------------------------

    return "Software Engineer"


# --------------------------------------------------
# MAIN RESUME ANALYZER
# --------------------------------------------------

def analyze_resume(pdf_file):
    """
    Main function used by CareerMatch AI.

    Returns:
        {
            "role": ...,
            "education": ...,
            "experience": ...,
            "skills": [...],
            "extraction_method": ...
        }
    """

    # Extract resume text
    text, extraction_method = extract_resume_text(pdf_file)

    # Extract information
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    role = extract_role(text)

    # Return structured result
    return {
        "role": role,
        "education": education,
        "experience": experience,
        "skills": skills,
        "extraction_method": extraction_method,
    }