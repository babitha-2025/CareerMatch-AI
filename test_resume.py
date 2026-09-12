from resume_parser import analyze_resume

resume_path = "resume.pdf"

result = analyze_resume(resume_path)

print("\n========== RESUME ANALYSIS ==========\n")

print("Education:")
print(result["education"])

print("\nExperience:")
print(result["experience"])

print("\nDetected Role:")
print(result["role"])

print("\nSkills:")
print(", ".join(result["skills"]))

print("\n=====================================\n")