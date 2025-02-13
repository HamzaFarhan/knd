from pathlib import Path
from uuid import uuid4

import chromadb
import pandas as pd
import streamlit as st

from hiring_force_app import (
    Resume,
    create_ideal_candidate,
    embedding_functions,
    index_resume_objects,
    save_resume_objects,
)

# Initialize paths
DATA_DIR = Path("hiring_force")
RESUMES_DIR = DATA_DIR / "resumes"
RESUME_OBJECTS_DIR = DATA_DIR / "resume_objects"
MEMORIES_DIR = DATA_DIR / "memories"
JOB_DESC_PATH = DATA_DIR / "job_posting.md"
IDEAL_CANDIDATE_PATH = DATA_DIR / "ideal_candidate.json"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESUMES_DIR, RESUME_OBJECTS_DIR, MEMORIES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def format_resume(resume: Resume) -> str:
    """Format resume data into a readable string."""
    sections = []

    sections.append(f"### Summary\n{resume.summary}\n")

    if resume.work_experience:
        sections.append("### Work Experience")
        for exp in resume.work_experience:
            sections.append(
                f"**{exp.title}** at {exp.company}\n"
                f"_{exp.start_date} - {'Present' if exp.is_current else exp.end_date}_\n"
                f"{exp.description}\n"
                + ("\nAchievements:\n" + "\n".join(f"- {a}" for a in exp.achievements) if exp.achievements else "")
            )

    if resume.skills:
        sections.append("### Skills")
        for skill in resume.skills:
            skill_str = f"- {skill.name}"
            if skill.level:
                skill_str += f" (Level: {skill.level}/10)"
            if skill.years_experience:
                skill_str += f" - {skill.years_experience} years"
            sections.append(skill_str)

    if resume.education:
        sections.append("### Education")
        for edu in resume.education:
            sections.append(
                f"**{edu.degree}** in {edu.field_of_study}\n"
                f"{edu.institution}\n" + (f"GPA: {edu.gpa}\n" if edu.gpa else "")
            )

    if resume.certifications:
        sections.append("### Certifications")
        sections.extend(f"- {cert}" for cert in resume.certifications)

    if resume.projects:
        sections.append("### Projects")
        for proj in resume.projects:
            sections.append(
                f"**{proj.name}**\n"
                f"{proj.description}\n"
                + ("\nDuties:\n" + "\n".join(f"- {d}" for d in proj.duties) if proj.duties else "")
            )

    return "\n\n".join(sections)


def load_chroma_collection():
    """Initialize and return ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="Alibaba-NLP/gte-modernbert-base",
    )
    chroma_client.delete_collection(name="resumes")
    return chroma_client.get_or_create_collection(name="resumes", embedding_function=sentence_transformer_ef)


def main():
    st.title("Hiring Force - Resume Matcher")

    # Section 1: Upload Resumes
    st.header("1. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume files", accept_multiple_files=True, type=["pdf", "docx", "md"]
    )

    if uploaded_files:
        if st.button("Process Resumes"):
            with st.spinner("Processing resumes..."):
                for file in uploaded_files:
                    file_path = RESUMES_DIR / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())

                st.success(f"Saved {len(uploaded_files)} resume(s)")

                # Convert to resume objects
                user_id = uuid4()
                st.session_state["processing_status"] = "Processing resumes..."
                save_resume_objects(
                    user_id=user_id,
                    resumes_path=RESUMES_DIR,
                    resume_objects_path=RESUME_OBJECTS_DIR,
                    memories_dir=MEMORIES_DIR,
                )
                st.session_state["processing_status"] = "Done processing resumes"
                st.success("Successfully processed resumes")

    # Check if resume objects exist
    resume_objects_exist = any(RESUME_OBJECTS_DIR.glob("*.json"))

    if resume_objects_exist:
        # Section 2: Job Description
        st.header("2. Job Description")
        job_desc_tab1, job_desc_tab2 = st.tabs(["Upload", "Write"])

        with job_desc_tab1:
            uploaded_jd = st.file_uploader("Upload job description", type=["txt", "md"])
            if uploaded_jd:
                JOB_DESC_PATH.write_bytes(uploaded_jd.getvalue())
                st.success("Job description uploaded")

        with job_desc_tab2:
            if JOB_DESC_PATH.exists():
                existing_jd = JOB_DESC_PATH.read_text()
            else:
                existing_jd = ""

            new_jd = st.text_area("Write job description", value=existing_jd, height=300)
            if st.button("Save Job Description"):
                JOB_DESC_PATH.write_text(new_jd)
                st.success("Job description saved")

        # Generate Ideal Candidate
        if JOB_DESC_PATH.exists() and st.button("Generate Ideal Candidate"):
            with st.spinner("Generating ideal candidate profile..."):
                user_id = uuid4()
                ideal_candidate = create_ideal_candidate(
                    user_id=user_id,
                    job_desc=JOB_DESC_PATH.read_text(),
                    path=IDEAL_CANDIDATE_PATH,
                    memories_dir=MEMORIES_DIR,
                )
                st.success("Generated ideal candidate profile")

        # Section 3: View Ideal Candidate
        if IDEAL_CANDIDATE_PATH.exists():
            st.header("3. Ideal Candidate Profile")
            if st.button("Show Ideal Candidate"):
                ideal_candidate = Resume.model_validate_json(IDEAL_CANDIDATE_PATH.read_text())
                st.markdown(format_resume(ideal_candidate))

            # Section 4: Find Matches
            st.header("4. Find Matches")
            if st.button("Find Best Matches"):
                with st.spinner("Finding matches..."):
                    # Initialize ChromaDB
                    collection = load_chroma_collection()

                    # Index resume objects
                    index_resume_objects(collection, RESUME_OBJECTS_DIR)

                    # Query for matches
                    ideal_candidate = Resume.model_validate_json(IDEAL_CANDIDATE_PATH.read_text())
                    results = collection.query(
                        query_texts=[ideal_candidate.model_dump_json()],
                        n_results=len(list(RESUME_OBJECTS_DIR.glob("*.json"))),
                    )

                    # Create results dataframe
                    matches_data = []
                    for name, distance in zip(results["ids"][0], results["distances"][0]):
                        matches_data.append(
                            {
                                "Resume": Path(name).stem,
                                "Match Score": 1000 - distance,  # Convert distance to score
                            }
                        )

                    matches_df = pd.DataFrame(matches_data)
                    matches_df = matches_df.sort_values("Match Score", ascending=False)

                    # Display results
                    st.dataframe(
                        matches_df,
                        column_config={
                            "Resume": st.column_config.TextColumn("Resume"),
                            "Match Score": st.column_config.ProgressColumn(
                                "Match Score", min_value=0, max_value=1000, format="%d"
                            ),
                        },
                    )


if __name__ == "__main__":
    main()
