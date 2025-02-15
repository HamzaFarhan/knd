import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import chromadb
import chromadb.api
import chromadb.api.client
import httpx
import pandas as pd
import pymupdf4llm
import streamlit as st
import torch
from chromadb.utils import embedding_functions
from loguru import logger
from markitdown import MarkItDown
from pydantic import UUID4

from hiring_force_app import AgentRequest, Resume, ResumeMatch

chromadb.api.client.SharedSystemClient.clear_system_cache()  # type:ignore


torch.classes.__path__ = []

MEMORIZE = False

PARENT_DIR = Path("hiring_force")
PARENT_DIR.mkdir(exist_ok=True)


def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    # Replace spaces and special chars with underscores
    return text.lower().replace(" ", "_").replace("-", "_")


# Initialize session state for project
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid4()
if "current_project" not in st.session_state:
    st.session_state.current_project = None


def get_existing_projects() -> list[str]:
    """Get list of existing project directories."""
    projects = [d for d in PARENT_DIR.iterdir() if d.is_dir() and d.name != "__pycache__"]
    # Sort projects by last modified time, newest first
    projects.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [d.name for d in projects]


def initialize_project_dirs(project_dir: Path) -> None:
    """Initialize all required subdirectories for a project."""
    subdirs = ["resumes", "resume_objects", "memories", "job_desc", "chroma_db"]
    for subdir in subdirs:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)


def project_creation_ui(label: str) -> None:
    """Handle project creation UI and logic."""
    new_project = st.text_input(label)
    if new_project:
        project_dir_name = to_snake_case(new_project)
        project_dir = PARENT_DIR / project_dir_name

        # Only show error if directory exists and user hasn't just created it
        if project_dir.exists() and st.session_state.current_project != project_dir_name:
            st.error("Project already exists!")
        elif st.button("Create Project"):
            initialize_project_dirs(project_dir)
            st.session_state.current_project = project_dir_name
            st.success(f"Created project: {new_project}")
            st.rerun()


def main():
    st.title("Hiring Force")

    # Project Selection Section
    st.header("Project Selection")
    existing_projects = get_existing_projects()

    col1, col2 = st.columns(2)
    with col1:
        if existing_projects:
            # Set the most recent project as default (index=1 since empty option is at index 0)
            selected_project = st.selectbox(
                "Select Existing Project",
                options=[""] + existing_projects,
                index=1 if "project_select" not in st.session_state else 0,
                key="project_select",
            )
            if selected_project:
                st.session_state.current_project = selected_project
        else:
            project_creation_ui("Create New Project")

    with col2:
        if existing_projects:
            project_creation_ui("Or Create New Project")

    if not st.session_state.current_project:
        st.warning("Please select or create a project to continue")
        return

    # Display resume count for current project
    BASE_DIR = PARENT_DIR / st.session_state.current_project
    RESUME_OBJECTS_DIR = BASE_DIR / "resume_objects"
    resume_count = len(list(RESUME_OBJECTS_DIR.glob("*.json")))
    st.info(f"Current project contains {resume_count} processed resumes")

    # Update directory paths based on current project
    RESUMES_DIR = BASE_DIR / "resumes"
    MEMORIES_DIR = BASE_DIR / "memories"
    JOB_DESC_DIR = BASE_DIR / "job_desc"
    CHROMA_DIR = BASE_DIR / "chroma_db"

    # Initialize ChromaDB client for current project
    with st.spinner("Initializing resume database... This may take a moment on first run."):
        try:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore
                model_name="Alibaba-NLP/gte-modernbert-base",
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            st.error("Failed to initialize resume database. Please try again.")
            return
    st.success("Ready!")

    def get_collection():
        try:
            collection = chroma_client.get_or_create_collection(
                name="resumes", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"}
            )
            return collection
        except Exception as e:
            logger.error(f"Error getting ChromaDB collection: {e}")
            st.error("Failed to initialize resume database. Please try again.")
            return None

    def doc_to_md(doc: Path | str) -> str:
        if not Path(doc).exists():
            raise FileNotFoundError(f"Document not found: {doc}")
        try:
            doc = Path(doc)
            if doc.suffix == ".md":
                return doc.read_text()
            elif doc.suffix == ".pdf":
                return pymupdf4llm.to_markdown(doc=str(doc))
            else:
                marker = MarkItDown()
                return marker.convert(source=str(doc)).text_content
        except Exception:
            logger.error(f"Error converting {doc} to markdown")
            return ""

    async def send_request(agent_request: AgentRequest) -> Resume:
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                response = await client.post("http://localhost:8000/run_agent", json=agent_request.model_dump())
                response.raise_for_status()
                return Resume(**response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    st.error("Rate limit reached. Please try again in a few minutes.")
                elif e.response.status_code == 400:
                    st.error(e.response.json()["detail"])
                elif e.response.status_code == 503:
                    st.error("The AI service is temporarily unavailable. Please try again later.")
                else:
                    st.error("An unexpected error occurred. Please try again later.")
                raise
            except Exception as e:
                st.error(f"Failed to process request: {str(e)}")
                raise

    async def create_ideal_candidate(
        user_id: UUID | str,
        job_desc: str,
        path: Path | str = "",
        memorize: bool = MEMORIZE,
        memories_dir: Path | str = "",
    ) -> Resume:
        agent_request = AgentRequest(
            user_prompt=job_desc,
            agent_name="ideal_candidate_agent",
            user_id=user_id,
            memorize=memorize,
            memories_dir=memories_dir,
        )
        ideal_candidate = await send_request(agent_request=agent_request)
        if path:
            Path(path).write_text(ideal_candidate.model_dump_json())
        return ideal_candidate

    async def process_resume(
        user_id: UUID4 | str,
        resume_path: Path,
        resume_objects_path: Path,
        memorize: bool = MEMORIZE,
        memories_dir: Path | str = "",
    ) -> Resume:
        agent_request = AgentRequest(
            user_prompt=doc_to_md(doc=resume_path),
            agent_name="resume_agent",
            user_id=user_id,
            memorize=memorize,
            memories_dir=memories_dir,
        )
        resume_object = await send_request(agent_request=agent_request)
        object_path = resume_objects_path / resume_path.name
        object_path = object_path.with_suffix(".json")
        object_path.write_text(resume_object.model_dump_json())

        # Add to ChromaDB collection
        collection = get_collection()
        if collection:
            collection.add(ids=[object_path.name], documents=[resume_object.model_dump_json()])
            logger.success(f"Added resume {object_path.name} to collection")

        return resume_object

    def display_resume(resume: Resume):
        st.subheader("Resume Details")
        st.write(f"**Title:** {resume.title}")
        st.write("**Summary**")
        st.write(resume.summary)

        st.write("**Years of Experience**:", resume.years_of_experience)

        if resume.work_experience:
            st.write("**Work Experience**")
            for exp in resume.work_experience:
                st.markdown(f"- **{exp.title}** at {exp.company}")
                st.markdown(f"  - {exp.description}")
                if exp.achievements:
                    st.markdown("  - **Achievements:**")
                    for achievement in exp.achievements:
                        st.markdown(f"    - {achievement}")

        if resume.skills:
            st.write("**Skills**")
            for skill in resume.skills:
                level_str = f" (Level: {skill.level})" if skill.level else ""
                years_str = f" ({skill.years_experience} years)" if skill.years_experience else ""
                st.markdown(f"- {skill.name}{level_str}{years_str}")

        if resume.certifications:
            st.write("**Certifications**")
            for cert in resume.certifications:
                st.markdown(f"- {cert}")

    async def get_resume_match(candidate_resume: Resume, ideal_candidate: Resume) -> ResumeMatch:
        agent_request = AgentRequest(
            user_prompt=f"Compare this candidate:\n{candidate_resume.model_dump_json()}\n\nTo this ideal profile:\n{ideal_candidate.model_dump_json()}",
            agent_name="resume_match_agent",
            user_id=st.session_state.user_id,
            memorize=False,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post("http://localhost:8000/run_agent", json=agent_request.model_dump())
            response.raise_for_status()
            return ResumeMatch(**response.json())

    def normalize_chroma_distance(distance):
        # ChromaDB distance is already in [0,1]
        # Invert so similar items (distance near 0) get high scores
        inverted = 1 - distance

        # Scale to [1,10] range
        normalized = 1 + (inverted * 9)

        return normalized

    # Section 1: Upload Resumes
    st.header("1. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, MD)", accept_multiple_files=True, type=["pdf", "docx", "md"]
    )

    if uploaded_files:
        for file in uploaded_files:
            resume_path = RESUMES_DIR / file.name
            resume_path.write_bytes(file.read())
        st.success(f"Uploaded {len(uploaded_files)} resumes")

        if st.button("Process Resumes"):
            with st.spinner("Processing resumes..."):
                for file in uploaded_files:
                    resume_path = RESUMES_DIR / file.name
                    resume = asyncio.run(
                        process_resume(
                            user_id=st.session_state.user_id,
                            resume_path=resume_path,
                            resume_objects_path=RESUME_OBJECTS_DIR,
                            memories_dir=MEMORIES_DIR,
                        )
                    )
                    st.success(f"Processed {file.name}")

    # Only show subsequent sections if resumes exist
    if not any(RESUME_OBJECTS_DIR.iterdir()):
        st.warning("Please upload and process some resumes first")
        return

    # Section 2: Job Description
    st.header("2. Job Description")

    # Initialize job description in session state if not present
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""

    # Handle file upload first
    job_desc_file = st.file_uploader("Upload job description file", type=["txt", "md"])
    if job_desc_file:
        st.session_state.job_description = job_desc_file.read().decode()

    # Show text area with current job description
    job_desc = st.text_area("Enter or edit job description", value=st.session_state.job_description)

    # Confirm button for job description
    if st.button("Generate Ideal Candidate") and job_desc:
        st.session_state.job_description = job_desc
        with st.spinner("Generating ideal candidate profile..."):
            ideal_candidate = asyncio.run(
                create_ideal_candidate(
                    user_id=st.session_state.user_id,
                    job_desc=job_desc,
                    path=JOB_DESC_DIR / "ideal_candidate.json",
                    memories_dir=MEMORIES_DIR,
                )
            )
        st.success("Generated ideal candidate profile")

    # Only show subsequent sections if ideal candidate exists
    ideal_candidate_path = JOB_DESC_DIR / "ideal_candidate.json"
    if not ideal_candidate_path.exists():
        return

    # Section 3: View Ideal Candidate
    st.header("3. Ideal Candidate Profile")
    with st.expander("Click to view Ideal Candidate Profile"):
        ideal_candidate = Resume.model_validate_json(ideal_candidate_path.read_text())
        display_resume(ideal_candidate)

    # Section 4: Find Matches
    st.header("4. Find Matches")
    if st.button("Find Best Matches"):
        collection = get_collection()
        if not collection:
            st.error("Could not access the resume collection")
            return

        with st.spinner("Finding and analyzing matches..."):
            # Query using both the job description and ideal candidate
            job_desc_results = collection.query(query_texts=[job_desc], n_results=5)
            ideal_candidate_results = collection.query(
                query_texts=[ideal_candidate.model_dump_json()], n_results=5
            )

            # Combine and process results
            matches = []
            seen_ids = set()

            for results in [job_desc_results, ideal_candidate_results]:
                if results is None:
                    continue
                if not results["ids"] or not results["distances"]:
                    continue
                for id_, distance in zip(results["ids"][0], results["distances"][0]):
                    if id_ not in seen_ids:
                        seen_ids.add(id_)
                        resume = Resume.model_validate_json((RESUME_OBJECTS_DIR / id_).read_text())
                        match_analysis = asyncio.run(get_resume_match(resume, ideal_candidate))

                        # Normalize the ChromaDB distance to a 1-10 score
                        semantic_score = normalize_chroma_distance(distance)

                        # Combine scores (70% semantic, 30% analysis)
                        combined_score = 0.7 * semantic_score + 0.3 * match_analysis.overall_score

                        matches.append(
                            {
                                "Name": Path(id_).stem,
                                "Overall Score": min(10.0, combined_score + 1.5),
                                "Semantic Score": min(10.0, semantic_score + 1.5),
                                "AI Analysis Score": min(10.0, match_analysis.overall_score + 1.5),
                                "Years of Experience": resume.years_of_experience,
                                "Title": resume.title,
                                "Key Strengths": "\n".join(f"• {s}" for s in match_analysis.key_strengths),
                                "Areas for Growth": "\n".join(f"• {g}" for g in match_analysis.gaps),
                                "Skills Analysis": match_analysis.skills_feedback,
                                "Experience Analysis": match_analysis.experience_feedback,
                            }
                        )

            if matches:
                df = pd.DataFrame(matches)
                # Sort by Match Score in descending order
                df = df.sort_values("Overall Score", ascending=False)
                st.dataframe(
                    df,
                    column_config={
                        "Overall Score": st.column_config.NumberColumn(format="%.1f"),
                        "Semantic Score": st.column_config.NumberColumn(format="%.1f"),
                        "AI Analysis Score": st.column_config.NumberColumn(format="%.1f"),
                        "Key Strengths": st.column_config.TextColumn(width="medium"),
                        "Areas for Growth": st.column_config.TextColumn(width="medium"),
                        "Skills Analysis": st.column_config.TextColumn(width="large"),
                        "Experience Analysis": st.column_config.TextColumn(width="large"),
                    },
                    hide_index=True,
                )
            else:
                st.warning("No matches found")


if __name__ == "__main__":
    main()
