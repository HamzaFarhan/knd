import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import chromadb
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

torch.classes.__path__ = []

MEMORIZE = False

# Initialize directories
BASE_DIR = Path("hiring_force")
RESUMES_DIR = BASE_DIR / "resumes"
RESUME_OBJECTS_DIR = BASE_DIR / "resume_objects"
MEMORIES_DIR = BASE_DIR / "memories"
JOB_DESC_DIR = BASE_DIR / "job_desc"

for dir_path in [BASE_DIR, RESUMES_DIR, RESUME_OBJECTS_DIR, MEMORIES_DIR, JOB_DESC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid4()

# Add these to your initial imports and directory setup
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore
    model_name="Alibaba-NLP/gte-modernbert-base",
)


def get_collection():
    try:
        collection = chroma_client.get_or_create_collection(
            name="resumes", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        logger.error(f"Error getting ChromaDB collection: {e}")
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
        response = await client.post("http://localhost:8000/run_agent", json=agent_request.model_dump())
        response.raise_for_status()
        return Resume(**response.json())


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


def main():
    st.title("Hiring Force")

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
    job_desc = st.text_area("Enter job description")
    job_desc_file = st.file_uploader("Or upload job description file", type=["txt", "md"])

    if job_desc_file:
        job_desc = job_desc_file.read().decode()

    if st.button("Generate Ideal Candidate") and job_desc:
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
                                "Match Score": combined_score,
                                "Semantic Score": semantic_score,
                                "Analysis Score": match_analysis.overall_score,
                                "Years of Experience": resume.years_of_experience,
                                "Current Title": resume.work_experience[0].title
                                if resume.work_experience
                                else "N/A",
                                "Key Strengths": "\n".join(f"• {s}" for s in match_analysis.key_strengths),
                                "Areas for Growth": "\n".join(f"• {g}" for g in match_analysis.gaps),
                                "Skills Analysis": match_analysis.skills_feedback,
                                "Experience Analysis": match_analysis.experience_feedback,
                            }
                        )

            if matches:
                df = pd.DataFrame(matches)
                # Sort by Match Score in descending order
                df = df.sort_values("Match Score", ascending=False)
                st.dataframe(
                    df,
                    column_config={
                        "Match Score": st.column_config.NumberColumn(format="%.1f"),
                        "Semantic Score": st.column_config.NumberColumn(format="%.1f"),
                        "Analysis Score": st.column_config.NumberColumn(format="%.1f"),
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
