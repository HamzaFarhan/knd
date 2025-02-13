from pathlib import Path
from uuid import uuid4

import chromadb
import httpx
import pymupdf4llm
from chromadb.utils import embedding_functions
from fastapi import UploadFile
from fasthtml.common import (
    H1,
    H2,
    H3,
    H4,
    H5,
    Article,
    Button,
    Div,
    Form,
    Grid,
    Img,
    Input,
    Label,
    Li,
    Link,
    Main,
    P,
    Textarea,
    Titled,
    Ul,
    fast_app,
    serve,
)
from loguru import logger
from markitdown import MarkItDown

from hiring_force_app import AgentRequest, Resume

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore
    model_name="Alibaba-NLP/gte-modernbert-base",
)
collection = chroma_client.get_or_create_collection(name="resumes", embedding_function=sentence_transformer_ef)


def divider():
    return Div(
        style="border-top: 1px solid #e0e0e0; margin: 30px 0;",
    )


def loader(id: str = "loader"):
    return Img(
        id=id,
        src="static/loader.svg",
        cls="ui-indicator",
        style="width: 40px; height: 40px; margin-left: 10px; margin-bottom: 10px;",
    )


def doc_to_md(doc: Path | str) -> str:
    if not Path(doc).exists():
        raise FileNotFoundError(f"Document not found: {doc}")
    md = ""
    try:
        doc = Path(doc)
        if doc.suffix == ".md":
            md = doc.read_text()
        elif doc.suffix == ".pdf":
            md = pymupdf4llm.to_markdown(doc=str(doc))
        else:
            marker = MarkItDown()
            md = marker.convert(source=str(doc)).text_content
    except Exception:
        logger.error(f"Error converting {doc} to markdown")
    return md


def format_resume(resume: Resume) -> Article:
    return Article(
        H3(f"Experience: {resume.years_of_experience} years"),
        P(resume.summary),
        H4("Skills"),
        Ul(*[Li(f"{skill.name} ({skill.years_experience} years)") for skill in resume.skills]),
        H4("Work Experience"),
        *[
            Article(
                H5(f"{exp.title} at {exp.company}"),
                P(exp.description),
                Ul(*[Li(achievement) for achievement in exp.achievements]),
            )
            for exp in resume.work_experience
        ],
    )


# Set up app with headers for styling
headers = (Link(rel="stylesheet", href="static/style.css"),)
app, rt = fast_app(hdrs=headers)


@rt("/")
def get():
    return Titled(
        "Hiring Force",
        Main(
            Grid(
                H1("AI-Powered Hiring Assistant"),
                Div(style="text-align: right"),
            ),
            divider(),
            Article(
                H2("Upload Job Description"),
                Form(hx_post="/create_ideal_candidate", hx_target="#ideal-candidate", hx_indicator="#job-loader")(
                    Label(
                        "Job Description",
                        Textarea(
                            name="job_desc", rows="10", required=True, placeholder="Paste job description here..."
                        ),
                    ),
                    Button("Generate Ideal Candidate", type="submit"),
                    loader(id="job-loader"),
                ),
                Div(id="ideal-candidate"),
                divider(),
                H2("Upload Resumes"),
                Form(
                    hx_post="/upload_files",
                    hx_target="#upload-results",
                    hx_indicator="#upload-loader",
                    enctype="multipart/form-data",
                )(
                    Input(type="file", name="resumes", multiple=True, accept=".pdf,.docx,.md"),
                    loader(id="upload-loader"),
                ),
                Div(id="upload-results"),
                Button(
                    "Process Resumes",
                    hx_post="/process_resumes",
                    hx_target="#resume-results",
                    hx_indicator="#resume-loader",
                    cls="secondary",
                ),
                loader(id="resume-loader"),
                Div(id="resume-results"),
                divider(),
                H2("Top Matches"),
                Button(
                    "Find Matches",
                    hx_get="/matches",
                    hx_target="#matches",
                    hx_indicator="#matches-loader",
                    cls="secondary",
                ),
                Div(id="matches"),
                loader(id="matches-loader"),
            ),
            cls="container",
        ),
    )


@rt
async def create_ideal_candidate(job_desc: str):
    agent_request = AgentRequest(
        user_prompt=job_desc, agent_name="ideal_candidate_agent", user_id=uuid4(), memorize=True
    )
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post("http://localhost:8000/run_agent", json=agent_request.model_dump())
        response.raise_for_status()
        ideal_candidate = Resume(**response.json())

    # Store ideal candidate for matching
    Path("ideal_candidate.json").write_text(ideal_candidate.model_dump_json())

    return Article(H3("Ideal Candidate Profile"), format_resume(ideal_candidate), hx_swap_oob="true")


@rt
async def upload_files(resumes: list[UploadFile]):
    resume_objects_path = Path("resume_objects")
    resume_objects_path.mkdir(exist_ok=True)

    uploaded_files = []
    for resume in resumes:
        file_path = resume_objects_path / resume.filename
        content = await resume.read()
        file_path.write_bytes(content)
        uploaded_files.append(resume.filename)

    return Article(H3(f"Uploaded {len(uploaded_files)} Files"), Ul(*[Li(filename) for filename in uploaded_files]))


@rt
async def process_resumes():
    results = []
    resume_objects_path = Path("resume_objects")

    for file_path in resume_objects_path.glob("*"):
        if file_path.suffix in [".pdf", ".docx", ".md"]:
            agent_request = AgentRequest(
                user_prompt=doc_to_md(file_path), agent_name="resume_agent", user_id=uuid4(), memorize=True
            )

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post("http://localhost:8000/run_agent", json=agent_request.model_dump())
                response.raise_for_status()
                resume_obj = Resume(**response.json())
                results.append(resume_obj)

                # Save resume object and index in ChromaDB
                object_path = resume_objects_path / f"{file_path.name}.json"
                object_path.write_text(resume_obj.model_dump_json())
                collection.add(ids=[file_path.name], documents=[resume_obj.model_dump_json()])
                logger.success(f"Added resume {file_path.name} to collection")

    return Article(H3(f"Processed {len(results)} Resumes"), *[format_resume(r) for r in results])


@rt
async def matches():
    ideal_candidate_path = Path("ideal_candidate.json")
    if not ideal_candidate_path.exists():
        return Article(
            H3("Top Matches"), P("Please upload a job description first to generate an ideal candidate profile")
        )

    ideal_candidate = Resume.model_validate_json(ideal_candidate_path.read_text())

    # Query ChromaDB for top matches
    top_candidates = collection.query(query_texts=[ideal_candidate.model_dump_json()], n_results=3)

    if not top_candidates["documents"][0]:
        return Article(H3("Top Matches"), P("No resumes have been uploaded yet for matching"))

    matches = []
    for doc in top_candidates["documents"][0]:
        resume = Resume.model_validate_json(doc)
        matches.append(resume)

    return Article(
        H3("Top Matches"),
        *[Article(H4(f"Match #{i + 1}"), format_resume(resume)) for i, resume in enumerate(matches)],
    )


serve()
