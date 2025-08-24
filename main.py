import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation',
)

st.title("PDF Summarizer App")

file_path=st.file_uploader("Upload a PDF file", type=["pdf"])
if file_path is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_path.read())
        tmp_file_path = tmp_file.name

    summary_template = """
    You are an expert text summarizer. 
    Your task is to read the following text and produce a clear, concise, and well-structured summary. 

    Guidelines:
    - Capture the most important points, arguments, and facts.
    - Remove unnecessary details, redundancies, or filler text.
    - Keep the summary objective and neutral.
    - If the text is long, organize the summary into sections or bullet points.
    - Ensure the summary length is proportionate (not too short, not overly long).

    Text to summarize:
    {text}

    Summary:
    """

    model=ChatHuggingFace(llm=llm)

    loader=PyPDFLoader(tmp_file_path)

    docs=loader.load()

    all_text=" ".join([doc.page_content for doc in docs])

    prompt=PromptTemplate(
        template=summary_template,
        input_variables=['text']
    )
    final_prompt=prompt.invoke({'text':all_text})

    result=model.invoke(final_prompt)

    st.markdown(result.content)