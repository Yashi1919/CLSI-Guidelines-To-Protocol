import os
import sys
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path
import pytesseract
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Step 1: Configure Gemini LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    print(f"Error initializing Gemini LLM: {str(e)}")
    sys.exit(1)

# Step 2: Extract text from PDF with OCR fallback
def extract_pdf_text(pdf_path):
    """Extract text from a PDF using PyMuPDF, with OCR as a fallback."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        if text.strip():
            return text
        print("No text extracted with PyMuPDF, attempting OCR...")
        images = convert_from_path(pdf_path)
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image) + "\n"
        return ocr_text if ocr_text.strip() else "No text extracted via OCR."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Step 3: Chunk text for processing
def chunk_text(text, max_chars=5000):
    """Split text into smaller chunks to handle token limits."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# Step 4: Validate PDF file
def validate_pdf(pdf_path):
    """Check if the PDF file exists and is valid."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        sys.exit(1)
    if not pdf_path.lower().endswith('.pdf'):
        print(f"File at {pdf_path} is not a PDF")
        sys.exit(1)
    return True

# Step 5: Estimate LaTeX word count
def estimate_latex_word_count(latex_file):
    """Estimate word count of a LaTeX file by stripping commands."""
    try:
        with open(latex_file, 'r', encoding='utf-8') as f:
            text = f.read()
        clean_text = re.sub(r'\\[^ ]+{.*?}|\\[a-zA-Z]+|\{|\}', '', text)
        words = len(clean_text.split())
        return words
    except Exception as e:
        print(f"Error estimating word count: {str(e)}")
        return 0

# Step 6: Trim LaTeX content
def trim_latex_content(latex_file, max_words=6000):
    """Trim LaTeX content to approximate max word count."""
    word_count = estimate_latex_word_count(latex_file)
    if word_count <= max_words:
        return
    try:
        with open(latex_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trimmed_lines = lines[:int(len(lines) * (max_words / word_count))]
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.writelines(trimmed_lines)
        print(f"Trimmed {latex_file} to approximate {max_words} words.")
    except Exception as e:
        print(f"Error trimming LaTeX file: {str(e)}")

# Step 7: Define Agents
metadata_agent = Agent(
    role="PDF Metadata Extractor",
    goal="Extract title, authors, and abstract/summary from a CLSI guideline PDF and format as JSON.",
    backstory="You are an expert in parsing CLSI guideline PDFs to extract structured metadata, including titles, authors, and abstracts, with a focus on handling variable document formats.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

protocol_agent = Agent(
    role="CLSI Protocol Generator",
    goal="Generate a concise LaTeX protocol based on the content of any CLSI guideline, tailored to its specific procedures, aiming for 1500–2000 words with examples and citations.",
    backstory="You are a specialist in clinical laboratory standards, adept at interpreting CLSI guidelines and creating practical protocols with relevant examples and cited sources for laboratory use.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

qa_agent = Agent(
    role="Protocol Quality Assurance Specialist",
    goal="Review extracted metadata and generated protocol for accuracy, consistency, and adherence to CLSI standards, ensuring examples and citations are relevant.",
    backstory="You are an expert in quality control for clinical laboratory documentation, ensuring metadata accuracy, protocol completeness, and proper use of examples and citations.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

enrichment_agent = Agent(
    role="Protocol Content Enhancer",
    goal="Enhance the protocol with concise background, examples, references, and citations to meet a 3000–4000 word target.",
    backstory="You are a specialist in scientific writing, skilled at enriching laboratory protocols with clear examples, cited sources, and concise explanations while maintaining practicality.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Step 8: Function to Create Tasks
def create_tasks(pdf_text, guideline_code):
    """Create tasks for metadata extraction, protocol generation, quality assurance, and content enrichment."""
    # Task 1: Extract metadata
    metadata_task = Task(
        description=f"""
        Extract the title, authors, and abstract (or summary if abstract is not present) from the provided CLSI guideline text.
        Format the output as a JSON object with keys 'title', 'authors', and 'abstract'.
        If any field is not found, set it to 'Not found'.
        Use patterns:
        - Title: Often bolded, prominent at the document's start, or includes the guideline code ({guideline_code}).
        - Authors: Typically listed below the title, possibly with affiliations or as 'CLSI' or a committee.
        - Abstract: A summary paragraph, often labeled 'Abstract', 'Summary', or 'Introduction'.
        Process the first 200,000 characters to focus on metadata.
        Text:
        {pdf_text[:200000]}
        """,
        expected_output="A JSON object with 'title', 'authors', and 'abstract' from the CLSI guideline.",
        agent=metadata_agent,
        output_file='metadata.json'
    )

    # Task 2: Generate initial protocol
    protocol_task = Task(
        description=f"""
        Analyze the provided CLSI guideline text (Guideline: {guideline_code}) and generate a concise protocol tailored to its core procedures (e.g., linearity evaluation for EP39).
        The protocol must:
        - Be approximately 1500–2000 words to allow for concise enrichment.
        - Be formatted as a valid LaTeX document for Overleaf with a preamble:
          \\documentclass{{article}}
          \\usepackage{{booktabs,natbib,bibentry}}
          \\nobibliography*
          \\begin{{document}}
          % Protocol content
          \\bibliography{{references}}
          \\end{{document}}
        - Use section headings (\section, \subsection), bullet points (\itemize), numbered lists (\enumerate), and tables (\begin{{tabular}}).
        - Include for each section:
          - A practical example (e.g., sample preparation steps, data analysis calculation, issue mitigation scenario).
          - Sources for examples (e.g., CLSI {guideline_code}, related CLSI standards like EP06 or M100, or peer-reviewed literature).
          - In-text citations using \\cite{{key}} (e.g., \\cite{{CLSI{guideline_code}}}).
        - Structure:
          - Author: CLSI Protocol Committee, dated {datetime.now().strftime('%m.%d.%Y')}.
          - Sections: Introduction, Objectives, Experimental Procedures, Data Collection, Data Analysis, Potential Issues.
          - One or two tables for data outputs (e.g., measurement results) and potential issues.
          - Bibliography with 1–2 entries using \\bibentry (e.g., CLSI {guideline_code}, another standard).
        - Create a clear, organized protocol, mirroring CLSI style, with concise sections, numbered steps, and cited examples.
        Process the text in chunks, prioritizing sections relevant to the guideline’s procedures.
        Text (in chunks):
        {"".join([f"\n\nChunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunk_text(pdf_text[:200000]))])}
        """,
        expected_output=f"""
        A LaTeX document (~1500–2000 words) containing a protocol for CLSI guideline {guideline_code}, formatted for Overleaf. The document must:
        - Include a preamble:
          \\documentclass{{article}}
          \\usepackage{{booktabs,natbib,bibentry}}
          \\nobibliography*
          \\begin{{document}}
          % Protocol content
          \\bibliography{{references}}
          \\end{{document}}
        - Include a title relevant to the guideline’s focus (e.g., 'Protocol for {guideline_code} Procedures').
        - Specify CLSI Protocol Committee as the author and {datetime.now().strftime('%m.%d.%Y')} as the date.
        - Feature concise sections with examples and citations:
          - Introduction (1–2 paragraphs, e.g., context for linearity evaluation, citing CLSI {guideline_code}).
          - Objectives (e.g., goal to validate assay linearity).
          - Step-by-step procedures (e.g., sample dilution steps, citing CLSI EP06).
          - Data collection (e.g., recording absorbance values, citing a related standard).
          - Data analysis (e.g., linear regression example, citing a textbook).
          - Potential issues with mitigation in a table (e.g., matrix effects, citing CLSI M100).
          - One or two tables for data outputs.
          - Bibliography with 1–2 \\bibentry entries (e.g., \\bibentry{{CLSI{guideline_code}}}).
        - Use valid LaTeX syntax with minimal markup.
        - Be structured for laboratory use, ready for enrichment.
        """,
        agent=protocol_agent,
        output_file=f'protocol_initial_{guideline_code}.tex'
    )

    # Task 3: Quality assurance
    qa_task = Task(
        description=f"""
        Review the extracted metadata and generated protocol for accuracy and adherence to CLSI standards.
        - For metadata (from metadata.json), verify:
          - Title matches the guideline code ({guideline_code}) or document content.
          - Authors are correctly identified (e.g., CLSI or committee).
          - Abstract/summary reflects the guideline’s purpose.
          If discrepancies are found, suggest corrections and output a revised JSON object.
        - For the protocol (from protocol_initial_{guideline_code}.tex), ensure:
          - Structure includes all required sections.
          - Procedures align with the guideline’s recommendations.
          - Examples are practical and relevant to the guideline.
          - Citations use \\cite and reference valid sources (e.g., CLSI documents).
          - Tables and references are complete and concise.
          - Content is suitable for laboratory use.
        Output a JSON object with:
        - 'metadata_corrections': Suggested changes to metadata (or 'None').
        - 'protocol_feedback': Feedback on protocol quality, examples, citations, and revisions.
        Input:
        - Metadata: {json.dumps(json.load(open('metadata.json', 'r', encoding='utf-8')) if os.path.exists('metadata.json') else {})}
        - Protocol: Content of protocol_initial_{guideline_code}.tex
        - Guideline text (in chunks):
        {"".join([f"\n\nChunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunk_text(pdf_text[:200000]))])}
        """,
        expected_output="A JSON object with 'metadata_corrections' and 'protocol_feedback' detailing any issues and suggested revisions.",
        agent=qa_agent,
        output_file=f'qa_report_{guideline_code}.json'
    )

    # Task 4: Content enrichment
    enrichment_task = Task(
        description=f"""
        Enhance the initial protocol (from protocol_initial_{guideline_code}.tex) to produce a final LaTeX document of 3000–4000 words, tailored to the CLSI guideline ({guideline_code}).
        Use the QA report (from qa_report_{guideline_code}.json) to address any identified issues.
        The enhanced protocol must:
        - Incorporate QA feedback to improve accuracy and completeness.
        - Expand sections concisely with examples and citations:
          - Brief background (1–2 paragraphs, e.g., clinical relevance of the guideline, citing CLSI {guideline_code}).
          - Clear objectives with an example (e.g., validating a specific assay, citing a related standard).
          - Actionable procedures with example steps (e.g., calibration procedure, citing CLSI EP06).
          - Data collection with an example (e.g., sample data table, citing a textbook).
           -Flow Charts have to be a saperate section and it has to be there try to find flowcharts
           -plan to draw a flow chart as well
          -Flow charts(find flow charts) and the relation with a protocol as well
          - Data analysis with an example (e.g., statistical method, citing a peer-reviewed article).
          - Potential issues and mitigations in a compact table (e.g., interference handling, citing CLSI M100).
          - Sources for examples (e.g., CLSI documents, peer-reviewed literature).
          - In-text citations using \\cite{{key}} (e.g., \\cite{{CLSI{guideline_code}}}).
        - Include 2–3 additional references using \\bibentry (e.g., CLSI standards, journal articles).
        - Use valid LaTeX syntax with a preamble:
          \\documentclass{{article}}
          \\usepackage{{booktabs,natbib,bibentry}}
          \\nobibliography*
          \\begin{{document}}
          % Protocol content
          \\bibliography{{references}}
          \\end{{document}}
        - Ensure the document is clear, practical, and suitable for laboratory use, ready for Overleaf or Google Docs.
        Input:
        - Initial protocol: Content of protocol_initial_{guideline_code}.tex
        - QA report: {json.dumps(json.load(open(f'qa_report_{guideline_code}.json', 'r', encoding='utf-8')) if os.path.exists(f'qa_report_{guideline_code}.json') else {})}
        - Guideline text (in chunks):
        {"".join([f"\n\nChunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunk_text(pdf_text[:200000]))])}
        """,
        expected_output=f"""
        A LaTeX document (~4000–5000 words) containing an enhanced protocol for CLSI guideline {guideline_code}, formatted for Overleaf. The document must:
        - Include a preamble:
          \\documentclass{{article}}
          \\usepackage{{booktabs,natbib,bibentry}}
          \\nobibliography*
          \\begin{{document}}
          % Protocol content
          \\bibliography{{references}}
          \\end{{document}}
        - Include a title relevant to the guideline’s focus (e.g., 'Protocol for {guideline_code} Procedures').
        - Specify CLSI Protocol Committee as the author and {datetime.now().strftime('%m.%d.%Y')} as the date.
        - Feature concise sections with examples and citations:
          - Brief background (1–2 paragraphs, e.g., guideline’s purpose, citing CLSI {guideline_code}).
          - Objectives (e.g., assay validation goal, citing a standard).
          - Procedures with example steps (e.g., calibration, citing CLSI EP06).
          - Data collection with example (e.g., data table, citing a textbook).
          - Data analysis with example (e.g., linear regression, citing a journal).
          - Potential issues and mitigations in a table (e.g., matrix effects, citing CLSI M100).
          -Flow Charts have to be a saperate section and it has to be there try to find flowcharts
          -try to draw a flowchart as well
          -represent flowcharts and the relation with guidelines and protocols as well for better understanding
          - 2–3 \\bibentry references (e.g., CLSI {guideline_code}, related standards, articles).
          -(important point "make sure you give a real world example or any example for important points or every point)
        - Use valid LaTeX syntax with minimal markup.
        - Be clear, practical, and ready for laboratory use or copying into a Google Doc.
        """,
        agent=enrichment_agent,
        output_file=f'protocol_{guideline_code}.tex'
    )

    return [metadata_task, protocol_task, qa_task, enrichment_task]

# Step 9: Main Function to Process Any CLSI Guideline
def process_clsi_guideline(pdf_path):
    """Process a CLSI guideline PDF to extract metadata and generate an enhanced protocol."""
    # Validate PDF
    validate_pdf(pdf_path)
    
    # Extract guideline code from filename
    guideline_code = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract text
    print(f"Extracting text from {pdf_path}...")
    pdf_text = extract_pdf_text(pdf_path)
    #if "Error" in pdf_text or not pdf_text.strip():
    #    print(pdf_text if "Error" in pdf_text else "No text extracted from PDF.")
    #    sys.exit(1)
    
    # Create tasks
    tasks = create_tasks(pdf_text, guideline_code)
    
    # Create and run Crew
    crew = Crew(
        agents=[metadata_agent, protocol_agent, qa_agent, enrichment_agent],
        tasks=tasks,
        verbose=True
    )
    
    # Execute tasks
    print(f"Processing CLSI guideline {guideline_code}...")
    try:
        results = crew.kickoff()
        
        # Process metadata task result
        try:
            metadata = json.loads(str(results.tasks_output[0].raw))
            print("\nExtracted Metadata (JSON):")
            print(json.dumps(metadata, indent=2))
            with open('metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print("Metadata saved to metadata.json")
        except json.JSONDecodeError:
            print("\nMetadata output is not valid JSON. Raw result:")
            print(results.tasks_output[0].raw)
            with open('metadata.json', 'w', encoding='utf-8') as f:
                f.write(str(results.tasks_output[0].raw))
            print("Raw metadata saved to metadata.json")
        
        # Process QA report
        try:
            qa_report = json.load(open(f'qa_report_{guideline_code}.json', 'r', encoding='utf-8'))
            print("\nQA Report (JSON):")
            print(json.dumps(qa_report, indent=2))
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"\nQA report (qa_report_{guideline_code}.json) not found or invalid.")
        
        # Trim final protocol to ensure size limit
        trim_latex_content(f'protocol_{guideline_code}.tex', max_words=4000)
        
        # Final protocol
        print(f"\nEnhanced protocol saved to protocol_{guideline_code}.tex")
        
        return results
    except Exception as e:
        print(f"Error during CrewAI execution: {str(e)}")
        sys.exit(1)

# Step 10: Run the Script
if __name__ == "__main__":
    # Example PDF path
    pdf_path = os.path.join(os.getcwd(), "EP39Ed1E.pdf")
    process_clsi_guideline(pdf_path)