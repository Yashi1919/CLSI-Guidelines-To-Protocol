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

# Step 2: Extract text from PDF (pages 1–76)
def extract_pdf_text(pdf_path, start_page=0, end_page=76):
    """Extract text from specified PDF pages using PyMuPDF, with OCR fallback."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(start_page, min(end_page, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        if text.strip():
            return text
        print(f"No text extracted from pages {start_page+1}–{end_page}, attempting OCR...")
        images = convert_from_path(pdf_path, first_page=start_page+1, last_page=end_page)
        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image) + "\n"
        return ocr_text if ocr_text.strip() else f"No text extracted from pages {start_page+1}–{end_page} via OCR."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Step 3: Chunk text for processing
def chunk_text(text, max_chars=5000):
    """Split text into chunks to handle token limits."""
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
def trim_latex_content(latex_file, max_words=12000):
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
    goal="Extract title, authors, and abstract/summary from CLSI guideline PDF (pages 1–76) and format as JSON.",
    backstory="Expert in parsing CLSI guideline PDFs to extract structured metadata, handling variable formats.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

content_extractor_agent = Agent(
    role="Content Extractor",
    goal="Extract all relevant content from CLSI guideline (pages 1–76, Objective to Conclusion) in simple language, producing a final LaTeX document (8000–10,000 words).",
    backstory="Specialist in extracting, organizing, and simplifying clinical laboratory guideline content, filtering out irrelevant sections, and producing clear, lab-ready documents.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Step 8: Create Tasks
def create_tasks(pdf_text, guideline_code):
    """Create tasks for metadata extraction and final content generation."""
    # Task 1: Extract metadata
    metadata_task = Task(
        description=f"""
Extract title, authors, and abstract/summary from CLSI guideline text (pages 1–76).
Format as JSON with keys 'title', 'authors', 'abstract'.
If not found, set to 'Not found'.
Patterns:
- Title: Bolded, at document start, or includes {guideline_code}.
- Authors: Below title, possibly CLSI or committee.
- Abstract: Labeled 'Abstract', 'Summary', or 'Introduction'.
Process first 50,000 characters.
Text:
{pdf_text[:50000]}
""",
        expected_output="JSON object with 'title', 'authors', 'abstract'.",
        agent=metadata_agent,
        output_file='metadata.json'
    )

    # Task 2: Generate final content
    final_content_task = Task(
        description=f"""
Extract all relevant content from CLSI guideline ({guideline_code}, pages 1–76, Objective to Conclusion) in simple language, producing a single LaTeX document (8000–10,000 words).
The document must:
- Use LaTeX format for Overleaf:
  \\documentclass{{article}}
  \\usepackage{{booktabs,natbib,bibentry}}
  \\nobibliography*
  \\begin{{document}}
  % Content
  \\bibliography{{references}}
  \\end{{document}}
- Use \\section, \\subsection, \\itemize, \\tabular for organization.
- Include:
  - Full relevant content from Objective to Conclusion, in simple language, avoiding jargon.
  - One practical example per major section (e.g., procedure step, data analysis).
  - Citations using \\cite{{key}} (e.g., \\cite{{CLSI{guideline_code}}}).
  - At least one table for key data/results (e.g., measurement outcomes).
- Exclude irrelevant content (e.g., appendices, glossary, references) using headings (e.g., 'Appendix', 'References', 'Glossary') or page numbers.
- Structure:
  - Author: CLSI Content Committee, dated {datetime.now().strftime('%m.%d.%Y')}.
  - Sections: Based on guideline structure (e.g., Objective, Procedures, Data Analysis, Conclusions).
  - Bibliography with 2–3 \\bibentry entries (e.g., CLSI {guideline_code}, related standards).
Text (chunks):
{"".join([f"\n\nChunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunk_text(pdf_text))])}
""",
        expected_output=f"""
LaTeX document (~10,000–12,000 words) with relevant CLSI guideline {guideline_code} content:
- Preamble: \\documentclass{{article}}, \\usepackage{{booktabs,natbib,bibentry}}, \\nobibliography*.
- Title: Relevant to {guideline_code} (e.g., 'Content of {guideline_code} Guidelines').
- Author: CLSI Content Committee, date: {datetime.now().strftime('%m.%d.%Y')}.
- Sections: Objective, Procedures, Data Analysis, Conclusions, etc.
- Features: Examples, \\cite citations, at least one table, 2–3 \\bibentry references.
- Simple, lab-ready, excludes irrelevant content, Overleaf/Google Docs compatible.
""",
        agent=content_extractor_agent,
        output_file=f'final_content_{guideline_code}.tex'
    )

    return [metadata_task, final_content_task]

# Step 9: Main Function
def process_clsi_guideline(pdf_path):
    """Process CLSI guideline PDF (pages 1–76) to extract relevant content."""
    # Validate PDF
    validate_pdf(pdf_path)
    
    # Extract guideline code
    guideline_code = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract text (pages 1–76)
    print(f"Extracting text from pages 1–76 of {pdf_path}...")
    pdf_text = extract_pdf_text(pdf_path, start_page=0, end_page=76)
    if "Error" in pdf_text or not pdf_text.strip():
        print(pdf_text if "Error" in pdf_text else "No text extracted.")
        sys.exit(1)
    
    # Create tasks
    tasks = create_tasks(pdf_text, guideline_code)
    
    # Create and run Crew
    crew = Crew(
        agents=[metadata_agent, content_extractor_agent],
        tasks=tasks,
        verbose=True
    )
    
    # Execute tasks
    print(f"Processing CLSI guideline {guideline_code} (pages 1–76)...")
    try:
        results = crew.kickoff()
        
        # Process metadata
        try:
            metadata = json.loads(str(results.tasks_output[0].raw))
            print("\nMetadata (JSON):")
            print(json.dumps(metadata, indent=2))
            with open('metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print("Metadata saved to metadata.json")
        except json.JSONDecodeError:
            print("\nMetadata not valid JSON. Raw result:")
            print(results.tasks_output[0].raw)
            with open('metadata.json', 'w', encoding='utf-8') as f:
                f.write(str(results.tasks_output[0].raw))
            print("Raw metadata saved to metadata.json")
        
        # Trim final content
        trim_latex_content(f'final_content_{guideline_code}.tex', max_words=10000)
        
        print(f"\nFinal content saved to final_content_{guideline_code}.tex")
        
        return results
    except Exception as e:
        print(f"Error during CrewAI execution: {str(e)}")
        sys.exit(1)

# Step 10: Run Script
if __name__ == "__main__":
    pdf_path = os.path.join(os.getcwd(), "EP39Ed1E.pdf")
    process_clsi_guideline(pdf_path)