# Multimodal RAG Pipeline for PDF Lecture Slides

This repository contains a Jupyter Notebook (`RAG_ML_1111.ipynb`) demonstrating an advanced Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about PDF-based academic material (like lecture slides or textbooks).

The primary challenge with academic PDFs is that simple text extraction fails to capture the rich information in diagrams, formulas, and visual layouts. This project tackles that problem by implementing a multimodal, multi-stage pipeline that "reads" the pages as images to understand their content and format.

The goal is not just to get the *correct answer* but to generate an answer that follows the specific **approach, format, and methodology** taught in the lecture slides.



## 🚀 Project Workflow

This pipeline is broken into three main stages:

### Stage 1: Preprocessing & Indexing (Page Summarization)

Instead of relying on brittle PDF text extraction, we treat each page as an image and generate a high-quality text summary of its visual content.

1.  **PDF to Image:** The `algo_merged.pdf` file is converted into 105 separate PNG images (one per page).
2.  **Image-based Summarization:** A powerful multimodal model (`microsoft/Phi-4-multimodal-instruct`) is used to "look" at each page's image and generate a concise, 6-sentence summary.
3.  **Vector Store Creation:** These 105 page summaries (now in text format) are embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed into a `Chroma` vector database. This DB maps text summaries back to their original page numbers.

### Stage 2: RAG Pipeline (Retrieval & Generation)

This is a two-step RAG process that uses different models for retrieval and generation.

1.  **Retrieval (Text-to-Page-Number):**
    * A user's question (e.g., "Sort this list using merge sort") is first given to a small, fast LLM (`microsoft/Phi-4-mini-instruct`).
    * This LLM's only job is to query the vector database (of page summaries) and **return the most relevant page numbers**.
    * The prompt is: `"On which page can we find related material to this problem? [question]"`

2.  **Generation (Image-to-Text):**
    * The retrieved page numbers are used to pull the *original page images* from disk.
    * These images, along with the original question, are sent to a high-capability multimodal model (`Gemini 1.5 Flash`).
    * The prompt is: `"[question] Use the same approach/format/graph in given context to answer the question."`
    * The `Gemini` model then generates a final answer by looking at the visual layout, graphs, and pseudocode on the source slides.

### Stage 3: Automated Evaluation (AI-as-Grader)

To prove the RAG pipeline is effective, we compare its output against a baseline (Gemini answering without context) and grade both.

1.  **Baseline Generation:** The question is sent to `Gemini 2.5 Flash` *without* any context slides.
2.  **Gold Standard Comparison:** A "gold standard" answer (as an image) is provided for each question.
3.  **AI Grading:** `Gemini 2.5 Flash` is used a third time, but as an **evaluator**. It is given the gold image, the RAG answer, and the baseline answer. It scores both candidates from 0.0 to 1.0 based on **approach and format similarity**.

The final results show that the RAG-powered candidate consistently scores higher in mimicking the correct academic format from the source material.

## 🤖 Models & Frameworks Used

This project orchestrates four different models for distinct tasks:

| Role | Model | Purpose |
| :--- | :--- | :--- |
| **Image Summarizer** | `microsoft/Phi-4-multimodal-instruct` | (Stage 1) Creates text summaries from page images. |
| **Retrieval LLM** | `microsoft/Phi-4-mini-instruct` | (Stage 2) Finds relevant page numbers from the vector DB. |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` | (Stage 1) Embeds the text summaries for retrieval. |
| **Generator & Grader**| `Gemini 2.5 Flash` | (Stage 2 & 3) Generates the final answer from images and evaluates the results. |
| **Framework** | `LangChain` | Orchestrates the RAG chain, prompts, and document loading. |
| **Vector DB** | `ChromaDB` | Stores and retrieves the page summary embeddings. |

## 🛠️ How to Run

> **Recommended Environment:**
> It is highly recommended to run this notebook in **Google Colab**. The environment comes with many of the necessary packages (like `torch`, `transformers`, etc.) pre-installed. The installation script in **Cell 1** is designed for Colab and only installs the *missing* dependencies. Running locally may require additional manual setup for packages like PyTorch with CUDA.

1.  **Open in Colab:**
    * Upload the `RAG.ipynb` file to your Google Drive.
    * Open the notebook with Google Colaboratory.
    * Change the runtime type to use a **GPU accelerator** (e.g., A100) via `Runtime > Change runtime type`. This is required for the `Phi-4` and `flash-attn` models.

2.  **Install Dependencies (Cell 1):**
    * Run the first code cell. This will install `langchain`, `chromadb`, `pdf2image`, `poppler-utils`, and other required packages that are not already in the Colab environment.

3.  **Set API Key (Cell 18):**
    * This project requires a Google Gemini API key.
    * Generate a key at [Google AI Studio](https://aistudio.google.com/app/apikey).
    * In **Cell 18**, replace the placeholder key with your own:
        ```python
        genai.configure(api_key="YOUR_API_KEY_HERE")
        ```
    * *Tip: For better security, use Colab's "Secrets" feature (click the key icon on the left) to store your API key.*

4.  **Execute the Notebook (Cells 2-22):**
    * Run all subsequent cells.
    * **Data Download (Cell 4):** The notebook will automatically download the `algo_merged.pdf` and the `algo_answer.zip` (containing questions and gold answers) using `gdown`.
    * **PDF Conversion (Cell 5):** This will convert the PDF to images and save them in the `/images` directory.
    * **Caption Generation (Cell 7):** This is the most time-consuming step (it can take a while). It will iterate through all 105 pages and generate summaries using the `Phi-4` model.
        * **To skip this step:** The notebook includes a commented-out cell (**Cell 10**) to upload a pre-computed `summary.txt` or `summary2.txt`. If you have this file, you can upload it and run that cell to skip the long summarization process.

5.  **Review Results:**
    * **Cell 17:** Shows the retrieved page numbers for each question.
    * **Cell 20:** Shows the final answers generated *with* context (the RAG pipeline) and *without* context (baseline).
    * **Cell 22:** Shows the final JSON-formatted evaluation, scoring both approaches against the gold standard.