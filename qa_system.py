import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os

# Image processing functions
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def enhance_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def post_process_physics_text(text):
    physics_replacements = {
        r'(\d+)\s*ms?\^?-?1\b': r'\1 m/s',
        r'(\d+)\s*m/s\^?2\b': r'\1 m/s²',
        r'\bAp\b': 'Δp',
        r'\bdv\b': 'Δv',
        r'\bdt\b': 'Δt',
        r'(\d+)\s*([NJ])\b': r'\1 \2',
        r'(\d+)\s*kg\b': r'\1 kg',
        r'(\d+)\s*m\b': r'\1 m',
        r'(\d+)\s*s\b': r'\1 s',
        r'(\d+)\s*K\b': r'\1 K',
        r'(\d+)\s*Pa\b': r'\1 Pa',
        r'(\d+)\s*W\b': r'\1 W',
        r'\bF\s*=\s*ma\b': 'F = ma',
        r'\bE\s*=\s*mc\^?2\b': 'E = mc²',
        r'\bPV\s*=\s*nRT\b': 'PV = nRT',
        r'\bv\^?2\b': 'v²',
        r'\ba\^?2\b': 'a²',
        r'\bπ': 'π',
        r'\b([pv])1\b': r'\1₁',
        r'\b([pv])2\b': r'\1₂',
        r'(\d+)([,.])(\d+)': r'\1.\3',
    }

    for pattern, replacement in physics_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text

def ocr_physics_question(image_path):
    try:
        image = cv2.imread(image_path)
        preprocessed = preprocess_image(image)
        enhanced = enhance_image(preprocessed)

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-+=^°ΔπμρλσωΩ'

        text = pytesseract.image_to_string(enhanced, config=custom_config)
        corrected_text = post_process_physics_text(text)

        return corrected_text
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# QA System functions
def read_pdf(file_path: str = "phys-tbk.pdf") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pdf_reader = PdfReader(file_path)
    content = ""
    for page in tqdm(pdf_reader.pages, desc="Reading PDF"):
        content += page.extract_text()

    print(f"PDF processed. Total characters: {len(content)}")
    return content

def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks")
    if chunks:
        print(f"First chunk length: {len(chunks[0])}")
        print(f"First chunk preview: {chunks[0][:100]}...")
        print(f"Last chunk preview: {chunks[-1][-100:]}...")
    else:
        print("Warning: No chunks created!")
    return chunks

def create_embeddings(chunks: list) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks, show_progress_bar=True)

def save_data(chunks: list, embeddings: np.ndarray, save_path: str):
    np.savez_compressed(save_path, chunks=chunks, embeddings=embeddings)
    print(f"Data saved to {save_path}")

def preprocess_dataset(dataset, save_path="physics_dataset_embeddings.npz"):
    if os.path.exists(save_path):
        print("Loading pre-processed dataset...")
        data = np.load(save_path, allow_pickle=True)
        if 'chunks' not in data or 'embeddings' not in data:  # Ensure this line has proper indentation
            raise KeyError('Required keys "chunks" or "embeddings" not found in the dataset.')
        return data['chunks'], data['embeddings']  # This line should be properly indented as well
  
    print("Processing dataset...")
    texts = [f"{item['instruction']} {item['output']}" for item in dataset]
    embeddings = create_embeddings(texts)
    save_data(texts, embeddings, save_path)
    return texts, embeddings

def process_textbook(file_path: str = "phys-tbk.pdf", save_path: str = "physics_textbook_data.npz"):
    if os.path.exists(save_path):
        print("Loading pre-processed textbook data...")
        data = np.load(save_path, allow_pickle=True)
        return data['chunks'], data['embeddings']

    print("Processing textbook...")
    content = read_pdf(file_path)
    chunks = create_chunks(content)
    if not chunks:
        raise ValueError("No chunks created. Check the text splitting process.")
    embeddings = create_embeddings(chunks)
    save_data(chunks, embeddings, save_path)
    return chunks, embeddings

def load_physics_dataset():
    dataset = load_dataset("Akul/alpaca_physics_dataset")
    return dataset['train']

def setup_qa_system():
    # Process textbook data
    textbook_chunks, textbook_embeddings = process_textbook()

    # Load and preprocess the dataset
    dataset = load_physics_dataset()
    dataset_texts, dataset_embeddings = preprocess_dataset(dataset)

    # Ensure both lists are of the same type before concatenation
    all_texts = list(textbook_chunks) + list(dataset_texts)
    all_embeddings = np.vstack([textbook_embeddings, dataset_embeddings])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings.tolist())),
        embedding=embeddings,
        metadatas=[{"source": "textbook"} if i < len(textbook_chunks) else {"source": "dataset"}
                   for i in range(len(all_texts))]
    )

    return db.as_retriever(search_kwargs={"k": 5})

def sanitize_text(text):
    # Replace problematic characters for Mermaid compatibility
    sanitized = text.replace("(", "").replace(")", "")#.replace("*", " × ").replace("/", " ÷ ")
    
    # Replace line breaks with a space and clean up multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Remove bullet points and any leading hyphens
    sanitized = sanitized.replace('-', '').strip()
    
    return sanitized

def generate_calculation_flowchart(answer):
    print("Generating flowchart...")
    print(f"Answer received: {answer}")

    # Extract the steps from the given answer using regular expressions
    steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', answer, re.DOTALL)
    print(f"Steps extracted: {steps}")

    # Initialize an empty list for nodes and edges
    elements = []

    # Loop through each step to create nodes and links between them
    for i, step in enumerate(steps):
        node_id = f"A{i}"

        # Sanitize the text to avoid problematic characters and multiline labels
        step = sanitize_text(step)

        # Add the node to the elements list
        elements.append({
            "data": {
                "id": node_id,
                "label": step
            }
        })

        # Create the link between the current node and the next one
        if i < len(steps) - 1:
            next_node_id = f"A{i+1}"
            elements.append({
                "data": {
                    "source": node_id,
                    "target": next_node_id
                }
            })

    print(f"Generated elements: {elements}")
    return elements

def ask_question(retriever, question, conversation_history):
    
    if not question:
        raise ValueError("Question cannot be None")
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel('gemini-pro')

    assistant_prompt = """
    You are an expert physics tutor explaining a problem to a student. Also Explain the question overall solving in brief words. Answer the question in a clear, step-by-step manner that can be easily translated into a flowchart. Follow these guidelines:

    1. Pay attention to the conversation history and previous questions to maintain context.
    2. If the question is a follow-up or relates to previous questions, use the information from earlier in the conversation.
    3. For calculation questions:
       - Clearly state the given information, including any from previous questions if relevant.
       - List the relevant formulas needed to solve the problem.
       - Explain the solution process step-by-step, showing all work and reasoning.
       - Provide the final answer with appropriate units.
    4. Use clear, numbered steps that can be easily translated into a flowchart.
    5. If asked to calculate something specific, provide the calculation without asking for more information unless absolutely necessary.

    Adapt your response style to the nature of the question, providing a clear, structured explanation that follows a logical problem-solving flow.
    """

    # Extract the relevant information (assuming conversation_history contains dictionaries)
    conversation_context = "\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in conversation_history if 'question' in entry and 'answer' in entry]
    )

    prompt = f"{assistant_prompt}\n\nConversation history:\n{conversation_context}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = model.generate_content(prompt)
    answer = response.text

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Generate a flowchart based on the answer
    print("Generating flowchart for the answer...")
    flowchart = generate_calculation_flowchart(answer)
    print("\nHere's a flowchart of the process:")
    print(flowchart)

    # Return both the answer and flowchart (elements)
    return answer, flowchart

# Initialize QA system
print("Initializing QA system...")
retriever = setup_qa_system()
print("QA system initialized.")

# Configure Google AI
genai.configure(api_key="AIzaSyAmnd7HUcyYpPo5ZMBz2qSAYK5zmZCQ1ao")  # Replace with your actual API key
