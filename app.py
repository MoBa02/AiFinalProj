import streamlit as st
import torch
import wikipediaapi
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
import glob

# --- Configuration ---
MODEL_PATH = "./QAWikiModel"  # Path to your fine-tuned model
WIKI_USER_AGENT = 'MyStreamlitQADemo/1.0 (your-email@example.com; for personal app)'

CHUNK_SIZE_CHARS = 1500  # Approximate size of each chunk in characters
CHUNK_OVERLAP_CHARS = 200  # Number of characters to overlap between chunks

# --- Caching Functions ---
@st.cache_resource  # Cache the model loading
def load_qa_pipeline():
    st.write("Loading QA model...")
    try:
        # Determine device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_num = 0 if device == "cuda" else -1
        
        if device == "cuda":
            st.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.write("Using CPU for inference.")

        # Check model directory contents
        if os.path.exists(MODEL_PATH):
            st.write(f"Model directory exists: {MODEL_PATH}")
            files = glob.glob(os.path.join(MODEL_PATH, "*"))
            st.write(f"Found {len(files)} files in model directory.")
            
            # List model files
            for f in files:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                st.write(f"File: {os.path.basename(f)} ({size_mb:.2f} MB)")
        else:
            st.error(f"Model directory not found: {MODEL_PATH}")
            return None

        # Load model and tokenizer directly with optimized settings
        st.write("Loading model and tokenizer...")
        try:
            # Try loading tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            st.write("Tokenizer loaded successfully")
            
            # Try to load model with specific settings to avoid HeaderTooLarge error
            model = AutoModelForQuestionAnswering.from_pretrained(
                MODEL_PATH,
                local_files_only=True,  # Don't try to download
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                # Use a more robust approach for loading safetensors
                use_safetensors=True if any(f.endswith('.safetensors') for f in files) else None
            )
            st.write("Model loaded successfully")
            
            # Create QA pipeline
            qa_pipe = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=device_num
            )
            st.success("QA Model pipeline created successfully!")
            return qa_pipe
            
        except Exception as e:
            st.error(f"Error loading model/tokenizer: {e}")
            
            # Fallback attempt with different settings
            st.warning("Attempting alternate loading method...")
            try:
                # Try model loading with minimum parameters
                model = AutoModelForQuestionAnswering.from_pretrained(
                    MODEL_PATH, 
                    local_files_only=True
                )
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                
                qa_pipe = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_num
                )
                st.success("QA Model loaded with fallback method!")
                return qa_pipe
            except Exception as fallback_error:
                st.error(f"Fallback loading failed: {fallback_error}")
                return None
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        st.error("Please ensure the model path is correct and all model files are present.")
        return None

# Cache Wikipedia API client
@st.cache_resource
def get_wiki_client():
    st.write("Initializing Wikipedia API client...")
    client = wikipediaapi.Wikipedia(user_agent=WIKI_USER_AGENT, language='en')
    st.success("Wikipedia API client initialized.")
    return client

# Cache fetching Wikipedia page content
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_wiki_content(_wiki_client, page_title):
    st.write(f"Fetching Wikipedia page: '{page_title}'...")
    page = _wiki_client.page(page_title)
    if not page.exists():
        st.warning(f"Wikipedia page '{page_title}' not found.")
        return None
    if not page.text or len(page.text.strip()) < 50:
        st.warning(f"Page '{page_title}' content is too short or empty.")
        return None
    st.info(f"Page '{page_title}' fetched (Length: {len(page.text)} chars).")
    return page.text

def get_answer_from_chunks(qa_pipeline, question, full_context):
    """
    Splits the full_context into chunks and gets the best answer.
    Uses character-based chunking for simplicity.
    """
    if not full_context:
        return None

    context_len = len(full_context)
    all_answers = []

    st.write(f"Processing context in chunks (Chunk size: {CHUNK_SIZE_CHARS} chars, Overlap: {CHUNK_OVERLAP_CHARS} chars)")

    # Process chunks with progress bar
    progress_bar = st.progress(0)
    total_chunks = max(1, (context_len // (CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS)))
    chunk_count = 0
    
    # Iterate through the context with a sliding window (character-based)
    for i in range(0, context_len, CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS):
        chunk_start = i
        chunk_end = min(i + CHUNK_SIZE_CHARS, context_len)
        context_chunk = full_context[chunk_start:chunk_end]

        if len(context_chunk.strip()) < 50:  # Skip very small or empty chunks
            continue
        
        try:
            # Update progress
            chunk_count += 1
            progress_bar.progress(min(1.0, chunk_count / total_chunks))
            
            # Process chunk
            result = qa_pipeline(question=question, context=context_chunk)
            if result and result['answer']:  # Ensure an answer was found
                # Adjust start/end character positions to be relative to the full_context
                result['start_global'] = chunk_start + result['start']
                result['end_global'] = chunk_start + result['end']
                all_answers.append(result)
        except Exception as e:
            st.warning(f"Error processing chunk {chunk_count}/{total_chunks}: {e}")
            continue  # Move to the next chunk

    progress_bar.empty()  # Remove progress bar when done

    if not all_answers:
        st.warning("No answer found in any of the processed chunks.")
        return None

    # Find the answer with the highest score
    best_answer = max(all_answers, key=lambda x: x['score'])
    st.info(f"Found {len(all_answers)} potential answers. Best score: {best_answer['score']:.4f}")
    return best_answer


# --- Streamlit App UI ---
st.set_page_config(page_title="Wikipedia QA Demo", layout="wide")
st.title("🤖 Question Answering with Wikipedia Articles")
st.markdown("""
Welcome! This app uses a fine-tuned DeBERTa model to answer questions based on Wikipedia articles.
Enter a Wikipedia article title and your question below.
""")

with st.spinner("Loading model (this may take a minute)..."):
    qa_pipeline_instance = load_qa_pipeline()
    wiki_client_instance = get_wiki_client()

if qa_pipeline_instance is None:
    st.error("Failed to initialize the QA model. Check the error messages above.")
    st.stop()

st.sidebar.header("Controls")
article_title_input = st.sidebar.text_input("Wikipedia Article Title:", placeholder="e.g., Albert Einstein")
question_input_raw = st.sidebar.text_area("Your Question:", placeholder="e.g., When was he born?")
submit_button = st.sidebar.button("Get Answer")

if submit_button:
    if not article_title_input:
        st.sidebar.error("Please enter a Wikipedia article title.")
    elif not question_input_raw:
        st.sidebar.error("Please enter your question.")
    else:
        with st.spinner(f"Fetching '{article_title_input}', chunking, and finding answer..."):
            article_context_full = fetch_wiki_content(wiki_client_instance, article_title_input)

            if article_context_full:
                # Input Normalization for the Question
                user_question = question_input_raw.strip()
                while user_question.endswith("?"):
                    user_question = user_question[:-1]
                normalized_question = user_question.strip() + "?"
                st.write(f"Normalized question: `{normalized_question}`")

                # Get answer using the chunking strategy
                best_result_from_chunks = get_answer_from_chunks(
                    qa_pipeline_instance,
                    normalized_question,
                    article_context_full
                )

                if best_result_from_chunks:
                    st.subheader("🔍 Best Answer Found:")
                    st.success(f"**{best_result_from_chunks['answer']}**")
                    st.caption(f"Confidence Score: {best_result_from_chunks['score']:.4f}")

                    with st.expander("View Context Snippet (from original article where best answer was found)"):
                        # Use 'start_global' and 'end_global' for the full context
                        start_char, end_char = best_result_from_chunks['start_global'], best_result_from_chunks['end_global']
                        snippet_radius = 200
                        snippet_start = max(0, start_char - snippet_radius)
                        snippet_end = min(len(article_context_full), end_char + snippet_radius)
                        
                        # Prepare context snippet with highlighted answer
                        context_before = article_context_full[snippet_start:start_char]
                        answer_text = article_context_full[start_char:end_char]
                        context_after = article_context_full[end_char:snippet_end]
                        
                        st.markdown(
                            f"...{context_before}<mark>**{answer_text}**</mark>{context_after}...", 
                            unsafe_allow_html=True
                        )
                else:
                    st.error("Could not find an answer in the article.")
else:
    st.info("Enter an article title and a question in the sidebar, then click 'Get Answer'.")

# Include system information for debugging
with st.expander("System Information"):
    st.markdown("### Environment Details")
    st.write(f"PyTorch Version: {torch.__version__}")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"CUDA Version: {torch.version.cuda}")
        st.write(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    st.markdown("### Model Configuration")
    st.write(f"Model Path: {MODEL_PATH}")
    
    # Print memory info - helpful to diagnose potential OOM issues
    if torch.cuda.is_available():
        st.write(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        st.write(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        st.write(f"GPU Memory Free: {torch.cuda.get_device_properties(0).total_memory/1024**3 - torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Streamlit](https://streamlit.io) & [Hugging Face Transformers](https://huggingface.co/transformers)")
