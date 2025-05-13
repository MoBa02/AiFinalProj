import streamlit as st
import torch
import wikipediaapi
from transformers import pipeline

# --- Configuration ---
# Path to your saved fine-tuned model and tokenizer directory
MODEL_PATH = "./QAWikiModel"
WIKI_USER_AGENT = 'MyStreamlitQADemo/1.0 (wiz.mohammed4444@gmail.com; for personal app)'
ADAPTERS_HUB_ID_OR_PATH = "your-hf-username/your-deberta-qlora-squad-model-id"

CHUNK_SIZE_CHARS = 1500  # Approximate size of each chunk in characters
CHUNK_OVERLAP_CHARS = 200 # Number of characters to overlap between chunks

# --- Caching Functions ---
# Cache the model loading to avoid reloading on every interaction
@st.cache_resource  # Use cache_resource for objects like models
def load_qa_pipeline(model_path):
    st.write(f"Loading QA model from {model_path}...")
    try:
        # Determine device (GPU if available, otherwise CPU)
        device_num = 0 if torch.cuda.is_available() else -1
        if device_num == 0:
            st.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.write("Using CPU for inference.")

        qa_pipe = pipeline(
            "question-answering",
            model=model_path,
            tokenizer=model_path,
            device=device_num
        )
        st.success("QA Model loaded successfully!")
        return qa_pipe
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

# Cache fetching Wikipedia page content for the session to avoid re-fetching on widget interactions
# You might want a more sophisticated caching or no caching if freshness is critical
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_wiki_content(_wiki_client, page_title): # _wiki_client to make it part of cache key if client changes
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

    # Split context into words to approximate token count for chunking later if needed
    # For now, using character based chunking
    
    context_len = len(full_context)
    all_answers = []

    st.write(f"Processing context in chunks (Chunk size: {CHUNK_SIZE_CHARS} chars, Overlap: {CHUNK_OVERLAP_CHARS} chars)")

    # Iterate through the context with a sliding window (character-based)
    for i in range(0, context_len, CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS):
        chunk_start = i
        chunk_end = min(i + CHUNK_SIZE_CHARS, context_len)
        context_chunk = full_context[chunk_start:chunk_end]

        if len(context_chunk.strip()) < 50: # Skip very small or empty chunks
            continue
        
        # st.write(f"Processing chunk {chunk_start}-{chunk_end}...") # Optional: for debugging
        try:
            result = qa_pipeline(question=question, context=context_chunk)
            if result and result['answer']: # Ensure an answer was found
                # Adjust start/end character positions to be relative to the full_context
                result['start_global'] = chunk_start + result['start']
                result['end_global'] = chunk_start + result['end']
                all_answers.append(result)
        except Exception as e:
            st.warning(f"Error processing a chunk: {e}")
            continue # Move to the next chunk

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
Welcome! This app uses a fine-tuned Transformer model to answer questions based on Wikipedia articles.
Enter a Wikipedia article title and your question below.
""")

qa_pipeline_instance = load_qa_pipeline(MODEL_PATH)
wiki_client_instance = get_wiki_client()

if qa_pipeline_instance is None:
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
                        
                        highlighted_text = (
                            article_context_full[snippet_start:start_char] +
                            f"<mark style='background-color: yellow;'>**{article_context_full[start_char:end_char]}**</mark>" +
                            article_context_full[end_char:snippet_end]
                        )
                        st.markdown(f"...{highlighted_text}...", unsafe_allow_html=True)
                else:
                    st.error("Could not find an answer in the article.")
            # else: article_context_full is None, error already shown by fetch_wiki_content
else:
    st.info("Enter an article title and a question in the sidebar, then click 'Get Answer'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Streamlit](https://streamlit.io) & [Hugging Face Transformers](https://huggingface.co/transformers)")
