import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline
import time

# --- IMPORT YOUR CUSTOM RAG ENGINE ---
from rag_engine import build_knowledge_base, retrieve_context

# --- 1. SETUP: Models & Knowledge ---
print("⏳ Initializing AI Models...")

# A. Emotion Detection (Local)
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=1
)

# B. Chat Model (Serverless)
# Make sure you have added your HF_TOKEN in the Space Settings > Secrets
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

# C. Knowledge Base (Local Vector DB)
# This will look for the 'vectorstore' folder or build it from 'data/*.pdf'
vector_db = build_knowledge_base()

# --- 2. THE CORE INTELLIGENCE PIPELINE ---
def agent_logic(user_message, history):
    """
    The 'Brain' of the application.
    Flow: User -> Emotion Detect -> Knowledge Search -> LLM Reasoning -> UI Decision
    """
    
    # Step A: PERCEPTION (Emotion)
    try:
        emotion_res = emotion_classifier(user_message)
        emotion = emotion_res[0][0]['label']
        confidence = emotion_res[0][0]['score']
    except:
        emotion = "neutral"
        confidence = 0.0

    # Step B: MEMORY (RAG Retrieval)
    # We retrieve specific advice based on the user's text
    knowledge_context = retrieve_context(vector_db, user_message)
    
    # Step C: REASONING (System Prompt)
    system_prompt = f"""
    You are 'Zen', a compassionate mental health companion for students.
    
    CURRENT USER STATE:
    - Emotion: {emotion} (Confidence: {confidence:.2f})
    
    RELEVANT KNOWLEDGE FROM DATABASE:
    {knowledge_context if knowledge_context else "No specific documents found. Use general psychological first aid."}
    
    INSTRUCTIONS:
    1. Validate the user's feelings first.
    2. Use the 'RELEVANT KNOWLEDGE' to provide specific, actionable advice (e.g. hotline numbers, specific techniques).
    3. If the user seems panicked or stressed, suggest a breathing exercise.
    4. Keep the response warm, short, and conversational.
    """

    # Prepare messages for Llama-3
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # Step D: GENERATION (Stream Response)
    partial_response = ""
    # We catch errors in case the Inference API is busy
    try:
        for msg in client.chat_completion(messages, max_tokens=512, stream=True):
            token = msg.choices[0].delta.content
            if token:  # Handle None tokens from streaming API
                partial_response += token
            yield partial_response, emotion
    except Exception as e:
        yield f"I'm having trouble connecting to my brain right now. (Error: {str(e)})", emotion

# --- 3. UI HELPER FUNCTIONS ---
def chat_wrapper(user_input, history):
    # This function bridges the UI and the Logic
    
    generated_text = ""
    detected_emotion = "neutral"
    
    # 1. Run the agent (Stream text)
    for text_chunk, emotion in agent_logic(user_input, history):
        generated_text = text_chunk
        detected_emotion = emotion
        # Stream the chat update immediately (Gradio 6 message format)
        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": generated_text}
        ]
        yield new_history, "", f"Detected: {detected_emotion.upper()}", gr.update(visible=False), gr.update(visible=False)

    # 2. Post-Processing: Decide which widget to show
    show_breathing = gr.update(visible=False)
    show_grounding = gr.update(visible=False)
    
    # Logic: Show breathing for high-arousal negative emotions
    if detected_emotion in ["fear", "anger", "sadness"]:
        show_breathing = gr.update(visible=True)
    
    # Logic: Show grounding for panic keywords
    if "panic" in user_input.lower() or "overwhelm" in user_input.lower():
        show_grounding = gr.update(visible=True)
    
    # Final Yield with widgets active (Gradio 6 message format)
    final_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": generated_text}
    ]
    yield final_history, "", f"Detected: {detected_emotion.upper()}", show_breathing, show_grounding


# --- 4. THE DASHBOARD UI (Blocks) ---
with gr.Blocks(title="Zen Student Companion") as demo:
    
    gr.Markdown("# 🌿 Zen: Context-Aware Student Companion")
    
    with gr.Row():
        # --- LEFT COLUMN: Chat Interface ---
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            msg = gr.Textbox(
                label="Your Message", 
                placeholder="Type here (e.g., 'I'm stressed about exams')...",
                autofocus=True
            )
            clear_btn = gr.Button("Clear Chat")

        # --- RIGHT COLUMN: Dynamic Wellness Panel ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🧠 Live Analysis")
            mood_badge = gr.Label(value="Waiting...", label="Emotional State")
            
            # --- WIDGET 1: Breathing Exercise (Hidden by default) ---
            with gr.Group(visible=False) as breathing_widget:
                gr.Markdown("---")
                gr.Markdown("### 🌬️ Box Breathing Tool")
                gr.HTML("""
                <div style="text-align:center; padding: 20px; background-color: #e0f7fa; border-radius: 10px;">
                    <div style="font-size: 40px; animation: pulse 4s infinite;">🔵</div>
                    <p>Inhale (4s) ... Hold (4s) ... Exhale (4s)</p>
                </div>
                """)
            
            # --- WIDGET 2: Grounding Checklist (Hidden by default) ---
            with gr.Group(visible=False) as grounding_widget:
                gr.Markdown("---")
                gr.Markdown("### 🦶 5-4-3-2-1 Grounding")
                gr.Markdown("You mentioned panic. Let's ground ourselves.")
                gr.Checkbox(label="👀 5 things I see")
                gr.Checkbox(label="✋ 4 things I can touch")
                gr.Checkbox(label="👂 3 things I hear")
                gr.Checkbox(label="👃 2 things I smell")
                gr.Checkbox(label="👅 1 thing I taste")

            # Static Resources
            with gr.Accordion("📚 Knowledge Source", open=False):
                gr.Markdown("This agent is grounded in the documents found in the `/data` folder.")

    # --- 5. EVENT WIRING ---
    msg.submit(
        chat_wrapper, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg, mood_badge, breathing_widget, grounding_widget]
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)