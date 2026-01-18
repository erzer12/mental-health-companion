import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline
import time

# --- IMPORT YOUR CUSTOM RAG ENGINE ---
from rag_engine import build_knowledge_base, retrieve_context

# --- CUSTOM CSS FOR CALMING DARK "ZEN" LOOK ---
custom_css = """
/* Dark calming gradient background */
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d1f2d 100%) !important;
    min-height: 100vh;
}

/* Dark chat window */
.chatbot {
    background: #1e293b !important;
    border-radius: 16px !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Style message bubbles */
.message {
    border-radius: 16px !important;
}

/* User messages - emerald accent */
.message.user {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
}

/* Bot messages - dark slate */
.message.bot, .message.assistant {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #e2e8f0 !important;
}

/* Input textbox */
.textbox, textarea, input[type="text"] {
    border-radius: 12px !important;
    border: 2px solid #475569 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

.textbox:focus-within, textarea:focus, input[type="text"]:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
}

/* Send button - vibrant emerald */
#send-btn {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
}

#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
}

/* Clear button - dark outline */
#clear-btn {
    background: #334155 !important;
    border: 2px solid #475569 !important;
    color: #10b981 !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    background: #475569 !important;
    border-color: #10b981 !important;
}

/* Right panel styling */
.panel {
    background: #1e293b !important;
    border-radius: 16px !important;
    border: 1px solid #334155 !important;
}

/* Labels and badges */
.label, .label-wrap {
    background: #334155 !important;
    border-radius: 12px !important;
    border: 1px solid #475569 !important;
    color: #e2e8f0 !important;
}

/* Accordion */
.accordion {
    background: #1e293b !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
}

/* Checkboxes in grounding widget */
.checkbox-group, .checkbox-label {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* Title styling */
h1, .markdown h1 {
    color: #10b981 !important;
    font-weight: 700 !important;
}

h3, .markdown h3 {
    color: #34d399 !important;
}

/* All text to light color */
.markdown-text, .prose, p, span, label {
    color: #e2e8f0 !important;
}

/* Group containers */
.group, .form, .block {
    background: transparent !important;
}

/* Dropdown and selects */
select, .dropdown {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
}
"""

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
    except Exception as e:
        print(f"Emotion detection error: {e}")
        emotion = "neutral"
        confidence = 0.0

    # Step B: MEMORY (RAG Retrieval)
    # We retrieve specific advice based on the user's text
    knowledge_context = retrieve_context(vector_db, user_message)
    
    # Step C: REASONING (System Prompt)
    # Map emotion to natural language hint (internal use only, not shown to user)
    emotion_hints = {
        "sadness": "They seem to be feeling down or sad.",
        "fear": "They appear anxious or worried about something.",
        "anger": "They might be frustrated or upset.",
        "joy": "They seem to be in a positive mood.",
        "surprise": "Something unexpected may have happened to them.",
        "disgust": "They may be experiencing aversion or discomfort.",
        "neutral": "Their emotional state is unclear."
    }
    emotion_hint = emotion_hints.get(emotion, "Their emotional state is unclear.")
    
    system_prompt = f"""You are Zen, a warm and supportive mental health companion for students. You're like a caring friend who happens to know a lot about mental wellness.

CONTEXT (use naturally, never mention these details directly):
- {emotion_hint}
- Reference material: {knowledge_context if knowledge_context else "Draw from general supportive counseling techniques."}

YOUR PERSONALITY:
- Warm, genuine, and never clinical or robotic
- You speak like a supportive friend, not a therapist reading from a textbook
- You use casual language, contractions, and occasional gentle humor when appropriate
- You NEVER mention "confidence levels", "scores", "databases", or any technical terms
- You NEVER say things like "I detect that you're feeling..." or "Your emotion is..."

HOW TO RESPOND:
1. Start by acknowledging what they shared (don't label their emotion, just reflect understanding)
2. Share a helpful insight or technique naturally woven into conversation
3. Keep it brief - 2-3 short paragraphs max
4. End with an open question or gentle suggestion, not a list of options

EXAMPLE OF WHAT NOT TO SAY: "Your confidence level is 0.37" or "I detect sadness in your message"
EXAMPLE OF NATURAL RESPONSE: "That sounds really tough. It makes total sense that you'd feel overwhelmed right now."
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
            # Guard against empty choices array
            if msg.choices and len(msg.choices) > 0:
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
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    css=custom_css,
    title="Zen Companion"
) as demo:
    
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
            with gr.Row():
                send_btn = gr.Button("Send", elem_id="send-btn", variant="primary")
                clear_btn = gr.Button("Clear Chat", elem_id="clear-btn")

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
    send_btn.click(
        chat_wrapper,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, mood_badge, breathing_widget, grounding_widget]
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)