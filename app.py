import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline
import time

# --- IMPORT YOUR CUSTOM RAG ENGINE ---
from rag_engine import build_knowledge_base, retrieve_context

# --- CUSTOM CSS: ZEN GLASSMORPHISM DARK THEME ---
custom_css = """
/* ===== BASE DARK THEME ===== */
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d1f2d 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
}

/* ===== GLASSMORPHISM CHAT WINDOW ===== */
.chatbot {
    background: rgba(30, 41, 59, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(52, 211, 153, 0.2) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

/* ===== MESSAGE BUBBLES ===== */
.message {
    border-radius: 16px !important;
    padding: 12px 16px !important;
}

.message.user {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
}

.message.bot, .message.assistant {
    background: rgba(51, 65, 85, 0.9) !important;
    border: 1px solid rgba(100, 116, 139, 0.5) !important;
    color: #e2e8f0 !important;
    border-radius: 20px 20px 20px 4px !important;
}

/* ===== INPUT TEXTBOX ===== */
.textbox, textarea, input[type="text"] {
    border-radius: 14px !important;
    border: 2px solid rgba(71, 85, 105, 0.8) !important;
    background: rgba(30, 41, 59, 0.9) !important;
    color: #e2e8f0 !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}

.textbox:focus-within, textarea:focus, input[type="text"]:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15) !important;
}

/* ===== BUTTONS ===== */
#send-btn {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
}

#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}

#clear-btn {
    background: rgba(51, 65, 85, 0.8) !important;
    border: 2px solid rgba(71, 85, 105, 0.8) !important;
    color: #10b981 !important;
    font-weight: 500 !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    background: rgba(71, 85, 105, 0.9) !important;
    border-color: #10b981 !important;
}

/* ===== RIGHT PANEL (GLASSMORPHISM) ===== */
.panel {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(52, 211, 153, 0.15) !important;
}

/* ===== LABELS & BADGES ===== */
.label, .label-wrap {
    background: rgba(51, 65, 85, 0.9) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(71, 85, 105, 0.6) !important;
    color: #e2e8f0 !important;
}

/* ===== ACCORDION ===== */
.accordion {
    background: rgba(30, 41, 59, 0.8) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(71, 85, 105, 0.5) !important;
    color: #e2e8f0 !important;
}

/* ===== CHECKBOXES ===== */
.checkbox-group, .checkbox-label, .form {
    background: transparent !important;
    color: #e2e8f0 !important;
}

input[type="checkbox"] {
    accent-color: #10b981 !important;
}

/* ===== TYPOGRAPHY ===== */
h1, .markdown h1 {
    color: #10b981 !important;
    font-weight: 700 !important;
    text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
}

h3, .markdown h3 {
    color: #34d399 !important;
}

.markdown-text, .prose, p, span, label {
    color: #cbd5e1 !important;
}

/* ===== BREATHING ANIMATION KEYFRAMES ===== */
@keyframes breathe {
    0%, 100% { 
        transform: scale(1); 
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
    }
    25% { 
        transform: scale(1.4); 
        box-shadow: 0 0 40px rgba(45, 212, 191, 0.6);
    }
    50% { 
        transform: scale(1.4); 
        box-shadow: 0 0 40px rgba(45, 212, 191, 0.6);
    }
    75% { 
        transform: scale(1); 
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
    }
}

@keyframes breathe-text {
    0%, 100% { opacity: 1; }
    12.5% { opacity: 0; }
    25% { opacity: 1; }
    37.5% { opacity: 0; }
    50% { opacity: 1; }
    62.5% { opacity: 0; }
    75% { opacity: 1; }
    87.5% { opacity: 0; }
}

.breathing-circle {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2dd4bf, #14b8a6);
    margin: 20px auto;
    animation: breathe 16s ease-in-out infinite;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 14px;
}

.breathing-container {
    text-align: center;
    padding: 20px;
    background: rgba(30, 41, 59, 0.9);
    border-radius: 16px;
    border: 1px solid rgba(45, 212, 191, 0.3);
}

.breathing-instruction {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 10px;
}

/* ===== GROUNDING CHECKLIST STYLING ===== */
.grounding-container {
    background: rgba(30, 41, 59, 0.9);
    border-radius: 16px;
    border: 1px solid rgba(251, 191, 36, 0.3);
    padding: 16px;
}

.grounding-title {
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 12px;
}
"""

# --- ANIMATED BREATHING WIDGET HTML ---
BREATHING_HTML = """
<div class="breathing-container">
    <h4 style="color: #2dd4bf; margin-bottom: 10px;">🌬️ Box Breathing</h4>
    <div class="breathing-circle">
        Breathe
    </div>
    <div class="breathing-instruction">
        <strong>Inhale</strong> (4s) → <strong>Hold</strong> (4s) → <strong>Exhale</strong> (4s) → <strong>Hold</strong> (4s)
    </div>
    <p style="color: #64748b; font-size: 12px; margin-top: 12px;">
        Follow the circle. It expands as you inhale, holds, then shrinks as you exhale.
    </p>
</div>
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
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

# C. Knowledge Base (Local Vector DB)
vector_db = build_knowledge_base()

# --- ENHANCED EMOTION CLASSIFICATION ---
def classify_emotion(user_message):
    """
    Enhanced emotion classification with context-aware overrides.
    Fixes: Procrastination/academic stress should NOT be classified as 'sadness'.
    """
    message_lower = user_message.lower()
    
    # --- PROCRASTINATION/ACADEMIC STRESS OVERRIDE ---
    procrastination_keywords = [
        'homework', 'assignment', 'exam', 'exams', 'test', 'study', 'studying',
        'lazy', 'procrastinating', 'procrastination', "can't focus", "don't want to work",
        'deadline', 'project', 'essay', 'paper', 'grades', 'school', 'college',
        'unmotivated', 'distracted', 'putting off', 'avoiding work'
    ]
    
    if any(keyword in message_lower for keyword in procrastination_keywords):
        # Check if it's more stress or actual sadness
        sadness_indicators = ['crying', 'cry', 'hopeless', 'worthless', 'grief', 'died', 'death', 'lost someone']
        if not any(sad in message_lower for sad in sadness_indicators):
            return "stress", 0.85  # Override to stress, not sadness
    
    # --- PANIC/ANXIETY OVERRIDE ---
    panic_keywords = ['panic', 'panicking', 'panic attack', 'heart racing', "can't breathe", 'hyperventilating']
    if any(keyword in message_lower for keyword in panic_keywords):
        return "panic", 0.90
    
    anxiety_keywords = ['anxious', 'anxiety', 'worried', 'nervous', 'scared', 'terrified', 'fear']
    if any(keyword in message_lower for keyword in anxiety_keywords):
        return "anxiety", 0.85
    
    # --- DISSOCIATION OVERRIDE ---
    dissociation_keywords = ['unreal', 'floating', 'disconnected', 'numb', 'out of body', 'watching myself', 'not real']
    if any(keyword in message_lower for keyword in dissociation_keywords):
        return "dissociation", 0.80
    
    # --- DEFAULT: Use ML Model ---
    try:
        result = emotion_classifier(user_message)
        emotion = result[0][0]['label']
        confidence = result[0][0]['score']
        return emotion, confidence
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "neutral", 0.0


# --- 2. THE CORE INTELLIGENCE PIPELINE ---
def agent_logic(user_message, history):
    """
    The 'Brain' of the application.
    Flow: User -> Emotion Detect -> Knowledge Search -> LLM Reasoning -> UI Decision
    """
    
    # Step A: PERCEPTION (Enhanced Emotion)
    emotion, confidence = classify_emotion(user_message)

    # Step B: MEMORY (RAG Retrieval)
    knowledge_context = retrieve_context(vector_db, user_message)
    
    # Step C: REASONING (System Prompt)
    emotion_hints = {
        "sadness": "They are experiencing genuine sadness, possibly grief or hopelessness.",
        "fear": "They appear anxious or fearful about something.",
        "anger": "They seem frustrated or upset.",
        "joy": "They seem to be in a positive mood.",
        "surprise": "Something unexpected may have happened.",
        "disgust": "They may be experiencing aversion.",
        "neutral": "Their emotional state is unclear.",
        "stress": "They are stressed, likely about academic work or responsibilities.",
        "panic": "They are experiencing panic or acute anxiety symptoms.",
        "anxiety": "They are feeling anxious or worried.",
        "dissociation": "They may be feeling disconnected or unreal."
    }
    emotion_hint = emotion_hints.get(emotion, "Their emotional state is unclear.")
    
    system_prompt = f"""You are Zen, a warm and supportive mental health companion for students. You're like a caring friend who knows about mental wellness.

CONTEXT (use naturally, never mention directly):
- {emotion_hint}
- Reference: {knowledge_context if knowledge_context else "Use general supportive techniques."}

YOUR PERSONALITY:
- Warm, genuine, never clinical or robotic
- Speak like a supportive friend, not a textbook therapist
- Use casual language and contractions
- NEVER mention "confidence levels", "scores", "databases", or technical terms
- NEVER say "I detect that you're feeling..." 

HOW TO RESPOND:
1. Acknowledge what they shared (don't label their emotion)
2. Share helpful insight or technique naturally
3. Keep it brief - 2-3 paragraphs max
4. End with an open question or gentle suggestion

Remember: Academic stress is NOT sadness. Procrastination needs motivation tips, not grief counseling."""

    # Prepare messages for Llama-3
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # Step D: GENERATION (Stream Response)
    partial_response = ""
    try:
        for msg in client.chat_completion(messages, max_tokens=512, stream=True):
            if msg.choices and len(msg.choices) > 0:
                token = msg.choices[0].delta.content
                if token:
                    partial_response += token
                yield partial_response, emotion
    except Exception as e:
        yield f"I'm having trouble connecting right now. Let's try again in a moment. (Error: {str(e)})", emotion


# --- 3. UI HELPER FUNCTIONS ---
def chat_wrapper(user_input, history):
    """Bridges the UI and the agent logic with widget visibility control."""
    
    if not user_input.strip():
        yield history, "", "Type a message...", gr.update(visible=False), gr.update(visible=False)
        return
    
    generated_text = ""
    detected_emotion = "neutral"
    
    # Stream the response
    for text_chunk, emotion in agent_logic(user_input, history):
        generated_text = text_chunk
        detected_emotion = emotion
        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": generated_text}
        ]
        yield new_history, "", f"Detected: {detected_emotion.upper()}", gr.update(visible=False), gr.update(visible=False)

    # --- WIDGET VISIBILITY LOGIC ---
    show_breathing = gr.update(visible=False)
    show_grounding = gr.update(visible=False)
    
    # Show BREATHING for panic/anxiety
    if detected_emotion in ["panic", "anxiety", "fear"]:
        show_breathing = gr.update(visible=True)
    
    # Show GROUNDING for dissociation/panic
    if detected_emotion in ["dissociation", "panic"]:
        show_grounding = gr.update(visible=True)
    
    # Final yield
    final_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": generated_text}
    ]
    yield final_history, "", f"Detected: {detected_emotion.upper()}", show_breathing, show_grounding


# --- 4. THE DASHBOARD UI ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    css=custom_css,
    title="Zen Companion"
) as demo:
    
    gr.Markdown("# 🌿 Zen: Your Mental Health Companion")
    gr.Markdown("*A safe space to talk. I'm here to listen and help.*")
    
    with gr.Row():
        # --- LEFT COLUMN: Chat Interface ---
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=480, 
                label="Conversation",
                show_copy_button=True,
                placeholder="Start a conversation... I'm here to help."
            )
            msg = gr.Textbox(
                label="Your Message", 
                placeholder="How are you feeling today?",
                autofocus=True,
                lines=2
            )
            with gr.Row():
                send_btn = gr.Button("Send", elem_id="send-btn", variant="primary")
                clear_btn = gr.Button("Clear Chat", elem_id="clear-btn")

        # --- RIGHT COLUMN: Dynamic Wellness Panel ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🧠 Emotional Insight")
            mood_badge = gr.Label(value="Waiting...", label="Current State")
            
            # --- BREATHING WIDGET (Animated) ---
            with gr.Group(visible=False) as breathing_widget:
                gr.Markdown("---")
                gr.HTML(BREATHING_HTML)
            
            # --- GROUNDING WIDGET (5-4-3-2-1) ---
            with gr.Group(visible=False) as grounding_widget:
                gr.Markdown("---")
                gr.Markdown("### 🦶 5-4-3-2-1 Grounding")
                gr.Markdown("*Focus on your senses to reconnect with the present.*")
                gr.CheckboxGroup(
                    choices=[
                        "👀 5 Things I See",
                        "✋ 4 Things I Touch", 
                        "👂 3 Things I Hear",
                        "👃 2 Things I Smell",
                        "👅 1 Thing I Taste"
                    ],
                    label="Check off as you go:",
                    interactive=True
                )

            # Static Resources
            with gr.Accordion("📚 About the Knowledge Base", open=False):
                gr.Markdown("""
                This companion is powered by a curated library of mental health resources including:
                - Breathing techniques
                - Grounding exercises  
                - Cognitive behavioral strategies
                - Sleep hygiene tips
                - Stress management guides
                """)

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
    
    clear_btn.click(lambda: ([], "", "Waiting..."), None, [chatbot, msg, mood_badge], queue=False)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)