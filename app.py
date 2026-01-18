import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline

# 1. Initialize the Sentiment Analysis Model (The "Empathy Sensor")
# We use a small, efficient model runs locally in the Space
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=1
)

# 2. Initialize the Chat Model (The "Brain")
# We use the Serverless API to call a powerful model (Llama-3-8B-Instruct)
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

# 3. Define the Safety & Crisis Logic
def get_system_prompt(emotion):
    """
    Adjusts the AI's persona based on the detected emotion.
    """
    base_prompt = (
        "You are a compassionate mental health support companion for students. "
        "Your goal is to listen, validate feelings, and offer gentle, non-medical advice. "
        "Keep responses concise, warm, and encouraging. "
    )
    
    if emotion == "sadness":
        return base_prompt + "The user feels sad. Focus on validation and gentle comfort."
    elif emotion == "fear":
        return base_prompt + "The user feels anxious or fearful. Suggest a grounding technique like 5-4-3-2-1."
    elif emotion == "anger":
        return base_prompt + "The user feels angry. Help them process this frustration calmly."
    else:
        return base_prompt

def respond(message, history):
    # --- Step A: Safety Check (Rule-based) ---
    crisis_keywords = ["suicide", "kill myself", "die", "end it all"]
    if any(word in message.lower() for word in crisis_keywords):
        yield "I am an AI, not a human. It sounds like you are in serious pain. Please reach out to a professional or call a crisis hotline immediately."
        return

    # --- Step B: Sentiment Analysis ---
    try:
        # Detect emotion
        emotion_result = emotion_classifier(message)
        detected_emotion = emotion_result[0][0]['label']
        confidence = emotion_result[0][0]['score']
        print(f"Detected: {detected_emotion} ({confidence:.2f})") # Logs for debugging
    except Exception as e:
        detected_emotion = "neutral"

    # --- Step C: Generate Response ---
    system_message = get_system_prompt(detected_emotion)
    
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
            
    messages.append({"role": "user", "content": message})

    response = ""
    # Stream the response token by token
    for message in client.chat_completion(
        messages,
        max_tokens=512,
        stream=True,
        temperature=0.7,
        top_p=0.95,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# 4. Build the UI
demo = gr.ChatInterface(
    respond,
    title="Student Wellness Companion 🌿",
    description="A safe space to chat. I can sense your mood and offer support.",
    examples=["I feel really overwhelmed with exams coming up.", "I feel lonely and isolated on campus.", "I'm just really angry at my professor."],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()