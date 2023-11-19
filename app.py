from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gradio as gr
import torch

# Initialize tokenizer and model
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_response(history):
    input_text = ' '.join(history)
    inputs = tok(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tok.eos_token_id)
    response = tok.decode(outputs[0], skip_special_tokens=True)
    return response.split(input_text)[-1].strip()  # Extract only the generated part

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message: str, history: list):
        return "", history + [[user_message, None]]

    def bot(history: list):
        bot_message = generate_response([h[0] for h in history])
        history[-1][1] = bot_message
        time.sleep(0.05)  # simulate typing delay
        return history

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        bot, inputs=chatbot, outputs=chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
if __name__ == "__main__":
    demo.launch()
