from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Initialize tokenizer and model
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def stream_response(input_text: str):
    # Encode the input text
    input_ids = tok.encode(input_text, return_tensors="pt")
    
    # Generate response in a streaming fashion
    chatbot_output = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tok.eos_token_id,
        do_sample=True
    )
    
    response_text = tok.decode(chatbot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear")

    def user(user_message: str, history: list):
        return "", history + [[user_message, None]]

    def bot(history: list):
        input_text = ' '.join([h[0] for h in history if h[0]])  # Join all user messages
        response = stream_response(input_text)
        for char in response:
            if history[-1][1] is None:
                history[-1][1] = char
            else:
                history[-1][1] += char
            yield history
            # Use time.sleep(0.1) if you want to simulate typing delay

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
        bot, inputs=chatbot, outputs=chatbot
    )
    clear.click(lambda: None, None, chatbot)
    
demo.launch()
