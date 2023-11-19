from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Initialize tokenizer and model
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def stream_response(input_text: str):
    # Encode the input text
    input_ids = tok.encode(input_text, return_tensors="pt")

    # Generate one token at a time and yield it
    for _ in range(50):  # Limiting to 50 tokens to prevent too long generation
        # Generate one token
        chatbot_output = model.generate(
            input_ids,
            max_new_tokens=1,  # Generate one token at a time
            pad_token_id=tok.eos_token_id,
            do_sample=True,
            temperature=0.7  # Sampling temperature, adjust as needed
        )
        # Extract the last token
        last_token = chatbot_output[:, -1].tolist()[0]
        # Update the input_ids
        input_ids = torch.cat((input_ids, torch.tensor([[last_token]])), dim=1)
        # Decode and yield the token
        yield tok.decode([last_token], skip_special_tokens=True)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear")

    def user(user_message: str, history: list):
        return "", history + [[user_message, None]]

    def bot(history: list):
        input_text = ' '.join([h[0] for h in history if h[0]])  # Join all user messages
        # Call stream_response which yields one token at a time
        for char in stream_response(input_text):
            if history[-1][1] is None:
                history[-1][1] = char
            else:
                history[-1][1] += char
            yield history

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
        bot, inputs=chatbot, outputs=chatbot
    )
    clear.click(lambda: None, None, chatbot)
    
demo.launch()
