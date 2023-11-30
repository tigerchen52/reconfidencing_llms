import gradio as gr
from confidence import run_nli

DESCRIPTION = """\
# Llama Chatbot with confidence scores 🩺
This space shows that we can teach LLMs to express how confident they are in their answers.
Since we can only access free CPUs, we use a tiny Llama ([TinyLlama-1.1B](https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3)) as the chatbot and an [NLI model](https://github.com/potsawee/selfcheckgpt) to get scores. <br/>
💯   There will be a score between 0 and 1 after each sentence, and a higher value means the sentence is more factual.<br/>
⏳ It takes 150s-300s to process each query, and we limit the token numbers of answers for saving time.
"""

def greet(query, history):
    results = run_nli(query, sample_size=3)
    return results
    #return "this is the result"


sample_list = [
    "Tell me something about Albert Einstein, e.g., a short bio with birth date and birth place",
    "Tell me something about Lihu Chen, e.g., a short bio with birth date and birth place",
    "How tall is the Eiffel Tower?"
]

iface = gr.ChatInterface(
    fn=greet,
    stop_btn=None,
    examples=sample_list,
    cache_examples=True
)

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    iface.render()
    #gr.Markdown(LICENSE)


demo.launch()