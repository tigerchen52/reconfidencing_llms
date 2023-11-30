import os
os.environ['TRANSFORMERS_CACHE'] = "data/parietal/store3/soda/lihu/hf_model/"
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from registry import register
from functools import partial
registry = {}
register = partial(register, registry=registry)


class LanguageModel(object):  
   
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
   
    def clean(self, answer):
        special_token = ['<s>', '</s>', '<unk>']
        result = answer.split("[/INST]")[-1].strip()
        for token in special_token:
            result = result.replace(token, '').strip()
        return result.strip()
    
    
    def generate_answer(self, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
        messages = [
        {"role": "user", "content": prompt}]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        self.model.to(device)

        generated_ids = self.model.generate(model_inputs, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, num_return_sequences=sample_num)
        decoded = self.tokenizer.batch_decode(generated_ids)
        print(decoded[0])
        return [self.clean(d) for d in decoded]


@register('mistal-7b')
class Mistral(LanguageModel):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        super(Mistral, self).__init__(model_name)
        

@register('llama')
class LLAMA(LanguageModel):
    def __init__(self, model_name):
        super(LLAMA, self).__init__(model_name)


@register('q-lllama')
class QLLAMA(LanguageModel):
    
    def __init__(self, model_name="TheBloke/Llama-2-7b-Chat-GPTQ"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=False,
            revision="main"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
   
    def clean(self, answer):
        special_token = ['<s>', '</s>', '<unk>']
        result = answer.split("[/INST]")[-1].strip()
        for token in special_token:
            result = result.replace(token, '').strip()
        return result.strip()
    
    
    def generate_answer(self, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
        messages = [{"role": "user", "content": prompt}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        self.model.to(device)
        
        results = list()
        for _ in range(sample_num):
            generated_ids = self.model.generate(model_inputs, max_new_tokens=max_new_tokens,  do_sample=do_sample, temperature=temperature)
            decoded = self.tokenizer.batch_decode(generated_ids)[0]
            answer = self.clean(decoded)
            results.append(answer)
        
        return results
    


@register('tiny_llama')
class TinyLLAMA(LanguageModel):  
   
    def __init__(self, model_name="PY007/TinyLlama-1.1B-Chat-v0.3"):
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        self.CHAT_EOS_TOKEN_ID = 32002

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token

    
    def generate_answer(self, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
        formatted_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        )
    
        sequences = self.model(
            formatted_prompt,
            do_sample=do_sample,
            temperature=temperature,
            top_k=50,
            top_p = 0.9,
            num_return_sequences=sample_num,
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.CHAT_EOS_TOKEN_ID,
        )
        results = list()
        for seq in sequences:
            answer = seq['generated_text'].replace(formatted_prompt, "")
            results.append(answer)
        return results


if __name__ == "__main__":
    sample_num = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # the device to load the model onto
    prompt = """
    Context: Suppose you know birth dates of all people.  
    Instructions: what is the birth date of {a}?   
    Constraint: The output formating should be this patter: yyyy-mm-dd.  
    Demonstration:  
    (1) Question: Albert Einstein?  
    Output is: 1879-04-14
    (2) Question: Michael Jackson?  
    Output is: 1958-08-29
    """.format(a="Joe Biden")
    
    
    
    mistral_7b = Mistral("mistralai/Mistral-7B-Instruct-v0.1")
    results = mistral_7b.generate_answer(prompt, sample_num, device)
    print(results)
    
    
    # q_llama_7b = QLLAMA("TheBloke/Llama-2-7b-Chat-GPTQ")
    # results = q_llama_7b.generate_answer(prompt, sample_num, device)
    # print(results)
    
    # tiny_llama = TinyLLAMA("PY007/TinyLlama-1.1B-Chat-v0.3")
    # results = tiny_llama.generate_answer(prompt, sample_num, device)
    # print(results)