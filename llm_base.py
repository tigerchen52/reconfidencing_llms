import os
os.environ['TRANSFORMERS_CACHE'] = "data/parietal/store3/soda/lihu/hf_model/"
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
from registry import register
from functools import partial
registry = {}
register = partial(register, registry=registry)


class LanguageModel(object):  
   
    def __init__(self, model_name):
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            #device_map="auto",
            #torch_dtype=torch.float16,
            trust_remote_code=False,
            revision="main"
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=True)
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
        return [self.clean(d) for d in decoded]


@register('mistal-7b')
class Mistral(LanguageModel):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        super(Mistral, self).__init__(model_name
            )
        

@register('llama')
# class LLAMA(LanguageModel):
#     def __init__(self, model_name):
#         super(LLAMA, self).__init__(model_name)

class LLAMA(object):
    def __init__(self, model_name):
        #model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.use_default_system_prompt = False
    
    def clean(self, answer):
        special_token = ['<s>', '</s>', '<unk>']
        result = answer.split("[/INST]")[-1].strip()
        for token in special_token:
            result = result.replace(token, '').strip()
        return result.strip()
    
    def generate_answer(self, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
        messages = list()
        messages.append({"role": "user", "content": prompt})
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        
        model_inputs = input_ids.to(device)
        self.model.to(device)

        generated_ids = self.model.generate(model_inputs, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, num_return_sequences=sample_num, top_p=0.9,
        top_k=50, repetition_penalty=1.2, remove_invalid_values=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return [self.clean(d) for d in decoded]
        #return decoded



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

@register("zephyr")
class Zephyr(LanguageModel):  
   
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
    def generate_answer(self, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": prompt},
        ]
        
        print(messages)
        prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.model(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=50, top_p=0.95)
        #print(outputs[0]["generated_text"])
        
        result = outputs[0]['generated_text'].split("<|assistant|>")[-1].strip()
        return result


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
    
    
    
    # mistral_7b = Mistral("mistralai/Mistral-7B-Instruct-v0.1")
    # results = mistral_7b.generate_answer(prompt, sample_num, device)
    # print(results)
    
    
    # q_llama_7b = QLLAMA("TheBloke/Llama-2-7b-Chat-GPTQ")
    # results = q_llama_7b.generate_answer(prompt, sample_num, device)
    # print(results)
    
    # tiny_llama = TinyLLAMA("PY007/TinyLlama-1.1B-Chat-v0.3")
    # results = tiny_llama.generate_answer(prompt, sample_num, device)
    # print(results)
    
    zephyr = Zephyr("HuggingFaceH4/zephyr-7b-beta")
    results = zephyr.generate_answer(prompt, sample_num, device)
    print(results)