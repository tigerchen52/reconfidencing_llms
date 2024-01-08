import os
os.environ['TRANSFORMERS_CACHE'] = "data/parietal/store3/soda/lihu/hf_model/"
import re
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from llm_base import registry as LLM
from prompt_template import JustAskForCalibration_Prompts, RELATION_PROMPTS
from registry import register
from functools import partial
registry = {}
register = partial(register, registry=registry)




@register('nli')
class NLIConfidence(object):
    
    def __init__(self, device):
        self.device = device
        self.check_nli = SelfCheckNLI(device=self.device) 
    
    
    def calculate_confidence(self, documents, answer):
        sent_scores_nli = self.check_nli.predict(
        sentences = answer,                          # list of sentences
        sampled_passages = documents, # list of sampled passages
    )
        #print(sent_scores_nli)
        sent_scores_nli = [round(1-s, 4) for s in sent_scores_nli]
        return sent_scores_nli


    
@register('llm_self_check')
class LLMSelfConfidence(object):
    
    def __init__(self, device, llm_type="q-lllama", llm_name="TheBloke/Llama-2-7b-Chat-GPTQ"):
        self.device = device
        self.llm = LLM[llm_type](llm_name) 
    
    
    def get_yes_or_no(self, result):
        if 'yes' in str.lower(result)[:5]:return 'Yes'
        if 'no' in str.lower(result)[:5]:return 'No'
        return 'N/A'
    
    
    def predict(self, document, answer):
        score_mapping = {'Yes':1.0, 'No':0.0}
        template = """
            Context: {a}
            Sentence: {b}
            is the sentence supported by the context above? 
            Answer "Yes" or "No"
        """
        scores, results = list(), list()
        for sentence in answer:
            temp_prompt = template.format(a=document.strip().replace('/n', ''), b=sentence.strip().replace('/n', ''))
            result = self.llm.generate_answer(temp_prompt, sample_num=1, device=self.device)[0]
            results.append(result)
        results = [self.get_yes_or_no(r) for r in results]
        scores = [score_mapping.get(result, 0.5) for result in results]
        return scores
    
    
    def calculate_confidence(self, documents, answer):
        
        all_scores = list()
        for doc in documents:
            sent_scores = self.predict(doc, answer)
            all_scores.append(sent_scores)

        all_scores = [round(sum(s)/len(s), 4) for s in zip(*all_scores)]
      
        return all_scores
    
@register('just_ask_for_calibration')
class JustAskForCalibration(object):
    
    def __init__(self, device, llm_type="llama", llm_name="meta-llama/Llama-2-7b-chat-hf"):
        self.device = device
        self.llm = LLM[llm_type](llm_name) 
    
    def extract_prob(self, answer):
        def is_float(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        probability = re.search(r'Probability: (\d+(\.\d+)?)', answer)

        if probability:
            extracted_value = probability.group(1)
            if is_float(extracted_value): return float(extracted_value)
            else: return 0.0
        return 0.0
    
    def extract_date(self, answer):
        date_text = re.search(r'Guess: \d{4}-\d{2}-\d{2}', answer)

        if date_text:
            extracted_date = date_text.group()
            return extracted_date
        else:
            return ""
        
        
    def calculate_confidence(self, query, sample_num=5, temperature=1.0, do_sample=True):
        
        confidence_prompt = JustAskForCalibration_Prompts["Verb. 1S top-1"].format(THE_QUESTION=query)
        #print(confidence_prompt)
        answers = self.llm.generate_answer(confidence_prompt, sample_num=sample_num, device=self.device, temperature=temperature, do_sample=do_sample)
        #answers = [a.replace(confidence_prompt, "") for a in ori_answers]
        #print(answers)
        #answers = ori_answers
        probs = [self.extract_prob(a) for a in answers]
        max_index = probs.index(max(probs))
        score, answer = probs[max_index], answers[max_index]
        return answer, score 
        
    
        
        

if __name__ == "__main__":
    from nltk import sent_tokenize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # the device to load the model onto
    # documents = ["""
    # Lihu Chen is an American writer and artist who works in comics. They received their degree in psychology from California State University, Fullerton and have worked on titles such as "The Gathering Storm" and "Heartthrob".
    # """,
    # """
    # Lihu Chen is an American writer and artist who works in comics. They received their degree in psychology from California State University, Fullerton and have worked on titles such as "The Gathering Storm" and "Heartthrob".
    # """
    
    # ]
    # sentences = sent_tokenize("""
    # Lihu Chen is an American writer and artist who works in comics. They received their degree in psychology from California State University, Fullerton and have worked on titles such as "The Gathering Storm" and "Heartthrob".
    # """)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # the device to load the model onto

    # checker = LLMSelfConfidence(device=device)
    
    # checker.calculate_confidence(documents, sentences)
    
   
    
    just_ask = JustAskForCalibration(device=device, llm_type='llama', llm_name="meta-llama/Llama-2-13b-chat-hf")
    birth_date_prompt = RELATION_PROMPTS["composer"].format(a="Devil in a New Dress")
    answer, score  = just_ask.calculate_confidence(query=birth_date_prompt, temperature=1.0)
    print("answer = ", answer, "score = ", score)
    
    
    
    
    