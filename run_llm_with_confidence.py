import nltk
nltk.download('punkt')
import json
import sys
import argparse
from nltk.tokenize import sent_tokenize
from prompt_template import ALL_PROMPTS
from llm_base import registry as LLM
from confidence_checker import registry as CHECKER


#hyper-parameters
parser = argparse.ArgumentParser(description='llms with confidence score')
parser.add_argument('-relation', type=str, default='birth_date')
parser.add_argument('-llm', type=str, default='q-lllama')
parser.add_argument('-model_name_on_hg', type=str, default='TheBloke/Llama-2-7b-Chat-GPTQ')
parser.add_argument('-confidence_checker', type=str, default='nli')
parser.add_argument('-data_file', type=str, default='benchmark/output/person_by_backlink.txt')
parser.add_argument('-out_file', type=str, default='benchmark/result/llama_birth_date.json')
parser.add_argument('-sample_num', type=int, default=5)
parser.add_argument('-temperature', type=float, default=1.0)
parser.add_argument('-max_new_tokens', type=int, default=200)
parser.add_argument('-do_sample', type=bool, default=True)
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-checkr_device', type=str, default='cpu')

def get_confidence(model, checker, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):
    sampled = model.generate_answer(prompt, sample_num, device, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample)
    answer = sampled[0]
    documents = sampled[1:]
    sentences = sent_tokenize(answer)
    scores = checker.calculate_confidence(documents=documents, answer=sentences)
    avg_confidence = round(sum(scores) / len(scores), 4) if len(scores)!=0 else 0

    return answer, avg_confidence


def run(model, checker, data_file, out_file, prompt, sample_num, device, temperature=1.0, max_new_tokens=500, do_sample=True):

    wf = open(out_file, "w", encoding="utf8")

    for index, line in enumerate(open(data_file, encoding="utf8")):
        
        row = line.strip().split("\t")
        name, backlink = row[0], row[1]
        print("index = {a}, name = {b}".format(a=index, b=name))
        temp_prompt = prompt.format(a=name)
        answer, avg_confidence = get_confidence(model, checker, temp_prompt, sample_num, device, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample)
        obj = {
            "name":name,
            "backlink":backlink,
            "answer":answer,
            "confidence":avg_confidence
        }
        json.dump(obj, wf, ensure_ascii=False)
        wf.write("\n")
        wf.flush()



if __name__ == '__main__':
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    data_file = args.data_file
    out_file = args.out_file
    relation = args.relation
    device = args.device
    checkr_device = args.checkr_device
    prompt = ALL_PROMPTS[relation]
    model_name = args.model_name_on_hg
    model = LLM[args.llm](model_name)
    checker = CHECKER[args.confidence_checker](device=checkr_device)
    sample_num = args.sample_num
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    do_sample = args.do_sample
    
    run(model, checker, data_file, out_file, prompt, sample_num, device=device, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample)
   