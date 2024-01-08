import re
import json


def get_name(url):
    name = url.split("/")[-1].replace(">", "").replace("_", " ")
    return name



def get_date(text):
    #input_text = '"1974-12-20"^^<http://www.w3.org/2001/XMLSchema#date>'

    # Define a regular expression pattern to match the date within double quotes
    pattern = r'"([^"]*)"'

    # Use re.search to find the match
    match = re.search(pattern, text)
    
    extracted_date = None
    # Extract the date from the match
    if match:
        extracted_date = match.group(1)
        #print(extracted_date)
    #else:
    #    print("Date not found.")

    return extracted_date




def get_birth_date():
    wf = open("output/yago_person_date.txt", 'w', encoding="utf8")
    for line in open("/data/parietal/store3/soda/lihu/yago_4/yago-wd-facts.nt", encoding="utf8"):
        row = line.strip().split("\t")
        if "birthDate" not in row[1]:continue
        head, tag, date = get_name(row[0]), get_name(row[1]), get_date(row[2])
        if len(date) < 10:continue
        #print(head, tag, date)
        if tag != "birthDate":continue
        #print(head, tag, date)
        print(head, tag, date)
        wf.write(head+"\t"+tag+"\t"+date+"\n")


def get_birth_place():
    wf = open("output/yago_person_place.txt", 'w', encoding="utf8")
    for line in open("/data/parietal/store3/soda/lihu/yago_4/yago-wd-facts.nt", encoding="utf8"):
        row = line.strip().split("\t")
        if "birthPlace" not in row[1]:continue
        head, tag, date = get_name(row[0]), get_name(row[1]), get_name(row[2])
        if len(date) < 10:continue
        #print(head, tag, date)
        if tag != "birthPlace":continue
        #print(head, tag, date)
        print(head, tag, date)
        wf.write(head+"\t"+tag+"\t"+date+"\n")
    

def get_father():
    wf = open("output/yago_person_father.txt", 'w', encoding="utf8")
    for line in open("/data/parietal/store3/soda/lihu/yago_4/yago-wd-facts.nt", encoding="utf8"):
        row = line.strip().split("\t")
        if "parent" not in row[1]:continue
        head, tag, date = get_name(row[0]), get_name(row[1]), get_name(row[2])
        if len(date) < 10:continue
        #print(head, tag, date)
        if tag != "parent":continue
        #print(head, tag, date)
        print(head, tag, date)
        wf.write(head+"\t"+tag+"\t"+date+"\n")


def get_music_composer():
    # music_entities = set()
    # for line in open("/data/parietal/store3/soda/lihu/yago_4/alternate_name_type.txt", encoding="utf8"):
    #     row = line.strip().split("\t")
    #     if "MusicComposition" not in row[1]:continue
    #     head, tag = get_name(row[0]), get_name(row[1])
    #     music_entities.add(head)
    # print("music entity = {a}".format(a=len(music_entities)))
    wf = open("/data/parietal/store3/soda/lihu/code/hallucination/benchmark/output/music.txt", "w", encoding="utf8")
    for line in open("/data/parietal/store3/soda/lihu/yago_4/yago-wd-facts.nt", encoding="utf8"):
        row = line.strip().split("\t")
        if "composer" not in row[1]:continue
   
        head, tag = get_name(row[0]), get_name(row[1])
        #if head not in music_entities:continue
        print(head, tag)
        wf.write(head+"\n")
        wf.flush()
        
def get_organization_founder():
    wf = open("/data/parietal/store3/soda/lihu/code/hallucination/benchmark/output/organization.txt", "w", encoding="utf8")
    for line in open("/data/parietal/store3/soda/lihu/yago_4/yago-wd-facts.nt", encoding="utf8"):
        row = line.strip().split("\t")
        if "founder" not in row[1]:continue
   
        head, tag = get_name(row[0]), get_name(row[1])
        #if head not in music_entities:continue
        print(head, tag)
        wf.write(head+"\n")
        wf.flush()
        

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_name_by_backlink():
    names = set()
    for line in open("output/yago_person_father.txt", encoding="utf8"):
        row = line.strip().split("\t")
        names.add(row[0])
        if len(names) > 10000:break


    name_links = dict()
    for line in open("output/backlinks.json", encoding="utf8"):
        obj = json.loads(line)
        name, back_link_num = obj["name"], obj["backlinks"]
        if name not in names:continue
        name_links[name] = back_link_num
    
    name_links = sorted(name_links.items(), key=lambda e:e[1], reverse=True)

    name_bins = list(chunks(name_links, len(name_links)//10))

    for bin in name_bins:
        print(len(bin), bin[-1])
    
    wf = open("output/person_father_backlink.txt", "w", encoding="utf8")

    for bin in name_bins:
        for name, bl in bin:
            wf.write(name + "\t" + str(bl) + "\n")
    


if __name__ == "__main__":

    #get_birth_date()
    #get_birth_place()
    #get_father()
    #generate_name_by_backlink()
    get_organization_founder()
