import wikipedia

# def get_wikipedia_backlinks(title):
#     try:
#         page = wikipedia.page(title)
#         backlinks = page.links
#         return backlinks
#     except wikipedia.exceptions.PageError:
#         print("Page '{}' does not exist on Wikipedia.".format(title))
#         return []
#     except wikipedia.exceptions.DisambiguationError as e:
#         print("Disambiguation Error: Please specify the title more precisely. Suggestions: ", e.options)
#         return []
    
def get_wikipedia_backlinks(title):
    try:
        # Try getting the page with the correct title case
        page = wikipedia.page(title, auto_suggest=False)
        backlinks = page.links
        return backlinks
    except wikipedia.exceptions.PageError:
        # If the page doesn't exist, try auto-suggesting the title
        suggested_title = wikipedia.suggest(title)
        if suggested_title:
            print(f"Page '{title}' does not exist. Did you mean '{suggested_title}'?")
        else:
            print(f"Page '{title}' does not exist on Wikipedia.")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        print("Disambiguation Error: Please specify the title more precisely. Suggestions: ", e.options)
        return []
    except Exception as e:
        return []


def load_titles(file_path):
    titles = list()
    for line in open(file_path):
        row = line.strip().split('\t')
        name = row[0]
        titles.append(name)
    return titles


def run(entity):
    base_path = "/data/parietal/store3/soda/lihu/code/hallucination/benchmark/output/"
    titles = load_titles(file_path=base_path+entity+".txt")
    wf = open(base_path+entity+"_backlink.txt", "w", encoding="utf8")
    print("loaded {a} titles".format(a=len(titles)))
    for index,title in enumerate(titles):
        print("this is the {a} title".format(a=index))
        backlinks = get_wikipedia_backlinks(title)
        print(title+"\t"+str(len(backlinks))+"\n")
        wf.write(title+"\t"+str(len(backlinks))+"\n")
        wf.flush()

if __name__ == "__main__":
        # Example usage:
    # input_title = "Axl Rose"  # Change this to the Wikipedia page title you want
    # backlinks = get_wikipedia_backlinks(input_title)

    # print("Backlinks for '{}' page:".format(input_title))
    # print(len(backlinks))
    run(entity="organization")

    