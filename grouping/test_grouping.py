import sys
from pathlib import Path
import tqdm
from typing import List
import pandas as pd
import json
import os
import re
import unicodedata
import numpy as np
from glest import GLEstimator

mistral_code_path = "/Users/alexandreperez/dev/rep/inr-phd-llm_reconf/mistral-src"
sys.path.append(mistral_code_path)
sys.path.append("/Users/alexandreperez/dev/rep/inr-phd-llm_reconf")

from prompt_template import ALL_PROMPTS
import matplotlib.pyplot as plt


def test_mistral():
    from mistral.model import Transformer
    from mistral.tokenizer import Tokenizer

    import torch

    print(Transformer)

    model_path = Path(
        "/home/soda/aperez/rep/inr-phd-llm_reconf/mistral-src/mistral_data/mistral-7B-v0.1"
    )

    model = Transformer.from_folder(
        model_path,
        dtype=torch.bfloat16,
    )
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))

    print(model)

    # prompts = [
    #     "The quick brown fox jumps over the lazy dog.",
    # ]

    relation = "birth_place"
    filename = "person_place_backlink.txt"
    template_prompt = ALL_PROMPTS[relation]
    prompts = get_prompts(template_prompt, filename)

    with torch.no_grad():
        featurized_x = []
        # compute an embedding for each sentence
        for i, x in enumerate(tqdm.tqdm(prompts[:10])):
            tokens = tokenizer.encode(x, bos=True)
            tensor = torch.tensor(tokens).to(model.device)
            features = model.forward_partial(
                tensor, [len(tokens)]
            )  # (n_tokens, model_dim)
            featurized_x.append(features.float().mean(0).cpu().detach().numpy())

    print(featurized_x)
    print(featurized_x[0].shape)


def get_prompts(template_prompt: str, filename: str) -> List[str]:
    """
    Get the prompts from the template and the filename
    """
    path = Path("benchmark/output")
    data_file = path / filename

    prompts = []
    for index, line in enumerate(open(data_file, encoding="utf8")):
        row = line.strip().split("\t")
        name, backlink = row[0], row[1]
        # print("index = {a}, name = {b}".format(a=index, b=name))
        prompt = template_prompt.format(a=name)
        prompts.append(prompt)

    return prompts


def test_prompt():
    relation = "birth_place"
    filename = "person_place_backlink.txt"
    template_prompt = ALL_PROMPTS[relation]

    prompts = get_prompts(template_prompt, filename)

    print(prompts[0])


def test_file():
    path = Path("benchmark/output")

    mistral_result_path = path / "mistral7b_birth_date.json"

    # Read file line by line
    for index, line in enumerate(open(mistral_result_path, encoding="utf8")):
        print(line)


def remove_accents(input_str):
    """
    Removes accents from a given string.
    """
    # Normalize the string to decompose the accents
    nfkd_form = unicodedata.normalize("NFKD", input_str)

    # Remove the combining characters (accents)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def sanitize_filename(filename):
    """
    Sanitize a string to make it safe to use as a filename.
    This function replaces spaces with underscores and removes
    or replaces characters that are not safe for filenames.
    """
    filename = remove_accents(filename)
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Remove or replace characters that are not safe for filenames
    filename = re.sub(r'[<>:"/\\|?*\']+', "", filename)

    return filename


def test_get_id():
    path = Path("benchmark/output")

    yago_person_date = path / "yago_person_date.txt"
    mistral_result_path = path / "mistral7b_birth_date.json"

    df = pd.read_csv(
        yago_person_date, sep="\t", header=None, names=["name", "relation", "date"]
    )

    print(df.iloc[0:10])

    for i, line in enumerate(open(mistral_result_path, encoding="utf8")):
        # print(line)
        mistral_result = json.loads(line)
        name = mistral_result["name"]
        # Get df index where df.name == line[name]
        df_index = df.index[df["name"] == name].tolist()[0]
        print(df_index, sanitize_filename(name))
        if i > 10:
            break


def test_mistral_forward_all():
    from mistral.model import Transformer
    from mistral.tokenizer import Tokenizer

    import torch

    print(Transformer)

    model_path = Path(
        "/home/soda/aperez/rep/inr-phd-llm_reconf/mistral-src/mistral_data/mistral-7B-v0.1"
    )

    model = Transformer.from_folder(
        model_path,
        dtype=torch.bfloat16,
    )
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))

    print(model)

    # prompts = [
    #     "The quick brown fox jumps over the lazy dog.",
    # ]

    path = Path("benchmark/output")

    yago_person_date = path / "yago_person_date.txt"
    mistral_result_path = path / "mistral7b_birth_date.json"

    df = pd.read_csv(
        yago_person_date, sep="\t", header=None, names=["name", "relation", "date"]
    )

    print(df.iloc[0:10])

    relation = "birth_date"
    template_prompt = ALL_PROMPTS[relation]

    forward_path = path / "mistral7b_birth_date"
    forward_path.mkdir(exist_ok=True)

    for i, line in tqdm.tqdm(enumerate(open(mistral_result_path, encoding="utf8"))):
        # print(line)
        mistral_result = json.loads(line)
        name = mistral_result["name"]
        df_index = df.index[df["name"] == name].tolist()[0]
        prompt = template_prompt.format(a=name)

        with torch.no_grad():
            tokens = tokenizer.encode(prompt, bos=True)
            tensor = torch.tensor(tokens).to(model.device)
            features = model.forward_partial(
                tensor, [len(tokens)]
            )  # (n_tokens, model_dim)
            featurized_x = features.float().mean(0).cpu().detach().numpy()

        # print(featurized_x.shape)

        filename = forward_path / (sanitize_filename(f"{df_index}_{name}") + ".npy")
        print(filename)
        # save the numpy array to disk
        featurized_x.tofile(filename)


def test_read_all():
    folder_path = "benchmark/output/mistral7b_birth_date"

    # List all .npy files in the directory
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

    # Initialize an empty list to store the arrays
    arrays = []

    path = Path("benchmark/output")
    yago_person_date = path / "yago_person_date.txt"
    mistral_result_path = path / "mistral7b_birth_date.json"
    df = pd.read_csv(
        yago_person_date, sep="\t", header=None, names=["name", "relation", "date"]
    )

    json_list = [
        json.loads(line)
        for line in tqdm.tqdm(open(mistral_result_path, encoding="utf8"))
    ]

    name_json_list = {d["name"]: d for d in json_list}

    print(name_json_list)

    rows = []

    # Loop through the files and append the arrays to the list
    for file in npy_files:
        # print(file)
        name_id = int(file.split("_")[0])
        d = df.loc[name_id]
        name = d["name"]

        # print(name_id, d["name"], file)
        array = np.fromfile(os.path.join(folder_path, file), dtype=np.float32)
        # array = np.load(os.path.join(folder_path, file))
        arrays.append(array)

        row = name_json_list[name]
        rows.append(row)
        # print(array.shape)

    # Concatenate the arrays along the first dimension
    X = np.stack(arrays, axis=0)
    print(X.shape)

    # GLEstimator()
    df = pd.DataFrame(rows)
    print(df)

    y_scores = df["confidence"].values
    y = df["tag"].values

    print(y_scores)
    print(np.unique(y, return_counts=True))

    gle = GLEstimator(y_scores, random_state=0)
    gle.fit(X, y)
    print(gle)

    # plt.tight_layout()
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = gle.plot()
    fig.savefig("test.pdf", bbox_inches="tight", pad_inches=0)
