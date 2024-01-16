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
import glest
import torch
from sktree.tree import ObliqueDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import pytest
from typing import Tuple

mistral_code_path = "/Users/alexandreperez/dev/rep/inr-phd-llm_reconf/mistral-src"
sys.path.append(mistral_code_path)
sys.path.append("/Users/alexandreperez/dev/rep/inr-phd-llm_reconf")

from grouping.utils import save_path

from prompt_template import RELATION_PROMPTS
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
    template_prompt = RELATION_PROMPTS[relation]
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
    template_prompt = RELATION_PROMPTS[relation]

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
    template_prompt = RELATION_PROMPTS[relation]

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


def test_llama():
    # print()
    from llm_base import registry as LLM
    import torch

    # print(LLM)
    # print(LLM["llama"])
    # llm = LLM["mistal-7b"]()
    llm = LLM["llama"]("TheBloke/Llama-2-7b-Chat-GPTQ")

    # print(llm)
    print(llm.model)

    # return
    # print(llm.model.model)
    # print(dir(llm.model))
    # methods = [
    #     method for method in dir(llm.model) if callable(getattr(llm.model, method))
    # ]
    # print(methods)

    # print(llm.model.forward_partial)

    prompt = "test"

    messages = list()
    messages.append({"role": "user", "content": prompt})
    input_ids = llm.tokenizer.apply_chat_template(messages, return_tensors="pt")

    device = "cuda"
    model_inputs = input_ids.to(device)
    llm.model.to(device)

    # output = llm.model.forward(model_inputs)
    # print(output)
    # print(type(output))
    # return
    # print(output.hidden_states)
    # print(output.logits)
    # print(output.last_hidden_state)
    output = llm.model.model.forward(model_inputs)
    # print(output)
    print(output.last_hidden_state)
    # print(type(output.last_hidden_state))
    print(output.last_hidden_state.shape)
    # print(type(llm.model.model))
    print(type(output))
    print(input_ids.shape)
    # print(torch.tensor(output, dtype=torch.float32))


def test_llama_forward_all():
    from llm_base import registry as LLM
    import torch
    # from mistral.model import Transformer
    # from mistral.tokenizer import Tokenizer

    # import torch

    # print(Transformer)

    # model_path = Path(
    #     "/home/soda/aperez/rep/inr-phd-llm_reconf/mistral-src/mistral_data/mistral-7B-v0.1"
    # )

    llm = LLM["llama"]("TheBloke/Llama-2-7b-Chat-GPTQ")

    messages = list()
    messages.append({"role": "user", "content": prompt})
    input_ids = llm.tokenizer.apply_chat_template(messages, return_tensors="pt")

    device = "cuda"
    model_inputs = input_ids.to(device)
    llm.model.to(device)

    # print(model)

    # prompts = [
    #     "The quick brown fox jumps over the lazy dog.",
    # ]

    path = Path("benchmark/output")

    yago_person_date = path / "yago_person_date.txt"
    result_path = path / "llama7b_birth_date.json"

    df = pd.read_csv(
        yago_person_date, sep="\t", header=None, names=["name", "relation", "date"]
    )

    print(df.iloc[0:10])

    relation = "birth_date"
    template_prompt = RELATION_PROMPTS[relation]

    forward_path = path / "mistral7b_birth_date"
    forward_path.mkdir(exist_ok=True)

    for i, line in tqdm.tqdm(enumerate(open(result_path, encoding="utf8"))):
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


def get_json_path(task: str) -> str:
    return f"/data/parietal/store3/soda/lihu/code/hallucination/benchmark/result_tag/{task}_tag.json"


def get_tensors(task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_dirpath = Path(
        f"/data/parietal/store3/soda/lihu/code/hallucination/benchmark/tensors/{task}/"
    )
    tensor_files = [file for file in tensor_dirpath.iterdir() if file.is_file()]
    json_path = Path(get_json_path(task))

    tensor_paths = {}
    for file in tensor_files:
        tensor_paths[file.stem] = str(file)

    rows = []
    for line in open(json_path, encoding="utf8"):
        result = json.loads(line)
        rows.append(result)
        tensor_uuid = result["uuid"]
        print(tensor_uuid)
        assert tensor_uuid in tensor_paths, f"{tensor_uuid} not in {tensor_paths}"

    df = pd.DataFrame(rows)
    print(df)

    # Load all the tensors in order
    tensors = []
    for uuid in tqdm.tqdm(df["uuid"]):
        tensor_path = tensor_paths[uuid]
        tensor: torch.Tensor = torch.load(tensor_path, map_location=torch.device("cpu"))
        tensor = tensor.detach()
        tensor = tensor.mean(1).squeeze()
        tensors.append(tensor)
        assert tensor.shape == (4096,)

    X = torch.stack(tensors, dim=0)
    X = X.numpy()

    S = df["confidence"].values
    y = df["tag"].values

    return X, S, y


@pytest.mark.parametrize(
    "binwise_fit",
    [
        # False,
        True,
    ],
)
@pytest.mark.parametrize(
    "partitioner_name",
    [
        "decision_tree",
        # "oblique_tree",
    ],
)
@pytest.mark.parametrize(
    "task",
    [
        # "mistral7b_composer_nli",
        "mistral7b_founder_nli",
        # "mistral7b_composer_jafc",
        "mistral7b_founder_jafc",
    ],
)
def test_tensors(binwise_fit, partitioner_name, task):
    # task = "mistral7b_composer_nli"
    strategy = "quantile"
    n_bins = 15
    # shuffle = False
    # binwise_fit = True
    # partitioner_name = "oblique_tree"

    X, S, y = get_tensors(task)
    json_path = get_json_path(task)

    # if shuffle:
    #     rng = np.random.default_rng(0)
    #     rng.shuffle(y)

    out_path = Path(f"./benchmark/gl/{task}/")
    out_path.mkdir(exist_ok=True, parents=True)

    if partitioner_name == "decision_tree":
        partitioner_est = DecisionTreeClassifier(random_state=0)

    elif partitioner_name == "oblique_tree":
        partitioner_est = ObliqueDecisionTreeClassifier(random_state=0)

    partitioner = glest.Partitioner(
        partitioner_est,
        predict_method="apply",
        n_bins=n_bins,
        strategy=strategy,
        binwise_fit=binwise_fit,
        verbose=10,
    )
    gle = GLEstimator(S, partitioner, random_state=0, verbose=10)
    gle.fit(X, y)
    metrics = gle.metrics()
    print(gle)

    metrics["source"] = str(json_path)
    metrics["n_samples"] = X.shape[0]

    d = dict(t=task, s=strategy, b=binwise_fit, n=n_bins, p=partitioner_name)

    # Write metrics to text file
    filepath = save_path(str(out_path), ext="json", _name="metrics", **d)
    with open(filepath, "w") as f:
        f.write(json.dumps(metrics))

    fig = gle.plot()
    filepath = save_path(str(out_path), ext="pdf", _name="diagram", **d)
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.1)


@pytest.mark.parametrize(
    "model",
    [
        "mistral7b",
    ],
)
@pytest.mark.parametrize(
    "partition_on_relation",
    [
        # False,
        True,
    ],
)
def test_tensors_all(model, partition_on_relation):
    partitioner_name = "decision_tree"
    n_bins = 15
    strategy = "quantile"
    binwise_fit = True
    method = "nli"
    # partition_on_relation = False
    relations = [
        "composer",
        "founder",
    ]

    tasks = [f"{model}_{relation}_{method}" for relation in relations]

    res = [get_tensors(task) for task in tasks]
    X, S, y = zip(*res)
    R = [np.full(len(x), i) for i, x in enumerate(X)]
    X = np.concatenate(X, axis=0)
    S = np.concatenate(S, axis=0)
    y = np.concatenate(y, axis=0)
    R = np.concatenate(R, axis=0)

    assert X.shape[0] == S.shape[0] == y.shape[0] == R.shape[0]

    out_path = Path(f"./benchmark/gl/merged/{model}_{method}/")
    out_path.mkdir(exist_ok=True, parents=True)

    if partitioner_name == "decision_tree":
        partitioner_est = DecisionTreeClassifier(random_state=0)

    elif partitioner_name == "oblique_tree":
        partitioner_est = ObliqueDecisionTreeClassifier(random_state=0)

    partitioner = glest.Partitioner(
        partitioner_est,
        predict_method="apply",
        n_bins=n_bins,
        strategy=strategy,
        binwise_fit=binwise_fit,
        verbose=10,
    )
    gle = GLEstimator(S, partitioner, random_state=0, verbose=10)
    if partition_on_relation:
        gle.fit(X, y, partition=R)
    else:
        gle.fit(X, y)
    metrics = gle.metrics()
    print(gle)

    metrics["sources"] = [get_json_path(task) for task in tasks]
    metrics["tasks"] = tasks
    metrics["n_samples"] = [r[0].shape[0] for r in res]

    d = dict(
        model=model,
        method=method,
        s=strategy,
        b=binwise_fit,
        n=n_bins,
        p=partitioner_name,
        por=partition_on_relation,
    )

    # Write metrics to text file
    filepath = save_path(str(out_path), ext="json", _name="metrics", **d)
    with open(filepath, "w") as f:
        f.write(json.dumps(metrics))

    fig = gle.plot()
    filepath = save_path(str(out_path), ext="pdf", _name="diagram", **d)
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
