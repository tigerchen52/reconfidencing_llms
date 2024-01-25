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
from joblib.memory import Memory
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

mistral_code_path = "/Users/alexandreperez/dev/rep/inr-phd-llm_reconf/mistral-src"
sys.path.append(mistral_code_path)
sys.path.append("/Users/alexandreperez/dev/rep/inr-phd-llm_reconf")

from grouping.utils import save_path, save_fig, set_latex_font

from prompt_template import RELATION_PROMPTS
import matplotlib.pyplot as plt

memory = Memory(location=".", verbose=0)


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


@memory.cache
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
    UUID = df["uuid"].values

    return X, S, y, UUID


@pytest.mark.parametrize(
    "binwise_fit",
    [
        False,
        # True,
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
        # "mistral7b_founder_nli",
        # "mistral7b_composer_jafc",
        # "mistral7b_founder_jafc",
        "llama7b_birth_date_nli",
        "llama7b_birth_date_jafc",
        "llama7b_composer_nli",
        "llama7b_composer_jafc",
        "llama7b_founder_nli",
        "llama7b_founder_jafc",
    ],
)
def test_tensors(binwise_fit, partitioner_name, task):
    # task = "mistral7b_composer_nli"
    strategy = "quantile"
    n_bins = 15
    # shuffle = False
    # binwise_fit = True
    # partitioner_name = "oblique_tree"

    X, S, y, UUID = get_tensors(task)
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
        # "mistral7b",
        "llama7b",
    ],
)
@pytest.mark.parametrize(
    "partition_on_relation",
    [
        False,
        # True,
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        "nli",
        "jafc",
    ],
)
# @memory.cache
def test_tensors_all(model, partition_on_relation, method):
    partitioner_name = "decision_tree"
    n_bins = 15
    strategy = "quantile"
    binwise_fit = True
    # method = "nli"
    # partition_on_relation = False
    relations = [
        "composer",
        "founder",
        "birth_date",
    ]

    tasks = [f"{model}_{relation}_{method}" for relation in relations]

    res = [get_tensors(task) for task in tasks]
    X, S, y, UUID = zip(*res)
    R = [np.full(len(x), i) for i, x in enumerate(X)]
    X = np.concatenate(X, axis=0)
    S = np.concatenate(S, axis=0)
    y = np.concatenate(y, axis=0)
    R = np.concatenate(R, axis=0)
    UUID = np.concatenate(UUID, axis=0)

    assert X.shape[0] == S.shape[0] == y.shape[0] == R.shape[0] == UUID.shape[0]

    out_path = Path(f"./benchmark/gl/merged/{model}_{method}/")
    out_path.mkdir(exist_ok=True, parents=True)

    if partition_on_relation:
        partitioner_name = None
        partitioner_est = None
        partition = R

    else:
        partition = None
        if partitioner_name == "decision_tree":
            partitioner_est = DecisionTreeClassifier(random_state=0)

        elif partitioner_name == "oblique_tree":
            partitioner_est = ObliqueDecisionTreeClassifier(random_state=0)

        sss = ShuffleSplit(n_splits=1, train_size=0.5, random_state=0)
        train_index, test_index = next(sss.split(X))

    print(test_index.shape)
    print(UUID.shape)

    print(UUID[test_index].shape)

    partitioner = glest.Partitioner(
        partitioner_est,
        predict_method="apply",
        n_bins=n_bins,
        strategy=strategy,
        binwise_fit=binwise_fit,
        verbose=10,
    )
    gle = GLEstimator(S[train_index], partitioner, random_state=0, verbose=10)
    gle.fit(
        X[train_index],
        y[train_index],
        test_data=(X[test_index], y[test_index], S[test_index]),
        partition=partition,
    )
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

    frac_pos = gle.frac_pos_
    counts = gle.counts_
    mean_scores = gle.mean_scores_
    label_ids = gle.label_ids_

    return
    #     return gle.frac_pos_, gle.counts_, gle.mean_scores_

    # def test_extract_uuid():
    #     frac_pos, counts, mean_scores = test_tensors_all("mistral7b", False)
    k = 5
    # idx1d = np.argsort(counts.ravel())[::-1][:k]
    # idx = np.unravel_index(idx1d, counts.shape)

    # print(idx)
    # print(counts[idx])
    # print(frac_pos[idx])
    # print(diff[idx].round(3))

    # weighted mean of frac_pos with counts
    C = np.average(frac_pos, weights=counts, axis=1)
    diff = frac_pos - C[:, None]
    idx1d = np.argsort(np.absolute(diff).ravel())[::-1]
    idx_counts = counts.ravel()[idx1d] >= 100
    idx1d = idx1d[idx_counts]
    idx1d = idx1d[:k]
    idx = np.unravel_index(idx1d, counts.shape)
    print(diff[idx].round(3))
    print(counts[idx].round(3))
    print(idx)

    # turn tuple of arrays into list of tuple
    idx = list(zip(*idx))
    print(idx)

    uuids = []

    labels = partitioner.predict(X[test_index], S[test_index])

    # for bin_id, label in idx:
    #     print(labels)
    # labels = partitioner.labels_
    for bin_id, label, _idx in partitioner.iter_region(labels):
        print(bin_id, label)
        bin_id = int(bin_id)
        label = int(label)
        label_id = label_ids[bin_id, label]
        if (bin_id, label_id) in idx:
            print(len(_idx))
            print(UUID[test_index][_idx])
            uuids.append(
                {
                    "mean_confidence": mean_scores[bin_id, label],
                    "mean_positives": frac_pos[bin_id, label],
                    "mean_positives_bin": C[bin_id],
                    "uuids": UUID[test_index][_idx].tolist(),
                }
            )
        else:
            print("not")

    filepath = save_path(str(out_path), ext="json", _name="uuids", **d)
    with open(filepath, "w") as f:
        f.write(json.dumps(uuids))


# @pytest.mark.parametrize(
#     "model",
#     [
#         "mistral7b",
#     ],
# )
# @pytest.mark.parametrize(
#     "partition_on_relation",
#     [
#         False,
#         True,
#     ],
# )
# def test_tensors_all_uuids(model, partition_on_relation):
#     partitioner_name = "decision_tree"
#     n_bins = 15
#     strategy = "quantile"
#     binwise_fit = True
#     method = "nli"
#     # partition_on_relation = False
#     relations = [
#         "composer",
#         "founder",
#     ]

#     tasks = [f"{model}_{relation}_{method}" for relation in relations]

#     res = [get_tensors(task) for task in tasks]
#     X, S, y = zip(*res)
#     R = [np.full(len(x), i) for i, x in enumerate(X)]
#     X = np.concatenate(X, axis=0)
#     S = np.concatenate(S, axis=0)
#     y = np.concatenate(y, axis=0)
#     R = np.concatenate(R, axis=0)


def fit_recalibrator(S_train, y_train):
    class DummyClassifier(ClassifierMixin, BaseEstimator):
        def __init__(self):
            self.classes_ = np.unique(y_train)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack((1 - X, X))

    custom_classifier = DummyClassifier()
    calibrated_classifier = CalibratedClassifierCV(
        estimator=custom_classifier, method="isotonic", cv="prefit"
    )
    calibrated_classifier.fit(S_train, y_train)
    return calibrated_classifier


def get_recalibrators(partitioner: glest.Partitioner, X, y, S):
    labels = partitioner.predict(X, S)

    uniques, counts = np.unique(labels[:, 1], return_counts=True)

    recalibrators = {}
    for label_id, c in zip(uniques, counts):
        print(f"label_id={label_id}, c={c}")
        # if c <= 5:
        #     continue
        idx = labels[:, 1] == label_id
        bin_ids = labels[idx, 0]
        # print(bin_ids)

        S_bin = S[idx]
        y_bin = y[idx]

        # S, S, S = recalibrate_scores(
        #     S_bin, y_bin, [S, S, S]
        # )
        recalibrator = fit_recalibrator(S_bin, y_bin)
        recalibrators[label_id] = recalibrator

    return recalibrators


def reconfidence(partitioner: glest.Partitioner, recalibrators: dict, X, S):
    labels = partitioner.predict(X, S)

    uniques, counts = np.unique(labels[:, 1], return_counts=True)

    S_recal = S.copy()

    for label_id, c in zip(uniques, counts):
        recalibrator = recalibrators.get(label_id, None)
        if recalibrator is None:
            continue
        idx = labels[:, 1] == label_id
        S_bin = S[idx]
        S_recal[idx] = recalibrator.predict_proba(S_bin)[:, 1]

    return S_recal


def recalibrate_scores(S_train, y_train, S_array):
    calibrated_classifier = fit_recalibrator(S_train, y_train)
    # Getting the calibrated probabilities
    return [calibrated_classifier.predict_proba(S)[:, 1] for S in S_array]


# @pytest.mark.parametrize(
#     "partitioner_name",
#     [
#         "decision_tree",
#         # "oblique_tree",
#     ],
# )
@pytest.mark.parametrize(
    "task",
    [
        "mistral7b_composer_nli",
        # "mistral7b_founder_nli",
        # "mistral7b_composer_jafc",
        # "mistral7b_founder_jafc",
        # "llama7b_birth_date_nli",
        # "llama7b_birth_date_jafc",
        # "llama7b_composer_nli",
        # "llama7b_composer_jafc",
        # "llama7b_founder_nli",
        # "llama7b_founder_jafc",
    ],
)
def test_reconfidence(task):
    X, S, y, UUID = get_tensors(task)
    json_path = get_json_path(task)

    idx = np.arange(len(X))
    idx_train_val, idx_test = train_test_split(idx, test_size=0.5, random_state=0)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.5, random_state=0)

    X_train = X[idx_train]
    S_train = S[idx_train]
    y_train = y[idx_train]
    UUID_train = UUID[idx_train]
    X_val = X[idx_val]
    S_val = S[idx_val]
    y_val = y[idx_val]
    UUID_val = UUID[idx_val]
    X_test = X[idx_test]
    S_test = S[idx_test]
    y_test = y[idx_test]
    UUID_test = UUID[idx_test]

    out_path = Path(f"./benchmark/gl/{task}/")
    out_path.mkdir(exist_ok=True, parents=True)

    # Estimate the CL/GL before
    partitioner_est1 = DecisionTreeClassifier(random_state=0)
    partitioner1 = glest.Partitioner(
        partitioner_est1,
        predict_method="apply",
        n_bins=15,
        strategy="quantile",
        binwise_fit=True,
        verbose=10,
    )
    # gle = GLEstimator(S_train, partitioner1, random_state=0, verbose=10)
    # gle.fit(X_train, y_train, test_data=(X_val, y_val, S_val))
    gle = GLEstimator(S_test, partitioner1, random_state=0, verbose=10)
    gle.fit(X_test, y_test)
    metrics_before = gle.metrics()

    set_latex_font()
    fig, ax = plt.subplots(figsize=(3, 2))
    gle.plot(ax=ax)
    ax.set_title(task, y=1.5)
    save_fig(fig, str(out_path), ext="pdf", _name="diagram", task=task)

    # Fit isotonics
    partitioner_est = DecisionTreeClassifier(random_state=0, max_leaf_nodes=5)
    partitioner = glest.Partitioner(
        partitioner_est,
        predict_method="apply",
        n_bins=15,
        strategy="quantile",
        binwise_fit=False,
        verbose=10,
    )
    partitioner.fit(X_train, S_train, y_train)
    recalibrators = get_recalibrators(partitioner, X_val, y_val, S_val)

    fig, ax = plt.subplots(figsize=(3, 2))
    fig = gle.plot(ax=ax)

    SS = np.linspace(0, 1, 100)

    for label_id, recalibrator in recalibrators.items():
        p = recalibrator.predict_proba(SS)
        print(label_id, p.shape)
        if p.shape[1] == 2:
            ax.plot(SS, p[:, 1], label=f"{int(label_id)}")

    ax.set_title(task, y=1.5)
    save_fig(fig, str(out_path), ext="pdf", _name="diagram_isos", task=task)

    # Reconfidence
    S_reconf_test = reconfidence(partitioner, recalibrators, X_test, S_test)

    # Recalibrate
    (S_recal_test,) = recalibrate_scores(S_val, y_val, [S_test])

    gle = GLEstimator(S_reconf_test, partitioner1, random_state=0, verbose=10)
    gle.fit(X_test, y_test)
    metrics_after_reconf = gle.metrics()
    print(gle)

    fig, ax = plt.subplots(figsize=(3, 2))
    gle.plot(ax=ax)
    ax.set_title(task, y=1.5)
    save_fig(fig, str(out_path), ext="pdf", _name="diagram_reconf", task=task)

    gle = GLEstimator(S_recal_test, partitioner1, random_state=0, verbose=10)
    gle.fit(X_test, y_test)
    metrics_after_recal = gle.metrics()
    print(gle)

    set_latex_font()
    fig, ax = plt.subplots(figsize=(3, 2))
    gle.plot(ax=ax)
    ax.set_title(task, y=1.5)
    save_fig(fig, str(out_path), ext="pdf", _name="diagram_recal", task=task)

    # Save scores
    df_S_before = pd.DataFrame(S_test, columns=["S"], index=UUID_test)
    df_S_after_reconf = pd.DataFrame(S_reconf_test, columns=["S"], index=UUID_test)
    df_S_after_recal = pd.DataFrame(S_recal_test, columns=["S"], index=UUID_test)
    df_y = pd.DataFrame(y_test, columns=["y"], index=UUID_test)

    df_S_before.to_csv(out_path / "S_before.csv")
    df_S_after_reconf.to_csv(out_path / "S_after_reconf.csv")
    df_S_after_recal.to_csv(out_path / "S_after_recal.csv")
    df_y.to_csv(out_path / "y.csv")

    # Save metrics
    df_metrics = pd.DataFrame(
        [metrics_before, metrics_after_reconf, metrics_after_recal],
        index=pd.Index(["before", "after_reconf", "after_recal"], name="which"),
    )
    df_metrics["task"] = task
    df_metrics.to_csv(out_path / "metrics.csv")
