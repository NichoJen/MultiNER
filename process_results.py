import json
import pandas as pd
from os import listdir
from os.path import isfile, join


def load_json_dict(filepath):
    """
    load json dictionary from file path and return it
    """
    with open(filepath) as f:
        json_dict = json.load(f)
        return json_dict


def flatten_seqeval_results(results_dict: dict, sep="-"):
    """
    flatten the seqeval output
    """
    flattened_results = dict()
    for key, value in results_dict.items():
        if isinstance(value, dict):
            for _key, _value in value.items():
                new_key = key + sep + _key
                flattened_results[new_key] = _value
        else:
            flattened_results[key] = value
    return flattened_results


def make_results_df(filepath):
    """
    extract results from json file an return dataframe
    """
    results_dict = load_json_dict(filepath)
    results_list = []
    for language in results_dict:
        flat_dict = {"lang": language}
        seqeval_flat = flatten_seqeval_results(results_dict[language]["seqeval"])
        flat_dict.update(seqeval_flat)
        results_list.append(flat_dict)
    df = pd.DataFrame(results_list)
    df.set_index('lang', inplace=True)
    return df


def make_metrics_dfs(directory, model_name_prefix="xlm-roberta-"):
    """
    create a dictionary containing dataframes of all seqeval metrics
    """
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    results_dfs = dict()
    # load results
    for file in files:
        model_language = file.split(sep=".")[0].split(sep="_")[-1]  # extract language from file name
        model_name = model_name_prefix + model_language
        model_df = make_results_df(directory + "/" + file)

        for metric in model_df.columns.values:
            # make metric df
            metric_df = pd.DataFrame(model_df[metric]).T
            metric_df = metric_df.rename(index={metric: model_name})

            if metric in results_dfs:
                results_dfs[metric] = pd.concat([results_dfs[metric], metric_df])
            else:
                results_dfs[metric] = metric_df
    return results_dfs
