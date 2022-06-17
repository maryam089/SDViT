# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os




import tqdm

from domainbed.lib.query import Q

def load_records(path,test_post_results=False,get_recursively=False):
    records = []
    if(get_recursively):
        filelist = []

        for root, dirs, files in os.walk(path):
            for file in files:
                #append the file name to the list
                filelist.append(os.path.join(root,file))
        if (test_post_results):
            results_ext = "results_test.jsonl"
        else:
            results_ext = "results.jsonl"
        for name in filelist:
            if (results_ext in name):
                results_path = name
            else:
                continue
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

    else:
        for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                                ncols=80,
                                leave=False):
            #we shall store all the file names in this list
            
            
            if (test_post_results):
                results_path = os.path.join(path, subdir, "results_test.jsonl")
            else:
                results_path = os.path.join(path, subdir, "results.jsonl")
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])