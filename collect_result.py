import os
import sys
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--include_exp", default=None, type=str)
parser.add_argument("-e", "--exclude_exp", default=None, type=str)
args = parser.parse_args()


experiment_dir = "sku110_logdir"


def find_and_print_result(file_path):
    if not os.path.exists(file_path):
        return
    print(file_path)
    result_dict = {}
    with open(file_path, 'r') as infile:
        for line in infile:
            if line.startswith(" Average Precision  (AP) @[ IoU") and line.find("area=   all") != -1:
                key, value = line.strip().split(" = ")
                key, value = key.strip(), float(value.strip())
                # result_dict[key] = result_dict.get(key, []).append(value)
                if key in result_dict:
                    result_dict[key].append(value)
                else:
                    result_dict[key] = [value,]
                # result_dict[key.strip()] = float(value.strip())

    for key, value in result_dict.items():
        print("\t{} = {}".format(key, value))

    print()

if __name__ == "__main__":


    for exp_name in os.listdir(experiment_dir):

        for file_name in [
            "workerlog.0",
            "eval_result.txt",
            "test_result.txt",
        ]:

            result_path = os.path.join(experiment_dir, exp_name, file_name)
            if args.include_exp is not None and not re.match(args.include_exp, result_path):
                continue
            if args.exclude_exp is not None and re.match(args.exclude_exp, result_path):
                continue

            find_and_print_result(result_path)
        