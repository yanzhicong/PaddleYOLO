import os
import sys
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--include_exp", default=None, type=str)
parser.add_argument("-e", "--exclude_exp", default=None, type=str)
parser.add_argument("-sr", "--show_ar", default=False, action="store_true")
parser.add_argument("-sa", "--show_area", default=False, action="store_true")
args = parser.parse_args()


experiment_dir = "sku110_logdir"


def find_and_print_result(file_path):
    if not os.path.exists(file_path):
        return
    print(file_path)
    result_dict = {}

    if not args.show_ar:
        if not args.show_area:
            filter_exp  = lambda line : line.startswith(" Average Precision  (AP)") and line.find("area=   all") != -1
        else:
            filter_exp  = lambda line : line.startswith(" Average Precision  (AP)") and line.find("area=   all") == -1
    else:
        if not args.show_area:
            filter_exp  = lambda line : line.startswith(" Average Recall     (AR)") and line.find("area=   all") != -1
        else:
            filter_exp  = lambda line : line.startswith(" Average Recall     (AR)") and line.find("area=   all") == -1


    with open(file_path, 'r') as infile:
        for line in infile:
            if filter_exp(line):
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

    exp_name_list = os.listdir(experiment_dir)
    exp_name_list = sorted(exp_name_list)

    for exp_name in exp_name_list:

        for file_name in [
            "workerlog.0",
            "eval_result.txt",
            "large_size_eval_result.txt",
            "large_large_size_eval_result.txt",
            "test_result.txt",
            "large_size_test_result.txt",
            "large_large_size_test_result.txt",
        ]:

            result_path = os.path.join(experiment_dir, exp_name, file_name)
            if args.include_exp is not None and not re.match(args.include_exp, result_path):
                continue
            if args.exclude_exp is not None and re.match(args.exclude_exp, result_path):
                continue

            find_and_print_result(result_path)
        