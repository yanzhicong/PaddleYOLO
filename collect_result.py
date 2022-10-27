import os
import sys




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
                result_dict[key.strip()] = float(value.strip())

    for key, value in result_dict.items():
        print("\t{} = {}".format(key, value))            

    print()

if __name__ == "__main__":
    for exp_name in os.listdir(experiment_dir):
        find_and_print_result(os.path.join(experiment_dir, exp_name, "workerlog.0"))
        find_and_print_result(os.path.join(experiment_dir, exp_name, "eval_result.txt"))
        find_and_print_result(os.path.join(experiment_dir, exp_name, "test_result.txt"))
