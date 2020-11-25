import os
import shutil

name = 'random_seeds'
# root_dir = "DOUBLE/"
root_dir = "Experiments-150-200-50-noisy/"

# if os.path.exists(root_dir):
#     shutil.rmtree(root_dir)

# Parameters for the test
# parameter_list = [
#     'True-not-noisy',
#     'True-low-noisy',
#     'True-high-noisy',
#     'False-not-noisy',
#     'False-low-noisy',
#     'False-high-noisy'
# ]
parameter_list = [
    'Clipping-noisy',
    'NocNoS-noisy',
    'NoCS-noisy',
]

for index in range(len(parameter_list)):
    name = 'Setting'
    name += '_' + str(parameter_list[index])
    print(name)
    for seed in [10, 20, 30, 40, 50]:
        command = "python run_naf2_for_tests.py " + name + ' ' + str(seed) + ' ' + root_dir + ' ' + str(index)
        os.system(command)

command = "python read_paper_tests.py " + root_dir

os.system(command)
