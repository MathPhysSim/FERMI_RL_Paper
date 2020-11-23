import os
import shutil

name = 'random_seeds'
# root_dir = "DOUBLE/"
root_dir = "Experiments-150-200-50/"

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

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
    'Clipping',
    'No-clipping-clipping-no-smoothing',
    'No-clipping-clipping-smoothing',
]

for index in range(len(parameter_list)):
    name = 'test-'
    name += '_' + str(parameter_list[index])
    print(name)
    for seed in [10]:
        command = "python run_naf2_single.py " + name + ' ' + str(seed) + ' ' + root_dir + ' ' + str(index)
        os.system(command)

command = "python read_paper_tests.py " + root_dir

os.system(command)
