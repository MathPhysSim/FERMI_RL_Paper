import os
import shutil

name = 'random_seeds'
# root_dir = "DOUBLE/"
root_dir = 'Data/Simulation/ModelSizeLong/'

# if os.path.exists(root_dir):
#     shutil.rmtree(root_dir)

parameter_list = [
    'Single',
    'Three',
    'Five',
    'Ten'
]

for index in range(len(parameter_list)):
    name = 'Setting'
    name += '_' + str(parameter_list[index])
    print(name)
    for seed in [10, 20, 30, 40, 50]:
        command = "python AEDYNA.py " + name + ' ' + str(seed) + ' ' + root_dir + ' ' + str(index)
        os.system(command)

# command = "python read_paper_tests.py " + root_dir

os.system(command)
