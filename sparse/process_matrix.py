import os
# from utils import get_dataset, ell
from torch.profiler import profile, ProfilerActivity, schedule
import torch
import numpy as np
import random
import argparse
import scipy.sparse as sp
import scipy
import time

dir_path = "/home/scratch.yinuol_gpu/CAG_Compiler_Project/data/"
# matrix_id_list = ['as-Skitter', 'human_gene2', 'kron_g500-logn19', 'ljournal-2008', 'rajat30', 'roadNet-CA', 'web-Google']
matrix_id_list = ['dielFilterV3real']
for matrix_id in matrix_id_list:
  print(f"processing {matrix_id}...")
  file_name = dir_path + matrix_id + '.mtx'
  print("file_name: ", file_name)
  data = scipy.io.mmread(file_name)
  # data = scipy.io.mmread('/home/v-yinuoliu/yinuoliu/code/SparseCodegen/dielFilterV3real/dielFilterV3real.mtx')
  scipy.io.mmwrite(os.path.join('.', 'data', f'{matrix_id}.mtx'),data,field="real",symmetry="general")