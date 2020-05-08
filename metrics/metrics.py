import os
import numpy as np

class Metrics():
    def __init__(self, f_name):
        self.f_name = f_name
        self.values = []

    def log_vals(self, value):
        self.values.append(value)

    def reset(self):
        self.values = []

    def save_to_disk(self, parent_dir, run_num):
        np.save(os.path.join(parent_dir, self.f_name + '_' + str(run_num)), np.array(self.values))

    def load_from_disk(self, parent_dir, run_num):
        self.values = list(np.load(os.path.join(parent_dir, self.f_name + '_' + str(run_num) + '.npy')))
