import datetime
import os

import numpy as np
from keras_preprocessing.image.utils import array_to_img
from matplotlib import pyplot as plt


class Reporter:
    def __init__(self, problem):
        self.problem = problem
        self.state = []
        self.data = {}
        self.run_folder = ""
        if problem.conf.SAVE_RESULTS:
            self.create_experiment_folders()

    def create_experiment_folders(self):
        if not os.path.exists(self.problem.conf.OUTPUTS_FOLDER_NAME):
            os.makedirs(self.problem.conf.OUTPUTS_FOLDER_NAME)
        experiment_folder = "%s/mio_%s_%s" % (self.problem.conf.OUTPUTS_FOLDER_NAME,
                                              self.problem.conf.MODEL_NAME,
                                              self.problem.conf.IMAGE_NAME.split('.')[0])
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)

        self.run_folder = "%s/%s" % (experiment_folder, self.problem.conf.RUN)
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

    def append_state(self, state):
        self.state.append(state)

    def report(self):
        self.save_matrix_overlay()
        self.show_and_save_final_image()
        self.show_and_save_line_plot()
        self.save_data_as_json()

    def save_matrix_overlay(self):
        matrix = self.problem.archive.populations
        img = np.zeros([224, 224, 3], dtype=np.uint8)
        img[:] = [255., 255., 255.]

        for point in matrix:
            img[int(point[0])][int(point[1])] = point[2]
        matrix_applied = array_to_img(img)
        matrix_applied.save(f"{self.run_folder}/matrix_overlay.png")

        if self.problem.conf.SHOW_PLOTS:
            plt.imshow(img)
            plt.show()

    def show_and_save_final_image(self):
        curr_image = self.problem.get_current_image()
        img = array_to_img(curr_image)
        img.save(f"{self.run_folder}/final_image.jpg")

        if self.problem.conf.SHOW_PLOTS:
            plt.imshow(img)
            plt.show()

    def show_and_save_line_plot(self):
        plt.clf()
        legend = []
        data = {}
        data_counter = 0
        for val in self.state:
            for s in val['predictions']:
                if s[1] not in data:
                    # if not exists on the beginning add zeros
                    data[s[1]] = [0] * data_counter
                data[s[1]].append(s[2])

        self.data = data
        for item in data:
            plt.plot(np.arange(len(data[item])), data[item], label=item)
            legend.append(item)

        plt.xlabel('Generations')
        plt.ylabel('Prediction Fitness Value')
        ax = plt.subplot(111)
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.8, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)
        # plt.legend(legend, bbox_to_anchor=(1, 1), shadow=True, ncol=1, loc="upper left")
        plt.savefig(f"{self.run_folder}/line.png", bbox_inches='tight')
        if self.problem.conf.SHOW_PLOTS:
            plt.show()

    def save_data_as_json(self):
        data = {}
        data['execution_time'] = self.problem.execution_time
        data['run'] = self.problem.conf.RUN
        data['eval_count'] = self.problem.get_eval_count()
        data['interval_count'] = self.problem.interval_count
        data['current_fitness'] = self.problem.current_fitness[0]
        if data['current_fitness'] < 0:
            data['flipped'] = True
        else:
            data['flipped'] = False

        data['predictions'] = self.data
        data['noise'] = self.problem.get_total_noise()
        data['matrix_size'] = len(self.problem.archive.populations)
        matrix = self.problem.archive.populations.copy()

        for i in range(len(matrix)):
            matrix[i].pop(4)

        data['matrix'] = matrix

        data['apc_pixel_start_value'] = self.problem.conf.APC_PIXEL_START_VALUE
        data['apc_pixel_end_value'] = self.problem.conf.APC_PIXEL_END_VALUE
        data['apc_noise_start_value'] = self.problem.conf.APC_NOISE_START_VALUE
        data['apc_noise_end_value'] = self.problem.conf.APC_NOISE_END_VALUE
        data['apc_threshold'] = self.problem.conf.APC_THRESHOLD
        data['apc_start_time'] = self.problem.conf.APC_START_TIME
        data['one_mutation_rate'] = self.problem.conf.ONE_MUTATION_RATE
        data['zero_mutation_rate'] = self.problem.conf.ZERO_MUTATION_RATE

        import json
        with open(f"{self.run_folder}/data.json", 'w') as outfile:
            json.dump(data, outfile, indent=6, sort_keys=True, default=str)
