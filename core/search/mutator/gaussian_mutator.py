import numpy as np

from core.search.mutator.mutator import Mutator
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl


# Based on https://zenodo.org/record/7741266

class GaussianMutator(Mutator):

    def __init__(self, problem, archive):
        super().__init__(problem, archive)
        self.apc = AdaptiveParameterControl(problem)

    def add_gaussian_mutation(self):
        # Create a copy of the seed image
        seed = self.problem.image_as_array.copy()

        already_changed = True
        while already_changed:
            already_changed = False
            row_index = self.problem.random.random_int(0, self.problem.rows - 1)
            col_index = self.problem.random.random_int(0, self.problem.cols - 1)

            existing_mutation = []
            for mutation in self.archive.populations:
                if mutation[0] == row_index and mutation[1] == col_index:
                    already_changed = True
                    existing_mutation = mutation
                    break

        value, noise = self.gaussian_noise(seed[row_index][col_index])
        return [row_index, col_index, value, noise]

    def gaussian_noise(self, pixel):
        sigma = self.apc.get_dpc_value(100, 20)
        delta = np.round(self.problem.random.random_gaussian(pixel * 0, sigma))

        if self.problem.random.random_float() <= 0.5:
            delta[delta == 0] = 1
        else:
            delta[delta == 0] = -1

        value = pixel + delta

        value[value < 0] = 0
        value[value > 255] = 255

        noise = self.calculate_noise(pixel, delta)

        return value, noise

    def calculate_noise(self, pixel, delta):
        value = pixel + delta
        noise = 0
        for i in range(0, len(delta)):
            if value[i] <= 0:
                noise += pixel[i]
            elif value[i] >= 255:
                noise += 255 - pixel[i]
            else:
                noise += abs(delta[i])

        return noise

    def apply_mutation(self, mutation):
        sigma = self.apc.get_dpc_value(self.apc.pixel_start_value, self.apc.pixel_end_value)
        row = np.round(self.problem.random.random_gaussian(mutation[0], sigma))
        col = np.round(self.problem.random.random_gaussian(mutation[1], sigma))
        pixel = mutation[2]
        sigma = self.apc.get_dpc_value(self.apc.noise_start_value, self.apc.noise_end_value)

        delta = np.round(self.problem.random.random_gaussian(pixel * 0, sigma))

        if self.problem.random.random_float() <= 0.5:
            delta[delta == 0] = 1
        else:
            delta[delta == 0] = -1

        value = pixel + delta

        value[value < 0] = 0
        value[value > 255] = 255

        if row < 0:
            row = 0
        elif row >= self.problem.rows:
            row = self.problem.rows - 1

        if col < 0:
            col = 0
        elif col >= self.problem.cols:
            col = self.problem.cols - 1

        noise = self.calculate_noise(pixel, delta)

        individual = [int(row), int(col), value, noise]

        res = self.calculate_fitness_and_append(individual)

        is_better = True if res[5] < mutation[5] else False

        return individual, is_better

    def apply_one_zero_mutation(self, mutation):
        sigma = self.apc.get_dpc_value(self.apc.pixel_start_value, self.apc.pixel_end_value)
        row = np.round(self.problem.random.random_gaussian(mutation[0], sigma))
        col = np.round(self.problem.random.random_gaussian(mutation[1], sigma))
        pixel = mutation[2]

        rand_prob = self.problem.random.random_float()
        if rand_prob <= self.problem.conf.ONE_MUTATION_RATE:
            value = np.array([255.0, 255.0, 255.0])
            delta = value - pixel
        elif rand_prob <= self.problem.conf.ONE_MUTATION_RATE + self.problem.conf.ZERO_MUTATION_RATE:
            value = np.array([0, 0, 0])
            delta = value - pixel
        else:
            sigma = self.apc.get_dpc_value(self.apc.noise_start_value, self.apc.noise_end_value)
            delta = np.round(self.problem.random.random_gaussian(pixel * 0, sigma))
            if self.problem.random.random_float() <= 0.5:
                delta[delta == 0] = 1
            else:
                delta[delta == 0] = -1

            value = pixel + delta

            value[value < 0] = 0
            value[value > 255] = 255

        if row < 0:
            row = 0
        elif row >= self.problem.rows:
            row = self.problem.rows - 1

        if col < 0:
            col = 0
        elif col >= self.problem.cols:
            col = self.problem.cols - 1

        noise = self.calculate_noise(pixel, delta)

        individual = [int(row), int(col), value, noise]

        res = self.calculate_fitness_and_append(individual)

        is_better = True if res[5] < mutation[5] else False

        return individual, is_better

    def _do(self):
        return self.add_gaussian_mutation()
