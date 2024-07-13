import numpy as np


class Archive:
    def __init__(self, problem):
        self.problem = problem
        self.populations = []
        self.sampling_counter = 0
        self.last_improvement = []
        self.last_chosen = []

    def clean_population(self):
        self.populations = []

    def add_archive_if_needed(self, individual):
        fitness_value = individual[4]

        if fitness_value[0] < self.problem.current_fitness[0]:
            individual.append(0)
            self.populations.append(individual)
            self.problem.current_fitness = individual[4]

    def shrink_archive(self):
        i = 0
        while i < len(self.populations):
            new_archive = self.populations.copy()
            new_archive.pop(i)
            new_img = self.problem.get_mutated_image(new_archive)
            fitness_value = self.problem.calculate_fitness(new_img)
            if fitness_value[0] < 0:
                self.populations.pop(i)
                self.problem.current_fitness = fitness_value
            else:
                i += 1

    def sample_individual(self):
        if self.populations.__len__() == 0:
            return

        ck_counters = np.array([i[-1] for i in self.populations])
        random_index = self.problem.random.random_choice(np.flatnonzero(ck_counters == min(ck_counters)))
        individual = self.populations[random_index].copy()
        self.last_chosen = individual
        self.sampling_counter += 1
        if self.problem.is_one_zero:
            new_mutate, is_better = self.problem.mutator.apply_one_zero_mutation(individual)
        else:
            new_mutate, is_better = self.problem.mutator.apply_mutation(individual)

        if is_better:
            self.populations[random_index][-1] = 0
        else:
            self.populations[random_index][-1] += 1

        return new_mutate

    def is_empty(self):
        return self.populations.__len__() == 0
