class Mutator:

    def __init__(self, problem, archive) -> None:
        super().__init__()
        self.problem = problem
        self.archive = archive

    def do(self):
        res = self._do()

        return self.calculate_fitness_and_append(res)

    def calculate_fitness_and_append(self, res):
        new_archive = self.archive.populations.copy()
        new_archive.append(res)

        new_img = self.problem.get_mutated_image(new_archive)
        only_one_mutated_img = self.problem.get_mutated_image([res])

        fitness_value = self.problem.calculate_fitness(new_img)
        one_mutated_fitness_value = self.problem.calculate_fitness(only_one_mutated_img)

        res.append(fitness_value)
        res.append(one_mutated_fitness_value)
        return res

    def _do(self):
        pass
