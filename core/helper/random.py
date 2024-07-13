import numpy as np

def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class Random:
    def __init__(self, conf):

        self.random = np.random
        if not conf.RANDOM_SEED:
            self.random.seed(conf.SEED)

    def random_float(self, min, max):
        return self.random.uniform(min, max)

    def random_float(self):
        return self.random.rand()

    def random_bool(self, value):
        return self.random.rand() < value

    def random_int(self, min=None, max=None):
        if min is None and max is None:
            return self.random.randint(0, 1)
        elif min is None:
            return self.random.randint(0, max)
        elif max is None:
            return self.random.randint(min, 1)
        else:
            return self.random.randint(min, max)

    # def random_int(self, max):
    #     return self.random.randint(0, max)

    def random_choice(self, max_size, selection_probs=None):
        if selection_probs is None:
            return self.random.choice(max_size)
        else:
            return self.random.choice(max_size, p=selection_probs)

    def random_gaussian(self, mean, std):
        return self.random.normal(mean, std)
