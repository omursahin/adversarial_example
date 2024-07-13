import time

import numpy as np
from keras.applications import vgg16, resnet
from keras.applications import vgg19
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import PIL.Image

from core.helper.config import Config
from core.helper.random import Random
from core.helper.reporter import Reporter
from core.search.mutator.gaussian_mutator import GaussianMutator
from core.search.service.archive import Archive


class DecreasePrediction:

    def __init__(self, argv):
        self.conf = Config(argv)
        self.random = Random(self.conf)
        self.reporter = Reporter(self)

        match self.conf.MODEL_NAME:
            case "vgg16":
                model = vgg16.VGG16(weights='imagenet')
            case "vgg19":
                model = vgg19.VGG19(weights='imagenet')
            case "r50":
                model = resnet.ResNet50(weights='imagenet')
            case "r101":
                model = resnet.ResNet101(weights='imagenet')
            case "r152":
                model = resnet.ResNet152(weights='imagenet')
            case _:
                print(f"Unknown model option: vgg16, vgg19, r50, r101, r152")

        self.model = model
        self.image = "./%s/%s" % (self.conf.IMAGE_FOLDER, self.conf.IMAGE_NAME)

        print("Model: %s Image: %s" % (self.conf.MODEL_NAME, self.conf.IMAGE_NAME))

        self.original_image = load_img(self.image).resize([224, 224], PIL.Image.BILINEAR)
        self.image_as_array = img_to_array(self.original_image, dtype=int)
        self.rows = len(self.image_as_array)
        self.cols = len(self.image_as_array[0])
        self.is_one_zero = True

        self.max_eval = self.conf.MAXIMUM_EVALUATION

        self.eval_count = 0
        self.interval_count = 1

        self.original_classification = self.get_predictions(self.image_as_array)
        self.current_fitness = self.calculate_fitness(self.image_as_array)

        self.archive = Archive(self)
        self.mutator = GaussianMutator(self, self.archive)

        self.start_time = time.time()
        self.execution_time = None


    def get_predictions(self, image_array):
        image_batch = np.expand_dims(image_array, axis=0)
        processed = preprocess_input(image_batch)
        predictions = self.model.predict(processed, verbose=0)
        return decode_predictions(predictions, top=10)[0]

    def get_mutated_image(self, archive):
        seed = self.image_as_array.copy()

        for pixel_change in archive:
            seed[pixel_change[0]][pixel_change[1]] = pixel_change[2]

        return seed

    def get_current_image(self):
        seed = self.image_as_array.copy()

        for pixel_change in self.archive.populations:
            seed[pixel_change[0]][pixel_change[1]] = pixel_change[2]

        return seed

    def calculate_fitness(self, image_array):
        values = self.get_predictions(image_array)
        f1 = 0

        if values[0][0] == self.original_classification[0][0]:
            f1 = np.array(values[0][2] - values[1][2], dtype=float)
        else:
            f1 = -np.array(values[0][2], dtype=float)
        self.increase_eval_count()
        return f1, values

    def write_fitness(self):
        self.execution_time = time.time() - self.start_time
        if self.conf.SHOW_PROGRESS:
            print("Eval count: " + str(self.get_eval_count()) + " Fitness: " + str(self.current_fitness[0]) + " Archive size: " + str(len(self.archive.populations))
                  + " Percentage used: " + str(self.percentage_used_budget()) + " Time elapsed: " + str(self.execution_time))

    def increase_eval_count(self):
        self.eval_count += 1
        if self.interval_count * self.conf.REPORT_INTERVAL <= self.get_eval_count():
            self.reporter.append_state(
                {
                    'eval_count': self.get_eval_count(),
                    'interval_count': self.interval_count,
                    'current_fitness': self.current_fitness[0],
                    'predictions': self.current_fitness[1],
                    'archive_size': len(self.archive.populations),
                    'total_noise': self.get_total_noise(),
                    'percentage_used_budget': self.percentage_used_budget(),
                    'archive': self.archive.populations.copy()
                }
            )
            self.interval_count += 1

    def get_total_noise(self):
        total_noise = 0
        for mutation in self.archive.populations:
            total_noise += mutation[3]
        return total_noise

    def get_eval_count(self):
        return self.eval_count

    def termination_criteria(self):
        return (self.get_eval_count() + self.archive.populations.__len__()) >= self.max_eval or self.current_fitness[0] < 0

    def percentage_used_budget(self):
        return self.get_eval_count() / self.max_eval

    def calculate_execution_time(self):
        self.execution_time = time.time() - self.start_time
        return self.execution_time
