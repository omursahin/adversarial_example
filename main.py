import os
import sys
import tensorflow as tf
from core.problem.decrease_prediction import DecreasePrediction

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
        print(e)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def main(argv):
    problem = DecreasePrediction(argv=argv)
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

    print("Starting...")
    while not problem.termination_criteria():
        if problem.archive.is_empty():
            mutated = problem.mutator.do()
            problem.archive.add_archive_if_needed(mutated)
            problem.write_fitness()

        sample = problem.archive.sample_individual()
        if sample is not None:
            problem.archive.add_archive_if_needed(sample)
            problem.write_fitness()
    print("Shrinking...")
    problem.archive.shrink_archive()
    problem.calculate_execution_time()
    problem.write_fitness()
    if problem.conf.SAVE_RESULTS:
        print("Saving...")
        problem.reporter.report()
    print("Done")


if __name__ == '__main__':
    main(sys.argv[1:])
