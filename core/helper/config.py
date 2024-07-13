import sys
import getopt
import configparser
import os


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


# @singleton
class Config:
    def __init__(self, argv):
        config = configparser.ConfigParser()
        config.read(os.path.dirname(os.path.abspath(__file__)) + '/conf.ini')
        # ####SETTINGS FILE######
        self.EXPERIMENT_NAME = config['DEFAULT']['ExperimentName']
        self.IMAGE_NAME = config['DEFAULT']['ImageName']
        self.IMAGE_FOLDER = config['DEFAULT']['ImageFolder']
        self.MODEL_NAME = config['DEFAULT']['ModelName']
        self.RUN = config['DEFAULT']['Run']

        self.SHOW_PLOTS = bool(config['REPORT']['ShowPlots'] == 'True')


        self.SHOW_PROGRESS = bool(config['REPORT']['ShowProgress'] == 'True')
        self.SAVE_RESULTS = bool(config['REPORT']['SaveResults'] == 'True')
        self.OUTPUTS_FOLDER_NAME = str(config['REPORT']['OutputsFolderName'])
        self.REPORT_INTERVAL = int(config['REPORT']['ReportInterval'])

        self.RANDOM_SEED = config['SEED']['RandomSeed'] == 'True'
        self.SEED = int(config['SEED']['Seed'])

        # Control Parameters
        self.MAXIMUM_EVALUATION = int(config['CONTROL_PARAMETERS']['MaximumEvaluation'])
        self.ONE_MUTATION_RATE = float(config['CONTROL_PARAMETERS']['OneMutationRate'])
        self.ZERO_MUTATION_RATE = float(config['CONTROL_PARAMETERS']['ZeroMutationRate'])
        self.APC_THRESHOLD = float(config['CONTROL_PARAMETERS']['APCThreshold'])
        self.APC_START_TIME = float(config['CONTROL_PARAMETERS']['APCStartTime'])
        self.APC_PIXEL_START_VALUE = float(config['CONTROL_PARAMETERS']['APCPixelStartValue'])
        self.APC_PIXEL_END_VALUE = float(config['CONTROL_PARAMETERS']['APCPixelEndValue'])
        self.APC_NOISE_START_VALUE = float(config['CONTROL_PARAMETERS']['APCNoiseStartValue'])
        self.APC_NOISE_END_VALUE = float(config['CONTROL_PARAMETERS']['APCNoiseEndValue'])

        # ####SETTINGS FILE######

        # ####SETTINGS ARGUMENTS######
        try:
            opts, args = getopt.getopt(argv, 'hm:r:i:o:',
                                       ['help', 'save_results=', 'show_progress=', 'show_plots=',
                                        'report_interval=', 'max_eval=', 'run=', 'image_name=', 'model_name=',
                                        'output_folder=', 'apc_threshold=', 'apc_start_time=', 'apc_pixel_start=',
                                        'apc_pixel_end=', 'apc_noise_start=', 'apc_noise_end=',
                                        'one_mutation_rate=', 'zero_mutation_rate='])
        except getopt.GetoptError:
            print('Usage: main.py -h or --help')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print('-h or --help : Show Usage')
                print('--show_progress : Show Progress')
                print('--save_results : Save Results')
                print('--show_plots : Show Plots')
                print('-m or --max_eval : Maximum Evaluation')
                print('-r or --run : Run Number')
                print('-i or --image_name= : image name')
                print('-o or --output_folder= [DEFAULT: Outputs]')
                sys.exit()
            if opt in ('-m', '--max_eval'):
                self.MAXIMUM_EVALUATION = int(arg)
            if opt in ('-i', '--image_name'):
                self.IMAGE_NAME = str(arg.strip())
            if opt == '--model_name':
                self.MODEL_NAME = str(arg.strip())
            elif opt in '--output_folder':
                self.OUTPUTS_FOLDER_NAME = arg
            elif opt in ('-r', '--run'):
                self.RUN = arg
            elif opt in ('--show_progress'):
                self.SHOW_PROGRESS = bool(arg == 'True')
            elif opt in ('--save_results'):
                self.SAVE_RESULTS = bool(arg == 'True')
            elif opt in ('--show_plots'):
                self.SHOW_PLOTS = bool(arg == 'True')
            elif opt == '--apc_threshold':
                self.APC_THRESHOLD = float(arg)
            elif opt == '--apc_start_time':
                self.APC_START_TIME = float(arg)
            elif opt == '--apc_pixel_start':
                self.APC_PIXEL_START_VALUE = float(arg)
            elif opt == '--apc_pixel_end':
                self.APC_PIXEL_END_VALUE = float(arg)
            elif opt == '--apc_noise_start':
                self.APC_NOISE_START_VALUE = float(arg)
            elif opt == '--apc_noise_end':
                self.APC_NOISE_END_VALUE = float(arg)
            elif opt == '--one_mutation_rate':
                self.ONE_MUTATION_RATE = float(arg)
            elif opt == '--zero_mutation_rate':
                self.ZERO_MUTATION_RATE = float(arg)

        # ####SETTINGS ARGUMENTS######
