import yaml

class train:
    def __init__(self):
        self.train_dir = None
        self.valid_dir = None
        self.use_valid_data = None
        self.output_dir = None
        self.train_split = None
        self.features = None
        self.add_range = None
        self.labels = None
        self.normalize = None
        self.binary = None
        self.device = None
        self.batch_size = None
        self.epochs = None
        self.init_lr = None
        self.output_classes = None
        self.epoch_timeout = None
        self.threshold_method = None
        self.termination_criteria = None
        self.model = None


class test:
    def __init__(self):
        self.test_dir  = None
        self.device = None
        self.batch_size = 1
        self.save_pred_clouds = None
        

class Config():

    def __init__(self, root_dir = ''):
        """Stores information about training and test configuration 

        Args:
            root_dir {string}: Absolute path to the config.yaml file.
        """
        with open(root_dir) as file:
            self.config = yaml.safe_load(file)

        # ---------------------------------------------------------------------#
        # TRAIN CONFIGURATION
        self.train = train()
        self.train.train_dir = self.config["train"]["TRAIN_DIR"]
        self.train.valid_dir = self.config["train"]["VALID_DIR"]
        self.train.use_valid_data = self.config["train"]["USE_VALID_DATA"]
        self.train.output_dir = self.config["train"]["OUTPUT_DIR"]
        self.train.train_split = self.config["train"]["TRAIN_SPLIT"]
        self.train.features = self.config["train"]["FEATURES"]
        self.train.labels = self.config["train"]["LABELS"]
        self.train.normalize = self.config["train"]["NORMALIZE"]
        self.train.binary = self.config["train"]["BINARY"]
        self.train.device = self.config["train"]["DEVICE"]
        self.train.batch_size = self.config["train"]["BATCH_SIZE"]
        self.train.epochs = self.config["train"]["EPOCHS"]
        self.train.init_lr = self.config["train"]["LR"]
        self.train.output_classes = self.config["train"]["OUTPUT_CLASSES"]
        self.train.threshold_method = self.config["train"]["THRESHOLD_METHOD"]
        self.train.termination_criteria = self.config["train"]["TERMINATION_CRITERIA"]
        self.train.epoch_timeout = self.config["train"]["EPOCH_TIMEOUT"]
        
        # ---------------------------------------------------------------------#
        # TEST CONFIGURATION
        self.test = test()
        self.test.test_dir = self.config["test"]["TEST_DIR"]
        self.test.device = self.config["test"]["DEVICE"]
        self.test.batch_size = self.config["test"]["BATCH_SIZE"]
        self.test.save_pred_clouds = self.config["test"]["SAVE_PRED_CLOUDS"]
