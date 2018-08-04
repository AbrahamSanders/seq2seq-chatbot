"""
TrainingStats class
"""
import jsonpickle

class TrainingStats(object):
    """Class that contains a set of metrics & stats that represent a model at a point in time.
    """

    def __init__(self, training_hparams):
        """Initializes the TrainingStats instance.

        Args:
            training_hparams: the training hyperparameters.
        """
        self.training_hparams = training_hparams
        self.best_validation_metric_value = self._get_metric_baseline(self.training_hparams.validation_metric)
        self.best_training_loss = self._get_metric_baseline("loss")
        self.learning_rate = self.training_hparams.learning_rate
        self.early_stopping_check = 0
        self.global_step = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["training_hparams"]
        state["best_validation_metric_value"] = float(self.best_validation_metric_value)
        state["best_training_loss"] = float(self.best_training_loss)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def compare_training_loss(self, new_value):
        """Compare the best training loss against a new value.

        Args:
            new_value: the new training loss value to compare against.

        Returns:
            True if new_value is better than the best training loss.
            False if the best training loss is better than or equal to new_value
        """
        if self._compare_metric("loss", self.best_training_loss, new_value):
            self.best_training_loss = new_value
            return True
        else:
            return False

    def compare_validation_metric(self, new_value):
        """Compare the best validation metric value against a new value.
        Validation metric is specified in training_hparams.

        Args:
            new_value: the new validation metric value to compare against.

        Returns:
            True if new_value is better than the best validation metric value.
            False if the best validation metric value is better than or equal to new_value
        """
        if self._compare_metric(self.training_hparams.validation_metric, self.best_validation_metric_value, new_value):
            self.best_validation_metric_value = new_value
            self.early_stopping_check = 0
            return True
        else:
            self.early_stopping_check += 1
            return False

    def decay_learning_rate(self):
        """Multiply the current learning rate by the decay coefficient specified in training_hparams.

        If the learning rate falls below the minimum learning rate, it is set to the minimum.
        """
        prev_learning_rate = self.learning_rate
        self.learning_rate *= self.training_hparams.learning_rate_decay
        if self.learning_rate < self.training_hparams.min_learning_rate:
            self.learning_rate = self.training_hparams.min_learning_rate
        return prev_learning_rate, self.learning_rate

    def save(self, filepath):
        """Saves the TrainingStats to disk.

        Args:
            filepath: The path of the file to save to
        """
        json = jsonpickle.encode(self)
        with open(filepath, "w") as file:
            file.write(json)
    
    def load(self, filepath):
        """Loads the TrainingStats from a JSON file.

        Args:
            filepath: path of the JSON file.
        """
        with open(filepath) as file:
            json = file.read()
        training_stats = jsonpickle.decode(json)
        self.best_validation_metric_value = training_stats.best_validation_metric_value
        self.best_training_loss = training_stats.best_training_loss
        self.learning_rate = training_stats.learning_rate
        self.early_stopping_check = training_stats.early_stopping_check
        self.global_step = training_stats.global_step
        

    def _compare_metric(self, metric, previous_value, new_value):
        """Compare a new metric value with its previous known value and determine which value is better.
        
        Which value is better is specific to the metric.
        For instance, loss is a lower-is-better metric while accuracy is a higher-is-better metric.

        Args:
            metric: The metric being compared

            previous_value: The previous known value for the metric.

            new_value: The new value to compare against the previous value.

        Returns:
            True if new_value is better than previous_value
            False if previous_value is better than or equal to new_value
        """
        if metric == "loss":
            return new_value < previous_value
        else:
            raise ValueError("Unsupported metric: '{0}'".format(metric))

    def _get_metric_baseline(self, metric):
        """Gets a baseline value for a metric that can be used to compare the first measurement against.

        For lower-is-better metrics such as loss, this will be a very large number (99999)

        For higher-is-better metrics such as accuracy, this will be 0.

        Args:
            metric: The metric for which to get a baseline value
        """
        if metric == "loss":
            return 99999
        else:
            raise ValueError("Unsupported metric: '{0}'".format(metric))