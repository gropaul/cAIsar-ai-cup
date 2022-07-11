from utils.util_functions import printc

import tensorflow as tf

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):

    LR_SCHEDULE = [
        # (epoch to start, learning rate) tuples
        (0, 0.05),
        (5, 0.01),
        (10, 0.005),
        (15, 0.001),
        (20, 0.0005),
        (25, 0.0001),
    ]
    
    def lr_schedule(epoch, lr):
        """Helper function to retrieve the scheduled learning rate based on epoch."""
        if epoch < CustomLearningRateScheduler.LR_SCHEDULE[0][0] or epoch > CustomLearningRateScheduler.LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(CustomLearningRateScheduler.LR_SCHEDULE)):
            if epoch ==CustomLearningRateScheduler.LR_SCHEDULE[i][0]:
                return CustomLearningRateScheduler.LR_SCHEDULE[i][1]

        return lr

    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule = lr_schedule, lr_per_epoch = LR_SCHEDULE ):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.LR_SCHEDULE = lr_per_epoch
        self.send_message(f'Initialization complete: LR_SCHEDULE = {self.LR_SCHEDULE}')

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


    def send_message(self, message: str, **kwargs) -> None:
        printc(source='[CustomLearningRateScheduler]', message=message, **kwargs)

  
