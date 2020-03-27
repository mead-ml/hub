import time
import numpy as np
import tensorflow as tf
from baseline.utils import get_model_file, listify
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.tf.classify.training.utils import to_tensors, ClassifyTrainerTf
from baseline.progress import create_progress_bar
from eight_mile.calibration import multiclass_calibration_bins, expected_calibration_error, maximum_calibration_error


# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2


@register_trainer(task='classify', name='temperature-scaling')
class CalibrationTrainer(ClassifyTrainerTf):
    def __init__(self, model_params, **kwargs):
        super().__init__(model_params, **kwargs)
        self.y = self.model.y
        self.probs = self.model.probs
        self.logits = self.model.logits

    def _train(self, loader, dataset=True, **kwargs):
        super()._train(loader, dataset=dataset, **kwargs)
        metrics = self._test(loader, dataset=dataset, phase="Train", **kwargs)
        return metrics

    def _test(self, loader, dataset=True, **kwargs):
        epoch_probs = []
        epoch_y = []
        epoch_loss = 0
        epoch_norm = 0
        pg = create_progress_bar(len(loader))
        for batch_dict in pg(loader):
            if dataset:
                probs, lossv, y = self.sess.run(
                    [self.probs, self.loss, self.y],
                    feed_dict={TRAIN_FLAG(): 0}
                )
                batchsz = len(y)
            else:
                feed_dict = self.model.make_input(batch_dict, False)
                probs, lossv, y = self.sess.run(
                    [self.probs, self.loss, self.y],
                    feed_dict=feed_dict
                )
                batchsz = self._get_batchsz(batch_dict)
            epoch_loss += lossv * batchsz
            epoch_norm += batchsz
            epoch_probs.append(probs)
            epoch_y.append(y)
        probs = np.concatenate(epoch_probs, axis=0)
        y = np.argmax(np.concatenate(epoch_y, axis=0), axis=1)
        bins = multiclass_calibration_bins(y, probs, bins=int(kwargs.get('bins', 10)))
        metrics = {
            "ECE": expected_calibration_error(bins.accs, bins.confs, bins.counts),
            "MCE": maximum_calibration_error(bins.accs, bins.confs, bins.counts),
            "avg_loss": epoch_loss / float(epoch_norm)
        }
        return metrics


@register_trainer(task='classify', name='temperature-scaling-lbfgs')
class CalibrationTrainerLBFGS(CalibrationTrainer):
    def _train(self, loader, dataset=True, **kwargs):
        reporting_fns = kwargs.get('reporting_fns', [])
        pg = create_progress_bar(len(loader))
        epoch_loss = 0
        epoch_norm = 0
        epoch_logits = []
        epoch_probs = []
        epoch_y = []
        start = time.time()
        for batch_dict in pg(loader):
            if dataset:
                logits, probs, lossv, y = self.sess.run(
                    [self.logits, self.probs, self.loss, self.y],
                    feed_dict={TRAIN_FLAG(): 0}
                )
                batchsz = len(y)
            else:
                feed_dict = self.model.make_input(batch_dict, False)
                logits, probs, lossv, y = self.sess.run(
                    [self.logits, self.probs, self.loss, self.y],
                    feed_dict=feed_dict
                )
                batchsz = self._get_batchsz(batch_dict)
            epoch_loss += lossv * batchsz
            epoch_norm += batchsz
            epoch_logits.append(logits)
            epoch_probs.append(probs)
            epoch_y.append(y)
        logits = np.concatenate(epoch_logits, axis=0)
        probs = np.concatenate(epoch_probs, axis=0)
        y = np.argmax(np.concatenate(epoch_y, axis=0), axis=1)
        bins = multiclass_calibration_bins(y, probs, bins=int(kwargs.get('bins', 10)))
        metrics = {
            "ECE": expected_calibration_error(bins.accs, bins.confs, bins.counts),
            "MCE": maximum_calibration_error(bins.accs, bins.confs, bins.counts),
            "avg_loss": epoch_loss / float(epoch_norm)
        }

        self.report(
            0, metrics, start, "Train-Before", "EPOCH", reporting_fns, 1
        )

        # Fit
        import tensorflow_probability as tfp
        x = tf.constant(logits)
        y = tf.constant(y)
        def scale(p):
            return tfp.math.value_and_gradient(
                lambda v: tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=x / v, labels=y
                )),
                p
            )

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=scale,
            initial_position=self.model.trainable_variables[0],
            max_iterations=100
        )
        self.sess.run(self.model.trainable_variables[0].assign(results.position))

        metrics = self._test(loader, dataset, phase="Train-After", **kwargs)
        return metrics



@register_training_func('classify', 'temperature-scaling-feed-dict')
def fit(model_params, _, ts, vs, **kwargs):

    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))
    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)
    epochs = kwargs.get('epochs', 2)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)

    test_metrics = trainer.test(vs, reporting_fns, phase="Test-Before", dataset=False)
    for epoch in range(epochs):
        trainer.train(ts, reporting_fns, dataset=False)
    test_metrics = trainer.test(vs, reporting_fns, phase="Test", dataset=False)

    trainer.checkpoint()
    trainer.model.save(model_file)


@register_training_func('classify', 'temperature-scaling-lbfgs-feed-dict')
def fit(model_params, _, ts, vs=None, **kwargs):

    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))
    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)

    test_metrics = trainer.test(vs, reporting_fns, phase="Test-Before", dataset=False)
    trainer.train(ts, reporting_fns, dataset=False)
    test_metrics = trainer.test(vs, reporting_fns, phase="Test", dataset=False)

    trainer.checkpoint()
    trainer.model.save(model_file)


@register_training_func('classify', name='temperature-scaling-lbfgs-dataset')
def fit(model_params, _, ts, vs=None, **kwargs):
    """Calibrate a model with temperature scaling"""

    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))

    batchsz = kwargs['batchsz']
    lengths_key = model_params.get('lengths_key')

    ## First, make tf.datasets for ts, vs and es
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, lengths_key))
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.repeat(2)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, lengths_key))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.repeat(2)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    iter = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_dataset),
                                                     tf.compat.v1.data.get_output_shapes(train_dataset))

    features, y = iter.get_next()
    # Add features to the model params
    model_params.update(features)
    model_params['y'] = tf.one_hot(tf.reshape(y, [-1]), len(model_params['labels']))
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)
    last_improved = 0

    trainer.sess.run(train_init_op)
    trainer.train(ts, reporting_fns)
    trainer.sess.run(valid_init_op)
    test_metrics = trainer.test(vs, reporting_fns, phase='Test')

    trainer.checkpoint()
    trainer.model.save(model_file)
