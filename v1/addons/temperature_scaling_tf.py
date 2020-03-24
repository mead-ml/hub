"""Implementations of various post-hoc calibrations outlined in https://arxiv.org/abs/1706.04599

The following are implemented:

    * Temperature Scaling: Divide each logit by a single learned parameter
    * Vector Scaling: Scale each logit with its own learned parameter and add a bias
    * Matrix Scaling: Project the logits with a single dense layer
"""

import tensorflow as tf
from baseline.model import register_model
from baseline.tf.classify.model import (
    ConvModel,
    LSTMModel,
    NBowModel,
    NBowMaxModel,
    FineTuneModelClassifier,
    CompositePoolingModel
)


class TemperatureScaling(tf.keras.layers.Layer):
    """Scale the output logits by a single learned parameter."""
    def __init__(self, num_classes, name="temperature_scaling", dtype=tf.float32, trainable=True):
        super().__init__(trainable=trainable, name=name, dtype=tf.float32)
        self.num_classes = num_classes
        self.temperature = None

    def build(self, input_shape):
        self.temperature = self.add_weight(
            "temperature",
            shape=[1],
            initializer=tf.compat.v1.constant_initializer(1.0)
        )

    def call(self, inputs):
        return inputs / self.temperature


class VectorScaling(tf.keras.layers.Layer):
    """Scale the output logits by a vector and bias."""
    def __init__(self, num_classes, name="vector_scaling", dtype=tf.float32, trainable=True):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.num_classes = num_classes
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight",
            shape=[input_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(1.0)
        )
        self.bias = self.add_weight(
            "bias",
            shape=[input_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(0.0)
        )

    def call(self, inputs):
        # Broadcast the weight over every things
        # [B, T, ..., C] * [C] -> [B, T, ..., C] * [1, 1, ..., C] = [B, T, ..., C]
        return tf.nn.bias_add(tf.multiply(inputs, self.weight), self.bias)


class MatrixScaling(tf.keras.layers.Layer):
    """Scale the output logits by a fully connected layer projection."""
    def __init__(self, num_classes, name="matrix_scaling", dtype=tf.float32, trainable=True):
        super().__init__(trainable=True, name=name)
        self.scale = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.compat.v1.keras.initializers.Identity(),
            bias_initializer=tf.compat.v1.constant_initializer(1.0)
        )

    def call(self, inputs):
        return self.scale(inputs)


class CalibrationClassifierMixin:
    """A base class for our post-hoc calibration methods."""
    def call(self, *args, **kwargs):
        """Get the logits with the implementation and then calibrate them.

        Here we can see that `super().call` is the way to call the call function of your parent
          https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/convolutional_recurrent.py#L933
        """
        return self.calibrate(super().call(*args, **kwargs))

    @property
    def trainable_variables(self):
        """This is a post-hoc method so make sure the only things that are trainable are the calibration variables."""
        return self.calibrate.trainable_variables


class TemperatureScaledClassifierMixin(CalibrationClassifierMixin):
    """Calibrate a model with temperature scaling."""
    def create_layers(self, **kwargs):
        super().create_layers(**kwargs)
        self.calibrate = TemperatureScaling(len(self.labels))


class VectorScaledClassifierMixin(CalibrationClassifierMixin):
    """Calibrate a model with vector scaling."""
    def create_layers(self, **kwargs):
        super().create_layers(**kwargs)
        self.calibrate = VectorScaling(len(self.labels))


class MatrixScaledClassifierMixin(CalibrationClassifierMixin):
    """Calibrate a model with matrix scaling."""
    def create_layers(self, **kwargs):
        super().create_layers(**kwargs)
        self.calibrate = MatrixScaling(len(self.labels))


@register_model(task='classify', name='temp-conv')
class TemperatureScaledConvModel(TemperatureScaledClassifierMixin, ConvModel):
    """Calibrate a conv model with temperature scaling.

    Note:
        We default the name of the model to conv_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "conv_model"
        super().__init__(name=name)


@register_model(task='classify', name='vector-conv')
class VectorScaledConvModel(VectorScaledClassifierMixin, ConvModel):
    """Calibrate a conv model with vector scaling.

    Note:
        We default the name of the model to conv_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "conv_model"
        super().__init__(name=name)

@register_model(task='classify', name='matrix-conv')
class MatrixScaledConvModel(MatrixScaledClassifierMixin, ConvModel):
    """Calibrate a conv model with matrix scaling.

    Note:
        We default the name of the model to conv_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "conv_model"
        super().__init__(name=name)


@register_model(task='classify', name='temp-lstm')
class TemperatureScaledLSTMModel(TemperatureScaledClassifierMixin, LSTMModel):
    """Calibrate a lstm model with temperature scaling.

    Note:
        We default the name of the model to lstm_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "lstm_model"
        super().__init__(name=name)

@register_model(task='classify', name='vector-lstm')
class VectorScaledLSTMModel(VectorScaledClassifierMixin, LSTMModel):
    """Calibrate a lstm model with vector scaling.

    Note:
        We default the name of the model to lstm_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "lstm_model"
        super().__init__(name=name)

@register_model(task='classify', name='matrix-lstm')
class MatrixScaledLSTMModel(MatrixScaledClassifierMixin, LSTMModel):
    """Calibrate a lstm model with matrix scaling.

    Note:
        We default the name of the model to lstm_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "lstm_model"
        super().__init__(name=name)


@register_model(task='classify', name='temp-nbow')
class TemperatureScaledNBowModel(TemperatureScaledClassifierMixin, NBowModel):
    """Calibrate a nbow model with temperature scaling.

    Note:
        We default the name of the model to n_bow_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_model"
        super().__init__(name=name)

@register_model(task='classify', name='vector-nbow')
class VectorScaledNBowModel(VectorScaledClassifierMixin, NBowModel):
    """Calibrate a nbow model with vector scaling.

    Note:
        We default the name of the model to n_bow_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_model"
        super().__init__(name=name)

@register_model(task='classify', name='matrix-nbow')
class MatrixScaledNBowModel(MatrixScaledClassifierMixin, NBowModel):
    """Calibrate a nbow model with matrix scaling.

    Note:
        We default the name of the model to n_bow_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_model"
        super().__init__(name=name)


@register_model(task='classify', name='temp-nbowmax')
class TemperatureScaledNBowMaxModel(TemperatureScaledClassifierMixin, NBowMaxModel):
    """Calibrate a nbow max over time model with temperature scaling.

    Note:
        We default the name of the model to n_bow_max_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_max_model"
        super().__init__(name=name)

@register_model(task='classify', name='vector-nbowmax')
class VectorScaledNBowMaxModel(VectorScaledClassifierMixin, NBowMaxModel):
    """Calibrate a nbow max over time model with vector scaling.

    Note:
        We default the name of the model to n_bow_max_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_max_model"
        super().__init__(name=name)


@register_model(task='classify', name='matrix-nbowmax')
class MatrixScaledNBowMaxModel(MatrixScaledClassifierMixin, NBowMaxModel):
    """Calibrate a nbow max over time model with matrix scaling.

    Note:
        We default the name of the model to n_bow_max_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "n_bow_max_model"
        super().__init__(name=name)



@register_model(task='classify', name='temp-fine-tune')
class TemperatureScaledFineTuneModelClassifier(TemperatureScaledClassifierMixin, FineTuneModelClassifier):
    """Calibrate a fine tuned model with temperature scaling.

    Note:
        We default the name of the model to fine_tune_model_classifier so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "fine_tune_model_classifier"
        super().__init__(name=name)

@register_model(task='classify', name='Vector-fine-tune')
class VectorScaledFineTuneModelClassifier(VectorScaledClassifierMixin, FineTuneModelClassifier):
    """Calibrate a fine tuned model with vector scaling.

    Note:
        We default the name of the model to fine_tune_model_classifier so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "fine_tune_model_classifier"
        super().__init__(name=name)


@register_model(task='classify', name='matrix-fine-tune')
class MatrixScaledFineTuneModelClassifier(MatrixScaledClassifierMixin, FineTuneModelClassifier):
    """Calibrate a fine tuned model with matrix scaling.

    Note:
        We default the name of the model to fine_tune_model_classifier so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "fine_tune_model_classifier"
        super().__init__(name=name)



@register_model(task='classify', name='temp-composite')
class TemperatureScaledCompositePoolingModel(TemperatureScaledClassifierMixin, CompositePoolingModel):
    """Calibrate a composite model with temperature scaling.

    Note:
        We default the name of the model to composite_pooling_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "composite_pooling_model"
        super().__init__(name=name)

@register_model(task='classify', name='vector-composite')
class VectorScaledCompositePoolingModel(VectorScaledClassifierMixin, CompositePoolingModel):
    """Calibrate a composite model with vector scaling.

    Note:
        We default the name of the model to composite_pooling_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "composite_pooling_model"
        super().__init__(name=name)

@register_model(task='classify', name='matrix-composite')
class MatrixScaledCompositePoolingModel(MatrixScaledClassifierMixin, CompositePoolingModel):
    """Calibrate a composite model with matrix scaling.

    Note:
        We default the name of the model to composite_pooling_model so that the weights from
        the conv_model are reloaded correctly, otherwise the scope would be
        based on this class name and weights would not be reloaded.
    """
    def __init__(self, name=None):
        name = name if name is not None else "composite_pooling_model"
        super().__init__(name=name)
