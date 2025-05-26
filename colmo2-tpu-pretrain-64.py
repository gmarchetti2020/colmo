import numpy as np
import os

# %%
# The distribution API is only implemented for the JAX backend for now.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["HF_HOME"] = 'huggingface'
SEED = 42 # the random seed for reproducibility
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# %%
import keras
keras.utils.set_random_seed(SEED)
keras.mixed_precision.set_global_policy("mixed_bfloat16")
keras.config.disable_flash_attention() #does not work on tpu
import keras_hub

# %%
import json
import re
import string
import warnings
warnings.filterwarnings("ignore")

# %%
#import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings


# %%
from keras import layers, models, losses, callbacks, ops, Layer
from keras.layers import Dense, Layer, Dropout,  RMSNormalization, LayerNormalization, Multiply
from keras.ops import softmax, silu
from keras_hub.layers import RotaryEmbedding

# %%
import jax
jax.distributed.initialize()
gpus=keras.distribution.list_devices()
num_gpus = len(gpus)

print("Num GPUs Available: ", num_gpus)

# %% [markdown]
# ## Pre-Training

# %%

#@title Define training params { run: "auto", display-mode: "form" }

#STAGING_BUCKET="gs://gm-llm-training/gm-models" #@param {type:"string"}
#INPUT_DIR = "/home/gimarchetti/mnt/llm-training/gm-models/datasets/dolma_ds" #@param {type:"string"}
#VOCAB_FILE = "./c4vocab.pkl" #@param {type:"string"}
#VOCAB_SIZE = 50000 # 50000 the size of the vocabulary for gpt2

#EMBEDDING_DIM = 2048 # the dimension of the word embeddings
#N_HEADS = 16 #  the number of attention heads
#N_KV_HEADS= 16 # N_HEADS//2 # KV heads in GQA , 1 per GPU
#NUM_BLOCKS = 16 # 12 in gpt2, 3 in gpt nano, 32 in LLama-7B
# The hidden dimension of the feed-forward network in the FF block = 4 * embedding with ReLU activation,
# 8/3 * embedding with SwiGLU to keep n. of computations constant
#FEED_FORWARD_DIM = 5120 # close to int(EMBEDDING_DIM * 8/3)
#VALIDATION_SPLIT = 0.2 # the fraction of data to be used for validation
#MIN_STRING_LEN = 1536  # Strings shorter than this will be discarded
#SEQ_LEN = 2048  # Length of training sequences, in tokens

#BATCH_SIZE_PER_GPU=1 #@param {"type":"integer"}
#BATCH_SIZE = BATCH_SIZE_PER_GPU * num_gpus

#DATA_DIM=4  #n. of data partitions #@param
#MODEL_DIM=16  #n. of model partitions #@param

#EPOCHS=100 #@param {"type":"integer"}

# %% [markdown]
# ## OR read the parameters in the selected text from a configuration file

# %%
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

STAGING_BUCKET = config['TRAINING']['STAGING_BUCKET']
INPUT_DIR = config['TRAINING']['INPUT_DIR']
VOCAB_FILE = config['TRAINING']['VOCAB_FILE']
VOCAB_SIZE = int(config['MODEL']['VOCAB_SIZE'])
EMBEDDING_DIM = int(config['MODEL']['EMBEDDING_DIM'])
N_HEADS = int(config['MODEL']['N_HEADS'])
N_KV_HEADS = int(config['MODEL']['N_KV_HEADS'])
NUM_BLOCKS = int(config['MODEL']['NUM_BLOCKS'])
FEED_FORWARD_DIM = int(config['MODEL']['FEED_FORWARD_DIM'])
VALIDATION_SPLIT = float(config['DATA']['VALIDATION_SPLIT'])
MIN_STRING_LEN = int(config['DATA']['MIN_STRING_LEN'])
SEQ_LEN = int(config['DATA']['SEQ_LEN'])
BATCH_SIZE_PER_GPU = int(config['TRAINING']['BATCH_SIZE_PER_GPU'])
BATCH_SIZE = BATCH_SIZE_PER_GPU * num_gpus
DATA_DIM = int(config['DISTRIBUTION']['DATA_DIM'])
MODEL_DIM = int(config['DISTRIBUTION']['MODEL_DIM'])
EPOCHS = int(config['TRAINING']['EPOCHS'])


# %%
from datasets import load_dataset, load_from_disk

raw_train_tf = (
    load_from_disk(f"{INPUT_DIR}/train",)
    .to_tf_dataset()
    .map(lambda x: x["text"], num_parallel_calls=tf_data.AUTOTUNE)
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
)
raw_val_tf = (
    load_from_disk(f"{INPUT_DIR}/test",)
    .to_tf_dataset()
    .map(lambda x: x["text"], num_parallel_calls=tf_data.AUTOTUNE)
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
)
# %% [markdown]
# ## OR read it back

# %%

# %%
#insert code to save and retrieve tokenizer
import pickle
with open(VOCAB_FILE, 'rb') as f:
    vocab = pickle.load(f)

print(vocab[100:120])

# %% [markdown]
# ## Instantiate tokenizer

# %%
tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

# %%
# packer adds a start token
start_packer = keras_hub.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

# %%
# Tokenize and split into train and label sequences.
train_ds = (raw_train_tf
            .map(preprocess, num_parallel_calls=tf_data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            #.shuffle(buffer_size=512) #there is no guarantee that the batch will be shuffled the same way by all nodes. avoid.
            #.cache() # too large to cache
            .prefetch(tf_data.AUTOTUNE)
           )
val_ds = (raw_val_tf
          .map(preprocess, num_parallel_calls=tf_data.AUTOTUNE)
          .batch(BATCH_SIZE, drop_remainder=True)
          #.cache()
          .prefetch(tf_data.AUTOTUNE)
         )

# %%
## Define Model
## Llama additions: SWiGlu, Grouped Query Attention (not used with <7B) and Rotary Positional Embedding

# This replaces the first Dense layer of the FF block
@keras.saving.register_keras_serializable()
class FFNSwiGLU2(Layer):
    def __init__(self, intermediate_dim, **kwargs):
        super(FFNSwiGLU2, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.gate = Dense(self.intermediate_dim, activation="silu", use_bias=False, kernel_initializer="glorot_uniform",
                          name="swiglu_gate",)
        self.linear = Dense(self.intermediate_dim, use_bias=False, kernel_initializer="glorot_uniform", name="swiglu_linear")
        self.multiply=Multiply(name="swiglu_out")

    def call(self, inputs):
        gated = self.gate(inputs)
        linear_out=self.linear(inputs)
        return self.multiply([gated, linear_out])

    def get_config(self):
        config = super(FFNSwiGLU2, self).get_config()
        config.update({'intermediate_dim': self.intermediate_dim})
        return config

import math

from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.backend.config import is_flash_attention_enabled
from keras.src.layers.activations.softmax import Softmax
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.layer import Layer
from keras.src.layers.regularization.dropout import Dropout

@keras.saving.register_keras_serializable()
class GQAwithRoPE(Layer):
    """Grouped Query Attention layer.

    This is an implementation of grouped-query attention introduced by
    [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here
    `num_key_value_heads` denotes number of groups, setting
    `num_key_value_heads` to 1 is equivalent to multi-query attention, and
    when `num_key_value_heads` is equal to `num_query_heads` it is equivalent
    to multi-head attention.

    This layer first projects `query`, `key`, and `value` tensors. Then, `key`
    and `value` are repeated to match the number of heads of `query`.

    Then, the `query` is scaled and dot-producted with `key` tensors. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities and concatenated back to a single
    tensor.

    Args:
        head_dim: Size of each attention head.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key and value attention heads.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        flash_attention: If `None`, the layer attempts to use flash
            attention for faster and more memory-efficient attention
            computations when possible. This behavior can be configured using
            `keras.config.enable_flash_attention()` or
            `keras.config.disable_flash_attention()`.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        seed: Optional integer to seed the dropout layer.

    Call arguments:
        query: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `target_seq_len` is the length of
            target sequence, and `feature_dim` is dimension of feature.
        value: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `source_seq_len` is the length of
            source sequence, and `feature_dim` is dimension of feature.
        key: Optional key tensor of shape
            `(batch_dim, source_seq_len, feature_dim)`. If not given, will use
            `value` for both `key` and `value`, which is most common case.
        attention_mask: A boolean mask of shape
            `(batch_dim, target_seq_len, source_seq_len)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, where 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output
            should be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: Result of the computation, of shape
            `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len`
            is for target sequence length and `feature_dim` is the query input
            last dim.
        attention_scores: (Optional) attention coefficients of shape
            `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        dropout=0.0,
        use_bias=True,
        flash_attention=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
        use_rope=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_query_heads` must be divisible by `num_key_value_heads`."
            )
        self.num_repeats = num_query_heads // num_key_value_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self._flash_attention = flash_attention or is_flash_attention_enabled()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seed = seed
        self.use_rope = use_rope

        self._inverse_sqrt_head_dim = 1.0 / math.sqrt(float(self.head_dim))
        self._return_attention_scores = False

        # Check for flash attention constraints
        if self._flash_attention and self.dropout > 0.0:
            raise ValueError(
                "Dropout is not supported when flash attention is enabled. "
                "Please set dropout to 0.0 to use flash attention."
            )

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        key_shape = value_shape if key_shape is None else key_shape
        self.feature_dim = query_shape[-1]
        self._query_dense = EinsumDense(
            "bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            bias_axes="uh" if self.use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)

        self._key_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)

        self._value_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        self._softmax = Softmax(axis=-1, dtype=self.dtype_policy)
        self._dropout_layer = Dropout(
            rate=self.dropout, dtype=self.dtype_policy, seed=self.seed
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self._output_dense = EinsumDense(
            "bquh,uhm->bqm",
            output_shape=(None, self.feature_dim),
            bias_axes="m" if self.use_bias else None,
            name="attention_output",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        self._rope=RotaryEmbedding()

        self.built = True

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype_policy,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self.kernel_initializer.__class__.from_config(
            self.kernel_initializer.get_config()
        )
        bias_initializer = self.bias_initializer.__class__.from_config(
            self.bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        self._return_attention_scores = return_attention_scores
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        if self.use_rope:
          query = self._rope(query)
          key = self._rope(key)


        key = ops.repeat(
            key, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)
        value = ops.repeat(
            value, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)

        output, scores = self._compute_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            training=training,
        )

        output = self._output_dense(
            output
        )  # (batch_dim, target_seq_len, feature_dim)

        if return_attention_scores:
            return output, scores
        return output

    def _compute_attention_mask(
        self,
        query,
        value,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, T, N, key_dim)`.
            value: Projected value tensor of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        auto_mask = None
        if query_mask is not None:
            query_mask = ops.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = ops.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = ops.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length
            mask = ops.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = ops.cast(key_mask, "bool")  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = ops.expand_dims(key_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else ops.cast(attention_mask, bool) & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query tensor of shape `(B, T, ...)`.
            value: value tensor of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`.
        """
        q_seq_length = ops.shape(query)[1]
        v_seq_length = q_seq_length if value is None else ops.shape(value)[1]
        ones_mask = ops.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        return ops.greater_equal(row_index, col_index)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        # Check for flash attention constraints
        if self._flash_attention and self._return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
            self.dropout > 0.0
            or self._return_attention_scores
            or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -1 * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                    len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = ops.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = ops.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                mask=attention_mask,
                scale=self._inverse_sqrt_head_dim,
                is_causal=False,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_head_dim, query.dtype)
        )
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        scores = ops.einsum(
            self._dot_product_equation, query, key
        )  # (batch_dim, query_heads, target_seq_len, source_seq_len)
        scores = self._masked_softmax(scores, attention_mask=attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout > 0.0:
            scores_dropout = self._dropout_layer(scores, training=training)
        else:
            scores_dropout = scores
        output = ops.einsum(self._combine_equation, scores_dropout, value)
        return output, scores

    def _masked_softmax(self, scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -1 * 2 - 1
            for _ in range(len(scores.shape) - len(attention_mask.shape)):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(scores, mask=attention_mask)

    def compute_output_shape(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        if key_shape is None:
            key_shape = value_shape

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )

        return query_shape

    def get_config(self):
        config = {
            "head_dim": self.head_dim,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "use_bias": self.use_bias,
            "dropout": self.dropout,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

@keras.saving.register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, num_heads, num_kv_heads, embed_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads=num_kv_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attn = GQAwithRoPE(
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=embed_dim // num_heads,
            use_rope=True,
            use_bias=False,
            dropout=0.0,
            flash_attention=None,
            seed=SEED,
            name="attn",
        )

        self.dropout_1 = Dropout(self.dropout_rate, seed=SEED)
        self.ln_1 = RMSNormalization()
        self.ffn_1 = FFNSwiGLU2(self.ff_dim, name="ffnswiglu", )
        self.ffn_2 = Dense(self.embed_dim, use_bias=False, name="ffnlinear",)
        self.dropout_2 = Dropout(self.dropout_rate, seed=SEED)
        self.ln_2 = RMSNormalization()

    def call(self, inputs):
      input_shape = keras.ops.shape(inputs)
      batch_size = input_shape[0]
      seq_len = input_shape[1]
      inputs_n=self.ln_1(inputs)
      attention_output = self.attn(
          query= inputs_n,
          #key= inputs_n,
          value= inputs_n, #quirk of implementation value is required, key is not and assumed = value
          use_causal_mask=True
      )
      attention_output = self.dropout_1(attention_output)
      out1 = self.ln_2(inputs + attention_output)

      ffn_1 = self.ffn_1(out1)
      ffn_2 = self.ffn_2(ffn_1)

      ffn_output = self.dropout_2(ffn_2)
      return (out1 + ffn_output)

    def get_config(self):
      config = super().get_config()
      config.update(
          {
              "num_kv_heads": self.num_kv_heads,
              "embed_dim": self.embed_dim,
              "num_heads": self.num_heads,
              "ff_dim": self.ff_dim,
              "dropout_rate": self.dropout_rate,
          }
      )
      return config

    def compute_output_shape(self, input_shape):
        # Assumes input_shape is [batch_size, sequence_length, hidden_size]
        #input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        hidden_size = input_shape[2]
        return [batch_size, sequence_length, hidden_size]



# %% [markdown]
# ## Distribution strategy
# 

# %% [markdown]
# ### FSDP

# %%

device_mesh=keras.distribution.DeviceMesh(
    shape=(DATA_DIM,MODEL_DIM), # arrays must be worker-local on tpuv3, hence 8 x 8 =64 tpus
    axis_names=["data","model" ],
    devices=gpus
)


# %%
layout_map=keras.distribution.LayoutMap(device_mesh)

# %%
# Partitioning for embeddings (regex)
layout_map["embedding/embeddings"] = (None, "model")
# Partitioning (regex) for attention layer weights
layout_map["transformer_block.*attn.*(query|key|value).*kernel"] = (None, "model", None)
layout_map["transformer_block.*attention_output.*kernel"] = (None, None, "model")
layout_map["transformer_block.*swiglu_gate.*kernel"] = ("model", None)
layout_map["transformer_block.*swiglu_linear.*kernel"] = ("model", None)
layout_map["transformer_block.*ffnlinear.*kernel"] = (None, "model")

##Hybrid DP/MP
model_parallel = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="data")
keras.distribution.set_distribution(model_parallel)

print(layout_map.__dict__)

# %% [markdown]
# ### OR data-parallel

# %%
#data_parallel = keras.distribution.DataParallel(devices=gpus)
#keras.distribution.set_distribution(data_parallel)

# %% [markdown]
# ## Create model

# %%

## From Llama
decay_steps = 1.2E6
initial_learning_rate = 0.0
alpha=0.1
warmup_steps = 2000
target_learning_rate = 3e-4
learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps,
    warmup_target=target_learning_rate,
    warmup_steps=warmup_steps,
    alpha=alpha
)
optimizer=keras.optimizers.AdamW(learning_rate=learning_rate,
                                weight_decay=0.1,
                                beta_1=0.9,
                                beta_2=0.95,
                                epsilon=1e-5,
                                )

# %%
## Model definition
inputs = keras.layers.Input(shape=(None,), dtype="int32")

## From Llama
x=layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,)(inputs)

for i in range(NUM_BLOCKS):

  # Create a new TransformerBlock instance in each iteration
  x = TransformerBlock(
      N_HEADS, N_KV_HEADS, EMBEDDING_DIM, FEED_FORWARD_DIM, dropout_rate=0.2,
      name=f"transformer_block_{i}",
  )(x)

x=RMSNormalization()(x)

output = Dense(VOCAB_SIZE,
              use_bias=False,
              #activation="softmax" #we'll use the logits
              #dtype="float32",
              kernel_initializer="glorot_uniform",
              name="output"
              )(x)

gpt = keras.Model(inputs=inputs, outputs=[output])
gpt.compile(optimizer=optimizer,
            loss=[losses.SparseCategoricalCrossentropy(from_logits=True)],
            metrics=[keras_hub.metrics.Perplexity(from_logits=True, mask_token_id=0)],
            jit_compile=True # Disable JIT compilation
           )



# %%

print (N_HEADS, N_KV_HEADS, NUM_BLOCKS, num_gpus)
gpt.summary()

# %%
for v in gpt.variables:
    print(v.path, v.shape, v.dtype, v.value.sharding)

# %% [markdown]
# ## Train Model

# %%
import datetime

# %%
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=f"{STAGING_BUCKET}/checkpoints/colmo2-tpu"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".keras",
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq="epoch",
    verbose=1)

backupandrestore=keras.callbacks.BackupAndRestore(
    backup_dir=f"{STAGING_BUCKET}/backupandrestore/colmo2-tpu", save_freq=10000, double_checkpoint=True, delete_checkpoint=False
)

# %%
earlystop=keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)

# %%
tensorboard=keras.callbacks.TensorBoard(
    log_dir=f"{STAGING_BUCKET}/logs/colmo2-tpu",
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq=10000, #"epoch",
    profile_batch=5000,
    embeddings_freq=10000,
    embeddings_metadata=None,
)

terminateonnan=keras.callbacks.TerminateOnNaN()

import socket
# on TPU clusters worker 0 hostname ends with -w-0
worker_0= True if socket.gethostname().endswith("-w-0") else False

callbacks=[checkpoint,
           earlystop, tensorboard,
           backupandrestore,
           terminateonnan]
# %%
history=gpt.fit(train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1)

# save only from worker 0
if worker_0:
    gpt.save(f"{STAGING_BUCKET}/models/colmo2-tpu.keras", save_format="keras")

# %%