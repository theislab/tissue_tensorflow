import tensorflow as tf


class ModelReg:

    def __init__(
            self,
            input_shapes,
            features,
            data_types,
            width: int,
            activation: str,
            l2_reg: float,
            depth: int
    ):
        self.args = {argument: value for argument, value in locals().items() if argument != 'self'}

        if isinstance(activation, str) and 'leakyrelu' in activation:
            alpha = float(activation.split('_')[-1])
            activation = tf.keras.layers.LeakyReLU(alpha=alpha)

        input_x = tf.keras.layers.Input(
            shape=(input_shapes[0][0], input_shapes[0][1]),
            name='input_features'
        )
        input_c = tf.keras.layers.Input(
            shape=(input_shapes[1][0]),
            name='input_covar'
        )
        # take relative cell type counts per image
        cell_count = tf.expand_dims(tf.reduce_sum(input_x, axis=[1, 2]), 1)

        # checking if cell count is 0
        cell_count = tf.where(tf.equal(cell_count, 0), tf.ones_like(cell_count), cell_count)
        x = tf.reduce_sum(input_x, axis=1) / cell_count
        # concatenate with cell_count
        # cell_count_scaled = cell_count / input_shapes[0][0]
        # x = tf.concat([x, cell_count_scaled], axis=1)

        output = []

        for i in range(depth):
            x = tf.keras.layers.Dense(
                units=width,
                activation=activation,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name="dense_" + str(i)
            )(x)

        # Map embedding to output space by each task in multitask setting (loop over tasks):
        for feature_name, feature_len in features.items():
            dt = data_types[feature_name]
            if dt == 'percentage':
                act = 'sigmoid'
            elif dt == 'categorical':
                act = 'softmax'
            elif dt == 'continuous':
                act = 'linear'
            elif dt == 'survival':
                act = 'relu'
            else:
                raise ValueError('Data type not recognized: Use \'categorical\', \'continuous\' or \'percentage\'.')
            x_out = tf.keras.layers.Dense(
                feature_len,
                activation=act,
                use_bias=True,
                name=feature_name
            )(x)
            output.append(x_out)

        self.training_model = tf.keras.models.Model(
            inputs=[input_x, input_c],
            outputs=output,
            name='regression'
        )



class ModelRegDispersion:

    def __init__(
            self,
            input_shapes,
            features,
            data_types,
            depth: int,
            width: int,
            activation: str,
            l2_reg: float,
    ):
        self.args = {argument: value for argument, value in locals().items() if argument != 'self'}

        if isinstance(activation, str) and 'leakyrelu' in activation:
            alpha = float(activation.split('_')[-1])
            activation = tf.keras.layers.LeakyReLU(alpha=alpha)
        print(f'{input_shapes=}')
        input_x = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1]),
            name='input_features'
        )
        input_c = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )
        output = []

        # mean cell type proportion
        x = tf.reduce_mean(input_x, axis=1)

        for i in range(depth):
            x = tf.keras.layers.Dense(
                units=width,
                activation=activation,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name="dense_" + str(i)
            )(x)

        # Map embedding to output space by each task in multitask setting (loop over tasks):
        for feature_name, feature_len in features.items():
            dt = data_types[feature_name]
            if dt == 'percentage':
                act = 'sigmoid'
            elif dt == 'categorical':
                act = 'softmax'
            elif dt == 'continuous':
                act = 'linear'
            elif dt == 'survival':
                act = 'relu'
            else:
                raise ValueError('Data type not recognized: Use \'categorical\', \'continuous\' or \'percentage\'.')
            x_out = tf.keras.layers.Dense(
                feature_len,
                activation=act,
                use_bias=True,
                name=feature_name
            )(x)
            output.append(x_out)

        self.training_model = tf.keras.models.Model(
            inputs=[input_x, input_c],
            outputs=output,
            name='regression'
        )

