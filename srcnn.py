from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, PReLU

def build_srcnn(num_channels=1):
    model = Sequential([
        Conv2D(56, kernel_size=(5, 5), padding='same', input_shape=(None, None, 1)),
        PReLU(shared_axes=[1, 2]),
        Conv2D(16, (1, 1), padding='same'),
        PReLU(shared_axes=[1, 2]),
        Conv2D(12, (3, 3), padding='same'),
        PReLU(shared_axes=[1, 2]),
        Conv2D(12, (3, 3), padding='same'),
        PReLU(shared_axes=[1, 2]),
        Conv2D(12, (3, 3), padding='same'),
        PReLU(shared_axes=[1, 2]),
        Conv2D(56, (1, 1), padding='same'),
        PReLU(shared_axes=[1, 2]),
        # Output layer: Adjust the number of filters to match the number of channels in the output images
        Conv2D(num_channels, (5, 5), padding='same')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
