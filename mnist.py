import numpy as np
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Optional: confusion matrix
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=10).numpy()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    main()