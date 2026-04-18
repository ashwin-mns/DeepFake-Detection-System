import tensorflow as tf
from tensorflow.keras import layers, models
import os
import random

def build_model():
    # Use MobileNetV2 as a base model for Transfer Learning
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model to retain learned features
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_dataset_folders():
    folders = [
        "dataset/real",
        "dataset/fake"
    ]
    created = False
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            created = True
    return created

if __name__ == "__main__":
    print("Checking dataset folders...")
    just_created = create_dataset_folders()
    
    dataset_dir = "dataset"
    
    real_count = len(os.listdir(os.path.join(dataset_dir, "real"))) if os.path.exists(os.path.join(dataset_dir, "real")) else 0
    fake_count = len(os.listdir(os.path.join(dataset_dir, "fake"))) if os.path.exists(os.path.join(dataset_dir, "fake")) else 0
    
    if real_count == 0 and fake_count == 0:
        print("\n=======================================================")
        print("📁 Dataset folders are ready but empty!")
        print("Please add your images to train the model:")
        print(" - dataset/real  (put original/authentic images here)")
        print(" - dataset/fake  (put manipulated/deepfake images here)")
        print("=======================================================\n")
        
        # Save a dummy model if it doesn't exist so UI can still run
        model_path = "model.h5"
        if not os.path.exists(model_path):
            print("Building dummy model so UI can run...")
            model = build_model()
            model.save(model_path)
            print(f"Saved initial untrained model to {model_path}")
    else:
        print(f"Found {real_count} real and {fake_count} fake total images.")
        print("Building model...")
        model = build_model()
        model.summary()
        
        import glob
        from sklearn.model_selection import train_test_split
        
        # Load all image paths
        real_images = glob.glob(os.path.join(dataset_dir, "real", "*.jpg")) + glob.glob(os.path.join(dataset_dir, "real", "*.png"))
        fake_images = glob.glob(os.path.join(dataset_dir, "fake", "*.jpg")) + glob.glob(os.path.join(dataset_dir, "fake", "*.png"))
        
        print(f"Original Dataset: {len(real_images)} Real, {len(fake_images)} Fake.")
        
        # Downsample the majority class to balance the dataset 1:1 perfectly!
        if len(fake_images) > len(real_images) and len(real_images) > 0:
            fake_images = random.sample(fake_images, len(real_images))
        elif len(real_images) > len(fake_images) and len(fake_images) > 0:
            real_images = random.sample(real_images, len(fake_images))
            
        print(f"Balanced Dataset for Training: {len(real_images)} Real, {len(fake_images)} Fake.")
        
        # Create full dataset lists. 0 = Real, 1 = Fake
        all_paths = real_images + fake_images
        all_labels = [0]*len(real_images) + [1]*len(fake_images)
        
        # Use Scikit-Learn to explicitly split exactly 80% for training and 20% for testing
        # stratify=all_labels ensures both REAL and FAKE images are included evenly in both subsets!
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_paths, all_labels, 
            test_size=0.20, 
            stratify=all_labels, 
            random_state=42
        )
        
        # Print explicitly for assurance
        train_reals = sum(1 for label in train_labels if label == 0)
        train_fakes = sum(1 for label in train_labels if label == 1)
        test_reals = sum(1 for label in test_labels if label == 0)
        test_fakes = sum(1 for label in test_labels if label == 1)
        
        print("\n--- Explicit 80/20 Split Breakdown ---")
        print(f"Training Set (80%): {len(train_paths)} total images -> ({train_reals} Real + {train_fakes} Fake)")
        print(f"Testing/Validation Set (20%): {len(test_paths)} total images -> ({test_reals} Real + {test_fakes} Fake)")
        
        # Custom TensorFlow DataLoader
        batch_size = 32
        img_height = 224
        img_width = 224
        
        def load_and_preprocess_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_height, img_width])
            img = img / 255.0 # Normalization (0-1)
            # Expand dimensions of label to (1,) for keras compatibility with binary crossentropy
            return img, tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
            
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print("\nStarting Keras Model Training...")
        epochs = 5
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        
        model_path = "model.h5"
        model.save(model_path)
        print(f"\n✅ Training complete! Model successfully saved to {model_path}")
        print("You can now run 'python -m streamlit run app.py' to test it.")
