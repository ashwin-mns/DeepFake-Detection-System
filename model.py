import tensorflow as tf                             #it defines the model
from tensorflow.keras import layers, models         #it defines the layers and models
import os                                           #it defines the os
import random                                       #it defines the random

def build_model():                                  #it defines the model

# Use MobileNetV2 as a base model for Transfer Learning

    base_model = tf.keras.applications.MobileNetV2( #it defines the model
        input_shape=(224, 224, 3),                  #it defines the input shape of the model
        include_top=False,                          #it defines the include top of the model
        weights='imagenet'                          #it defines the weights of the model
    )
    
    # 1. FINE-TUNING: Unfreeze the last 20 layers to learn deepfake-specific features
    base_model.trainable = True                     #it defines the trainable layers. true defines that the layers are trainable
    for layer in base_model.layers[:-20]:           #it defines the layers
        layer.trainable = False                     #it defines the trainable layers. false defines that the layers are not trainable

    # 2. DATA AUGMENTATION: Helps prevent overfitting by artificially expanding the dataset
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),            #it defines the data augmentation
        layers.RandomRotation(0.1),                 #it defines the data augmentation
        layers.RandomZoom(0.1),                     #it defines the data augmentation
    ])

    model = models.Sequential([                     #it defines the model
        data_augmentation,                          #it defines the data augmentation
        base_model,                                 #it defines the base model
        layers.GlobalAveragePooling2D(),            #it defines the global average pooling
        layers.Dense(256, activation='relu'),       # Slightly larger dense layer
        layers.Dropout(0.5),                        # Dropout to prevent overfitting
        layers.Dense(128, activation='relu'),       #it defines the dense layer
        layers.Dropout(0.3),                        #it defines the dropout
        layers.Dense(1, activation='sigmoid')       #it defines the dense layer
    ])

    # 3. LOWER LEARNING RATE: Crucial when fine-tuning a pre-trained model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #it defines the optimizer
                  loss='binary_crossentropy',                               #it defines the loss
                  metrics=['accuracy'])                                     #it defines the metrics
    return model

def create_dataset_folders():                                               #it defines the dataset folders
    folders = [
        "dataset/real",                                                     #it defines the dataset real
        "dataset/fake"                                                      #it defines the dataset fake
    ]
    created = False                                                         #it defines the created
    for folder in folders:                                                  #it defines the folders
        if not os.path.exists(folder):                                      #it defines the folders
            os.makedirs(folder)                                             #it defines the folders
            created = True                                                  #it defines the created
    return created

if __name__ == "__main__":                                                  #it defines the main function
    print("Checking dataset folders...")                                    #it prints the dataset folders
    just_created = create_dataset_folders()                                 #it defines the dataset folders
    
    dataset_dir = "dataset"                                                 #it defines the dataset directory
    
    real_count = len(os.listdir(os.path.join(dataset_dir, "real"))) if os.path.exists(os.path.join(dataset_dir, "real")) else 0 #it defines the real count
    fake_count = len(os.listdir(os.path.join(dataset_dir, "fake"))) if os.path.exists(os.path.join(dataset_dir, "fake")) else 0 #it defines the fake count
    
    if real_count < 2 or fake_count < 2:                                 #it defines the real count and fake count
        print("\n=======================================================")
        print("📁 Dataset folders are ready but empty!")                    #it prints the dataset folders
        print("Please add your images to train the model:")                  #it prints the dataset folders
        print(" - dataset/real  (put original/authentic images here)")       #it prints the dataset folders
        print(" - dataset/fake  (put manipulated/deepfake images here)")     #it prints the dataset folders
        print("=======================================================\n")
        
        # Save a dummy model if it doesn't exist so UI can still run
        model_path = "model.h5"                                                 #it defines the model path
        if not os.path.exists(model_path):                                      #it defines the model path
            print("Building dummy model so UI can run...")                      #it prints the model path
            model = build_model()                                               #it defines the model
            model.save(model_path)                                              #it defines the model path
            print(f"Saved initial untrained model to {model_path}")             #it prints the model path
        exit()
    else:
        print(f"Found {real_count} real and {fake_count} fake total images.")   #it prints the real count and fake count
        print("Building model...")                                              #it prints the model
        model = build_model()                                                   #it defines the model
        model.summary()                                                         #it defines the model summary
        
        import glob                                                             #it defines the glob
        from sklearn.model_selection import train_test_split                    #it defines the train test split
        
        # Load all image paths
        real_images = glob.glob(os.path.join(dataset_dir, "real", "*.jpg")) + glob.glob(os.path.join(dataset_dir, "real", "*.png")) #it defines the real images
        fake_images = glob.glob(os.path.join(dataset_dir, "fake", "*.jpg")) + glob.glob(os.path.join(dataset_dir, "fake", "*.png")) #it defines the fake images
        
        print(f"Original Dataset: {len(real_images)} Real, {len(fake_images)} Fake.") #it prints the real count and fake count
        
        # Downsample the majority class to balance the dataset 1:1 perfectly!
        if len(fake_images) > len(real_images) and len(real_images) > 0:              #it defines the fake count and real count
            fake_images = random.sample(fake_images, len(real_images))                #it defines the fake images
        elif len(real_images) > len(fake_images) and len(fake_images) > 0:            #it defines the real count and fake count
            real_images = random.sample(real_images, len(fake_images))                #it defines the real images
            
        print(f"Balanced Dataset for Training: {len(real_images)} Real, {len(fake_images)} Fake.") #it prints the real count and fake count
        
        # Create full dataset lists. 0 = Real, 1 = Fake
        all_paths = real_images + fake_images                                       #it defines the all paths
        all_labels = [0]*len(real_images) + [1]*len(fake_images)                    #it defines the all labels
        
        # Use Scikit-Learn to explicitly split exactly 80% for training and 20% for testing
        # stratify=all_labels ensures both REAL and FAKE images are included evenly in both subsets!
        train_paths, test_paths, train_labels, test_labels = train_test_split(        #it defines the train test split
            all_paths, all_labels,                                                    #it defines the all paths and all labels
            test_size=0.20,                                                           #it defines the test size
            stratify=all_labels,                                                      #it defines the stratify
            random_state=42                                                           #it defines the random state
        )
        
        # Print explicitly for assurance
        train_reals = sum(1 for label in train_labels if label == 0)                #it defines the train reals
        train_fakes = sum(1 for label in train_labels if label == 1)                #it defines the train fakes
        test_reals = sum(1 for label in test_labels if label == 0)                  #it defines the test reals
        test_fakes = sum(1 for label in test_labels if label == 1)                  #it defines the test fakes
        
        print("\n--- Explicit 80/20 Split Breakdown ---")                           #it prints the train test split
        print(f"Training Set (80%): {len(train_paths)} total images -> ({train_reals} Real + {train_fakes} Fake)")        #it prints the train test split
        print(f"Testing/Validation Set (20%): {len(test_paths)} total images -> ({test_reals} Real + {test_fakes} Fake)") #it prints the train test split
        
        # Custom TensorFlow DataLoader
        batch_size = 32                                                             #it defines the batch size
        img_height = 224                                                            #it defines the image height
        img_width = 224                                                             #it defines the image width
        
        def load_and_preprocess_image(path, label):                                 #it defines the load and preprocess image
            img = tf.io.read_file(path)                                             #it defines the load and preprocess image
            img = tf.io.decode_image(img, channels=3, expand_animations=False)      #it defines the load and preprocess image
            img.set_shape([None, None, 3])                                          #it defines the load and preprocess image
            img = tf.image.resize(img, [img_height, img_width])                     #it defines the load and preprocess image
            img = (img / 127.5) - 1.0                                                       #it defines the load and preprocess image
            # Expand dimensions of label to (1,) for keras compatibility with binary crossentropy
            return img, tf.expand_dims(tf.cast(label, tf.float32), axis=-1)                         #it defines the load and preprocess image
            
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))                  #it defines the train ds
        train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)     #it defines the train ds
        train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)  #it defines the train ds

        val_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))                       #it defines the val ds
        val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)          #it defines the val ds
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)                                 #it defines the val ds

        print("\nStarting Keras Model Training...")                                                  #it prints the model training
        epochs = 20                                                                                  #it defines the epochs
        
        # 4. CALLBACKS: Reduce learning rate if stagnating (Early stopping disabled to force full 20 epochs)
        callbacks = [                                                                                                #it defines the callbacks
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1) #it defines the reduce learning rate
        ]

        history = model.fit(                                                                            #it defines the model fit
            train_ds,                                                                                   #it defines the train ds
            validation_data=val_ds,                                                                     #it defines the val ds
            epochs=epochs,                                                                              #it defines the epochs
            callbacks=callbacks                                                                         #it defines the callbacks
        )
        
        model_path = "model.h5"                                                                         #it defines the model path
        model.save(model_path)                                                                          #it defines the model path
        print(f"\n[SUCCESS] Training complete! Model successfully saved to {model_path}")                      #it prints the model path
        print("You can now run 'python -m streamlit run app.py' to test it.")                           #it prints the model path
