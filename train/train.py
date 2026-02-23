import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers

IMG_SIZE = (224, 224)


def build_and_train(train_dir, val_dir, model_out, labels_out, epochs=5, batch_size=16):
    if not os.path.exists(train_dir):
        raise SystemExit(f'dataset not found at {train_dir}. Expect train/<class>/')

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical')

    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)

    labels = [None] * len(train_gen.class_indices)
    for k, v in train_gen.class_indices.items():
        labels[v] = k
    with open(labels_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    print('Saved model to', model_out)


def parse_args():
    p = argparse.ArgumentParser(description='Train mango leaf classifier')
    p.add_argument('--data-dir', type=str, default=None, help='Root dataset dir containing train/ and val/ subfolders')
    p.add_argument('--train-dir', type=str, default=None, help='Specific train directory (overrides --data-dir)')
    p.add_argument('--val-dir', type=str, default=None, help='Specific val directory (overrides --data-dir)')
    p.add_argument('--model-out', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'backend', 'model.h5'), help='Model output path')
    p.add_argument('--labels-out', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'backend', 'labels.txt'), help='Labels output path')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    workspace_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if args.train_dir and args.val_dir:
        train_dir = args.train_dir
        val_dir = args.val_dir
    elif args.data_dir:
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
    else:
        # default to workspace dataset folder
        train_dir = os.path.join(workspace_base, 'dataset', 'train')
        val_dir = os.path.join(workspace_base, 'dataset', 'val')

    model_out = os.path.abspath(args.model_out)
    labels_out = os.path.abspath(args.labels_out)

    build_and_train(train_dir, val_dir, model_out, labels_out, epochs=args.epochs, batch_size=args.batch)


if __name__ == '__main__':
    main()
