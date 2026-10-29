"""
EcoSort: Intelligent Waste Management Assistant
Consolidated Python script from waste_management.ipynb
"""

import warnings
import os
import logging
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import TextVectorization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
pd.options.mode.chained_assignment = None
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# --- Cell 3 ---

import warnings
import os
import logging

# 1. Suppress standard Python warnings
warnings.filterwarnings('ignore')

# 2. Suppress TensorFlow C++ backend warnings (0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3. Suppress TensorFlow Python backend warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 4. Suppress Pandas chained assignment warnings
import pandas as pd
pd.options.mode.chained_assignment = None


# --- Cell 5 ---

from pathlib import Path

PROJECT_DIR = Path('/content/drive/MyDrive/EcoSort')
data_dir = PROJECT_DIR / 'RealWaste'

assert PROJECT_DIR.is_dir(), 'Check your EcoSort folder location.'
assert data_dir.is_dir(), 'Check your RealWaste folder location.'


# --- Cell 6 ---

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Find the category folders and count their images.
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
class_names = sorted(
    folder.name for folder in data_dir.iterdir()
    if folder.is_dir() and not folder.name.startswith('.')
)

image_files_by_class = {
    category: sorted(
        path for path in (data_dir / category).rglob('*')
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    for category in class_names
}

counts = pd.Series(
    {category: len(paths)
     for category, paths in image_files_by_class.items()},
    name='image_count'
)
display(counts.to_frame())
print('Number of categories:', len(class_names))
print('Total images:', int(counts.sum()))

assert len(class_names) == 9, (
    'Expected 9 category folders. Check whether RealWaste is nested '
    'inside another folder.'
)
assert (counts > 0).all(), 'At least one category contains no images.'

counts.plot.bar(figsize=(10, 4), title='Images per waste category')
plt.ylabel('Number of images')
plt.tight_layout()
plt.show()

# Display one image per category and report its original size.
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
image_details = []
for ax, category in zip(axes.flat, class_names):
    path = image_files_by_class[category][0]
    with Image.open(path) as image:
        image_details.append({
            'category': category,
            'width': image.width,
            'height': image.height,
            'mode': image.mode
        })
        ax.imshow(image.convert('RGB'))
    ax.set_title(category)
    ax.axis('off')

plt.tight_layout()
plt.show()
display(pd.DataFrame(image_details))


# --- Cell 8 ---

df = pd.read_csv(PROJECT_DIR / 'waste_descriptions.csv')

print('Rows and columns:', df.shape)
display(df.head())
display(df.isna().sum().to_frame('missing_values'))
display(df['category'].value_counts().to_frame('description_count'))

# Check the fields required for text classification.
assert df['description'].notna().all(), 'Some descriptions are missing.'
assert df['category'].notna().all(), 'Some categories are missing.'
assert set(df['category']) == set(class_names), (
    'Text categories and image folder names do not match.'
)

word_counts = df['description'].str.split().str.len()
vocabulary = set(' '.join(df['description'].str.lower()).split())

print('Unique descriptions:', df['description'].nunique())
print('Duplicate descriptions:', df['description'].duplicated().sum())
print('Vocabulary size (whitespace-based):', len(vocabulary))
print('Average words per description:', round(word_counts.mean(), 2))
display(word_counts.describe().to_frame('words_per_description'))


# --- Cell 9 ---

import json # Add this line

with open(PROJECT_DIR / 'waste_policy_documents.json', encoding='utf-8') as f:
    policies = json.load(f)

policy_df = pd.DataFrame(policies)

print('Number of policy documents:', len(policy_df))
display(policy_df[
    ['policy_id', 'policy_type', 'categories_covered',
     'effective_date', 'jurisdiction']
])

print('\nFirst policy document:\n')
print(policy_df.iloc[0]['document_text'])

covered_categories = {
    category
    for categories in policy_df['categories_covered']
    for category in categories
}

print('\nImage categories without a policy:',
      set(class_names) - covered_categories)


# --- Cell 11 ---

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Use your Google Drive folder
data_dir = PROJECT_DIR / 'RealWaste'


# --- Cell 12 ---

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

data_dir = PROJECT_DIR / 'RealWaste'
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
SEED = 42

if not data_dir.is_dir():
    raise FileNotFoundError(f'Folder not found: {data_dir}')

# Find the categories and their images
class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
num_classes = len(class_names)
image_paths, labels = [], []
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

for label, category in enumerate(class_names):
    for path in sorted((data_dir / category).rglob('*')):
        if path.is_file() and path.suffix.lower() in extensions:
            image_paths.append(str(path))
            labels.append(label)

# Separate files: about 80% training, 10% validation, 10% test
train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
    image_paths, labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    remaining_paths, remaining_labels,
    test_size=0.5,
    random_state=SEED,
    stratify=remaining_labels
)

# Check that no file appears in more than one group
assert set(train_paths).isdisjoint(val_paths)
assert set(train_paths).isdisjoint(test_paths)
assert set(val_paths).isdisjoint(test_paths)
assert len(train_paths) + len(val_paths) + len(test_paths) == len(image_paths)

print('Number of classes:', num_classes)
print('Class names:', class_names)
print('Total images found:', len(image_paths))
print('Training images:', len(train_paths))
print('Validation images:', len(val_paths))
print('Test images:', len(test_paths))

# Read and resize each image; create one-hot encoded labels
# Keep pixels in the 0–255 range for now.
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_paths, train_labels, training=True)
validation_ds = make_dataset(val_paths, val_labels)
test_dataset = make_dataset(test_paths, test_labels)


# --- Cell 13 ---

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Use the description data loaded in section 1.2
print('Available columns:', df.columns.tolist())

# Find likely description and category columns.
# If detection fails, the error will show what to change.
column_lookup = {str(c).strip().lower(): c for c in df.columns}

text_column = next(
    (column_lookup[name] for name in
     ['description', 'text', 'waste_description', 'item_description']
     if name in column_lookup),
    None
)
label_column = next(
    (column_lookup[name] for name in
     ['category', 'label', 'waste_category', 'waste_type']
     if name in column_lookup),
    None
)

if text_column is None or label_column is None:
    raise ValueError(
        'Could not identify the description/category columns. '
        f'Your columns are: {df.columns.tolist()}. '
        'Send this output so we can set the correct names.'
    )

print('Description column:', text_column)
print('Category column:', label_column)

# Clean text while preserving words such as "not".
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

text_data = df[[text_column, label_column]].dropna().copy()
text_data['clean_text'] = text_data[text_column].map(clean_text)
text_data[label_column] = (
    text_data[label_column].astype(str).str.strip()
)
text_data = text_data[
    text_data['clean_text'].ne('') & text_data[label_column].ne('')
].copy()

# Remove repeated descriptions to prevent train/test overlap.
# Stop if the same cleaned description has conflicting labels.
label_counts = text_data.groupby('clean_text')[label_column].nunique()
if (label_counts > 1).any():
    raise ValueError(
        'Some identical descriptions have different categories. '
        'Resolve those labels before splitting the data.'
    )

text_data = text_data.drop_duplicates(subset='clean_text')

# Encode category names as numbers.
text_label_encoder = LabelEncoder()
encoded_labels = text_label_encoder.fit_transform(text_data[label_column])
text_class_names = text_label_encoder.classes_.tolist()

# Reserve 20% for testing, keeping category proportions similar.
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    text_data['clean_text'],
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

assert set(X_train_text).isdisjoint(set(X_test_text))

# Tokenize text and create numerical TF-IDF features.
# Learn the vocabulary ONLY from training descriptions.
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    token_pattern=r'(?u)\b\w+\b'
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print('Categories:', text_class_names)
print('Training descriptions:', len(X_train_text))
print('Test descriptions:', len(X_test_text))
print('Training feature shape:', X_train_tfidf.shape)
print('Test feature shape:', X_test_tfidf.shape)
print('Example cleaned text:', X_train_text.iloc[0])
print('Example tokens:', tfidf_vectorizer.build_tokenizer()(X_train_text.iloc[0]))


# --- Cell 14 ---

# Install the library for local document embeddings.
%pip install -q sentence-transformers

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import os
import warnings

# Suppress Hugging Face unauthenticated request warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_DIR = Path('/content/drive/MyDrive/EcoSort')
# CORRECTED PATH: Point directly to the JSON file
policy_file = PROJECT_DIR / 'waste_policy_documents.json'

if not policy_file.is_file():
    raise FileNotFoundError(
        f'Policy file not found: {policy_file}. '
        'Check that Google Drive is connected and the filename is correct.'
    )

with policy_file.open('r', encoding='utf-8') as file:
    policy_docs = json.load(file)

print('Policy documents loaded successfully.')


# Use policy_docs loaded from policy.json in section 1.2.
# Support a list, a wrapped list, or documents keyed by name.
if isinstance(policy_docs, list):
    documents = policy_docs
elif isinstance(policy_docs, dict):
    documents = None
    for key in ['documents', 'policies', 'data']:
        if isinstance(policy_docs.get(key), list):
            documents = policy_docs[key]
            break

    if documents is None:
        if policy_docs and all(
            isinstance(value, dict) for value in policy_docs.values()
        ):
            documents = [
                {'document_key': key, 'content': value}
                for key, value in policy_docs.items()
            ]
        else:
            documents = [policy_docs]
else:
    raise ValueError('Expected policy_docs to be a list or dictionary.')

# Convert nested policy fields into readable text.
# Keep headings, categories, and instructions together.
def policy_to_text(value):
    if isinstance(value, dict):
        return '\n'.join(
            f'{str(key).replace("_", " ")}: {policy_to_text(item)}'
            for key, item in value.items()
            if item is not None
        )
    if isinstance(value, list):
        return '; '.join(policy_to_text(item) for item in value)
    return str(value)

# First run downloads the pretrained model.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Split long policies into overlapping chunks so their ends
# are not silently lost when creating embeddings.
rag_chunks = []
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40

for document_index, document in enumerate(documents):
    text = re.sub(r'\s+', ' ', policy_to_text(document)).strip()
    if not text:
        continue

    tokens = embedding_model.tokenizer.encode(
        text, add_special_tokens=False
    )

    for start in range(0, len(tokens), CHUNK_TOKENS - OVERLAP_TOKENS):
        chunk_text = embedding_model.tokenizer.decode(
            tokens[start:start + CHUNK_TOKENS],
            skip_special_tokens=True
        )
        rag_chunks.append({
            'document_index': document_index,
            'text': chunk_text,
            'source_document': document
        })
        if start + CHUNK_TOKENS >= len(tokens):
            break

if not rag_chunks:
    raise ValueError('No policy text found. Check policy_docs.')

policy_embeddings = embedding_model.encode(
    [chunk['text'] for chunk in rag_chunks],
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Retrieve the policy passages most similar to a question.
def retrieve_policy_documents(query, top_k=3):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Enter a non-empty question.')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer.')

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Normalized dot products give cosine similarity scores.
    scores = policy_embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:min(top_k, len(rag_chunks))]

    return [
        {**rag_chunks[int(i)], 'similarity': float(scores[i])}
        for i in indices
    ]

print('Policy documents:', len(documents))
print('Policy chunks:', len(rag_chunks))
print('Embedding shape:', policy_embeddings.shape)

# Check retrieval with an example question.
results = retrieve_policy_documents('How should I dispose of a glass bottle?')
for number, result in enumerate(results, start=1):
    print(f'\nResult {number} — similarity: {result["similarity"]:.3f}')
    print(result['text'])


# --- Cell 17 ---

import tensorflow as tf
from tensorflow.keras import layers, models

# Data augmentation to increase dataset diversity and prevent overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# MobileNetV2 expects pixel values scaled to [-1, 1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

print("Preprocessing pipeline defined successfully.")


# --- Cell 19 ---

# Define the input shape matching the dataset image sizes
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Load the base MobileNetV2 model pre-trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model so its weights aren't updated during initial training
base_model.trainable = False

# Build the complete model architecture
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False) # Keep BatchNormalization layers in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x) # Regularization

# Output layer with softmax for 9 categories
outputs = layers.Dense(num_classes, activation='softmax')(x)

cnn_model = tf.keras.Model(inputs, outputs)

# Compile the model
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()


# --- Cell 21 ---

import matplotlib.pyplot as plt

# 1. Train the model
epochs = 10
print("Starting initial training...")
history = cnn_model.fit(
    train_ds,
    epochs=epochs,
    validation_data=validation_ds
)

# 2. Evaluate on the test dataset
print("\nEvaluating on test set...")
loss, accuracy = cnn_model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")

# 3. Plot training vs validation metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()


# --- Cell 22 ---

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Gather true labels and model predictions from the test set
y_true = []
y_pred_probs = []

print("Generating predictions on test set...")
for images, labels in test_dataset:
    # Convert one-hot encoded labels back to integers
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    # Get model predictions for this batch
    y_pred_probs.extend(cnn_model.predict(images, verbose=0))

# Convert probabilities to actual class predictions
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Generate and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Waste Category')
plt.ylabel('True Waste Category')
plt.title('CNN Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Analyze error patterns using a classification report
print("\n--- Error Pattern Analysis (Classification Report) ---")
print(classification_report(y_true, y_pred, target_names=class_names))


# --- Cell 24 ---

# 1. Unfreeze the base model
base_model.trainable = True

# 2. Freeze all layers except for the last 50
# This prevents destroying the fundamental features (like edges/colors) learned previously
for layer in base_model.layers[:-50]:
    layer.trainable = False

# 3. Recompile the model with a very low learning rate
# A low learning rate is crucial for fine-tuning so we don't drastically alter the weights
cnn_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

print("Starting fine-tuning...")
fine_tune_epochs = 5
total_epochs = epochs + fine_tune_epochs

# 4. Resume training from the last epoch of the initial training phase
history_fine = cnn_model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_ds
)

# 5. Evaluate the fine-tuned model on the test set
print("\nEvaluating fine-tuned model on test set...")
ft_loss, ft_accuracy = cnn_model.evaluate(test_dataset)
print(f"Fine-tuned Test Accuracy: {ft_accuracy:.4f}")


# --- Cell 27 ---

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import TextVectorization

# 1. Extract texts and labels from the dataframe loaded in Part 1
texts = df['description'].values
labels = df['category'].values

# 2. Encode the string labels into integers, then to one-hot encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_text_classes = len(label_encoder.classes_)

y_one_hot = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_text_classes)

# 3. Split the data (80% train, 10% validation, 10% test)
X_train_text, X_temp_text, y_train_text, y_temp_text = train_test_split(
    texts, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
)
X_val_text, X_test_text, y_val_text, y_test_text = train_test_split(
    X_temp_text, y_temp_text, test_size=0.5, random_state=42, stratify=y_temp_text
)

# 4. Create and adapt a TextVectorization layer
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 20 # Captures the short descriptions

text_vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

# Adapt the vectorizer to learn the vocabulary strictly from the training data
text_vectorizer.adapt(X_train_text)

print("Text preprocessing completed.")
print(f"Training samples: {len(X_train_text)}")
print(f"Validation samples: {len(X_val_text)}")
print(f"Test samples: {len(X_test_text)}")
print(f"Vocabulary size: {len(text_vectorizer.get_vocabulary())}")


# --- Cell 29 ---

from tensorflow.keras import layers, models

# Build the text classification model (RNN/LSTM)
text_model = models.Sequential([
    # Add an explicit input layer for strings so the model builds immediately
    tf.keras.Input(shape=(1,), dtype=tf.string),

    # The TextVectorization layer we adapted in 3.1
    text_vectorizer,

    # Embedding layer to convert integer sequences to dense vectors
    layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=64, mask_zero=True),

    # Bidirectional LSTM to understand text context from both directions
    layers.Bidirectional(layers.LSTM(64)),

    # Dense classifier head with dropout for regularization
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    # Output layer for our categories
    layers.Dense(num_text_classes, activation='softmax')
])

# Compile the model
text_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

text_model.summary()


# --- Cell 31 ---

# 1. Train the text classification model
epochs = 10
batch_size = 32

print("Training text classification model...")
text_history = text_model.fit(
    X_train_text, y_train_text,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_text, y_val_text)
)


# --- Cell 32 ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Calculate accuracy on test set
print("\nEvaluating text model on test set...")
text_loss, text_accuracy = text_model.evaluate(X_test_text, y_test_text)
print(f"Text Model Test Accuracy: {text_accuracy:.4f}")

# 2. Plot training vs validation metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(text_history.history['accuracy'], label='Training Accuracy')
plt.plot(text_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Text Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(text_history.history['loss'], label='Training Loss')
plt.plot(text_history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Text Model Loss')
plt.show()

# 3. Generate confusion matrix and analyze error patterns
print("\nGenerating predictions for error analysis...")
y_pred_text_probs = text_model.predict(X_test_text)
y_pred_text = np.argmax(y_pred_text_probs, axis=1)
y_true_text = np.argmax(y_test_text, axis=1)

cm_text = confusion_matrix(y_true_text, y_pred_text)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_text, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Category')
plt.ylabel('True Category')
plt.title('Text Classification Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Text Model Error Pattern Analysis ---")
print(classification_report(y_true_text, y_pred_text, target_names=label_encoder.classes_))


# --- Cell 34 ---

# Install the transformers library if you haven't already
%pip install -q transformers accelerate torch

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Create the retrieval mechanism
def retrieve_policy_documents(query, top_k=3):
    """Finds the most relevant policy chunks for a given query."""
    # Create an embedding for the user's query
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Calculate cosine similarity (dot product of normalized embeddings)
    scores = policy_embeddings @ query_embedding

    # Get the indices of the highest scoring chunks
    indices = np.argsort(scores)[::-1][:min(top_k, len(rag_chunks))]

    # Return the chunks and their similarity scores
    return [
        {**rag_chunks[int(i)], 'similarity': float(scores[i])}
        for i in indices
    ]

# 2. Select and load a pre-trained generative language model
print("Loading generative language model (FLAN-T5)... this may take a minute.")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Automatically use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on: {device}")
print("RAG components successfully initialized!")

# Helper function for generation (use this instead of the old pipeline)
def generate_answer(prompt, max_new_tokens=150):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the retrieval part quickly
test_query = "What should I do with a glass bottle?"
top_docs = retrieve_policy_documents(test_query, top_k=1)
print(f"\nTest Retrieval for '{test_query}':")
print(top_docs[0]['text'])


# --- Cell 37 ---

# !pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# 1. Use the policies data you already loaded in section 1.2
documents = policies

# 2. Helper function to convert the JSON policy fields into readable text strings
def policy_to_text(value):
    if isinstance(value, dict):
        return '\n'.join(
            f'{str(key).replace("_", " ")}: {policy_to_text(item)}'
            for key, item in value.items()
            if item is not None
        )
    if isinstance(value, list):
        return '; '.join(policy_to_text(item) for item in value)
    return str(value)

# 3. Load a pre-trained embedding model
print("Loading embedding model (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Split long policies into overlapping chunks so context isn't lost at the edges
rag_chunks = []
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40

print("Chunking policy documents...")
for document_index, document in enumerate(documents):
    text = re.sub(r'\s+', ' ', policy_to_text(document)).strip()
    if not text:
        continue

    # Tokenize the text
    tokens = embedding_model.tokenizer.encode(
        text, add_special_tokens=False
    )

    # Create overlapping chunks
    for start in range(0, len(tokens), CHUNK_TOKENS - OVERLAP_TOKENS):
        chunk_text = embedding_model.tokenizer.decode(
            tokens[start:start + CHUNK_TOKENS],
            skip_special_tokens=True
        )
        rag_chunks.append({
            'document_index': document_index,
            'text': chunk_text,
            'source_document': document
        })
        if start + CHUNK_TOKENS >= len(tokens):
            break

# 5. Generate numerical embeddings for all the chunks
print(f"Creating embeddings for {len(rag_chunks)} chunks...")
policy_embeddings = embedding_model.encode(
    [chunk['text'] for chunk in rag_chunks],
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print('Document preprocessing complete!')
print('Embedding matrix shape:', policy_embeddings.shape)


# --- Cell 39 ---

# Install if needed
%pip install -q transformers accelerate torch

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Retrieval function
def retrieve_policy_documents(query, top_k=3):
    """Finds the most relevant policy chunks for a given query."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = policy_embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:min(top_k, len(rag_chunks))]

    return [
        {**rag_chunks[int(i)], 'similarity': float(scores[i])}
        for i in indices
    ]

# 2. Load FLAN-T5
print("Loading generative language model (FLAN-T5)... this may take a minute.")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on: {device}")
print("RAG components successfully initialized!")

# 3. Generation function (this is what replaces the old pipeline)
def generate_answer(prompt, max_new_tokens=150):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test retrieval
test_query = "What should I do with a glass bottle?"
top_docs = retrieve_policy_documents(test_query, top_k=1)
print(f"\nTest Retrieval for '{test_query}':")
print(top_docs[0]['text'])


# --- Cell 41 ---

def generate_answer_tuned(prompt, max_new_tokens=180, num_beams=4, temperature=0.7):
    """slightly tuned generation function"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate the response using the provided parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=True, # Required when using temperature
        early_stopping=True # Recommended when using num_beams
    )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Cell 42 ---

# List of diverse waste categories to test
test_categories = [
    "Plastic Water Bottles",
    "Electronic Waste (E-Waste)",
    "Hazardous Household Chemicals",
    "Cardboard and Paper Packaging"
]

print("--- EVALUATION OF GENERATED RECYCLING INSTRUCTIONS ---\n")

for category in test_categories:
    prompt = f"Provide clear, step-by-step recycling instructions for {category}."

    # Generate instructions using the tuned function
    output = generate_answer_tuned(prompt, max_new_tokens=150)

    # Basic relevance & structural evaluation checks
    has_steps = any(indicator in output.lower() for indicator in ["1.", "step", "first", "next"])
    has_category_kw = any(word.lower() in output.lower() for word in category.split())

    print(f"Category: {category}")
    print(f"Generated Instructions:\n{output}\n")
    print(f"  • Relevance Check: {'Passed' if has_category_kw else 'Needs Review'}")
    print(f"  • Structure Check: {'Passed' if has_steps else 'Needs Review'}")
    print("=" * 60)


# --- Cell 44 ---

def generate_recycling_instructions(waste_category):
    """
    Generates detailed recycling instructions for a given waste category.

    Args:
        waste_category (str): Waste category

    Returns:
        str: Detailed recycling instructions
        list: Relevant policy documents
    """
    # 1. Retrieve relevant policy documents based on the waste category
    query = f"How to recycle and dispose of {waste_category}"

    # NOTE: If your retriever variable is named something else (like vectorstore or db),
    # adjust the line below to match (e.g., vectorstore.similarity_search(query))
    relevant_docs = retriever.get_relevant_documents(query)

    # 2. Combine the retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 3. Construct the prompt for the language model
    prompt = f"""Based on the following policy guidelines, provide clear, step-by-step recycling instructions for the waste category: {waste_category}.

Policy Context:
{context}

Category: {waste_category}
Instructions:"""

    # 4. Generate the response using your tuned generation function
    instructions = generate_answer_tuned(prompt, max_new_tokens=200)

    return instructions, relevant_docs


# --- Cell 47 ---

class WasteManagementPipeline:
    """
    Architecture class integrating the three core models:
    1. Vision/NLP Classifier (determines the waste category)
    2. Document Retriever (finds relevant policies)
    3. LLM Generator (creates human-readable instructions)
    """

    def __init__(self):
        # Define allowed inputs to handle the input/output flow securely
        self.supported_types = ["image", "text"]

    def _classification_interface(self, input_data, input_type):
        """
        Interface 1: Handles the input flow to the classification models.
        Takes raw image or text and outputs a standardized string category.
        """
        if input_type not in self.supported_types:
            raise ValueError(f"Invalid input type. Expected one of {self.supported_types}")

        # (The actual calls to your image or text models will go here in 5.2)
        # e.g., if input_type == 'image': return predict_image(input_data)
        pass

    def _rag_interface(self, waste_category):
        """
        Interface 2: Handles the flow from the classifier to the RAG generator.
        Takes a category string and returns recycling instructions and sources.
        """
        # (This will call your generate_recycling_instructions function from 4.4)
        pass

    def execute_flow(self, input_data, input_type="image"):
        """
        Handles the end-to-end input/output data flow between all components.
        """
        print(f"Processing {input_type} input...")

        # Step 1: Run Classification
        category = self._classification_interface(input_data, input_type)

        # Step 2: Run RAG Generation based on the detected category
        instructions, docs = self._rag_interface(category)

        # Step 3: Format final structured output
        return {
            "input_type": input_type,
            "detected_category": category,
            "recycling_instructions": instructions,
            "policy_sources": docs
        }


# --- Cell 49 ---

# Section 5.2: Pipeline Implementation and Testing

class IntegratedWastePipeline(WasteManagementPipeline):
    """
    Concrete implementation of the Waste Management Pipeline.
    Connects the blueprint to our actual working models.
    """

    def _classification_interface(self, input_data, input_type):
        """
        Routes the input to the correct classification model (Vision or NLP).
        """
        # Security check from the parent class
        if input_type not in self.supported_types:
            raise ValueError(f"Invalid input type. Expected one of {self.supported_types}")

        # 1. Route Image Inputs
        if input_type == "image":
            # TODO: Replace 'predict_image' with the exact name of your image
            # classification function from the earlier vision section (e.g., Section 1 or 2)
            category = predict_image(input_data)

        # 2. Route Text Inputs
        elif input_type == "text":
            # TODO: Replace 'predict_text' with the exact name of your text
            # classification function from the earlier NLP section
            category = predict_text(input_data)

        return category

    def _rag_interface(self, waste_category):
        """
        Connects the output of the classifier to the RAG Generator.
        """
        # Calls the function we just built in Section 4.4
        instructions, docs = generate_recycling_instructions(waste_category)
        return instructions, docs


# --- 1. Instantiate the Pipeline ---
pipeline = IntegratedWastePipeline()

# --- 2. Test the Pipeline (Text Input) ---
print("--- TESTING TEXT PIPELINE ---")
test_description = "A broken smartphone with a cracked screen and swollen battery."

try:
    # Run the full flow
    result_text = pipeline.execute_flow(input_data=test_description, input_type="text")

    print(f"\nDetected Category: {result_text['detected_category']}")
    print(f"Recycling Instructions:\n{result_text['recycling_instructions']}")

except Exception as e:
    print(f"Pipeline Error: {e}\n(Make sure your text prediction function is named correctly in the class above!)")

print("\n" + "="*60 + "\n")

# --- 3. Test the Pipeline (Image Input) ---
# Uncomment and modify the lines below to test an image if you have a sample image path
"""
print("--- TESTING IMAGE PIPELINE ---")
test_image_path = "path/to/your/sample_waste_image.jpg"

try:
    result_image = pipeline.execute_flow(input_data=test_image_path, input_type="image")
    print(f"\nDetected Category: {result_image['detected_category']}")
    print(f"Recycling Instructions:\n{result_image['recycling_instructions']}")
except Exception as e:
    print(f"Pipeline Error: {e}")
"""


# --- Cell 51 ---

# 5.3 Evaluate the Integrated System

# 1. Define Test Data (Text descriptions & Image paths with ground truth labels)
test_text_dataset = [
    {"input": "Empty clear plastic beverage bottle with cap on.", "expected": "Plastic"},
    {"input": "Old broken smartphone with a cracked screen and lithium battery.", "expected": "E-Waste"},
    {"input": "Flattened cardboard shipping box and brown paper packaging.", "expected": "Cardboard"},
    {"input": "Rinsed glass pickle jar with metal lid separate.", "expected": "Glass"}
]

# Optional: Add paths to sample images from your test dataset
test_image_dataset = [
    # {"input": "dataset/test/plastic_1.jpg", "expected": "Plastic"},
    # {"input": "dataset/test/cardboard_1.jpg", "expected": "Cardboard"}
]

print("       INTEGRATED SYSTEM EVALUATION & PERFORMANCE       ")

# --- A. Evaluate Text Descriptions ---
print("-Evaluating Text Inputs")
text_correct = 0

for idx, item in enumerate(test_text_dataset, 1):
    text_input = item["input"]
    expected_cat = item["expected"]

    try:
        # Pass input through the full pipeline
        output = pipeline.execute_flow(input_data=text_input, input_type="text")
        detected_cat = output["detected_category"]
        instructions = output["recycling_instructions"]

        # Check if classification matches expectation
        is_correct = expected_cat.lower() in detected_cat.lower() or detected_cat.lower() in expected_cat.lower()
        if is_correct:
            text_correct += 1

        print(f"\n[Test Case {idx}]")
        print(f"  • Input Description : '{text_input}'")
        print(f"  • Expected Category : {expected_cat}")
        print(f"  • Detected Category : {detected_cat}")
        print(f"  • Category Match    : {'✓ PASS' if is_correct else '✗ MISMATCH'}")
        print(f"  • Generated RAG Preview:\n    \"{instructions[:120].strip()}...\"")

    except Exception as e:
        print(f"[Test Case {idx}] Failed with error: {e}")

# Text Performance Metric
text_acc = (text_correct / len(test_text_dataset)) * 100
print(f"Text Input Accuracy: {text_correct}/{len(test_text_dataset)} ({text_acc:.1f}%)")

# B. Evaluate Image Inputs
if test_image_dataset:
    print("--- Evaluating Image Inputs ---")
    img_correct = 0

    for idx, item in enumerate(test_image_dataset, 1):
        img_path = item["input"]
        expected_cat = item["expected"]

        try:
            output = pipeline.execute_flow(input_data=img_path, input_type="image")
            detected_cat = output["detected_category"]

            is_correct = expected_cat.lower() in detected_cat.lower() or detected_cat.lower() in expected_cat.lower()
            if is_correct:
                img_correct += 1

            print(f"Image {idx} | Expected: {expected_cat} | Detected: {detected_cat} | Match: {'✓' if is_correct else '✗'}")
        except Exception as e:
            print(f"Image {idx} Failed: {e}")

    img_acc = (img_correct / len(test_image_dataset)) * 100
    print(f"\nImage Input Accuracy: {img_correct}/{len(test_image_dataset)} ({img_acc:.1f}%)")
