import matplotlib.pyplot as plt
import numpy as np
import shutil


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, 21)  # Start from 1 to 20 for better readability

    plt.figure(figsize=(20, 8))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xticks(epochs_range)  # Ensuring x-axis has integer labels

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xticks(epochs_range)  # Ensuring x-axis has integer labels

    plt.show()

def plot_class_distribution(dataset, class_names):
    # Initialize a dictionary to count the occurrences of each class
    class_counts = {name: 0 for name in class_names}
    
    # Iterate over the dataset to count each class
    for images, labels in dataset.unbatch().as_numpy_iterator():
        # labels are one-hot encoded, so convert them back to index
        label_index = np.argmax(labels)
        class_name = class_names[label_index]
        class_counts[class_name] += 1
    
    # Prepare data for plotting
    class_labels = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)
    percentages = [f"{(count/total)*100:.2f}%" for count in counts]

    # Creating the bar chart

    plt.figure(figsize=(30, 8))
    fig, ax = plt.subplots()
    bars = ax.bar(class_labels, counts)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of Classes in the Dataset')
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")

    # Add the percentage annotations
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, percentage, va='bottom')  # va: vertical alignment
    
    plt.tight_layout()
    plt.show()

def evaluate_model(test_ds, model):
    # Evaluate the model on the validation dataset
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

def remove_output_dir(data_dir_train):
    # Iterate through each subdirectory in Train
    for sub_dir in data_dir_train.iterdir():
        if sub_dir.is_dir():  # Confirm it's a directory
            output_dir = sub_dir / "output"
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"Removed 'output' directory in {sub_dir}")