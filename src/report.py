# This file handles the reporting utilities and other helper functions

import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import csv

# def plot_loss(train_losses, val_loss, val_acc, file_name):
#     plt.plot(train_losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
    
#     image_filename = f"{file_name}.jpg"
#     plt.savefig(image_filename)  # Save as PNG, you can change to other formats like .pdf, .jpeg
#     print(f"Loss curve saved as {image_filename}")


def plot_loss(train_losses, val_losses, val_accs, file_name):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # Create 3 subplots

    # Training Loss
    axes[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Validation Loss
    axes[1].plot(epochs, val_losses, label='Validation Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True)

    # Validation Accuracy
    axes[2].plot(epochs, val_accs, label='Validation Accuracy', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()  # Adjust spacing between subplots

    # Save and show the figure
    image_filename = f"{file_name}.jpg"
    plt.savefig(image_filename)
    #plt.show()

    print(f"Loss and accuracy curves saved as {image_filename}")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Function to append model data to csv file
def addToCSV(cfg, nncfg, model, accuracy, precision, recall, f1, parameters):

    file_exists = os.path.isfile(cfg.CSV_FILE)

    with open(cfg.CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['Model ID', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Model Parameters', 'Learning Rate', 'Batch size', 'Epoch', 'Optimizer'])

        val_acc_index = nncfg.val_acc.index(max(nncfg.val_acc))
        writer.writerow([model.model_id, f"{accuracy:.4f}%", f"{precision:.4f}%", f"{recall:.4f}%", f"{f1:.4f}%", parameters, nncfg.learning_rate, nncfg.batch_size, nncfg.epoch_count, nncfg.optimizer, nncfg.conv1_size, nncfg.conv2_size, nncfg.fc1_size, nncfg.kernal_size1, nncfg.kernal_size2, f"{max(nncfg.val_acc):.4f}%", val_acc_index])
    print(f"Model details for {model.model_id} appended to {cfg.CSV_FILE} CSV.")


# Function to dump all model details into a seperate pdf file
def test_report(cfg, nncfg, model, true_tensor, predicted_classes):

    TP = ((predicted_classes == 1) & (true_tensor == 1)).sum().item()  # True Positives
    TN = ((predicted_classes == 0) & (true_tensor == 0)).sum().item()  # True Negatives
    FP = ((predicted_classes == 1) & (true_tensor == 0)).sum().item()  # False Positives
    FN = ((predicted_classes == 0) & (true_tensor == 1)).sum().item()  # False Negatives

    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = 100*((TP + TN) / (TP + TN + FP + FN))
    precision = 100*(TP / (TP + FP)) if (TP + FP) != 0 else 0
    recall = 100*(TP / (TP + FN)) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    parameters = count_parameters(model)

    # Print the results
    print(f'Accuracy: {accuracy:.4f}%')
    print(f'Precision: {precision:.4f}%')
    print(f'Recall: {recall:.4f}%')
    print(f'F1 Score: {f1:.4f}%')
    print(f'Parameters: {parameters}')

    pdf = FPDF()
    pdf.add_page()

    # Add the model ID as the title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=f"Model: {model.model_id}", ln=True, align='C')

    # Add the loss curve title and image
    pdf.cell(200, 10, txt="Training Loss Curve", ln=True, align='C')

    loss_curve_image = cfg.MODEL_PATH + model.model_id + ".jpg"
    
    if os.path.exists(loss_curve_image):
        pdf.image(loss_curve_image, x=10, y=30, w=180)  # Adjust the position (x, y) and size (w)
        pdf.ln(300)  
    else:
        print(f"Loss curve not found for {model.model_id}")

    pdf.set_font("Arial", size=12)

    # Add the calculated metrics to the PDF
    pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Precision: {precision:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Recall: {recall:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"F1 Score: {f1:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Parameters: {parameters}", ln=True, align='L')
    # pdf.cell(200, 10, txt=f"Max val acc: {max(nncfg.val_acc)}", ln=True, align='L')
    # pdf.cell(200, 10, txt=f"Max val acc index: {nncfg.val_acc.index(max(nncfg.val_acc))}", ln=True, align='L')

    pdf_filename = cfg.MODEL_PATH + model.model_id + ".pdf"
    pdf.output(pdf_filename)
    print(f"Write output to {pdf_filename}")    
    addToCSV(cfg, nncfg, model, accuracy, precision, recall, f1, parameters)


def find_latest_file(directory, prefix, extension):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        raise ValueError(f"No files with prefix '{prefix}' and extension '{extension}' found in {directory}")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return latest_file


def getLatestModelName(cfg):
    # Fine the latest file name
    directory = cfg.MODEL_PATH  # Change to your model directory
    model_prefix = (cfg.MODEL_TYPE.name).lower()
    model_extension = ".pt"
    latest_model_file = find_latest_file(directory, model_prefix, model_extension)
    return latest_model_file