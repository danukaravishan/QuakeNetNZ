# This file handles the reporting utilities and other helper functions

import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import csv
from utils import *
from dataprep import pre_proc_data, normalize_data, pre_process_real_time_2s, apply_wavelet_denoise
import h5py
import numpy as np
import torch
from scipy.signal import resample
from obspy import UTCDateTime
from obspy import Stream
from obspy import Trace
import shutil  # For deleting the temporary directory


def generate_output_for_events(event_ids, hdf5_file_path, output_dir, model, nncfg, sampling_rate=50, window_size=100, stride=10):
    import os
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    with h5py.File(hdf5_file_path, 'r') as hdf:
        for idx, event_id in enumerate(event_ids, 1):  # Add counter using enumerate
            dataset = hdf.get(event_id)
            if dataset is None:
                print(f"Dataset {event_id} not found.")
                continue

            data = np.array(dataset)  # Shape: (num_channels, num_samples)
            if data.shape[0] != 3:
                print(f"Skipping {event_id}: Expected 3 channels, found {data.shape[0]}")
                continue

            event_magnitude = dataset.attrs.get("magnitude", "N/A")
            epicentral_distance = dataset.attrs.get("epicentral_distance", "N/A")

            # Adjust P-wave and S-wave indices for sampling rate
            p_arrival_index = int(dataset.attrs["p_arrival_sample"])
            s_arrival_index = int(dataset.attrs["s_arrival_sample"])
            original_rate = dataset.attrs["sampling_rate"]

            # Resample if necessary
            data_pp = pre_proc_data(data, dataset.attrs["sampling_rate"])

            if dataset.attrs["sampling_rate"] != sampling_rate:
                original_rate = dataset.attrs["sampling_rate"]
                num_samples = int(data.shape[1] * sampling_rate / original_rate)
                data = np.array([resample(channel, num_samples) for channel in data])
                data_pp = np.array([resample(channel, num_samples) for channel in data_pp])
                downsample_factor = original_rate / sampling_rate
                p_arrival_index = int(p_arrival_index / downsample_factor)
                s_arrival_index = int(s_arrival_index / downsample_factor)

            # Convert indices to time
            p_wave_time = p_arrival_index / sampling_rate
            s_wave_time = s_arrival_index / sampling_rate

            # Get total samples and time duration
            total_samples = data.shape[1]
            total_seconds = total_samples / sampling_rate
            timestamps = np.arange(0, total_seconds, 1 / sampling_rate)

            # Simulate real-time inference
            real_time_window = int(2 * sampling_rate)  # Last 2 seconds of data
            time_preds = []
            raw_output = []
            classified_output = []

            for current_time in range(0, total_samples, stride):
                # Extract the last 2 seconds of data (or less if at the start)
                start = max(0, current_time - real_time_window)
                segment = data[:, start:current_time]  # (3, variable_window_size)

                if segment.shape[1] < window_size:
                    # Skip if the segment is smaller than the required window size
                    continue

                segment = normalize_data(apply_wavelet_denoise(segment, nncfg.wavelet_name, nncfg.wavelet_level))
                input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # (1, 3, window_size)
                output = model(input_tensor)

                cl_out = output > 0.9
                classified_output.append(cl_out.item())
                raw_output.append(output.item())
                time_preds.append(current_time / sampling_rate)

            # Convert to NumPy array
            classified_output = np.array(classified_output)

            # Plot the waveform
            fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)  # Adjusted for 4 subplots

            colors = ["b", "g", "r"]
            labels = ["Z Component", "N Component", "E Component"]

            # Plot raw waveform
            for i in range(3):
                axes[0].plot(timestamps, data[i], color=colors[i], label=labels[i])
            axes[0].set_title(f"Waveform ({idx}/{len(event_ids)}) - Event {event_id} | Magnitude: {event_magnitude} | Distance: {epicentral_distance} km")
            axes[0].set_ylabel("Amplitude")
            axes[0].legend()

            # Add vertical lines for P-wave and S-wave pick times
            axes[0].axvline(x=p_wave_time, color='orange', linestyle='--', label='P-wave Pick')
            axes[0].axvline(x=s_wave_time, color='purple', linestyle='--', label='S-wave Pick')
            axes[0].legend()

            # Plot pre-processed waveform
            for i in range(3):
                axes[1].plot(timestamps, data_pp[i], color=colors[i], label=labels[i])
            axes[1].set_title("Pre-processed Waveform")
            axes[1].set_ylabel("Amplitude")
            axes[1].legend()

            # Plot raw model output
            axes[2].vlines(time_preds, ymin=0, ymax=raw_output, color="k", linewidth=1)
            axes[2].plot(time_preds, raw_output, 'o', color="r", markersize=3, label="Model Predictions - Earthquake")
            axes[2].axhline(y=0.9, color='blue', linestyle='--', label="Cutoff Threshold (0.9)")
            axes[2].set_title("Raw Model Output")
            axes[2].set_xlabel("Time (seconds)")
            axes[2].legend()

            # Plot classified output
            axes[3].vlines(time_preds, ymin=0, ymax=classified_output, color="k", linewidth=1)
            axes[3].plot(time_preds, classified_output, 'o', color="k", markersize=3, label="Model Predictions - Noise")
            axes[3].set_title("Classified Output")
            axes[3].set_xlabel("Time (seconds)")
            axes[3].set_ylabel("Prediction (scaled)")
            axes[3].legend()

            plt.tight_layout()
            output_file = os.path.join(output_dir, f"{event_id}.png")
            plt.savefig(output_file)
            plt.close()
            print(f"Saved output for event {event_id} to {output_file}")


def plot_loss(train_losses, val_losses, val_accs, file_name):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Create 2 subplots now

    # Combined Training and Validation Loss
    axes[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Validation Accuracy
    axes[1].plot(epochs, val_accs, label='Validation Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

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
        writer.writerow([model.model_id, f"{accuracy:.4f}%", f"{precision:.4f}%", f"{recall:.4f}%", f"{f1:.4f}%", parameters, nncfg.learning_rate, nncfg.batch_size, nncfg.epoch_count, nncfg.optimizer, nncfg.conv1_size, nncfg.conv2_size, nncfg.conv3_size, nncfg.fc1_size, nncfg.fc2_size, nncfg.kernal_size1, nncfg.kernal_size2, nncfg.kernal_size3, f"{max(nncfg.val_acc):.4f}%", val_acc_index, nncfg.wavelet_name, nncfg.wavelet_level])
    print(f"Model details for {model.model_id} appended to {cfg.CSV_FILE} CSV.")


# Function to dump all model details into a seperate pdf file
def test_report(cfg, nncfg, model, true_tensor, predicted_classes):
    
    TP = ((predicted_classes == 1) & (true_tensor == 1)).sum().item()  # True Positives
    TN = ((predicted_classes == 0) & (true_tensor == 0)).sum().item()  # True Negatives
    FP = ((predicted_classes == 1) & (true_tensor == 0)).sum().item()  # False Positives
    FN = ((predicted_classes == 0) & (true_tensor == 1)).sum().item()  # False Negatives

    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = 100 * ((TP + TN) / (TP + TN + FP + FN))
    precision = 100 * (TP / (TP + FP)) if (TP + FP) != 0 else 0
    recall = 100 * (TP / (TP + FN)) if (TP + FN) != 0 else 0
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
        os.remove(loss_curve_image)  # Remove the image file after adding to the PDF
    else:
        print(f"Loss curve not found for {model.model_id}")

    pdf.set_font("Arial", size=12)

    # Add the calculated metrics to the PDF
    pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Precision: {precision:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Recall: {recall:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"F1 Score: {f1:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Parameters: {parameters}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Max val acc: {max(nncfg.val_acc):.4f}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Max val acc index: {nncfg.val_acc.index(max(nncfg.val_acc))}", ln=True, align='L')

    param_txt1 = f"LR={nncfg.learning_rate}, Batch={nncfg.batch_size}, Epoch={nncfg.epoch_count}, c1={nncfg.conv1_size}, c2={nncfg.conv2_size}, c3={nncfg.conv3_size}, f1={nncfg.fc1_size}, f2={nncfg.fc2_size}, k1=={nncfg.kernal_size1}, k2={nncfg.kernal_size2}, k3={nncfg.kernal_size3}"
    pdf.cell(200, 10, txt=param_txt1, ln=True, align='L')

    param_txt2 = f"L2_decay={nncfg.l2_decay}, droput1={nncfg.dropout1}, droput2={nncfg.dropout2}, wavelet_name={nncfg.wavelet_name}, wavelet_level={nncfg.wavelet_level}"
    pdf.cell(200, 10, txt=param_txt2, ln=True, align='L')

    param_txt2 = nncfg.model_note
    pdf.cell(200, 10, txt=param_txt2, ln=True, align='L')

    # Create a temporary directory for waveform graphs
    temp_dir = os.path.join(cfg.MODEL_PATH, f"{model.model_id}_waveforms")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate waveform graphs
    # Read event IDs from the first column of the CSV file
    event_ids = []
    with open('data/sample_event_list.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            event_ids.append(row[0])  # Assuming the first column contains event IDs

    generate_output_for_events(event_ids, cfg.ORIGINAL_DB_FILE, temp_dir, model, nncfg)

    # Append waveform graphs to the PDF in the correct order
    for event_id in event_ids:
        image_file = f"{event_id}.png"
        image_path = os.path.join(temp_dir, image_file)
        if os.path.exists(image_path):
            pdf.add_page()
            pdf.image(image_path, x=10, y=10, w=180)

    # Save the PDF
    pdf_filename = cfg.MODEL_PATH + model.model_id + ".pdf"
    pdf.output(pdf_filename)
    print(f"Write output to {pdf_filename}")

    # Delete the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Deleted temporary directory: {temp_dir}")

    # Append model details to CSV
    addToCSV(cfg, nncfg, model, accuracy, precision, recall, f1, parameters)