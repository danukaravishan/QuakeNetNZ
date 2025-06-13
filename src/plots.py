import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_accuracy_vs_metadata(true_tensor, predicted_classes, metadata, acc_vs_metadata_img, p_data, bins=10):
    
    metadata = np.array(metadata)
    event_info = metadata[:, 0]  # Assuming first column is event ID or name
    magnitudes = metadata[:, 1]
    epicentral_distances = metadata[:, 2]
    # Convert to proper numeric type
    magnitudes = np.array(magnitudes, dtype=float)
    epicentral_distances = np.array(epicentral_distances, dtype=float)

    true_tensor = np.array(true_tensor.to("cpu").numpy())
    predicted_classes = np.array(predicted_classes.to("cpu").numpy())
    magnitudes = np.array(magnitudes)
    epicentral_distances = np.array(epicentral_distances)

    # Metadata already contains only positive samples.
    mask = true_tensor == 1
    # Changed: predictions now only for true positives.
    preds = predicted_classes[mask]

    mags = magnitudes
    dists = epicentral_distances

    # New: ignore entries with epicentral_distance > 100
    valid = (dists <= 100)
    mags = mags[valid]
    dists = dists[valid]
    preds = preds[valid]
    event_info = event_info[valid]
    p_data = p_data[valid]

    # Remove incorrect_sample_list from evaluation
    incorrect_sample_list = ["2013p543832_WDFS", "2013p613809_NNZ", "2015p718332_TLZ", "2016p858314_TCW", "2016p935725_CMWZ"]

    # Mask to ignore incorrect_sample_list
    ignore_mask = np.isin(event_info, incorrect_sample_list, invert=True)
    mags = mags[ignore_mask]
    dists = dists[ignore_mask]
    preds = preds[ignore_mask]
    event_info = event_info[ignore_mask]
    p_data = p_data[ignore_mask]

    # Print information about incorrect samples (after removing incorrect_sample_list)
    incorrect_mask = preds != 1
    # if np.any(incorrect_mask):
    #     print("Incorrect samples (event_info):")
    #     for info in event_info[incorrect_mask]:
    #         print(info)
    # else:
    #     print("No incorrect samples found.")


    # Bin by magnitude
    mag_bins = np.linspace(mags.min(), mags.max(), bins+1)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
    mag_acc = []
    for i in range(bins):
        idx = (mags >= mag_bins[i]) & (mags < mag_bins[i+1])
        if idx.sum() > 0:
            acc = (preds[idx] == 1).sum() / idx.sum()
            mag_acc.append(acc)
        else:
            mag_acc.append(np.nan)

    # Bin by epicentral distance
    dist_bins = np.linspace(dists.min(), dists.max(), bins+1)
    dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
    dist_acc = []
    for i in range(bins):
        idx = (dists >= dist_bins[i]) & (dists < dist_bins[i+1])
        if idx.sum() > 0:
            acc = (preds[idx] == 1).sum() / idx.sum()
            dist_acc.append(acc)
        else:
            dist_acc.append(np.nan)

    # Updated Plot: combine all plots into one figure with 3 subplots
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.plot(mag_centers, mag_acc, marker='o')
    plt.xlabel('Earthquake Magnitude', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Accuracy as a Function of Magnitude', fontsize=14)
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot(dist_centers, dist_acc, marker='o')
    plt.xlabel('Epicentral Distance (km)', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Accuracy as a Function of Epicentral Distance', fontsize=14)
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.scatter(mags, dists, alpha=0.6)
    plt.xlabel('Earthquake Magnitude', fontsize=12)
    plt.ylabel('Epicentral Distance (km)', fontsize=12)
    plt.title('Magnitude vs. Epicentral Distance', fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(acc_vs_metadata_img)
    
    # Additional plot: Bivariate Accuracy Heatmap showing relationship of magnitude and epicentral distance with accuracy
    heatmap_img = acc_vs_metadata_img.replace("acc_metadata", "bivariate_heatmap")
    # Compute 2D bins using the previously filtered arrays
    mag_bins_2 = np.linspace(mags.min(), mags.max(), bins+1)
    dist_bins_2 = np.linspace(dists.min(), dists.max(), bins+1)
    heatmap = np.empty((bins, bins))
    for i in range(bins):
        for j in range(bins):
            bin_mask = (mags >= mag_bins_2[i]) & (mags < mag_bins_2[i+1]) & (dists >= dist_bins_2[j]) & (dists < dist_bins_2[j+1])
            if bin_mask.sum() > 0:
                heatmap[i,j] = (preds[bin_mask] == 1).sum() / bin_mask.sum()
            else:
                heatmap[i,j] = np.nan

    plt.figure(figsize=(6,5))
    plt.imshow(heatmap, extent=[dist_bins_2[0], dist_bins_2[-1], mag_bins_2[0], mag_bins_2[-1]],
               origin='lower', aspect='auto', interpolation='nearest')
    plt.xlabel('Epicentral Distance (km)', fontsize=12)
    plt.ylabel('Earthquake Magnitude', fontsize=12)
    plt.title('Bivariate Heatmap of Classification Accuracy', fontsize=14)
    plt.colorbar(label='Classification Accuracy')
    plt.tight_layout()
    plt.savefig(heatmap_img)


## Plot the ROC curves
def plot_roc_curve(true_tensor, predicted_probs, roc_img_path, model_name=None):
    """
    Plots the ROC curve for a binary classifier and also plots precision, recall, and F1 score vs threshold in the same image.
    Args:
        true_tensor: torch.Tensor or np.ndarray of true binary labels (0 or 1)
        predicted_probs: torch.Tensor or np.ndarray of predicted probabilities (floats in [0,1])
        roc_img_path: Path to save the ROC curve image
        model_name: Optional string for plot title
    """
    # Convert to numpy arrays if needed
    if hasattr(true_tensor, 'cpu'):
        y_true = true_tensor.cpu().numpy().flatten()
    else:
        y_true = np.array(true_tensor).flatten()
    if hasattr(predicted_probs, 'cpu'):
        y_score = predicted_probs.cpu().numpy().flatten()
    else:
        y_score = np.array(predicted_probs).flatten()

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute precision, recall, F1 vs threshold
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    thresholds_pr = np.append(thresholds_pr, 1.0)  # To match array lengths for plotting
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Plot both ROC and PR/F1 in the same figure (side by side)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate', fontsize=12)
    axs[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    title = 'Receiver Operating Characteristic (ROC)'
    if model_name:
        title += f' - {model_name}'
    axs[0].set_title(title, fontsize=14)
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Precision, Recall, F1 vs Threshold
    axs[1].plot(thresholds_pr, precision, label='Precision', color='blue')
    axs[1].plot(thresholds_pr, recall, label='Recall', color='green')
    axs[1].plot(thresholds_pr, f1, label='F1 Score', color='red')
    axs[1].set_xlabel('Threshold', fontsize=12)
    axs[1].set_ylabel('Score', fontsize=12)
    prf1_title = 'Precision, Recall, and F1 Score vs. Threshold'
    if model_name:
        prf1_title += f' - {model_name}'
    axs[1].set_title(prf1_title, fontsize=14)
    axs[1].legend(loc='best')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(roc_img_path)
    plt.close()

    return fpr, tpr, thresholds_roc, roc_auc, thresholds_pr, precision, recall, f1


