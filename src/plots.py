import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import h5py
import requests


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





## Plot the srtation coordinates on a map of New Zealand
## Usage (plot_stations_on_nz_map(hdf5_file_path))
## hdf5_file_path should have the trace_name with event_id_station_id format, where station names are extracted

def plot_stations_on_nz_map(hdf5_path):

    def fetch_station_coordinates(hdf5_path):
        station_codes = set()
        with h5py.File(hdf5_path, 'r') as f:
            group = f['positive_samples_p']
            for key in group.keys():
                if '_' in key:
                    parts = key.split('_')
                    if len(parts) > 1:
                        station_codes.add(parts[1])
        
        base_url = "https://service.geonet.org.nz/fdsnws/station/1/query"
        params = {
            "network": "NZ",
            "level": "station",
            "format": "text",
            "station": ",".join(station_codes)
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        station_coords = {}
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                station_code = parts[1]
                latitude = float(parts[2])
                longitude = float(parts[3])
                station_coords[station_code] = (longitude, latitude)
        return station_coords


    station_coords = fetch_station_coordinates(hdf5_path)

    data = {
        "station": list(station_coords.keys()),
        "geometry": [Point(lon, lat) for lon, lat in station_coords.values()]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")  # WGS84 lat/lon
    gdf = gdf.to_crs(epsg=3857)  # Web Mercator for basemap overlay

    # New: Set NZ bounds in lat/lon and convert to Web Mercator
    min_lat = -47.5617
    max_lat = -34.2192
    min_lon = 165.8271
    max_lon = 179.6050
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 14), dpi=100)
    # Reduce marker size for stations
    gdf.plot(ax=ax, color='red', marker='^', markersize=30, alpha=0.8, label='Stations')

    # Remove station names for cleaner map

    # Set axis limits to NZ bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Add reliable basemap with error handling
    try:
        #ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=8) # This is good.
    except Exception as e:
        print(f"Error adding basemap: {e}")

    # Add scale bar
    scale_bar_length_km = 40  # Length of scale bar in kilometers
    scale_bar_x = min_x + (max_x - min_x) * 0.05
    scale_bar_y = min_y + (max_y - min_y) * 0.05
    ax.annotate(f'{scale_bar_length_km} km', xy=(scale_bar_x, scale_bar_y), fontsize=10, color='black')

    # Add title and legend
    ax.set_title("GeoNet Strong Motion Sensor Distribution", fontsize=18)
    ax.legend(loc='upper right', fontsize=12)

    # Add grid with latitude and longitude labels
    lat_ticks = np.linspace(min_lat, max_lat, 7)
    lon_ticks = np.linspace(min_lon, max_lon, 7)
    xticks, _ = transformer.transform(lon_ticks, [min_lat]*len(lon_ticks))
    _, yticks = transformer.transform([min_lon]*len(lat_ticks), lat_ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{lon:.2f}" for lon in lon_ticks], fontsize=10)
    ax.set_yticklabels([f"{lat:.2f}" for lat in lat_ticks], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Draw a thick border for the NZ bounding box
    from matplotlib.patches import Rectangle
    border = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                       linewidth=4, edgecolor='black', facecolor='none', zorder=10)
    ax.add_patch(border)

    # Ensure axis labels are visible
    ax.set_axis_on()

    plt.tight_layout()
    plt.show()
