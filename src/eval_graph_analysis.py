import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 


def accuracy_vs_parameters():
    data_path = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/analysis and graphs/kernal.xlsx"
    df = pd.read_excel(data_path, sheet_name="Sheet2")  
    df = df.sort_values("Parameter Count")

    #df = df[df["filter_size"] == 4]

    print(df.head())  # View first few rows
    print(df.columns)  # List of column names
    print(df.info())

    plt.figure(figsize=(10, 5))

    lowess = sm.nonparametric.lowess(df["Accuracy"], df["Parameter Count"], frac=0.2)  # Adjust `frac` for smoothing

    # Plot original data
    sns.scatterplot(data=df, x="Parameter Count", y="Accuracy", alpha=0.5)  # Scatter for reference

    # Plot smoothed line
    plt.plot(lowess[:, 0], lowess[:, 1], color="red", label="Smoothed Curve")


    plt.title("Model Accuracy over number of trainable Parameters", fontsize=15, fontweight='bold')
    plt.xlabel("Parameter count", fontsize=15, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=15, fontweight='bold')
    plt.legend(title="Accuracy vs Parameter Count", fontsize=14, title_fontsize=14)
    plt.savefig("/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/analysis and graphs//parameters_vs_accuracy.png")
    plt.show()


def multi_x():

    data_path = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/analysis and graphs/kernal.xlsx"
    df = pd.read_excel(data_path, sheet_name="Sheet2")  

    print(df.head())  # View first few rows
    print(df.columns)  # List of column names
    print(df.info())

    plt.figure(figsize=(10, 5))

    batch_sizes = df["Batch size"].unique()

    # Plot for each batch size
    for batch in batch_sizes:
        subset = df[df["Batch size"] == batch]  # Filter data
        sns.lineplot(data=subset, x="Learning Rate", y="Accuracy", label=f"Batch Size {batch}")

    # Customize the graph
    plt.title("Accuracy vs learning rate for Different Batch Sizes, for filter size =  4", fontsize=15, fontweight='bold')
    plt.xlabel("Learning Rate", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.legend(title="Batch Size", fontsize=14, title_fontsize=14)
    plt.grid(True)
    plt.show()


def literature_window_size():

    # Load the data
    data_path = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/Earthquake detection/analysis.xlsx"
    df = pd.read_excel(data_path, sheet_name="Graphs")  

    # Replace "ND" with NaN
    df.replace("ND", pd.NA, inplace=True)

    # Drop rows with NaN in "Input window" or "Title"
    df = df.dropna(subset=["Input window", "Title"])

    # Convert "Input window" to numeric
    df["Input window"] = pd.to_numeric(df["Input window"], errors="coerce")

    # Drop any new NaNs caused by conversion
    df = df.dropna(subset=["Input window"])

    # Plot the bar graph
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Title", y="Input window")

    # Customize the graph
    plt.title("Input Window vs Title", fontsize=15, fontweight='bold')
    plt.xlabel("Title", fontsize=14, fontweight='bold')
    plt.ylabel("Input Window", fontsize=14, fontweight='bold')
    plt.ylim(0, 20)
    plt.xticks(rotation=45)  # Rotate x-axis labels if titles are long
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.figure(figsize=(14, 8))  # Increase figure size
    sns.barplot(data=df, x="Title", y="Input window")
    plt.xticks(rotation=90, ha="right")  # Rotate labels fully & align right
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()


    # Show the plot
    plt.show()

def roc_curve():
    data_path = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/analysis and graphs/kernal.xlsx"
    df = pd.read_excel(data_path, sheet_name="ROC")  
    #df = df[df["filter_size"] == 4]

    print(df.head())  # View first few rows
    print(df.columns)  # List of column names
    print(df.info())

    plt.figure(figsize=(10, 5))

    lowess = sm.nonparametric.lowess(df["Precision"], df["Recall"], frac=0.2)  # Adjust `frac` for smo

    # Plot original data
    sns.scatterplot(data=df, x="Recall", y="Precision", alpha=0.5)  # Scatter for reference

    # Plot smoothed line
    plt.plot(lowess[:, 0], lowess[:, 1], color="red", label="Smoothed Curve")


    plt.title("Precision vs Recall variation with detection threshold", fontsize=15, fontweight='bold')
    plt.xlabel("Precision", fontsize=15, fontweight='bold')
    plt.ylabel("Recall", fontsize=15, fontweight='bold')
    plt.legend(title="Recall vs Precision", fontsize=14, title_fontsize=14)
    plt.savefig("/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/analysis and graphs//precision_vs_recall.png")
    plt.show()


def input_window2():
    data_path = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Writting/paper1/Earthquake detection/analysis.xlsx"
    df = pd.read_excel(data_path, sheet_name="Graphs")  

    # Replace "ND" with NaN
    df.replace("ND", pd.NA, inplace=True)

    # Drop rows with NaN in "Input window" or "Title"
    df = df.dropna(subset=["Input window", "Title"])

    # Convert "Input window" to numeric
    df["Input window"] = pd.to_numeric(df["Input window"], errors="coerce")

    # Drop any new NaNs caused by conversion
    df = df.dropna(subset=["Input window"])

    # Count occurrences of each unique time value
    count_df = df["Input window"].value_counts().reset_index()
    count_df.columns = ["Time (seconds)", "Number of Names"]

    # Sort by time values
    count_df = count_df.sort_values(by="Time (seconds)")
    #y_ticks = sorted(count_df["Number of Names"].unique())
    y_ticks  = [x for x in range (1,20)]
    # Plot the new bar graph
    plt.figure(figsize=(12, 6))
    sns.barplot(data=count_df, x="Time (seconds)", y="Number of Names", color="b")

    # Customize the graph
    plt.title("Input  Window vs Number of Resrach", fontsize=15, fontweight='bold')
    plt.xlabel("Input Window Time (seconds)", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Research", fontsize=14, fontweight='bold')
    plt.yticks(y_ticks)
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()  # Adjust layout
    plt.show()

#accuracy_vs_parameters()
roc_curve()
