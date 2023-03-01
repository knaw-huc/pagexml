import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def plot_clusters(df, data_cols, x_col='doc_num', y_col='num_words', n_clusters: int = 2):
    data = df[data_cols].values.tolist()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    offset = df[y_col].max() / 50
    pc = plt.scatter(df[x_col], df[y_col], c=kmeans.labels_)
    [plt.text(x=row[x_col], y=row[y_col] + offset, s=row[x_col]) for k, row in df.iterrows()]

    pc.figure.set_size_inches(16, 8)
    plt.show()


def cluster_plot_elbow(df, data_cols, max_clusters: int = 10):
    data = df[data_cols].values.tolist()
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def plot_line_widths_boundary_points(line_widths, line_width_break_points) -> None:
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(data=line_widths, ax=ax, clip=(0, max(line_widths)))
    fig.set_size_inches(16, 6)
    plt.vlines(line_width_break_points, 0.0, ax.get_ylim()[1], colors='green', linestyles='dashed')
    return None


sns.set_theme()
sns.set_theme(style="whitegrid", palette="pastel")
