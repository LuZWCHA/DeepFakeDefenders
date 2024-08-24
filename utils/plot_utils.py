import torch
from builtins import isinstance
from copy import deepcopy
import io
from pathlib import Path
from typing import List
import warnings
import matplotlib.pyplot as plt
import numpy as np


from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}


def calculate_cpm(x, axis=1):
    """Calculate counts-per-million on data where the rows are genes.

    Parameters
    ----------
    x : array_like
    axis : int
        Axis accross which to compute CPM. 0 for genes being in rows and 1 for
        genes in columns.

    """
    normalization = np.sum(x, axis=axis)
    # On sparse matrices, the sum will be 2d. We want a 1d array
    normalization = np.squeeze(np.asarray(normalization))
    # Straight up division is not an option since this will form a full dense
    # matrix if `x` is sparse. Divison can be expressed as the dot product with
    # a reciprocal diagonal matrix
    normalization = sp.diags(1 / normalization, offsets=0)
    if axis == 0:
        cpm_counts = np.dot(x, normalization)
    elif axis == 1:
        cpm_counts = np.dot(normalization, x)
    return cpm_counts * 1e6


def log_normalize(data):
    """Perform log transform log(x + 1).

    Parameters
    ----------
    data : array_like

    """
    if sp.issparse(data):
        data = data.copy()
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def select_genes(
    data,
    threshold=0,
    atleast=10,
    yoffset=0.02,
    xoffset=5,
    decay=1,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    legend_names=[],
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                legend_names,
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


def evaluate_embedding(
    embedding, labels, projection_embedding=None, projection_labels=None, sample=None
):
    """Evaluate the embedding using Moran's I index.

    Parameters
    ----------
    embedding: np.ndarray
        The data embedding.
    labels: np.ndarray
        A 1d numpy array containing the labels of each point.
    projection_embedding: Optional[np.ndarray]
        If this is given, the score will relate to how well the projection fits
        the embedding.
    projection_labels: Optional[np.ndarray]
        A 1d numpy array containing the labels of each projection point.
    sample: Optional[int]
        If this is specified, the score will be computed on a sample of points.

    Returns
    -------
    float
        Moran's I index.

    """
    has_projection = projection_embedding is not None
    if projection_embedding is None:
        projection_embedding = embedding
        if projection_labels is not None:
            raise ValueError(
                "If `projection_embedding` is None then `projection_labels make no sense`"
            )
        projection_labels = labels

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The shape of the embedding and labels don't match")

    if projection_embedding.shape[0] != projection_labels.shape[0]:
        raise ValueError("The shape of the reference embedding and labels don't match")

    if sample is not None:
        n_samples = embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        embedding = embedding[sample_indices]
        labels = labels[sample_indices]

        n_samples = projection_embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        projection_embedding = projection_embedding[sample_indices]
        projection_labels = projection_labels[sample_indices]

    weights = projection_labels[:, None] == labels
    if not has_projection:
        np.fill_diagonal(weights, 0)

    mu = np.asarray(embedding.mean(axis=0)).ravel()

    numerator = np.sum(weights * ((projection_embedding - mu) @ (embedding - mu).T))
    denominator = np.sum((projection_embedding - mu) ** 2)

    return projection_embedding.shape[0] / np.sum(weights) * numerator / denominator

def plot_many(image_list, titles=None, rows=None, cols=None, show=True, default_type="image",  hist_params={}, cmps=[], **kwargs):
    assert isinstance(image_list, np.ndarray) or isinstance(image_list, List)
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        
    image_num = len(image_list)
    layout_flag = False
    if rows is not None:
        cols = image_num // rows
    elif cols is not None:
        rows = image_num // cols
    elif rows is None and cols is None:
    
        if image_num == 1:
            cols = rows = 1
        else:
            layout_flag = True
            raise RuntimeWarning("Rows/columns should be assigned.")

        # TODO
    plt.figure(dpi=100)
    # plt.subplots_adjust(left = 0.1, top = 0.9, right = 0.9, bottom = 0.1, hspace = 0.5, wspace = 0.5)
    # 改变文字大小参数-fontsize
    cache = []
    
    for i, image in enumerate(image_list):
        assert isinstance(image, np.ndarray) and 3 >= image.ndim >= 1
        
        type_ = default_type
        
        if image.ndim == 2 and default_type == "image":
            # expand one channel
            image = image[..., None]
        
        if image.ndim == 3:
            if default_type == "image":
                channels = image.shape[-1]
                assert channels in [1, 2, 3, 4]
                type_ = "image"
            elif default_type == "points":
                assert image.shape[1] in [2, 3]
                type_ = f"{image.shape[1]}d_points"
        elif image.ndim == 1 and type_=="image":
            type_ = "hist"
        else: 
            type_ = default_type
            
        if not layout_flag:
            if type_ == "3d_points":
                plt.subplot(rows, cols, i + 1 ,  projection='3d', **kwargs)
            else:
                plt.subplot(rows, cols, i + 1 ,  **kwargs)

        if type_ == "image" and image.ndim > 1:
            plt.imshow(image, cmap=cmps[i] if i < len(cmps) else None)
            plt.axis("off")
        if type_ == "hist":
            plt.hist(image, **hist_params)
        if type_ == "bar":
            plt.bar(image, **kwargs)
        if type_ == "2d_points":
            plt.scatter(image[:, 0], image[:, 1], **kwargs)
        elif type_ == "3d_points":
            plt.scatter(image[:, 0], image[:, 1],  image[:, 2], **kwargs)
        elif type_ == "plot":
            plt.plot(image, **kwargs)
        
        
        if titles is not None and len(titles) > i:
            title = titles[i]
            plt.title(title)
    plt.tight_layout()
    
    if show:    
        plt.show()
    return plt

def plt2arr(fig, DPI=250):
    """
    need to draw if figure is not drawn yet
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def plot_roc(y_true, y_probas, title='ROC Curves', class_names=None,
                   plot_micro=True, plot_macro=True, classes_to_plot=None,
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            class_name = classes[i] if class_names is None else class_names[i]
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(class_name, roc_auc))

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='deeppink', linestyle=':', linewidth=4)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


def plot_roc_curve(y_true, y_probas, class_names):
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_probas, torch.Tensor):
        y_probas = y_probas.cpu().numpy()
    if y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    # skplt.metrics.plot_roc_curve(y_true, y_probas, ax=ax)
    plot_roc(y_true, y_probas, ax=ax, class_names=class_names)
    res = plt2arr(fig)
    plt.close(fig)
    return res


def plot_precision_recall(y_true, y_probas, class_names=None,
                          title='Precision-Recall Curve',
                          plot_micro=True,
                          classes_to_plot=None, ax=None,
                          figsize=None, cmap='nipy_spectral',
                          title_fontsize="large",
                          text_fontsize="medium"):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the precision-recall
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            average_precision = average_precision_score(
                binarized_y_true[:, i],
                probas[:, i])
            class_name = classes[i]
            if class_names is not None:
                class_name = class_names[i]
            
            precision, recall, _ = precision_recall_curve(
                y_true, probas[:, i], pos_label=classes[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall, precision, lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(class_name,
                                                     average_precision),
                    color=color)

    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            binarized_y_true.ravel(), probas.ravel())
        average_precision = average_precision_score(binarized_y_true,
                                                    probas,
                                                    average='micro')
        ax.plot(recall, precision,
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax


def plot_pr_curve(y_true, y_probas, class_names):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_probas, torch.Tensor):
        y_probas = y_probas.cpu().numpy()
    if y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    # skplt.metrics.plot_roc_curve(y_true, y_probas, ax=ax)
    plot_precision_recall(y_true, y_probas, ax=ax, class_names=class_names)
    res = plt2arr(fig)
    plt.close(fig)
    return res

def plot_f1_threshold(y_true, y_probas, class_names=None,
                          title='F1-Threshold Curve',
                          plot_micro=True,
                          classes_to_plot=None, ax=None,
                          figsize=None, cmap='nipy_spectral',
                          title_fontsize="large",
                          text_fontsize="medium"):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the precision-recall
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            class_name = classes[i]
            if class_names is not None:
                class_name = class_names[i]
            
            precision, recall, t = precision_recall_curve(
                y_true, probas[:, i], pos_label=classes[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            
            f1_score = 2 * precision * recall / (precision + recall + 1e-5)
            max_f1 = f1_score.max()
            max_f1_ts = t[f1_score.argmax()]
            
            f1_score = [0] + f1_score.tolist()
            t = [0] + t.tolist() + [1]
            ax.plot(t, f1_score, lw=2,
                    label='F1 curve of class {0} '
                    '(f1 = {1:0.3f} at {2:0.3f})'.format(class_name, max_f1, max_f1_ts),
                    color=color)

    precision, recall, t = precision_recall_curve(
        binarized_y_true.ravel(), probas.ravel())

    f1_score = 2 * precision * recall / (precision + recall + 1e-5)
    max_f1 = f1_score.max()
    max_f1_ts = t[f1_score.argmax()]
    f1_score = [0] + f1_score.tolist()
    t = [0] + t.tolist() + [1]
    ax.plot(t, f1_score, 
            label='F1 curve of all classes '
                    '(f1 = {0:0.3f} at {1:0.3f})'.format(max_f1, max_f1_ts),
                    color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax

def plot_f1_curve(y_true, y_probas, class_names):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_probas, torch.Tensor):
        y_probas = y_probas.cpu().numpy()
    if y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    # skplt.metrics.plot_roc_curve(y_true, y_probas, ax=ax)
    plot_f1_threshold(y_true, y_probas, ax=ax, class_names=class_names)
    res = plt2arr(fig)
    plt.close(fig)
    return res

def plot_confusion_matrix(matrix, nc, normalize=True, save_dir=None, names=(), show=True, add_backgound=False):
    import seaborn as sn

    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    # if not normalize:
    #     array = array.astype(np.uint16)
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = nc, len(names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    ticklabels = (list(names) + (['background'] if add_backgound else [])) if labels else 'auto'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30,
                   annot_kws={
                       'size': 8},
                   cmap='Blues',
                   fmt='.2f' if normalize else 'g',
                   square=True,
                   vmin=0.0,
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
    if show:
        plt.show()
    res = plt2arr(fig)
    plt.close(fig)
    return res
    
    
import os
import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=False,
                               font_path=None):
    """Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                      (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    try:
        font = ImageFont.truetype('SemHei.ttf', 64)
    except IOError:
        if font_path is None:
            font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=64)
        

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=(),
                                       use_normalized_coordinates=False,
                                       box_mode="yxyx", font_path=None):
    """Draws bounding boxes on image (numpy array).
    Args:
        image: a numpy array object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
            The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.
    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(image)
    for box, display_str_list in zip(boxes, display_str_list_list):
        if box_mode == "yxyx":
            ymin, xmin, ymax, xmax  = box
        elif box_mode =="xyxy":
            xmin, ymin, xmax, ymax  = box
        elif box_mode =="xxyy":
            xmin, xmax, ymin, ymax  = box
        else:
            raise RuntimeError(f"Not support box mode: {box_mode}")
        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax,
		color, thickness, display_str_list, use_normalized_coordinates=use_normalized_coordinates, font_path=font_path)
    np.copyto(image, np.array(image_pil))
    

def draw_distribution(dataset_name, groups, bins=40, _range=None, show=True):
    fig = plt.figure()
    min_ = 1e10
    max_ = -1e10

    for g in groups:
        # sorted = np.percentile(np.sort(deepcopy(g[1])), 99)
        max_ = max(g[1].max(), max_)
        min_ = min(g[1].min(), min_)
    
    if _range is None:
        _range = (min_, max_)
    
    for g in groups:
        name = g[0]
        # name = f"{name}"
        his, range_ = np.histogram(g[1], bins=bins, range=_range)
        # f = np.percentile(f, 95, keepdims=True)
        plt.stairs(his, range_, fill=True, alpha=1/len(groups), label=name)
        
    plt.legend(loc='upper right')
    plt.title(f"Histogram of {dataset_name}")

    fig.savefig(f"/nasdata/private/zwlu/segmentation/segment-anything/.data/analysis/histograms/{dataset_name}.png")
    if show:
        plt.show()
    # plt.savefig(f"/nasdata/private/zwlu/segmentation/segment-anything/.data/analysis/histograms/{dataset_name}.png")
    
    plt.close()
    fig.clf()
    return fig
