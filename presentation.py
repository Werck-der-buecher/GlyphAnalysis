import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors as mcolors

import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import base64

from typing import Literal
from functools import partial
from pystackreg import StackReg
from umap import UMAP
from PIL import Image
from io import BytesIO

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Button
from IPython.display import display
import pandas as pd


def display_images(imga_stack: np.ndarray,
                   imgb_stack: np.ndarray,
                   imga_z: int,
                   imgb_z: int,
                   cmap: matplotlib.colors.Colormap
                   ) -> None:
    plt.subplots(1, 2, figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(imga_stack[imga_z], cmap=cmap)
    plt.title("Image Stack (A)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(imgb_stack[imgb_z], cmap=cmap)
    plt.title("Image Stack (B)")
    plt.axis("off")

    plt.show()


def display_registration_results(reference_method: Literal['first', 'mean'],
                                 img_stack_unreg: np.ndarray,
                                 img_stack_reg: np.ndarray,
                                 img_z: int,
                                 cmap_stack: matplotlib.colors.Colormap,
                                 cmap_diff: matplotlib.colors.Colormap
                                 ) -> None:
    if reference_method == 'mean':
        img_reference = np.mean(img_stack_unreg, axis=0)
    elif reference_method == 'first':
        img_reference = img_stack_unreg[0]
    else:
        raise ValueError("Chosen reference method is invalid.")

    plt.subplots(1, 5, figsize=(14, 8))
    plt.title("Hello World")

    plt.subplot(1, 5, 1)
    plt.imshow(img_reference, cmap=cmap_stack)
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(img_stack_unreg[img_z], cmap=cmap_stack)
    plt.title("Unregistered Stack")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(img_stack_reg[img_z], cmap=cmap_stack)
    plt.title("Registered Stack")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(img_stack_unreg[img_z] - img_stack_reg[img_z], cmap=cmap_diff)
    plt.title("Diff. Image (Unreg/Reg)")
    # plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(img_reference - img_stack_reg[img_z], cmap=cmap_diff)
    plt.title("Diff. Image (Ref/Reg)")
    plt.axis("off")

    plt.show()


def display_images_with_alpha(img_z, alpha, imga, imgb):
    img = (1.0 - alpha) * imga[img_z] + alpha * imgb[img_z]
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()


def add_headers(
        fig,
        *,
        row_headers=None,
        col_headers=None,
        row_pad=1,
        col_pad=5,
        rotate_row_headers=True,
        **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


reduction_ops = {'mean': partial(np.mean, axis=0),
                 'std': partial(np.std, axis=0),
                 'median': partial(np.median, axis=0),
                 'max': partial(np.amax, axis=0),
                 'min': partial(np.amin, axis=0)}


def cluster_overview(
        img_stack: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_labels_sorted: np.ndarray,
        tgt_num_rows: int = 10,
        cmap: matplotlib.colors.Colormap = plt.cm.inferno
) -> None:
    # Visualize similarity groups for Stack A
    nrows = min(len(cluster_labels_sorted), tgt_num_rows)
    ncols = len(reduction_ops)

    subplots_kwargs = dict(sharex=True, sharey=True, figsize=(14, 20))
    fig, axes = plt.subplots(nrows, ncols, **subplots_kwargs)
    for lbl_idx, lbl in enumerate(cluster_labels_sorted[:10], ):
        probmaps_labels = img_stack[cluster_labels == lbl]
        for ops_idx, (op_name, op_func) in enumerate(reduction_ops.items()):
            img_reduced = op_func(probmaps_labels)
            if img_reduced.ndim == 3:
                img_reduced = np.moveaxis(img_reduced, 0, -1)
            axes[lbl_idx, ops_idx].imshow(img_reduced, cmap=cmap)

    row_headers = [f"C-{i}" for i in range(10)]
    col_headers = list(reduction_ops.keys())
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.show()


def umap_embd(x_flat: np.ndarray, n_neighbors: int = 15, n_components: int = 2, min_dist: float = 0.1,
              random_state: int = 42) -> np.ndarray:
    reducer = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                   random_state=random_state)
    reducer.fit(x_flat)
    embedding = reducer.transform(x_flat)

    return embedding


def viz_embedding(embedding, labels=None, cmap='Spectral', s=0.1, alpha=1.0, **kwargs) -> None:
    sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=labels, palette=cmap, s=s, alpha=alpha, **kwargs)


def viz_hungarian_cluster_matching(matched_pairs,
                                   matching_cost,
                                   cluster_centroids_A,
                                   cluster_centroids_B,
                                   cmap: matplotlib.colors.Colormap = plt.cm.inferno
                                   ) -> None:
    fig, ax = plt.subplots(len(matched_pairs),3, figsize=(14,10))

    row_headers = []
    for i, (pair, cost) in enumerate(zip(matched_pairs, matching_cost)):
        row_headers.append(f"A{pair[0]}-B{pair[1]}")
        # a = cluster_centroids_A[list(cluster_centroids_A.keys())[pair[0]]]
        # b = cluster_centroids_B[list(cluster_centroids_B.keys())[pair[1]]]
        a = cluster_centroids_A[pair[0]]
        b = cluster_centroids_B[pair[1]]
        sr = StackReg(StackReg.SCALED_ROTATION)
        tmat = sr.register(a, b)
        b_mapped = sr.transform(b, tmat)

        ax[i, 0].imshow(a)
        ax[i, 1].imshow(b_mapped)
        ax[i, 2].imshow(a-b_mapped, cmap=cmap)

    col_headers = ["(A)", "(B)", "Diff. (A)-(B)"]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.show()


def display_cluster_mappings(mapping_dict, mapping_cost, clusters_A, clusters_B):
    cluster_a_labels = list(mapping_dict.keys())
    cluster_b_labels = [mapping_dict[a] for a in cluster_a_labels]
    cluster_a_sizes = [len(clusters_A[cidx]) for cidx in mapping_dict.keys()]
    cluster_b_sizes = [len(clusters_B[cidx]) for cidx in mapping_dict.values()]

    if mapping_cost != None:
        mappings_df = pd.DataFrame({
            "Stack A Cluster": cluster_a_labels,
            "Stack B Cluster": cluster_b_labels,
            "A/B Matching Cost ↓": mapping_cost,
            "Stack A Size": cluster_a_sizes,
            "Stack B Size": cluster_b_sizes
        })
    else:
        mappings_df = pd.DataFrame({
            "Stack A Cluster": cluster_a_labels,
            "Stack B Cluster": cluster_b_labels,
            "Stack A Size": cluster_a_sizes,
            "Stack B Size": cluster_b_sizes
        })

    mappings_df = mappings_df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])]).hide(axis='index')

    display(mappings_df)


def interactive_cluster_visualization(images_a, images_b, embeddings_a, embeddings_b, labels_a, labels_b,
                                      sample_size=50, max_img_size=(100, 100)):

    def image_to_base64(image_array, max_size=max_img_size):
        if image_array.dtype in [np.float32, np.float64]:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        elif image_array.max() > 255:
            image_array = image_array.clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(image_array)
        pil_img.thumbnail(max_size, Image.LANCZOS)

        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    if len(images_a) > sample_size:
        indices_a = np.random.choice(len(images_a), sample_size, replace=False)
    else:
        indices_a = np.arange(len(images_a))

    if len(images_b) > sample_size:
        indices_b = np.random.choice(len(images_b), sample_size, replace=False)
    else:
        indices_b = np.arange(len(images_b))

    images_a = images_a[indices_a]
    embeddings_a = embeddings_a[indices_a]
    labels_a = [labels_a[i] for i in indices_a]

    images_b = images_b[indices_b]
    embeddings_b = embeddings_b[indices_b]
    labels_b = [labels_b[i] for i in indices_b]

    unique_labels_a = sorted(set(labels_a))
    unique_labels_b = sorted(set(labels_b))
    colors_a = {label: mcolors.to_hex(plt.get_cmap('Oranges')(i / (len(unique_labels_a) - 1))) for i, label in
                enumerate(unique_labels_a)}
    colors_b = {label: mcolors.to_hex(plt.get_cmap('Blues')(i / (len(unique_labels_b) - 1))) for i, label in
                enumerate(unique_labels_b)}

    x_coords_a, y_coords_a = embeddings_a[:, 0], embeddings_a[:, 1]
    x_coords_b, y_coords_b = embeddings_b[:, 0], embeddings_b[:, 1]

    image_base64_a = [image_to_base64(img) for img in images_a]
    image_base64_b = [image_to_base64(img) for img in images_b]

    image_output = widgets.HTML(value="<h3>Hover over a point to see the image</h3>",
                                layout=widgets.Layout(width="200px", height="200px"))

    fig = go.FigureWidget()

    trace_a = go.Scatter(
        x=x_coords_a,
        y=y_coords_a,
        mode='markers+text',
        text=labels_a,
        marker=dict(size=15,
                    color=[colors_a[label] for label in labels_a],
                    line=dict(width=1, color='gray')),
        hoverinfo="text",
        name="Stack A",
        showlegend=False
    )
    fig.add_trace(trace_a)

    trace_b = go.Scatter(
        x=x_coords_b,
        y=y_coords_b,
        mode='markers+text',
        text=labels_b,
        marker=dict(size=15,
                    color=[colors_b[label] for label in labels_b],
                    line=dict(width=1, color='gray')),
        hoverinfo="text",
        name="Stack B",
        showlegend=False
    )
    fig.add_trace(trace_b)

    for label, color in colors_a.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            legendgroup="Stack A",
            showlegend=True,
            name=f"Stack A - {label}"
        ))

    for label, color in colors_b.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            legendgroup="Stack B",
            showlegend=True,
            name=f"Stack B - {label}"
        ))

    # Customize layout with larger size
    fig.update_layout(
        title="Cluster Visualization for Image Stacks (A) & (B)",
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        width=1200,
        height=800,
        hovermode="closest"
    )

    display_area = HBox([fig, image_output])

    # Button to toggle label display
    toggle_button = Button(description="Toggle Labels")
    labels_visible = True

    def toggle_labels(b):
        nonlocal labels_visible
        labels_visible = not labels_visible
        fig.data[0].text = labels_a if labels_visible else ["" for _ in labels_a]
        fig.data[1].text = labels_b if labels_visible else ["" for _ in labels_b]

    toggle_button.on_click(toggle_labels)

    display(VBox([toggle_button, display_area]))

    def display_image(trace, points, selector):
        if points.point_inds:
            idx = points.point_inds[0]
            if "Stack A" in trace.name:
                img_html = f"<h3>(A) C-{labels_a[idx]}</h3><img src='data:image/png;base64,{image_base64_a[idx]}' width='150' height='150'>"
            elif "Stack B" in trace.name:
                img_html = f"<h3>(B) C-{labels_b[idx]}</h3><img src='data:image/png;base64,{image_base64_b[idx]}' width='150' height='150'>"
            else:
                img_html = "<h3>Hover over a point to see the image</h3>"
            image_output.value = img_html

    fig.data[0].on_hover(display_image)  # Stack A
    fig.data[1].on_hover(display_image)  # Stack B
    fig.show()