import pickle
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tissue.estimators as estimators
import tissue.models as models


class InterpreterBase:
    """
    Class for model interpretation. Inherits Estimator functionality and loads data
    and a model based on information saved with TrainModel.
    """

    def __init__(
            self,
            results_path: str,
            model_id: str,
            cv: str,
            model_class: str,
    ):
        """
        Parameters
        ----------
        results_path : Path to folder where model information was stored with TrainModel.
        model_id : Identifier given to model in TrainModel.
        model_class : Model class that was trained in TrainModel. Could be:
            - GCN
            - GAT
            - MI
            - MLP
            - GSUM
        """
        self.results_path = results_path
        self.model_id = model_id
        self.model_class = model_class.upper()
        self.cv = cv

    def load_model(self):
        fn = f'{self.results_path}{self.model_id}_{self.cv}_model_weights.tf'
        fn_args = f'{self.results_path}{self.model_id}_{self.cv}_model_args.pickle'
        with open(fn_args, 'rb') as f:
            model_args = pickle.load(f)
        # print(self.model_class)
        if 'GCN' in self.model_class:
            model = models.ModelGcn(**model_args)
        elif self.model_class == 'GAT':
            model = models.ModelGat(**model_args)
        elif self.model_class == 'MI':
            model = models.ModelMultiInstance(**model_args)
        elif self.model_class == 'MLP':
            model = models.ModelReg(**model_args)
        elif self.model_class == 'GSUM':
            model = models.ModelGsum(**model_args)
        self.model = model.training_model
        self.model.load_weights(fn)

    def get_data_again(
            self,
            data_path,
            buffered_data_path=None,
            radius=None,
    ):
        """
        Loads data as previously done during model training.
        """
        fn_args = f'{self.results_path}{self.model_id}_{self.cv}_get_data_args.pickle'
        with open(fn_args, 'rb') as f:
            get_data_args = pickle.load(f)
        if radius:
            get_data_args["radius"] = radius
        if buffered_data_path:
            get_data_args["buffered_data_path"] = buffered_data_path
        self.get_data(
            data_path=data_path,
            **get_data_args,
        )

    def get_partition_idx(
            self,
            partition: str
    ):
        """
        Returns indices of samples in selected partition.

        Parameters
        ----------
        partition : "test", "val" or "train

        Returns
        -------
        Observation indices of selected partition.
        """
        if partition.lower() == "train":
            return self.img_keys_train
        elif partition.lower() == "val":
            return self.img_keys_eval
        elif partition.lower() == "test":
            return self.img_keys_test
        elif partition.lower() == "all":
            return self.complete_img_keys
        else:
            raise ValueError("partition %s not recognized" % partition)

    def plot_graph(
            self,
            image_key,
            panel_width=4.,
            panel_height=4.,
            edge_width=1,
            node_size=5,
            cmap='hsv',
            save: Union[str, None] = None,
            suffix: str = "_graphs.png",
            show: bool = True,
            return_axs: bool = False,
    ):
        """
        Plot graph with cell type as color.

        Parameters
        ----------
        image_key : Identifier of graph to plot.
        panel_width :
        panel_height :
        edge_width :
        node_size :
        cmap :
        save : Whether (if not None) and where (path as string given as save) to save plot.
        suffix : Suffix of file name to save to.
        show : Whether to display plot.
        return_axs: Whether to return axis objects.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib.cm as cmx
        import matplotlib.colors as colors

        if not return_axs:
            plt.ioff()

        # Make sure image_key is a two dimensional array of image identifiers
        if isinstance(image_key, str):
            image_key = np.array([[image_key]])
        elif isinstance(image_key, List):
            image_key = np.array(image_key)
        if len(image_key.shape) == 1:
            image_key = np.array([image_key]).T
        image_key_array = image_key

        shape = image_key_array.shape
        fig, ax = plt.subplots(
            nrows=shape[0],
            ncols=shape[1],
            figsize=(panel_width * shape[1], panel_height * shape[0])
        )

        for i, image_key_list in enumerate(image_key_array):
            for j, image_key in enumerate(image_key_list):
                a = self.a[image_key]
                cluster_key = self.data.img_celldata[image_key].uns['metadata']['cluster_col_preprocessed']
                cell_types = self.data.img_celldata[image_key].obs[cluster_key]

                cNorm = colors.Normalize(vmin=0, vmax=len(np.unique(cell_types)))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))

                g = nx.from_numpy_matrix(a.todense())
                dict_nodes = {i: self.data.img_celldata[image_key].obsm['spatial'][i] for i in
                              np.arange(cell_types.shape[0])}

                if shape[0] * shape[1] == 1:
                    ax_tmp = ax
                elif shape[0] == 1:
                    ax_tmp = ax[j]
                elif shape[1] == 1:
                    ax_tmp = ax[i]
                else:
                    ax_tmp = ax[i][j]

                nx.draw_networkx_edges(
                    g,
                    pos=dict_nodes,
                    width=edge_width,
                    ax=ax_tmp,
                )

                for k, ctype in enumerate(np.unique(cell_types)):
                    color = [scalarMap.to_rgba(k)]
                    idx_c = list(np.where(cell_types == ctype)[0])
                    nx.draw_networkx_nodes(
                        g,
                        node_size=node_size,
                        nodelist=idx_c,
                        node_color=color,
                        pos=dict_nodes,
                        label=ctype,
                        ax=ax_tmp,
                    )
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], linestyle='None', color=scalarMap.to_rgba(k), label=ctype, marker='o') for
                   k, ctype in enumerate(np.unique(cell_types))]
        lgd = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='cell type')

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)
        if show:
            plt.show()
        if return_axs:
            return ax
        else:
            plt.close(fig)
            plt.ion()
            return None

    def _get_graph_embeddings(
            self,
            idx,
            layers=['final_dense_0']
    ) -> list:
        """
        Get embedding of graphs from one layer after node aggregation.
        """
        if isinstance(idx, int) or isinstance(idx, np.int32) or isinstance(idx, np.int64) or isinstance(idx, str):
            idx = [idx]
        ds = self._get_dataset(
            keys=idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=1234
        )
        out = []
        for l in layers:
            out.append(self.model.get_layer(l).output)
        model = tf.keras.Model(self.model.input, out)
        acts = {l: [] for l in layers}
        ind_to_layer = {i: layer for i, layer in enumerate(layers)}
        for step, (x_batch, y_batch) in enumerate(ds):
            temp = model(x_batch)
            if len(layers) == 1:
                temp = [temp]
            temp = [xx.numpy().squeeze() for xx in temp]
            for i, t in enumerate(temp):
                acts[ind_to_layer[i]].append(t)
        return acts

    def plot_umap_graphs(
            self,
            layer_name='final_dense_0',
            label='Group',
            partitions=['train', 'test'],
            embedding_method='umap',
            palette="colorblind",
            save: Union[str, None] = None,
            suffix: str = "_umap_graphs.pdf",
            show: bool = True,
            return_axs: bool = False,
            data_key=None,
            hue_order=None,
            return_embeddings=None,
    ):
        """
        Plots a PCA or UMAP based on activations of one model layer as graph representations
        colored by a categorical graph label.

        Parameters
        ----------
        layer_name : Layer name. Activations of that layer are used as graph representation.
        label : Categorical graph label to plot on UMAP.
        partitions : Partitions to highlight in separate subplots.
        embedding_method : Either 'UMAP' or 'PCA'.
        palette
        save : Whether (if not None) and where (path as string given as save) to save plot.
        suffix : Suffix of file name to save to.
        show : Whether to display plot.
        return_axs: Whether to return axis objects.

        Returns
        -------

        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.ioff()
        fig = plt.figure(figsize=(3 * len(partitions), 3))

        idx = []
        part_name = []
        for p in partitions:
            idx_part = list(self.get_partition_idx(p))
            idx.append(idx_part)
            part_name += [p] * len(idx_part)
        idx = np.concatenate(idx)
        part_name = np.array(part_name)

        labels = [
            self.data.img_celldata[image_key].uns['graph_covariates']['label_tensors'][label]
            for image_key in idx
        ]
        label_names = [
            l.split('>')[-1]
            for l in self.data.img_celldata[idx[0]].uns['graph_covariates']['label_names'][label]
        ]
        hue = np.argmax(labels, axis=1)
        hue = [np.round(float(label_names[i]),0) for i in hue]        

        if data_key=='bz':
            palette = {
                            1: sns.color_palette("colorblind")[0], 
                            2: sns.color_palette("colorblind")[1], 
                            3: sns.color_palette("colorblind")[2]
                        }
            hue_order = [1, 2, 3]
        elif data_key == 'mb':
            palette = {
                            1: sns.color_palette("colorblind")[0], 
                            2: sns.color_palette("colorblind")[1], 
                            3: sns.color_palette("colorblind")[2]
                        }
            hue_order=[1, 2, 3]
        else:
            palette = {
                            1: sns.color_palette("colorblind")[0], 
                            2: sns.color_palette("colorblind")[1], 
                        }
            hue_order = [1, 2]

        graph_embeddings = self._get_graph_embeddings(idx=idx, layers=[layer_name])

        embedding_method = embedding_method.upper()
        if embedding_method == 'UMAP':
            import umap
            reducer = umap.UMAP()
        elif embedding_method == 'PCA':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(np.array(graph_embeddings[layer_name]))

        for col, part in enumerate(partitions):
            sizes = {p: 6. if p == part else 1. for p in partitions}
            ax = fig.add_subplot(1, len(partitions), col + 1)
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=hue,
                hue_order=hue_order,
                ax=ax,
                size=part_name,
                sizes=sizes,
                linewidth=.1,
                palette=palette
            )
            ax.set_xlabel('')
            ax.set_title(part)
            plt.tick_params(
                axis='both',
                which='both',
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )
            ax.legend_.remove()
            ax.grid(False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        handles, labels = ax.get_legend_handles_labels()
        handles = list(np.array(handles)[[l in label_names for l in labels]])
        labels = list(np.array(labels)[[[l in label_names for l in labels]]])
        fig.legend(handles=list(handles), labels=list(labels), title=label, loc='center left', bbox_to_anchor=(1, 0.5))

        # Save, show and return figure.
        plt.tight_layout()
        plt.grid(False)
        if save is not None:
            plt.tight_layout()
            plt.savefig(save + suffix)
        if show:
            plt.show()
        if return_axs:
            if return_embeddings:
                return ax, embedding, hue
            return ax
        else:
            plt.close(fig)
            plt.ion()
            if return_embeddings:
                return embedding, hue
            return None

    def _get_node_embeddings(
            self,
            idx,
            layers,
    ) -> list:
        """
        Computes the node embeddings after chosen layers.

        :param idx: Indices to compute embeddings for.
        :param layers: Layers for which to return embeddings.
        :return:
        """
        if isinstance(idx, int) or isinstance(idx, np.int32) or isinstance(idx, np.int64):
            idx = [idx]
        ds = self._get_dataset(
            keys=idx,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=1234
        )
        out = []
        for l in layers:
            out.append(self.model.get_layer(l).output)
        model = tf.keras.Model(self.model.input, out)
        acts = {l: [] for l in layers}
        ind_to_layer = {i: layer for i, layer in enumerate(layers)}
        for step, (x_batch, y_batch) in enumerate(ds):
            h = x_batch[0]
            h = h.numpy().squeeze()
            # Mask node embedding:
            node_idx = np.arange(0, np.max(np.where(np.sum(np.abs(h), axis=1) > 0)[0]) + 1)  # excluded padded cells
            temp = model(x_batch)
            if len(layers) == 1:
                temp = [temp]
            for i, o in enumerate(temp):
                if isinstance(o, List):  # DiffPool output = [A, F]
                    o = [m.numpy().squeeze() for m in o]
                else:
                    o = o.numpy().squeeze()
                    if o.shape[0] == h.shape[0]:
                        o = o[node_idx, :]
                acts[ind_to_layer[i]].append(o)
        return acts

    def _compute_umap_embedding(
            self,
            idx,
            layer_name=None,
            n_neighbors=15,
    ):
        import umap

        if layer_name == 'input' or layer_name != layer_name:
            h = [self.h[image_key] for image_key in idx]
        else:
            acts = self._get_node_embeddings(
                idx=idx,
                layers=[layer_name]
            )
            h = acts[layer_name]
        cells = np.concatenate(h)
        cells = cells[:, cells.std(axis=0) > 0]
        cells = (cells - cells.mean(axis=0)) / cells.std(axis=0)

        reducer = umap.UMAP(n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(cells)

        umap_emb = pd.DataFrame({
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
        })
        return umap_emb

    def plot_umap_nodes(
            self,
            idx,
            layer_names=['input'],
            plot_types=[],
            panel_width=3.,
            panel_height=3.,
            save: Union[str, None] = None,
            suffix: str = "_umap.png",
            show: bool = True,
            return_axs: bool = False,
            data_key=None,
            hue_order=None,
            return_embeddings=False,
            n_neighbors=15,
            tumor_given=False,
    ):
        """
        Plots a grid of UMAPs of node features / node feature embeddings after multiple layers of a model (rows)
        and overlayed with different characteristics (columns).

        Parameters
        ----------
        idx : Indices of graphs to use nodes from.
        layer_names : A list of layer names (check out model.summary()) for which UMAP of embedded nodes will be
                shown. Use 'input' for original node features. In that case loading data with get_data() is enough and
                no model is needed.
        plot_types : A list of characteristics that should be overlayed on the UMAP of each layer.
            - images: colors the nodes based on the image id
            - degree: indicates the node degree
            - a categorical graph/image label: indicates the label of the corresponding graph/image
        panel_width
        panel_height
        save : Whether (if not None) and where (path as string given as save) to save plot.
        suffix : Suffix of file name to save to.
        show : Whether to display plot.
        return_axs: Whether to return axis objects.

        Returns
        -------

        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if not return_axs:
            plt.ioff()

        if not isinstance(plot_types, List):
            plot_types = [plot_types]
        if not isinstance(layer_names, List):
            layer_names = [layer_names]

        fig, axs = plt.subplots(
            nrows=len(layer_names),
            ncols=len(plot_types),
            figsize=(panel_width * len(plot_types), panel_height * len(layer_names))
        )
        if len(layer_names) == 1:
            axs = np.expand_dims(np.asarray(axs), axis=0)
        if len(plot_types) == 1:
            axs = np.expand_dims(np.asarray(axs), axis=-1)

        hue = {}
        palette = {}
        hue_order = {}
        for pl in plot_types:
            hue_order[pl] = None
            graph_labels = self.data.img_celldata[idx[0]].uns['graph_covariates']['label_selection']
            if pl in graph_labels:
                labels = np.concatenate([
                    [self.data.img_celldata[image_key].uns['graph_covariates']['label_tensors'][pl]]
                    for image_key in idx
                ])
                if len(labels.shape) == 2:
                    labels = np.argmax(labels, axis=1)
                if pl != 'grade':
                    labels = np.concatenate([[labels[i]] * self.h[id].shape[0] for i, id in enumerate(idx)])
                    hue[pl] = labels
                    palette[pl] = 'Paired'
                else:
                    labels = np.concatenate([[labels[i] + 1] * self.h[id].shape[0] for i, id in enumerate(idx)])
                    hue[pl] = labels
                    if data_key=='bz' or data_key=='mb':
                        palette[pl] = {
                            1: sns.color_palette("colorblind")[0], 
                            2: sns.color_palette("colorblind")[1], 
                            3: sns.color_palette("colorblind")[2],
                        }
                        hue_order[pl]=[1, 2, 3]
                    elif data_key=='sch':
                        palette[pl] = {
                            1: sns.color_palette("colorblind")[0], 
                            2: sns.color_palette("colorblind")[1],
                        }
                        hue_order[pl]=[1, 2]

            if pl == 'types':
                cell_types = np.concatenate([self.data.img_celldata[id].obsm['node_types'] for i, id in enumerate(idx)])
                cell_types = np.argmax(cell_types, axis=1)
                cell_types_names = list(self.data.img_celldata[idx[0]].uns["node_type_names"].values())
                cell_types = [cell_types_names[type] for type in cell_types]
                hue[pl] = cell_types
                hue_order[pl] = None
                palette[pl] = 'Paired'

            if pl == 'tumor':
                cell_tumor = np.concatenate([self.data.img_celldata[id].obsm['node_types'] for i, id in enumerate(idx)])
                cell_tumor = np.argmax(cell_tumor, axis=1)
                if tumor_given:
                    cell_tumor = [
                        'tumor cell' if 'tumor' in cell_type_names[type] else 'non-tumor cell'
                        for type in cell_tumor
                    ]
                else:
                    cell_type_tumor_dict = {
                        'B cells': 'non-tumor cell',
                        'Basal CKlow': 'tumor cell',
                        'endothelial': 'non-tumor cell',
                        'Endothelial': 'non-tumor cell',
                        'Fibroblasts': 'non-tumor cell',
                        'Fibroblasts CD68+': 'non-tumor cell',
                        'HER2+': 'tumor cell',
                        'HR+ CK7-': 'tumor cell',
                        'HR+ CK7- Ki67+': 'tumor cell',
                        'HR+ CK7- Slug+': 'tumor cell',
                        'HR- CK7+': 'tumor cell',
                        'HR- CK7-': 'tumor cell',
                        'HR- CKlow CK5+': 'tumor cell',
                        'HR- Ki67+': 'tumor cell',
                        'HRlow CKlow': 'tumor cell',
                        'Hypoxia': 'tumor cell',
                        'macrophages': 'non-tumor cell',
                        'Macrophages': 'non-tumor cell',
                        'Macrophages Vim+ CD45low': 'non-tumor cell',
                        'Macrophages Vim+ Slug+': 'non-tumor cell',
                        'Macrophages Vim+ Slug-': 'non-tumor cell',
                        'Myoepithelial': 'non-tumor cell',
                        'Myofibroblasts': 'non-tumor cell',
                        'stromal cells': 'non-tumor cell',
                        'T cells': 'non-tumor cell',
                        'T and B cells': 'non-tumor cell',
                        'Vascular SMA+': 'non-tumor cell',
                        'Tumor cells': 'tumor cell',
                        'tumor cells': 'tumor cell'
                    }
                    cell_types_names = list(self.data.img_celldata[idx[0]].uns["node_type_names"].values())
                    cell_tumor = [cell_type_tumor_dict[cell_types_names[type]] for type in cell_tumor]
                cell_tumor = np.array(cell_tumor)
                hue[pl] = cell_tumor
                hue_order[pl] = None
                palette[pl] = 'Paired'

            if pl == 'images':
                images = np.concatenate([[id] * self.h[id].shape[0] for id in idx])
                hue[pl] = images
                hue_order[pl] = None
                palette[pl] = 'Paired'
            if pl == 'degree':
                degrees = [
                    self.data.img_celldata[image_key].obsp['adjacency_matrix_connectivities'].sum(axis=1)
                    for image_key in idx
                ]
                degrees = np.squeeze(np.array(np.concatenate(degrees)))
                degrees = degrees.astype(int)
                quant = np.quantile(degrees, 0.95)
                degrees = np.array([deg if deg < quant else round(quant) for deg in degrees])
                hue[pl] = degrees
                hue_order[pl] = None
                palette[pl] = 'coolwarm'
        for row, layer_name in enumerate(layer_names):
            umap_emb = self._compute_umap_embedding(
                idx=idx,
                layer_name=layer_name,
                n_neighbors=n_neighbors,
            )
            for col, pl in enumerate(plot_types):
                if len(idx) == 1:
                    s = 10.
                else:
                    s = 1.
                sns.scatterplot(
                    data=umap_emb,
                    x="umap1",
                    y="umap2",
                    hue=hue[pl],
                    hue_order=hue_order[pl],
                    palette=palette[pl],
                    ax=axs[row, col],
                    s=s,
                    linewidth=.01,
                    rasterized=True
                )
                if col == 0:
                    if 'Layer_dense_feature_embedding' in layer_name:
                        yname = 'dense ' + layer_name[-1]
                    elif 'Layer_gcn' in layer_name:
                        yname = 'gcn ' + layer_name[-1]
                    elif 'Layer_gat' in layer_name:
                        yname = 'gat' + layer_name[-1]
                    elif layer_name == 'node_labels':
                        yname = 'node labels'
                    else:
                        yname = layer_name
                    axs[row, col].set_ylabel(yname, rotation=90)
                else:
                    axs[row, col].set_ylabel('')
                axs[row, col].set_xlabel('')
                axs[row, col].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False
                )
                if pl == 'images':
                    axs[row, col].get_legend().remove()
                    if row == 0:
                        axs[row, col].set_title('image')
                elif pl == 'degree':
                    axs[row, col].get_legend().remove()
                    if row == 0:
                        axs[row, col].set_title('node degree')
                elif pl in graph_labels:
                    if row == 0:
                        axs[row, col].set_title(pl)
                axs[row, col].grid(False)

        # Save, show and return figure.
        plt.tight_layout()
        plt.grid(False)
        if save is not None:
            plt.savefig(save + suffix)
        if show:
            plt.show()
        if return_axs:
            if return_embeddings:
                return axs, umap_emb, hue
            return axs
        else:
            plt.close(fig)
            plt.ion()
            if return_embeddings:
                return umap_emb, hue
            return None

    def _compute_gradients_filters(
            self,
            layer,
            image_key: Union[np.ndarray, List[int]],
            target_label: str,
    ) -> Tuple[Tuple[list, list, list], list, list, list]:
        """
        Compute gradients with respect to input data. (A list of gradients per sample if target is multidimensional.)

        :param idx: Observation indices.
        :param image_key: Keys of images.
        :return:
        """
        if isinstance(image_key, str):
            image_key = [image_key]
        ds = self._get_dataset(
            keys=image_key,
            batch_size=1,
            shuffle_buffer_size=1,
            train=False,
            seed=1234
        )
        grads = []

        for x_batch, y_batch in ds:
            h = x_batch[0]
            node_idx = np.arange(0, np.max(np.where(np.sum(np.abs(h), axis=0) > 0)[0]) + 1)  # excluded padded cells
            with tf.GradientTape(persistent=True) as g:
                activations = self.model.get_layer(layer).output
                new_model = tf.keras.models.Model([self.model.inputs], [self.model.output, activations])
                out, act = new_model(x_batch)
                if not isinstance(out, List):
                    out = [out]
                    model_output = [self.model.output]
                else:
                    model_output = self.model.output
                target_output = [o for i, o in enumerate(out) if target_label in model_output[i].name][0][0]

                gr = []
                for o in target_output:  # iterate e.g. over grade 1, grade 2, grade 3
                    gr.append(g.gradient(o, act).numpy().squeeze())
                gr = [g[node_idx] for g in gr]
                grads.append(gr)
        # returns an array of shape (number outputs, number filters) with the mean gradients of the activations of the
        # respective filters
        if len(grads[0][0].shape) > 1:
            grads = [[np.mean(grad_grade, axis=0) for grad_grade in grad_im] for grad_im in grads]
        grads = np.mean(np.array(grads), axis=0)
        return grads

    def plot_weight_matrix(
            self,
            layer_name,
            signed=True,
            is_type_space=True,
            data_key: str = "bz",
            target_label='grade',
            save: Union[str, None] = None,
            suffix: str = "_weight_matrix.pdf",
            panel_width: float = 4.,
            panel_height: float = 4.,
            show: bool = True,
            return_axs: bool = False,
    ):
        """
        Plots the weight matrix of a model trained on cell types.
        :param layer_name: The layer from which the weight is taken.
        :param signed: Whether to show the raw or absolute values of this matrix.
        :param bz: If True, the cell types are nicely sorted. Only for BZ!
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cm

        if not return_axs:
            plt.ioff()

        fig = plt.figure(figsize=(panel_width, panel_height))
        ax1 = fig.add_subplot(111)

        weights = self.model.get_layer(layer_name).get_weights()[0]
        if len(weights) == 1:
            weights = weights[0]
        if not signed:
            weights = np.abs(weights)
        self.test = weights
        gradients = self._compute_gradients_filters(
            layer=layer_name,
            image_key=self.img_keys_test,
            target_label=target_label
        )
        # group filters into the grades that they are most important for and sort them within
        # that group
        grade = np.argmax(gradients, axis=0)
        new_order = []
        for gr in np.unique(grade):
            filter = (grade != gr) * 10
            grad = gradients[gr] + filter
            indices = np.argsort(grad)[:np.sum(grade == gr)][::-1]# reversed filter order before [:np.sum(grade == gr)]
            
            new_order += list(indices)
        weights = weights[:, new_order]
        gradients = gradients[:, new_order]
        if is_type_space:
            cluster_key = self.data.img_celldata[self.img_keys_test[0]].uns['metadata']['cluster_col_preprocessed']
            self.data.celldata.obs[cluster_key]=self.data.celldata.obs[cluster_key].astype(str)

            cell_types = np.unique(self.data.celldata.obs[cluster_key])

            new_order = np.arange(0, len(cell_types))
            weights = weights[new_order]
            yticks = cell_types[new_order]
            ylabel = "cell types"
        else:
            yticks = self.data.node_feature_names
            ylabel = "features"

        min_weight = np.min(weights)
        max_weight = np.max(weights)
        abs_weight = np.max([np.abs(min_weight), np.abs(max_weight)])
        im_pos = ax1.matshow(weights, cmap=cm.get_cmap('seismic'), vmin=-abs_weight, vmax=abs_weight)

        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("top", size='100%', pad=1.0)
        cax_pos = divider.append_axes("right", size='2%', pad=1.0)
        fig.colorbar(im_pos, ax=ax1, cax=cax_pos)
        ax1.set_ylabel(ylabel)
        ax1.set_yticks(ticks=np.arange(0, len(yticks), 1))
        ax1.set_yticklabels(yticks)
        ax1.set_xticks([], [])
        ax1.axis('image')

        min_weight = np.min(gradients)
        max_weight = np.max(gradients)
        abs_weight = np.max([np.abs(min_weight), np.abs(max_weight)])
        im2 = ax2.matshow(gradients, cmap=cm.get_cmap('seismic'), vmin=-abs_weight, vmax=abs_weight)
        
        n_filter = gradients.shape[1]
        ax2.set_xticks(np.arange(n_filter))
        ax2.set_xticklabels(np.arange(1, n_filter + 1))
        ax2.set_xlabel('filter')
        ax2.xaxis.set_label_position('top')
        ax2.axis('image')
        if target_label == "grade":
            ax2.set_yticklabels(['', 'grade 1', 'grade 2', 'grade 3'])
        elif target_label == "Group":
            ax2.set_yticklabels(['', 'Group 1', 'Group 2'])
        else:
            ax2.set_yticklabels(
                [''] +
                [x.split(target_label + ">")[-1] for x in list(self.data.label_names[target_label])]
            )
        ax1.grid(False)
        ax2.grid(False)
        cax3 = divider.append_axes("right", size='2%', pad=1.0)
        cbar = fig.colorbar(im2, ax=ax2, cax=cax3)

        # Save, show and return figure.
        plt.tight_layout()
        plt.grid(False)
        if save is not None:
            plt.savefig(save + suffix)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        if save is not None:
            plt.grid(False)
            plt.savefig(save + suffix)
        if show:
            plt.grid(False)
            plt.show()
        if return_axs:
            return ax
        else:
            plt.close(fig)
            plt.ion()
            return None

class InterpreterGraph(InterpreterBase, estimators.EstimatorGraph):
    pass


class InterpreterNoGraph(InterpreterBase, estimators.EstimatorNoGraph):
    pass
