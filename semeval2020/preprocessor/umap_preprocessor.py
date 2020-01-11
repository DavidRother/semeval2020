from semeval2020.factory_hub import preprocessor_factory
import umap

preprocessor_factory.register("UMAP", umap.UMAP)
