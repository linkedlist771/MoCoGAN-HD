"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os
import numpy as np
from loguru import logger
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import time

from options.pca_options import PCAOptions
from models.models import create_model


class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components,
                                          whiten=self.whiten,
                                          batch_size=max(
                                              100, 2 * n_components))
        self.fit_time = None

    def fit(self, X):
        print(f"Starting PCA fit on data with shape: {X.shape}")
        start_time = time.time()
        
        # If X is large, we can track progress through partial_fit
        batch_size = self.transformer.batch_size
        n_samples = X.shape[0]
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, n_samples, batch_size), 
                     desc="PCA fitting", 
                     total=(n_samples + batch_size - 1) // batch_size,
                     unit="batch"):
            batch_end = min(i + batch_size, n_samples)
            self.transformer.partial_fit(X[i:batch_end])
        
        self.fit_time = time.time() - start_time
        print(f"PCA fit completed in {self.fit_time:.2f} seconds")

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_)  # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio  # PCA outputs are normalized


def main():
    args = PCAOptions().parse(save=False)
    modelS = create_model(args)
    pca = IPCAEstimator(n_components=args.latent_dimension)
    z = torch.FloatTensor(args.batchSize, args.latent_dimension)
    if args.gpu is not None:
        logger.info(f"Using GPU: {args.gpu}")
        z = z.cuda(args.gpu)

    style_list = []
    with torch.no_grad():
        for _ in tqdm(range(args.pca_iterations), desc="Computing PCA", unit="iteration"):
            z.data.normal_()
            styles = modelS(z).data
            style_list.append(styles.cpu().numpy())

        styles_all = np.concatenate(style_list, axis=0)
        style_mean = np.mean(styles_all, axis=0)
        style_var = np.var(styles_all, axis=0)

        pca.fit(styles_all)
        comp, stdev, var_ratio = pca.get_components()

        os.makedirs(args.save_pca_path, exist_ok=True)
        np.save(os.path.join(args.save_pca_path, 'pca_comp.npy'), comp)
        np.save(os.path.join(args.save_pca_path, 'pca_stdev.npy'), stdev)
        np.save(os.path.join(args.save_pca_path, 'pca_var_ratio.npy'),
                var_ratio)
        np.save(os.path.join(args.save_pca_path, 'style_mean.npy'), style_mean)
        np.save(os.path.join(args.save_pca_path, 'style_var.npy'), style_var)


if __name__ == "__main__":
    main()
