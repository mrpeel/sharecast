
print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    #
    # plt.figure()
    # ax = plt.subplot(111)
    #for i in range(X.shape[0]):
        #plt.text(X[i, 0], X[i, 1], #str(digits.target[i]),
                 # color=plt.cm.Set1(y[i] / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)

    # plot the result
    vis_x = X[:, 0]
    vis_y = X[:, 1]

    plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(np.min(y), np.max(y))

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()

def execute_analysis(X, y):
    n_samples, n_features = X.shape
    n_neighbors = 30


    #----------------------------------------------------------------------
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, y,
                   "t-SNE embedding of sharecast data  (time %.2fs)" %
                   (time() - t0))

    #----------------------------------------------------------------------
    # Projection on to the first 2 principal components

    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca, y,
                   "Principal Components projection of sharecast data (time %.2fs)" %
                   (time() - t0))



    #----------------------------------------------------------------------
    # Isomap projection of the digits dataset
    print("Computing Isomap embedding")
    t0 = time()
    X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
    print("Done.")
    plot_embedding(X_iso, y,
                   "Isomap projection of the sharecast data  (time %.2fs)" %
                   (time() - t0))


    #----------------------------------------------------------------------
    # Locally linear embedding of the digits dataset
    print("Computing LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='standard')
    t0 = time()
    X_lle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_lle, y,
                   "Locally Linear Embedding of sharecast data (time %.2fs)" %
                   (time() - t0))


    #----------------------------------------------------------------------
    # Modified Locally linear embedding of the digits dataset
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='modified')
    t0 = time()
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_mlle, y,
                   "Modified Locally Linear Embedding of sharecast data (time %.2fs)" %
                   (time() - t0))


    #----------------------------------------------------------------------
    # HLLE embedding of the digits dataset
    print("Computing Hessian LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='hessian')
    t0 = time()
    X_hlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_hlle, y,
                   "Hessian Locally Linear Embedding of sharecast data (time %.2fs)" %
                   (time() - t0))


    #----------------------------------------------------------------------
    # LTSA embedding of the digits dataset
    print("Computing LTSA embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='ltsa')
    t0 = time()
    X_ltsa = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_ltsa, y,
                   "Local Tangent Space Alignment of the digits (time %.2fs)" %
                   (time() - t0))

    #----------------------------------------------------------------------
    # MDS  embedding of the digits dataset
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    t0 = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding(X_mds, y,
                   "MDS embedding of sharecast data  (time %.2fs)" %
                   (time() - t0))

    #----------------------------------------------------------------------
    # Random Trees embedding of the digits dataset
    print("Computing Totally Random Trees embedding")
    hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                           max_depth=5)
    t0 = time()
    X_transformed = hasher.fit_transform(X)
    pca = decomposition.TruncatedSVD(n_components=2)
    X_reduced = pca.fit_transform(X_transformed)

    plot_embedding(X_reduced, y,
                   "Random forest embedding of sharecast data (time %.2fs)" %
                   (time() - t0))

    #----------------------------------------------------------------------
    # Spectral embedding of the digits dataset
    print("Computing Spectral embedding")
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                          eigen_solver="arpack")
    t0 = time()
    X_se = embedder.fit_transform(X)

    plot_embedding(X_se, y,
                   "Spectral embedding of sharecast data  (time %.2fs)" %
                   (time() - t0))



# digits = datasets.load_digits(n_class=6)
# X = digits.data
# y = digits.target

# boston = datasets.load_boston()
# X = boston.data
# y = boston.target
#
# execute_analysis(X, y)