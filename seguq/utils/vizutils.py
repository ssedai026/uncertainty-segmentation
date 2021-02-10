import numpy as np
import colorsys
import os

"""
Functions that aids data/image visualization
"""


def __minmax(arr):
    return '(' + str(np.max(arr)) + ',' + str(np.min(arr)) + ')'


def viz_overlaymask(im, out, overlay_intensity=0.3):
    """
    Performs visualization by overlaying the segmentation mask on the image
    :param im: input image
    :param out: segmentation mask  which pixel values should either 0 or 255
    :return: overlayed image
    """
    im = np.asarray(im, np.uint8)
    out = np.asarray(out, np.uint8)

    assert (set(out.flatten())) == set(
        np.asarray([0, 255])), ' Invalid  mask pixels value. The valid values are 0 and 255, found' + __minmax(out)
    assert overlay_intensity < 1 and overlay_intensity > 0, 'The value of overlay_intensity should be between 0 and 1'

    if (len(im.shape) == 2 or im.shape[2] == 1):
        im = np.transpose([im, im, im], [1, 2, 0])  # single to 3 channel

    out = out / 255.0
    mask3 = np.transpose([out, out, out], [1, 2, 0])
    invmask3 = 1 - mask3

    label = [out, np.multiply(out, 255), out]  # green
    label = np.transpose(label, [1, 2, 0])
    imreg = np.multiply(im, mask3)

    x = np.multiply(imreg, (1 - overlay_intensity)) + np.multiply(label, overlay_intensity)
    x = np.asarray(x, np.uint8)
    y = np.multiply(im, invmask3)
    z = np.asarray(x + y, np.uint8)

    return z









def _pad_patches_stack(patches, N):
    """
     Appends the image stack with the blank images to make total number of images in stack to be N
    :param patches:
    :param N:
    :return: padded patches
    """
    n_blank = N - patches.shape[0]

    # assert (n_blank <= ngrid_x and n_blank <= ngrid_y), ' too many blank grids, ' + str(n_blank)

    if n_blank > 0:
        ss = list(patches.shape)
        ss[0] = n_blank
        blank = np.zeros(tuple(ss))
        patches = np.vstack([patches, blank])

    return patches

def stack_patches(patches, nr_row, nr_col):
    """
    Stack patches, i.e,  convert the image patches to a single image
    :param patches:  N x H x W x c
    :return: nr_row*H x nr_col*W image
    """
    assert (patches.shape[0] <= nr_row * nr_col), 'The number of patches should be equal to nr_row*nr_col' + str(
        patches.shape[0]) + '<=' + str(nr_row) + 'x' + str(nr_col)

    n_blank = nr_col * nr_row - patches.shape[0]
    assert (n_blank <= nr_row and n_blank <= nr_col), ' too many blank grids,' + str(n_blank)

    patches = _pad_patches_stack(patches, nr_row * nr_col)



    rows = []
    for r in range(nr_row):
        cols = []
        for c in range(nr_col):
            # print r,c, r * nr_row + c
            cols.append(patches[r * nr_col + c, :, :, :])
        col = np.concatenate(cols, axis=1)
        rows.append(col)

    row = np.concatenate(rows, axis=0)
    return row



def ensure3channels(im):
    # im - N x H x W x C

    assert len(im.shape) == 3 or len(im.shape) == 4, ' invalid dimension'
    if (len(im.shape) == 3):
        im = np.expand_dims(im, -1)
    else:
        assert im.shape[3] == 1 or im.shape[3] == 3, 'expected dimension for channel is 1 or 3 imdim '+ str(im.shape)

    if (im.shape[3] == 1):
        im = np.transpose([im, im, im], [1, 2, 3, 0, 4])  # single to 3 channel
        im = np.squeeze(im, axis=4)

    return im








def visualize_probmaps(prob_maps, colors=None):
    """
        Visualizes a multi channel probability map  (output from semantic segmentation)
        :param prob_maps: HxWxc or NxHxWxc
        :return: viz rgb image HxWx3 or NxHxWx3
        """

    if(len(prob_maps.shape)==4):
        pmaps=[]
        for pm in prob_maps:
            pmaps.append(np.expand_dims(visualize_probmaps_one(pm, colors),0))
        vizim = np.vstack(pmaps)
        return vizim
    else:
        return visualize_probmaps_one(prob_maps,colors)



def visualize_probmaps_one(prob_maps, colors=None, background_index=None):
    """
    Visualizes a multi channel probability map  (output from semantic segmentation)
    :param prob_maps: HxWxc
    :return: viz rgb image
    """

    N_class = prob_maps.shape[2]

    if (background_index is None):
        background_index = N_class - 1

    if (colors is None):
        colors = [colorsys.hsv_to_rgb(x * 1.0 / N_class, 0.6, 1) for x in range(N_class)]
        colors[background_index] = (0, 0, 0)
    else:
        assert len(colors) == N_class

    # (H,W)
    alabel = np.argmax(prob_maps, axis=2)

    viz = np.zeros((alabel.shape[0], alabel.shape[1], 3))
    for c in range(N_class):
        viz += _visualize_probmaps_oneclass(alabel, c, colors[c])
    return viz


def visualize_labelmaps(label_maps, N_class, colors=None, background_index=None):
    """

    :param label_maps:
    :param N_class:
    :param colors:
    :param background_index:  is none, the last the background index is taken as N_class-1
    :return:
    """
    if(background_index is None):
        background_index = N_class -1

    if (colors is None):
        colors = [colorsys.hsv_to_rgb(x * 1.0 / N_class, 0.6, 1) for x in range(N_class)]
    else:
        assert len(colors) == N_class

    colors[background_index]=(0,0,0)

    def get_labelmap_viz(alabel):
        viz = np.zeros((alabel.shape[0], alabel.shape[1], 3))
        for c in range(N_class):
            viz += _visualize_probmaps_oneclass(alabel, c, colors[c])
        return viz

    if(len(label_maps.shape)==3):
        labels_viz = []
        for alabel in label_maps:
            viz = get_labelmap_viz(alabel)
            labels_viz.append(np.expand_dims(viz, 0))
        vizim = np.vstack(labels_viz)
        return vizim
    else:
        alabel=label_maps
        viz = get_labelmap_viz(alabel)
        return viz


def _visualize_probmaps_oneclass(alabel, class_label, color):
    alabel_r = np.zeros((alabel.shape[0], alabel.shape[1], 1))
    alabel_g = np.zeros((alabel.shape[0], alabel.shape[1], 1))
    alabel_b = np.zeros((alabel.shape[0], alabel.shape[1], 1))

    mask_class = alabel == class_label
    np.place(alabel_r, mask_class, color[0] * 255)
    np.place(alabel_g, mask_class, color[1] * 255)
    np.place(alabel_b, mask_class, color[2] * 255)

    im = np.concatenate([alabel_r, alabel_g, alabel_b], axis=2)
    return im.astype(np.uint8)






def visualize_labels_overlay_labelmap(label_maps, val_imgs, n_classes, alpha_label=0.6, BG_INDEX=None, stack_images=True):
    """

    :param label_maps: (n,H,W) where each pixel contain class index
    :param val_imgs: (n,H,W,C) where C=1 or 3
    :n_classes number of classes used in visualization. It should be consistent with label_maps
    :param BG_INDEX: index of background class, if None then n_classes - 1 will be used
    :return:
    """

    num_images = label_maps.shape[0]

    if (BG_INDEX is None):
        BG_INDEX = n_classes - 1

    alphas = np.ones_like(label_maps) * alpha_label
    alphas[label_maps == BG_INDEX] = 0
    # alphas[label_maps != BG_INDEX] = 0.2
    alphas = np.expand_dims(alphas, -1)
    alphas = np.repeat(alphas, 3, axis=3)
    viz = visualize_labelmaps(label_maps, N_class=n_classes)

    if (val_imgs.shape[3] == 1):
        val_imgs = np.repeat(val_imgs, 3, axis=3)
    else:
        assert val_imgs.shape[3] == 3, 'channels not supported for visualization ' + str(val_imgs.shape[3])

    # alphas =0.5
    # alphas = alphas.astype(np.float32)
    final_viz = val_imgs * (1 - alphas) + (alphas) * viz
    if(stack_images):
        a, b = get_factors(num_images)
        final_viz = stack_patches(final_viz, a, b)
    return final_viz



def get_factors(x):
    # This function takes a number x  and returns the two numbers a,b both factors of x
    # such  that abs(a-b) is minimum among all factors of x

    z = []
    for i in range(1, x + 1):
        if x % i == 0:
            z.append(i)
    n = int(len(z) / 2) - 1
    return z[n], z[n + 1]
