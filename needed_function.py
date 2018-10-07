import h5py
import math
import numpy as np
def random_mini_batches(image, anchor_label,true_box_label, prior_boxes,mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (image, anchor_label,true_box_label)

    Arguments:
    image -- matrix of shape (None ,256 ,256,3)
    anchor_label --  of shape (None, grid,gird,num_anchor_per_cell,5+num_classes)
    true_box_label -- of shape ( None, num_boxes ,5)
    prior_boxes -- of shape(None,grid,grid,num_anchor_per_cell,4)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = image.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (image,anchor_label, true_box_label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation]
    shuffled_anchor = anchor_label[permutation]
    shuffled_true_box = true_box_label[permutation]
    shuffled_prior_box = prior_boxes[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_anchor = shuffled_anchor[ k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_true_box = shuffled_true_box[k*mini_batch_size :k * mini_batch_size + mini_batch_size]
        mini_batch_prior_box = shuffled_prior_box[ k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_anchor, mini_batch_true_box, mini_batch_prior_box)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size: m]
        mini_batch_anchor = shuffled_anchor[ num_complete_minibatches * mini_batch_size: m]
        mini_batch_true_box = shuffled_true_box[num_complete_minibatches * mini_batch_size: m]
        mini_batch_prior_box = shuffled_prior_box[ k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_anchor, mini_batch_true_box, mini_batch_prior_box)
        mini_batches.append(mini_batch)

    return mini_batches


def readh5(h5_path ):
    f = h5py.File(h5_path, 'r')
    train_images = np.array(f['images'][0:])
    anchor_labels = np.array(f['anchor_labels'][0:])
    true_box_labels = np.array(f["true_box_labels"][0:])
    prior_boxes = np.array(f["prior_boxes"][0:])
    f.close()
    return train_images, anchor_labels, true_box_labels ,prior_boxes