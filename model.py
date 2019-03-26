import os.path
import sys

ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import tensorflow as tf
import util.tf_util as tf_util
from util.pointnet_util import pointnet_sa_module, pointnet_fp_module


def get_placeholders(num_point, hyperparams):
    feature_size = 3 * int(hyperparams["use_color"])
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, num_point, 4 + feature_size))
    labels_pl = tf.placeholder(tf.int32, shape=(None, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(None, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, hyperparams, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}

    if hyperparams["use_color"]:
        feature_size = 3 * int(hyperparams["use_color"])
        l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, feature_size])
    else:
        l0_xyz = point_cloud
        ###########print(l0_xyz.shape)
        l0_points = None
    end_points["l0_xyz"] = l0_xyz

    #####Shape should be (-1,-1,4) which indicates intensity has been integrated
    print("L0:", l0_xyz.shape)

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(
        xyz=l0_xyz,
        points=l0_points,
        npoint=hyperparams["l1_npoint"],
        radius=hyperparams["l1_radius"],
        nsample=hyperparams["l1_nsample"],
        mlp=[32, 32, 64],
        mlp2=None,
        group_all=False,
        is_training=is_training,
        bn_decay=bn_decay,
        scope="layer1",
    )
    print('\nLayer1')
    print('l1xyz:',l1_xyz.shape, 'l1points:',l1_points.shape, 'l1indices:',l1_indices.shape)
    
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(
        l1_xyz,
        l1_points,
        npoint=hyperparams["l2_npoint"],
        radius=hyperparams["l2_radius"],
        nsample=hyperparams["l2_nsample"],
        mlp=[64, 64, 128],
        mlp2=None,
        group_all=False,
        is_training=is_training,
        bn_decay=bn_decay,
        scope="layer2",
    )
    print('\nLayer2')
    print('l2xyz:',l2_xyz.shape, 'l2points:',l2_points.shape, 'l2indices:',l2_indices.shape)
    
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(
        l2_xyz,
        l2_points,
        npoint=hyperparams["l3_npoint"],
        radius=hyperparams["l3_radius"],
        nsample=hyperparams["l3_nsample"],
        mlp=[128, 128, 256],
        mlp2=None,
        group_all=False,
        is_training=is_training,
        bn_decay=bn_decay,
        scope="layer3",
    )
    print('\nLayer3')
    print('l3xyz:',l3_xyz.shape, 'l3points:',l3_points.shape, 'l3indices:',l3_indices.shape)

    l4_xyz, l4_points, l4_indices = pointnet_sa_module(
        l3_xyz,
        l3_points,
        npoint=hyperparams["l4_npoint"],
        radius=hyperparams["l4_radius"],
        nsample=hyperparams["l4_nsample"],
        mlp=[256, 256, 512],
        mlp2=None,
        group_all=False,
        is_training=is_training,
        bn_decay=bn_decay,
        scope="layer4",
    )
    print('\nLayer4')
    print('l4xyz:',l4_xyz.shape, 'l4points:',l4_points.shape, 'l4indices:',l4_indices.shape)

    # Feature Propagation layers
    print('\nBackProp to 4')
    l3_points = pointnet_fp_module(
        l3_xyz,
        l4_xyz,
        l3_points,
        l4_points,
        [256, 256],
        is_training,
        bn_decay,
        scope="fa_layer1",
    )
    
    print('\nBackProp to 3')
    l2_points = pointnet_fp_module(
        l2_xyz,
        l3_xyz,
        l2_points,
        l3_points,
        [256, 256],
        is_training,
        bn_decay,
        scope="fa_layer2",
    )
    
    print('\nBackProp to 2')
    l1_points = pointnet_fp_module(
        l1_xyz,
        l2_xyz,
        l1_points,
        l2_points,
        [256, 128],
        is_training,
        bn_decay,
        scope="fa_layer3",
    )
    
    print('\nBackProp to 1')
    l0_points = pointnet_fp_module(
        l0_xyz,
        l1_xyz,
        l0_points,
        l1_points,
        [128, 128, 128],
        is_training,
        bn_decay,
        scope="fa_layer4",
    )

    # FC layers
    net = tf_util.conv1d(
        l0_points,
        128,
        1,
        padding="VALID",
        bn=True,
        is_training=is_training,
        scope="fc1",
        bn_decay=bn_decay,
    )
    end_points["feats"] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope="dp1")
    net = tf_util.conv1d(
        net, num_class, 1, padding="VALID", activation_fn=None, scope="fc2"
    )

    return net, end_points


# For get_loss I added the end_points parameter. Like in pointnet2_cls_ssg.py, it's not used in the function.
def get_loss(pred, label, smpw, end_points):
    """ pred: BxNxC, #one score per class per batch element (N is the nb of points)
        label: BxN,  #one label per batch element
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label, logits=pred, weights=smpw
    )
    tf.summary.scalar("classify loss", classify_loss)
    tf.add_to_collection("losses", classify_loss)
    return classify_loss
