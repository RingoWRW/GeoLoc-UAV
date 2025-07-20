from models import group
from models import aggregators
from models import backbone

def get_groupnet(groupnet_arch='groupnet', group_config={}):
    
    if  "groupnet" in groupnet_arch.lower():
        return group.GroupNet(**group_config)

def get_groupdinonet(groupnet_arch='groupdinonet', group_config={}):
    
    if  "groupdinonet" in groupnet_arch.lower():
        return group.GroupDinoNet(**group_config)
    
def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'multiconvap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.MulConvAP(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)

    elif 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config
        assert 'num_clusters' in agg_config
        assert 'cluster_dim' in agg_config
        assert 'token_dim' in agg_config
        return aggregators.SALAD(**agg_config)

    elif 'netvlad' in agg_arch.lower():
        return aggregators.NetVLAD()

def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 pretrain_flag=False):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where we sometimes need to crop the last residual block (ex. [4]). Defaults to [].

    Returns:
        model: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbone.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop, pretrain_flag)
    
    elif 'dinov2' in backbone_arch.lower():
        return backbone.DINOv2(model_name=backbone_arch,  num_trainable_blocks=4,
            norm_layer=True,
            return_token=True,
            pretrain_flag=pretrain_flag)