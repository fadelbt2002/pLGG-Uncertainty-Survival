import os
import torch
from torch import nn
import resnet_seg as resnet_Seg


def generate_model(opt):
    assert opt.model in ['resnet']
    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

    builder = {
        10:  resnet_Seg.resnet10,
        18:  resnet_Seg.resnet18,
        34:  resnet_Seg.resnet34,
        50:  resnet_Seg.resnet50,
        101: resnet_Seg.resnet101,
        152: resnet_Seg.resnet152,
        200: resnet_Seg.resnet200,
    }[opt.model_depth]

    model = builder(
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes,
    )

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)

    net_dict = model.state_dict()

    if opt.phase != 'test' and opt.pretrain_path:
        print(f'Loading pretrained model: {opt.pretrain_path}')
        pretrain = torch.load(opt.pretrain_path, map_location='cpu')
        pretrain_dict = {
            k.replace("module.", ""): v
            for k, v in pretrain['state_dict'].items()
            if k.replace("module.", "") in net_dict
        }
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [
            p for pname, p in model.named_parameters()
            if any(pname.find(ln) >= 0 for ln in opt.new_layer_names)
        ]
        new_parameters_id = set(map(id, new_parameters))
        base_parameters = [p for p in model.parameters() if id(p) not in new_parameters_id]
        return model, {'base_parameters': base_parameters, 'new_parameters': new_parameters}

    return model, model.parameters()
