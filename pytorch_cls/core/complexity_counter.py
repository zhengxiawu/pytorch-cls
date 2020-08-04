# modified from https://github.com/Lyken17/pytorch-OpCounter
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1


def prRed(skk):
    print("\033[91m{}\033[00m".format(skk))


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = total_params


def zero_ops(m, x, y):
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_acts += torch.DoubleTensor([int(0)])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(y.nelement())])


def count_bn(m, x, y):
    # x = x[0]

    # nelements = x.numel()
    # if not m.training:
    #     # subtract, divide, gamma, beta
    #     total_ops = 2 * nelements

    # m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_acts += torch.DoubleTensor([int(0)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(0)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(0)])


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor(
        [*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(0)])


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(y.nelement())])


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    # nn.AvgPool1d: count_avgpool,
    # nn.AvgPool2d: count_avgpool,
    # nn.AvgPool3d: count_avgpool,
    # nn.AdaptiveAvgPool1d: count_adap_avgpool,
    # nn.AdaptiveAvgPool2d: count_adap_avgpool,
    # nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.AvgPool1d: zero_ops,
    nn.AvgPool2d: zero_ops,
    nn.AvgPool3d: zero_ops,
    nn.AdaptiveAvgPool1d: zero_ops,
    nn.AdaptiveAvgPool2d: zero_ops,
    nn.AdaptiveAvgPool3d: zero_ops,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
}


def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_acts', torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." %
                      (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." %
                      (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                prRed(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(
                fn), m.register_forward_hook(count_parameters))
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        total_acts = 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
                m_acts = m.total_acts.item()
            else:
                m_ops, m_params, m_acts = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
            total_acts += m_acts
        print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, total_acts

    total_ops, total_params, total_acts = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return {"flops": total_ops, "params": total_params, "acts": total_acts}
