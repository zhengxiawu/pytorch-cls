import torch
import torch.nn as nn

multiply_adds = 1


def count_convNd(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    total_acts = output_elements
    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([int(total_acts)])


def count_linear(m, _, y):
    total_ops = m.in_features * m.out_features
    m.total_acts += torch.DoubleTensor([int(y.nelement())])
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_acts += torch.DoubleTensor([0])
    m.total_ops += torch.DoubleTensor([total_ops])


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.total_acts += torch.DoubleTensor([0])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    ######################################
    nn.Linear: count_linear,
    ######################################
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: count_bn,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
}


def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))
        m_.register_buffer('total_acts', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            # print("Not implemented for ", m_)
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params, total_acts = 0, 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params, m_acts = m.total_ops.item(
                ), m.total_params.item(), m.total_acts.item()
            else:
                m_ops, m_params, m_acts = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
            total_acts += m_acts
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params, total_acts

    total_ops, total_params, total_acts = dfs_count(model)

    # total_ops = 0
    # total_params = 0
    # total_acts = 0
    # for m in model.modules():
    #     if len(list(m.children())) > 0:  # skip for non-leaf module
    #         continue
    #     total_ops += m.total_ops
    #     total_params += m.total_params
    #     total_acts += m.total_acts

    # total_ops = total_ops.item()
    # total_params = total_params.item()
    # total_acts = total_acts.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return {"flops": total_ops, "params": total_params, "acts": total_acts}
