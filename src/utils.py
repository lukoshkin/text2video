import torch
from torch import autograd

def to_video(tensor):
    """
    Takes generator output and converts it to
    the format which can later be written by
    SummaryWriter's instance

    input: tensor of shape (N, C, D, H, W)
           obtained passed through nn.Tanh
    """
    generated = (tensor + 1) / 2 * 255
    generated = generated.cpu() \
                         .numpy() \
                         .transpose(0, 2, 1, 3, 4)

    return generated.astype('uint8')


def selectFramesRandomly(N, k):
    """
    N - total number of frames
    k - number to choose
    """
    frame_ids = torch.multinomial(torch.ones(N), k)
    frame_ids, _ = torch.sort(frame_ids)
    return frame_ids


def calc_grad_penalty(real_samples, fake_samples, net_D, condition):
    """
    Evaluates the D's gradient penalty and allows other gradients
    to backpropogate through the penalty term
    Args:
        real_samples - a tensor (presumably, without `grad` attribute)
        fake_samples - tensor of the same shape as `real_samples`
        net_D - conditional discriminator
    """
    alpha = real_samples.new(
            real_samples.size(0),
            *([1]*(real_samples.dim()-1))
            ).uniform_().expand(*real_samples.shape)

    inputs = alpha * real_samples + (1-alpha) * fake_samples.detach()
    inputs.requires_grad_(True)
    outputs = net_D(inputs, condition)
    jacobian = autograd.grad(
                outputs=outputs, inputs=inputs,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True)[0]
       
    # flatten each sample grad. and apply 2nd norm to it
    jacobian = jacobian.view(jacobian.size(0), -1)
    return (jacobian.norm(dim=1)**2).mean()


def vanilla_DLoss(M, GM, E, NE, net_D):
    pos, neg, gen = map(net_D, (M, M, GM), (E, NE, E))
    L = pos.log().mean() + (-neg).log1p().mean() + (-gen).log1p().mean()
    return -.33 * L

def vanilla_GLoss1(GM, E, net_D):
    return (-net_D(GM, E)).log1p().mean()

def vanilla_GLoss2(GM, E, net_D):
    return -net_D(GM, E).log().mean()


eps = torch.tensor(1e-12)

def batchGAN_DLoss(u, multibatch, net_D):
    E, M = multibatch
    v = net_D(M, E).view(len(u), -1).mean(1)
    L = u * ((u+eps)/(v+eps)).log() + (1-u) * ((1-u+eps)/(1-v+eps)).log()
    return L.mean()

def batchMGAN_DLoss(u, multibatch, net_D):
    E, M = multibatch
    v = net_D(M, E)
    u = u.repeat_interleave(len(v)//len(u))
    L = u * ((u+eps)/(v+eps)).log() + (1-u) * ((1-u+eps)/(1-v+eps)).log()
    return L.mean()

def batchGAN_GLoss(multibatch, net_D):
    E, M = multibatch
    return vanilla_GLoss2(M, E, net_D)
