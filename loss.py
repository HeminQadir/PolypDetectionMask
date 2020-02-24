from torch import nn
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules import Module


class LossBinary(_Loss):
    def __init__(self, jaccard_weight=0):
        super(LossBinary, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss += self.jaccard_weight * (1 - (intersection + eps) / (union - intersection + eps))
        return loss
    
    
def gan_loss(predict_fake):
    gen_loss_GAN = torch.mean(-torch.log(predict_fake + 1e-5))
    return gen_loss_GAN


class GAN_loss:
    def __init__(self, gan_weight=1):
        self.gan_weight = gan_weight

    def __call__(self, predict_fake):
        Gan_loss = self.gan_weight * gan_loss(predict_fake)
        return Gan_loss
    
    
def discrim_loss(predict_real, predict_fake):
    discrim_loss_GAN =  torch.mean(-(torch.log(predict_real + 1e-5) + torch.log(1 - predict_fake + 1e-5)))
    return discrim_loss_GAN

class Discrim_loss:
    def __init__(self, dircrim_weight=1):
        self.dircrim_weight = dircrim_weight

    def __call__(self, predict_real, predict_fake):
        discrim_loss_all = self.dircrim_weight * discrim_loss(predict_real, predict_fake)
        return discrim_loss_all
    

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        

        
class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per batch element instead and ignores size_average.
           Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    Examples::
        >>> loss = nn.L1Loss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=True, reduce=True):
        super(L1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _assert_no_grad(target)
        target = target.float() 
        #input = torch.tanh(input)
        return F.l1_loss(input, target, size_average=self.size_average,
                         reduce=self.reduce)


#torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    
class KLDivLoss(_Loss):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
    *log-probabilities* and is not restricted to a 2D Tensor.
    The targets are given as *probabilities* (i.e. without taking the logarithm).

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        l(x,y) = L = \{ l_1,\dots,l_N \}, \quad
        l_n = y_n \cdot \left( \log y_n - x_n \right)

    where the index :math:`N` spans all dimensions of ``input`` and :math:`L` has the same
    shape as ``input``. If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';} \\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    In default :attr:`reduction` mode ``'mean'``, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. ``'batchmean'`` mode gives the correct KL divergence where losses
    are averaged over batch dimension only. ``'mean'`` mode's behavior will be changed to the same as
    ``'batchmean'`` in the next major release.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied.
            ``'batchmean'``: the sum of the output will be divided by batchsize.
            ``'sum'``: the output will be summed.
            ``'mean'``: the output will be divided by the number of elements in the output.
            Default: ``'mean'``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. note::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as ``'batchmean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, *)`, the same shape as the input

    """
    __constants__ = ['reduction']
        
    def __init__(self, size_average=True, reduce=True):
        super(KLDivLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _assert_no_grad(target)
        target = target.float() 
        #input = torch.tanh(input)
        return F.kl_div(input.log(), target, size_average=self.size_average, reduce=self.reduce)
    
