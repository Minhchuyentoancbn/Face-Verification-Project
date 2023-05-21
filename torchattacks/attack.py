from collections import OrderedDict

import torch
import torch.nn as nn


class FGSM:
    """
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255):
        self._attacks = OrderedDict()
        self.model = model
        self.device = next(model.parameters()).device

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

        self._normalization_applied = False
        self.targeted = False

        self.eps = eps


    def __call__(self, images, labels=None, *args, **kwargs):
        given_training = self.model.training
        self._change_model_mode(given_training)
        adv_images = self.forward(images, labels, *args, **kwargs)
        self._recover_model_mode(given_training)
        return adv_images

    
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()


    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()


    def forward(self, images, labels=None, *args, **kwargs):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        images.requires_grad = True

        outputs, _ = self.model(images)
        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                    retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=-127.5/128, max=127.5/128).detach()

        return adv_images