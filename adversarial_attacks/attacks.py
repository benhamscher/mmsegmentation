import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import trange
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.interpolate import NearestNDInterpolator
from torchvision.transforms import ToPILImage
from typing import Optional, Tuple, Union
import cv2
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
import math
from torch.autograd import Function, grad
from typing import Callable, Optional

from functools import partial

from typing import Optional

import torch
#from adv_lib.utils.losses import difference_of_logits
#from adv_lib.utils.visdom_logger import VisdomLogger
from torch import Tensor
from torch.autograd import grad

class P(Function):
    @staticmethod
    def forward(ctx, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        y_sup = μ * y + μ * ρ * y ** 2 + 1 / 6 * ρ ** 2 * y ** 3
        y_inf = μ * y / (1 - ρ.clamp(min=1) * y.clamp(max=0))
        sup = y >= 0
        ctx.save_for_backward(y, ρ, μ, sup)
        return torch.where(sup, y_sup, y_inf)

    @staticmethod
    def backward(ctx, grad_output):
        y, ρ, μ, sup = ctx.saved_tensors
        grad_y_sup = μ * y + 2 * μ * ρ * y + 1 / 2 * ρ ** 2 * y ** 2
        grad_y_inf = μ / (1 - ρ.clamp(min=1) * y.clamp(max=0)).square_()
        return grad_output * torch.where(sup, grad_y_sup, grad_y_inf), None, None, None



colors = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]



class Attacks():
    """
    Attacks class contains different attacks
    """
    def __init__(self, model, device='cpu'): #, params=[]):
        """ Initialize the FGSM class
        Args:
            model (torch.nn model): model to be attacked
            device  (device):       device
        """
        self.model = model
        self.device = device
        # self.params = params

    def model_pred(self, img):
        """ individual model prediction
        Args:
            img (torch.tensor): input image
        Returns:
           pred (torch.tensor): predicted semantic segmentation
        """
        pred = (self.model.test_step(img)[0].seg_logits.data).unsqueeze(0)

    
        return pred


    

    def difference_of_logits_ratio(self, logits: Tensor, labels: Tensor, labels_infhot: Optional[Tensor] = None,
                               targeted: bool = False, ε: float = 0) -> Tensor:
        """Difference of Logits Ratio from https://arxiv.org/abs/2003.01690. This version is modified such that the DLR is
        always positive if argmax(logits) == labels"""
        logit_dists = self.difference_of_logits(logits=logits, labels=labels, labels_infhot=labels_infhot)

        if targeted:
            top4_logits = torch.topk(logits, k=4, dim=1).values
            logit_normalization = top4_logits[:, 0] - (top4_logits[:, -2] + top4_logits[:, -1]) / 2
        else:
            top3_logits = torch.topk(logits, k=3, dim=1).values
            logit_normalization = top3_logits[:, 0] - top3_logits[:, -1]

        return (logit_dists + ε) / (logit_normalization + 1e-8)

    def difference_of_logits(self, logits: Tensor, labels: Tensor, labels_infhot: Optional[Tensor] = None) -> Tensor:
        if labels_infhot is None:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

        class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = (logits - labels_infhot).amax(dim=1)
        return class_logits - other_logits
    

    def projected_gradient_descent(self,  x, y, step_size=2, step_norm="inf", eps=10, num_steps=40, 
                                   clamp=(0,255), eps_norm = "inf", y_target=None):
        """Performs the projected gradient descent attack on a batch of images."""
        lo = nn.CrossEntropyLoss(ignore_index=255)
        x = x["inputs"][0].unsqueeze(0)
        if y_target != None:
            y_target = y_target.to(self.device)
        x = x.to(self.device)
        print(x.shape)
        print(y.shape)

        x_adv = x.clone().detach().float().requires_grad_(True).to(self.device)
        y = y.to(self.device)
        targeted = y_target is not None


        num_channels = x.shape[1]

        for i in range(num_steps):
            print(i)
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = self.model_pred({"inputs":_x_adv})
            print(prediction.shape)
            print(y.shape)
            
            #exit()
            loss = lo(prediction, y_target if targeted else y)
            loss.backward()
            print(i)

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                if step_norm == 'inf':
                    gradients = _x_adv.grad.sign() * step_size
                else:
                    # Note .view() assumes batched image data as 4D tensor
                    gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                        .norm(step_norm, dim=-1)\
                        .view(-1, num_channels, 1, 1)

                if targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t. the image data
                    print("target done")
                    x_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                    # the model parameters
                    x_adv += gradients

            # Project back into l_norm ball and correct range
            if eps_norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            else:
                delta = x_adv - x

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
                scaling_factor[mask] = eps

                # .view() assumes batched images as a 4D Tensor
                delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                x_adv = x + delta
                
            x_adv = x_adv.clamp(*clamp)

        return x_adv.detach()

    def prox_linf_indicator(self, δ: Tensor, λ: Tensor, lower: Tensor, upper: Tensor, H: Optional[Tensor] = None,
                        ε: float = 1e-6, section: float = 1 / 3) -> Tensor:
        """Proximity operator of λ||·||_∞ + \iota_Λ in the diagonal metric H. The lower and upper tensors correspond to
        the bounds of Λ. The problem is solved using a ternary search with section 1/3 up to an absolute error of ε on the
        prox. Using a section of 1 - 1/φ (with φ the golden ratio) yields the Golden-section search, which is a bit faster,
        but less numerically stable."""
        δ_, λ_ = δ.flatten(1), 2 * λ.unsqueeze(1)
        H_ = H.flatten(1) if H is not None else None
        δ_proj = δ_.clamp(min=lower.flatten(1), max=upper.flatten(1))
        right = δ_proj.norm(p=float('inf'), dim=1, keepdim=True)
        left = torch.zeros_like(right)
        steps = (ε / right.max()).log_().mul_(math.log(math.e, 1 - section)).ceil_().long()
        prox, left_third, right_third, f_left, f_right, cond = (None,) * 6
        for _ in range(steps):
            left_third = torch.lerp(left, right, weight=section, out=left_third)
            right_third = torch.lerp(left, right, weight=1 - section, out=right_third)

            prox = torch.clamp(δ_proj, min=-left_third, max=left_third, out=prox).sub_(δ_).square_()
            if H_ is not None:
                prox.mul_(H_)
            f_left = torch.sum(prox, dim=1, keepdim=True, out=f_left)
            f_left.addcmul_(left_third, λ_)

            prox = torch.clamp(δ_proj, min=-right_third, max=right_third, out=prox).sub_(δ_).square_()
            if H_ is not None:
                prox.mul_(H_)
            f_right = torch.sum(prox, dim=1, keepdim=True, out=f_right)
            f_right.addcmul_(right_third, λ_)

            cond = torch.ge(f_left, f_right, out=cond)
            left = torch.where(cond, left_third, left, out=left)
            right = torch.where(cond, right, right_third, out=right)
        left.lerp_(right, weight=0.5)
        return δ_proj.clamp_(min=-left, max=left).view_as(δ)





    def alma_prox(self, #model: nn.Module,
                #inputs: Tensor,
                data,
                labels: Tensor,
                masks: Tensor = None,
                targeted: bool = False,
                adv_threshold: float = 0.99,
                penalty: Callable = P.apply,
                num_steps: int = 500,
                #lr_init: float = 0.001,
                lr_init: float = 0.2,
                #lr_reduction: float = 0.1,
                lr_reduction: float = 0.1,
                μ_init: float = 1,
                ρ_init: float = 0.1,
                #ρ_init: float = 0.01,
                check_steps: int = 10,
                τ: float = 0.95,
                γ: float = 2,
                α: float = 0.8,
                α_rms: float = None,
                scale_min: float = 0.1,
                scale_max: float = 1,
                scale_init: float = 1,
                #scale_γ: float = 0.01,
                scale_γ: float = 0.05,
                logit_tolerance: float = 1e-4,
                constraint_masking: bool = True,
                mask_decay: bool = True):
        
        """
        ALMA prox attack from https://arxiv.org/abs/2206.07179 to find $\ell_\infty$ perturbations.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : Tensor
            Inputs to attack. Should be in [0, 1].
        labels : Tensor
            Labels corresponding to the inputs if untargeted, else target labels.
        masks : Tensor
            Binary mask indicating which pixels to attack, to account for unlabeled pixels (e.g. void in Pascal VOC)
        targeted : bool
            Whether to perform a targeted attack or not.
        adv_threshold : float
            Fraction of pixels required to consider an attack successful.
        penalty : Callable
            Penalty-Lagrangian function to use. A good default choice is P2 (see the original article).
        num_steps : int
            Number of optimization steps. Corresponds to the number of forward and backward propagations.
        lr_init : float
            Initial learning rate.
        lr_reduction : float
            Reduction factor for the learning rate. The final learning rate is lr_init * lr_reduction
        μ_init : float
            Initial value of the penalty multiplier.
        ρ_init : float
            Initial value of the penalty parameter.
        check_steps : int
            Number of steps between checks for the improvement of the constraint. This corresponds to M in the original
            article.
        τ : float
            Constraint improvement rate.
        γ : float
            Penalty parameter increase rate.
        α : float
            Weight for the exponential moving average.
        α_rms : float
            Smoothing constant for gradient normalization. If none is provided, defaults to α.
        scale_min : float
            Minimum constraint scale, corresponding to w_min in the paper.
        scale_max : float
            Maximum constraint scale.
        scale_init : float
            Initial constraint scale w^{(0)}.
        scale_γ : float
            Constraint scale adjustment rate.
        logit_tolerance : float
            Small quantity added to the difference of logits to avoid solutions where the difference of logits is 0, which
            can results in inconsistent class prediction (using argmax) on GPU. This can also be used as a confidence
            parameter κ as in https://arxiv.org/abs/1608.04644, however, a confidence parameter on logits is not robust to
            scaling of the logits.
        constraint_masking : bool
            Discard (1 - adv_threshold) fraction of the largest constraints, which are less likely to be satisfied.
        mask_decay : bool
            Linearly decrease the number of discarded constraints.
        callback : VisdomLogger
            Callback to visualize the progress of the algorithm.

        Returns
        -------
        best_adv : Tensor
            Perturbed inputs (inputs + perturbation) that are adversarial and have smallest perturbation norm.

        """
        attack_name = f'ALMA prox'
      
        inputs = data['inputs'][0].unsqueeze(0).float()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        masks = masks.bool()
        masks = masks.to(self.device)
        #batch_size = len(inputs)
        batch_size = 1
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        multiplier = -1 if targeted else 1
        α_rms = α if α_rms is None else α_rms

        # Setup variables
        δ = torch.zeros_like(inputs, requires_grad=True)
        lr = torch.full((batch_size,), lr_init, device=self.device, dtype=torch.float)
        s = torch.zeros_like(δ)
        lower, upper = -inputs, 255 - inputs
        prox_func = partial(self.prox_linf_indicator, lower=lower, upper=upper)

        # Init constraint parameters
        μ = torch.full_like(labels, μ_init, device=self.device, dtype=torch.double)
        ρ = torch.full_like(labels, ρ_init, device=self.device, dtype=torch.double)
        if scale_init is None:
            scale_init = math.exp(math.log(scale_min * scale_max) / 2)
        w = torch.full_like(lr, scale_init)  # constraint scale

        # Init trackers
        best_dist = torch.full_like(lr, float('inf'))
        best_adv_percent = torch.zeros_like(lr)
        adv_found = torch.zeros_like(lr, dtype=torch.bool)
        best_adv = inputs.clone()
        pixel_adv_found = torch.zeros_like(labels, dtype=torch.bool)
        step_found = torch.full_like(lr, num_steps // 2)

        for i in range(num_steps):  
            if i % 100 == 0:
                print(i)

            adv_inputs = inputs + δ


            logits = self.model_pred({"inputs":adv_inputs})
            #logits = self.model(adv_inputs)
            dist = δ.data.flatten(1).norm(p=float('inf'), dim=1)

            if i == 0:
                # initialize variables based on model's output
                num_classes = logits.size(1)
                if masks is None:
                    masks = labels < num_classes
                masks_sum = masks.flatten(1).sum(dim=1)
                labels_ = labels * masks
                masks_inf = torch.zeros_like(masks, dtype=torch.float).masked_fill_(~masks, float('inf'))
                labels_infhot = torch.zeros_like(logits.detach()).scatter_(1, labels_.unsqueeze(1), float('inf'))
                diff_func = partial(self.difference_of_logits_ratio, labels=labels_, labels_infhot=labels_infhot,
                                    targeted=targeted, ε=logit_tolerance)
                k = ((1 - adv_threshold) * masks_sum).long()  # number of constraints that can be violated
                constraint_mask = masks

            # track progress
            pred = logits.argmax(dim=1)
            pixel_is_adv = (pred == labels) if targeted else (pred != labels)
            pixel_adv_found.logical_or_(pixel_is_adv)
            adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
            if i % 100 == 0:
                print("adv_percent", adv_percent)
            is_adv = adv_percent >= adv_threshold
            is_smaller = dist <= best_dist
            improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
            is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
            if i < num_steps // 2: step_found.masked_fill_((~adv_found) & is_adv, i)  # reduce lr before num_steps // 2
            adv_found.logical_or_(is_adv)
            best_dist = torch.where(is_better_adv, dist.detach(), best_dist)
            best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
            best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

            # adjust constraint scale
            w.div_(torch.where(is_adv, 1 + scale_γ, 1 - scale_γ)).clamp_(min=scale_min, max=scale_max)

            dlr = multiplier * diff_func(logits)
            constraints = w.view(-1, 1, 1) * dlr

            if constraint_masking:
                if mask_decay:
                    k = ((1 - adv_threshold) * masks_sum).mul_(i / (num_steps - 1)).long()
                if k.any():
                    top_constraints = constraints.detach().sub(masks_inf).flatten(1).topk(k=k.max()).values
                    ξ = top_constraints.gather(1, k.unsqueeze(1) - 1).squeeze(1)
                    constraint_mask = masks & (constraints <= ξ.view(-1, 1, 1))

            # adjust constraint parameters
            if i == 0:
                prev_constraints = constraints.detach()
            elif (i + 1) % check_steps == 0:
                improved_constraint = (constraints.detach() * constraint_mask <= τ * prev_constraints)
                ρ = torch.where(~(pixel_adv_found | improved_constraint), γ * ρ, ρ)
                prev_constraints = constraints.detach()
                pixel_adv_found.fill_(False)

            if i:
                c = constraints.to(dtype=μ.dtype)
                new_μ = grad(penalty(c, ρ, μ)[constraint_mask].sum(), c, only_inputs=True)[0]
                μ.lerp_(new_μ, weight=1 - α).clamp_(1e-12, 1)

            loss = penalty(constraints, ρ, μ).mul(constraint_mask).flatten(1).sum(dim=1)
            δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

            if lr_reduction != 1:
                tangent = lr_reduction / (1 - lr_reduction) * (num_steps - step_found).clamp_(min=1)
                decay = tangent / ((i - step_found).clamp_(min=0) + tangent)
                λ = lr * decay
            else:
                λ = lr

            s.mul_(α_rms).addcmul_(δ_grad, δ_grad, value=1 - α_rms)
            H = s.div(1 - α_rms ** (i + 1)).sqrt_().clamp_(min=1e-8)

            # gradient step
            δ.data.addcmul_(δ_grad, batch_view(λ) / H, value=-1)

            # proximal step
            δ.data = prox_func(δ=δ.data, λ=λ, H=H)

        print("adv_percent", adv_percent)
        return best_adv

    def _pgd(self,
            model: nn.Module,
            inputs: Tensor,
            labels: Tensor,
            ε: Tensor,
            masks: Tensor = None,
            targeted: bool = False,
            norm: float = float('inf'),
            num_steps: int = 40,
            random_init: bool = False,
            loss_function: str = 'ce',
            relative_step_size: float = 0.1 / 3,
            absolute_step_size: Optional[float] = None) -> Tuple[Tensor, Tensor]:
        _loss_functions = {
            'ce': (partial(F.cross_entropy, reduction='none'), 1),
            #'dl': (difference_of_logits, -1),
            #'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
        }
        device = inputs.device
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        neg_inputs = torch.zeros_like(inputs)
        one_minus_inputs = torch.full_like(inputs, 255)  # Adjust the clamp range to the data range

        loss_func, multiplier = _loss_functions[loss_function.lower()]
        if targeted:
            multiplier *= -1

        if absolute_step_size is not None:
            step_size = torch.full((len(inputs),), absolute_step_size, dtype=torch.float, device=inputs.device)
        else:
            step_size = ε * relative_step_size

        δ = torch.zeros_like(inputs, requires_grad=True)
        best_percent = torch.zeros(batch_size, device=device)
        best_adv = inputs.clone()

        if random_init:
            if norm == float('inf'):
                δ.data.uniform_().sub_(0.5).mul_(2 * batch_view(ε))
            elif norm == 2:
                δ.data.normal_()
                δ.data.mul_(batch_view(ε / δ.data.flatten(1).norm(p=2, dim=1)))
            δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

        for i in range(num_steps):
            adv_inputs = inputs + δ
            #print("inputs + δ" , (inputs + δ).shape)
            
            #print(model)
            
            logits = model({"inputs": (inputs + δ)})

            if i == 0:
                if masks is None:
                    num_classes = logits.size(1)
                    masks = labels < num_classes
                masks_sum = masks.flatten(1).sum(dim=1)
                labels_ = labels * masks

                if loss_function.lower() in ['dl', 'dlr']:
                    labels_infhot = torch.zeros_like(logits).scatter(1, labels_.unsqueeze(1), float('inf'))
                    loss_func = partial(loss_func, labels_infhot=labels_infhot)

            
            loss = multiplier * loss_func(logits, labels_).masked_select(masks)
            δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

            pred = logits.argmax(dim=1)
            pixel_is_adv = (pred == labels) if targeted else (pred != labels)
            adv_percent = (pixel_is_adv.float() * masks).flatten(1).sum(dim=1) / masks_sum
            is_better = adv_percent >= best_percent
            best_percent = torch.where(is_better, adv_percent, best_percent)
            best_adv = torch.where(batch_view(is_better), adv_inputs.detach(), best_adv)

            if norm == float('inf'):
                δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
            elif norm == 2:
                δ_grad.div_(δ_grad.flatten(1).norm(p=2, dim=1).clamp_min_(1e-6))
                δ.data.add_(batch_view(step_size) * δ_grad)
                δ_norm = δ.data.flatten(1).norm(p=2, dim=1)
                δ.data.mul_(batch_view(ε / torch.where(δ_norm > ε, δ_norm, ε)))
            δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

        return best_percent, best_adv

    def pgd(self,
            model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        ε: Union[float, Tensor],
        masks: Tensor = None,
        targeted: bool = False,
        norm: float = float('inf'),
        num_steps: int = 40,
        random_init: bool = False,
        restarts: int = 1,
        loss_function: str = 'ce',
        relative_step_size: float = 0.1 / 3,
        absolute_step_size: Optional[float] = None) -> Tensor:
        if isinstance(ε, (int, float)):
            ε = torch.full((len(inputs),), ε, dtype=torch.float, device=inputs.device)

        adv_inputs = inputs.clone()
        adv_percent = torch.zeros_like(ε)
        pgd_attack = partial(self._pgd, model=model, ε=ε, targeted=targeted, masks=masks, norm=norm, num_steps=num_steps,
                            loss_function=loss_function, relative_step_size=relative_step_size,
                            absolute_step_size=absolute_step_size)

        for i in range(restarts):
            adv_percent_run, adv_inputs_run = pgd_attack(inputs=inputs, labels=labels, random_init=random_init or (i != 0))
            better_adv = adv_percent_run >= adv_percent
            adv_inputs[better_adv] = adv_inputs_run
            adv_percent[better_adv] = adv_percent_run

        return adv_inputs

    def minimal_pgd(self, 
                    #model ,#inputs#: Tensor,
                    data,
                    labels,#: Tensor, 
                    max_ε: float,
                    masks: Tensor = None,
                    targeted: bool = False,
                    adv_threshold: float = 0.99,
                    binary_search_steps: int = 20, **kwargs) -> Tensor:
        device = self.device
        
        batch_size = 1
        inputs = data['inputs'][0].unsqueeze(0).float()
        inputs = inputs.to(self.device)
        

        labels = labels.to(self.device)

        masks = masks.bool()
        masks = masks.to(self.device)
    
        adv_inputs = data["inputs"][0].unsqueeze(0).clone()


        best_ε = torch.full((batch_size,), 2 * max_ε, dtype=torch.float, device=device)
        ε_low = torch.zeros_like(best_ε)

        attack = partial(self.pgd, inputs=inputs, labels=labels, model=self.model_pred, targeted=targeted, masks=masks, **kwargs)
        
        for i in range(binary_search_steps):
            
            ε = (ε_low + best_ε) / 2

            adv_inputs_run = attack(ε=ε)
            logits = self.model_pred({"inputs": adv_inputs_run})

            if i == 0:
                num_classes = logits.size(1)
                if masks is None:
                    masks = labels < num_classes
                masks_sum = masks.flatten(1).sum(dim=1)

            pred = logits.argmax(dim=1)
            pixel_is_adv = (pred == labels) if targeted else (pred != labels)
            adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum

            better_adv = (adv_percent >= adv_threshold) & (ε < best_ε)
      
            adv_inputs[0] = adv_inputs_run[0]

            ε_low = torch.where(better_adv, ε_low, ε)
            best_ε = torch.where(better_adv, ε, best_ε)

        return adv_inputs

    def dag(self,
            data, #inputs: Tensor,
            labels, #labels: Tensor,
            masks: Tensor = None,
            targeted: bool = False,
            adv_threshold: float = 0.99,
            max_iter: int = 200,
            γ: float = 0.5,
            p: float = float('inf')) ->  Tensor:
            """DAG attack from https://arxiv.org/abs/1703.08603"""

            inputs = data['inputs'][0].unsqueeze(0).clone().detach()
            inputs = inputs.float().to(self.device)
            inputs.requires_grad = True
       
            labels = labels.to(self.device)
    
            batch_size = 1

            batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))

            multiplier = -1 if targeted else 1

            r = torch.zeros_like(inputs)

            # Init trackers
            best_adv_percent = torch.zeros(batch_size, device=self.device)
           
            adv_found = torch.zeros_like(best_adv_percent, dtype=torch.bool)
            
            best_adv = inputs.clone()


            for i in range(max_iter):
                

                active_inputs = ~adv_found
      
                
                inputs_ = inputs[active_inputs]
     
            
                r_ = r[active_inputs]
                r_.requires_grad_(True)
  
                
                adv_inputs_ = inputs_.clone()
                adv_inputs_ = (inputs_ + r_).clamp(0, 255)
         
                logits = self.model_pred({"inputs":adv_inputs_})
  

                if i == 0:
                    num_classes = logits.size(1)
                    if masks is None:
                        masks = labels < num_classes
                      
                        
               
                    else:
                        masks = masks.to(self.device)

                    masks_sum = masks.flatten(1).sum(dim=1)
                    masked_labels = labels * masks
                
                    labels_infhot = torch.zeros_like(logits.detach()).scatter(1, masked_labels.unsqueeze(1), float('inf'))


                dl = multiplier * self.difference_of_logits(logits, labels=masked_labels[active_inputs],
                                                   labels_infhot=labels_infhot[active_inputs])

                pixel_is_adv = dl < 0


                active_masks = masks[active_inputs]
                adv_percent = (pixel_is_adv & active_masks).flatten(1).sum(dim=1) / masks_sum[active_inputs]
               
                is_adv = adv_percent >= adv_threshold
                adv_found[active_inputs] = is_adv
             
                
                best_adv[active_inputs] = torch.where(batch_view(is_adv), adv_inputs_.detach(), best_adv[active_inputs])

                if is_adv.all():
        
                    break
                
                loss = (dl[~is_adv] * active_masks[~is_adv]).relu()
                r_grad = grad(loss.sum(), r_, only_inputs=True)[0]
                r_grad.div_(batch_view(r_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
                r_grad.div_(batch_view(r_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
                r_.data.sub_(r_grad, alpha=γ)
           

                r[active_inputs] = r_
                

            return adv_inputs_ 

    def FGSM_untargeted(self, data, label, eps=2):
        """ FGSM untargeted attack (FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label of the input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        # eps = eps / 255

        loss = nn.CrossEntropyLoss(ignore_index=-1)
        print(self.device)
        # zero gradients
        self.model.zero_grad()

        data['inputs'][0] = (data['inputs'][0].float()).to(self.device)
        data['inputs'][0].requires_grad = True

        pred = self.model_pred(data)

        lo = loss(pred, label.detach())
        lo.backward()
        im_grad = data['inputs'][0].grad
        # Check if gradient exists
        assert(im_grad is not None)
        # print(lo, im_grad)
        print("eps =", eps)
        noise = eps * torch.sign(im_grad)
        adv_img = data['inputs'][0] + noise

        return adv_img, noise


    def FGSM_targeted(self, data, eps=2):
        """ FGSM targeted attack (FGSM ll)
        Args:
            img    (torch.tensor): input image
            eps  (float):          size of adversarial perturbation
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        # eps = eps / 255


        loss = nn.CrossEntropyLoss(ignore_index=-1)

        # zero gradients
        self.model.zero_grad()

        data['inputs'][0] = (data['inputs'][0].float()).to(self.device)
        data['inputs'][0].requires_grad = True

        pred = self.model_pred(data)
        target = (torch.argmin(pred[0],0)).unsqueeze(0)


        lo = loss(pred, target.detach())
        lo.backward()
        im_grad = data['inputs'][0].grad

        # Check if gradient exists
        assert(im_grad is not None)
      
        noise = -eps * torch.sign(im_grad)
        adv_img = data['inputs'][0] + noise

        return adv_img, noise



    def FGSM_untargeted_iterative(self, data, label, alpha=1, eps=2, num_it=None):
        """ FGSM iterative untargeted (I-FGSM)
        Args:
            img    (torch.tensor): input image
            label  (torch.tensor): label
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        data_adv = data
        data_adv['inputs'][0] = (data_adv['inputs'][0].float()).to(self.device)
        data_adv['inputs'][0].requires_grad = True

        tbar=trange(num_it)
        for i in tbar:

            # zero gradients for each iteration
            self.model.zero_grad()

            pred = self.model_pred(data_adv)

            lo = loss(pred, label.detach())
            lo.backward()
            im_grad = data_adv['inputs'][0].grad
            # Check if gradient exists
            assert(im_grad is not None)
            print("eps =", eps)
            noise = (alpha * torch.sign(im_grad)).clamp(-eps,eps)
            data_adv['inputs'][0] = (data_adv['inputs'][0] + noise).clamp(data['inputs'][0]-eps,data['inputs'][0]+eps)
            data_adv['inputs'][0] = Variable(data_adv['inputs'][0], requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM attack'.format(i, num_it))
        return data_adv['inputs'][0], noise


    def FGSM_targeted_iterative(self, data, alpha=1, eps=2, num_it=None):
        """ FGSM iterative least likely class (targeted) attack (I-FGSM ll)
        Args:
            img    (torch.tensor): input image
            alpha  (float):        step size of the attack
            eps    (float):        size of adversarial perturbation
            num_it (int):          number of attack iterations
        Returns:
           adv_img (torch.tensor): pertubed image
           noise   (torch.tensor): adversarial noise
        """

        if num_it == None:
            num_it = min(int(eps+4), int(1.25*eps))
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        with torch.no_grad():
            pred = self.model_pred(data)
            target = ((torch.argmin(pred[0],0)).unsqueeze(0)).detach()

        data_adv = data
        data_adv['inputs'][0] = (data_adv['inputs'][0].float()).to(self.device)
        data_adv['inputs'][0].requires_grad = True

        tbar = trange(num_it)
        for i in tbar:

            # zero gradients for each iteration
            self.model.zero_grad()

            pred = self.model_pred(data_adv)

            # lo = loss(pred, target.detach())
            lo = loss(pred, target)
            lo.backward()
            im_grad = data_adv['inputs'][0].grad
            # Check if gradient exists
            assert(im_grad is not None)
            # print(lo, im_grad)
            print("eps =", eps)
            noise = (-alpha * torch.sign(im_grad)).clamp(-eps,eps)
            data_adv['inputs'][0] = (data_adv['inputs'][0] + noise).clamp(data['inputs'][0]-eps,data['inputs'][0]+eps)
            data_adv['inputs'][0] = Variable(data_adv['inputs'][0], requires_grad=True)

            tbar.set_description('Iteration: {}/{} of I-FGSM ll attack'.format(i, num_it))
        return data_adv['inputs'][0], noise


    ### SSMM & DNNM ###
    
    def universal_adv_pert_static_2(self, train_loader, target_img_name=None, alpha=0.9999, eps=10, num_it=60):
        """ Computes the universal adversarial pertubations (static)
        Args:
            train_loader    (dataloader):   training data
            target_img_name (path):         target image filename
            alpha           (float):        step size of the attack
            eps             (float):        size of adversarial perturbation
            num_it          (int):          number of attack iterations
        Returns:
            noise            (torch.tensor): adversarial noise
        """

        print("eps", eps)
        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        
        print(train_loader[0]["data_samples"][0].img_path.split("/")[-1])
 
        
        for i in range(len(train_loader)):
            if i == 0:
                _, H, W = train_loader[i]["inputs"][0].size()
                #print("H, W: ", H, W)
                if target_img_name == None:
                    target_img = train_loader[i].copy()
                    break
            if target_img_name == train_loader[0]["data_samples"][0].img_path.split("/")[-1]:
                target_img = train_loader[i].clone()
                break
        
        #target_img = train_loader[target_i]
        target_img['inputs'][0] = (target_img['inputs'][0].float()).to(self.device)
        target_img['inputs'][0].requires_grad = True
        with torch.no_grad():
            pred = self.model_pred(target_img)

        target = (torch.argmax(pred[0],0)).unsqueeze(0)


        num_img = len(train_loader)
        noise = torch.zeros(target_img['inputs'][0].size())
        noise = noise.to(self.device)
        print(self.device)



        # num_it = 40
        for it in trange(num_it):

            sum_grads = torch.zeros(noise.size()).to(self.device)
            #img_noise = train_loader.copy()

            for i, data in enumerate(train_loader):
                if i % 100 == 0:
                    print(i)

                # zero gradients for each iteration
                self.model.zero_grad()

                image = data['inputs'][0].clone().detach()
                image = image.float().to(self.device)
                image.requires_grad = True

                def model_pred_with_noise(image, noise):
                    noisy_input = image + noise
                    return self.model_pred({'inputs': [noisy_input]})
                
                pred = model_pred_with_noise(image, noise)

                J_cls = loss(pred, target.detach())
                pred_softmax = torch.softmax(pred[0],0)
                J_cls[0,torch.logical_and(torch.argmax(pred_softmax,0)==target.squeeze(0), torch.max(pred_softmax,0)[0]>0.75)] = 0
                J_ss = torch.sum(J_cls) / (H * W)

                #print("Before backward pass")
                if i == 0:
                    print(f"J_ss: {J_ss}")

                J_ss.backward()
                im_grad = image.grad

                # Check if gradient exists
                assert(im_grad is not None)


                sum_grads = sum_grads + im_grad

            noise = (noise - alpha * torch.sign(sum_grads / num_img)).clamp(-eps,eps)

        return noise
    

    def universal_adv_pert_static(self, train_loader, target_img_name=None, alpha=1, eps=10, num_it=60):
        """ Computes the universal adversarial pertubations (static)
        Args:
            train_loader    (dataloader):   training data
            target_img_name (path):         target image filename
            alpha           (float):        step size of the attack
            eps             (float):        size of adversarial perturbation
            num_it          (int):          number of attack iterations
        Returns:
            noise            (torch.tensor): adversarial noise
        """

        print("eps", eps)
        #alpha = alpha
        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        #print (train_loader[0])
        # create static target
        target_i = 0

        for i in range(len(train_loader)):

            if i == 0:
                _, H, W = train_loader[i]["inputs"][0].size()
                #print("H, W: ", H, W)
                #if target_img_name == None:
                    #target_img = image.clone()
                   # break
            #if target_img_name == filename[0]:
                #target_i = i
                #target_img = image.clone()
                #break

        target_img = train_loader[target_i]
        target_img['inputs'][0] = (target_img['inputs'][0].float()).to(self.device)
        target_img['inputs'][0].requires_grad = True
        with torch.no_grad():
            pred = self.model_pred(target_img)
        target = (torch.argmax(pred[0],0)).unsqueeze(0)




        num_img = len(train_loader)
        noise = torch.zeros(target_img['inputs'][0].size())
        noise = noise.to(self.device)

        #noise.requires_grad = True
        #print(f"noise size {noise.size()}")

        # num_it = 40
        for it in trange(num_it):

            sum_grads = torch.zeros(noise.size()).to(self.device)
            #img_noise = train_loader.copy()

            for i, data in enumerate(train_loader):
                if i % 100:
                    print(i)

                # zero gradients for each iteration
                self.model.zero_grad()

                # if i == 30:
                #     break
                image = data['inputs'][0].clone().detach()
                
                image = image.float().to(self.device)
                
                image.requires_grad = True



                def model_pred_with_noise(image, noise):
                    noisy_input = image + noise
                    return self.model_pred({'inputs': [noisy_input]})



                pred = model_pred_with_noise(image, noise)

                J_cls = loss(pred, target.detach())
                
                pred_softmax = torch.softmax(pred[0],0)
                J_cls[0,torch.logical_and(torch.argmax(pred_softmax,0)==target.squeeze(0), torch.max(pred_softmax,0)[0]>0.75)] = 0

                J_ss = torch.sum(J_cls) / (H * W)


                #print("Before backward pass")
                if i == 5:
                    print(f"J_ss: {J_ss}")


                J_ss.backward()

                im_grad = image.grad


                # Check if gradient exists
                assert(im_grad is not None)



                sum_grads = sum_grads + im_grad

              

            noise = (noise - alpha * torch.sign(sum_grads / num_img)).clamp(-eps,eps)
            

        print("Total training samples: {:d}".format(num_img))


        return noise

    def universal_adv_pert_dynamic(self, train_loader, labels, class_id=11, alpha=1, eps=10, num_it=60):
        """ Computes the universal adversarial pertubations (dynamic)
        Args:
            train_loader (dataloader):   training data
            class_id     (int):          removing class id
            alpha        (float):        step size of the attack
            eps          (float):        clipping value for iterative attacks
            num_it       (int):          number of attack iterations
        Returns:
        noise         (torch.tensor): adversarial noise
        """

        # eps = eps / 255
        # alpha = alpha / 255
        loss = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        #for i, (image, label, filename) in enumerate(train_loader):
            #_, _, H, W = image.size()
            #break
        for i in range(len(train_loader)):
            _, H, W = train_loader[i]["inputs"][0].size()
            #print(len(train_loader[i]["inputs"]))
            break


        num_img = 0

        #noise = torch.zeros(image.size()
        noise = torch.zeros(train_loader[0]['inputs'][0].size())
        noise = noise.to(self.device)
        preds_all = torch.zeros((len(train_loader), H, W), dtype=torch.uint8).to(self.device)
        targets_all = torch.zeros((len(train_loader), H, W), dtype=torch.long).to(self.device)

        # num_it = 40
        for it in trange(num_it):
            print(it)

            sum_grads = torch.zeros(noise.size()).to(self.device)

            for i, data in enumerate(train_loader):
                
                print(f"it {it}, image munber {i}")

                # use image for training if class_id is included
                if class_id in torch.unique(labels[i]):

                    # zero gradients for each iteration
                    self.model.zero_grad()

                    data['inputs'][0] = (data['inputs'][0].float()).to(self.device)
                    data['inputs'][0].requires_grad = True


                    # create dynamic target for each image ones
                    if it == 0:
                        # print(i, filename)
                        num_img = num_img + 1

                        # save memory: save not in first interation, instead run every interation
                        with torch.no_grad():
                            pred1 = self.model_pred(data)
                        pred_orig = (torch.argmax(pred1[0],0))
                        preds_all[i] = pred_orig.clone()
                        pred_orig_np = pred_orig.cpu().data.numpy()
                        pred_orig_np[pred_orig_np==class_id] = -2
                        mask = np.where(~(pred_orig_np==-2))
                        interp = NearestNDInterpolator(np.transpose(mask), pred_orig_np[mask])
                        pred_filled_np = interp(*np.indices(pred_orig_np.shape))
                        targets_all[i] = torch.from_numpy(pred_filled_np) #.unsqueeze(0) #.to(self.device)

      
                    def model_pred_with_noise(data, noise):
                        noisy_input = data['inputs'][0] + noise
                        
            
                        pred = self.model_pred({'inputs': [noisy_input]})
                      
                        return pred
                    
                    pred = model_pred_with_noise(data, noise)

                    J_cls = loss(pred, targets_all[i].unsqueeze(0).detach())

                    # loss of target pixels predicted as deesired class
                    # with confidence aboцve tau=0.75 is set to 0
                    pred_softmax = torch.softmax(pred[0],0)
                    J_cls[0,torch.logical_and(torch.argmax(pred_softmax,0)==targets_all[i], torch.max(pred_softmax,0)[0]>0.75)] = 0

                    w = 0.9999 # 0.9
                    J_ss_w = (w * torch.sum(J_cls[0,preds_all[i]==class_id]) + (1-w) * torch.sum(J_cls[0,preds_all[i]!=class_id])) / (H * W)

                    J_ss_w.backward()
                    im_grad = data['inputs'][0].grad
                    # Check if gradient exists
                    assert(im_grad is not None)

                    sum_grads = sum_grads + im_grad

            noise = (noise - alpha * torch.sign(sum_grads / num_img)).clamp(-eps,eps)
            
        print("Total training samples: {:d}".format(num_img))

        return noise

