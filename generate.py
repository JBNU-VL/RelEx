import torch
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import collections

from models.adversarial import ProjectedGradientDescent, ManipulationMethod, IterativeAttack
from models.saliency import RelEx, RealTimeSaliency, GradCAM, SmoothGrad, IntegratedGradient
from models.network import load_network
from utils import load_image
from options import get_opts

r'''
This script performs to generate Adversarial example and Saliency.
'''

'''
NaturalGenerator generates Adversarial example and Saliency based on pretrained-model
pretrained-model is torchvison.models.resnet50(pretrained=True)
'''


class NaturalGenerator:
    def __init__(self, opts):
        net = load_network(opts.network, encoder=False, robust=False)
        rts_net = load_network(opts.network, encoder=True, robust=False)
        if opts.gpu:
            net = DataParallel(net).to(0)
            rts_net = DataParallel(net).to(0)

        self.opts = opts

        self.pgd_generator = ProjectedGradientDescent(net,
                                                      eps=opts.untargeted.eps,
                                                      a=opts.untargeted.a,
                                                      K=opts.untargeted.K,
                                                      norm=opts.untargeted.norm,
                                                      max_min_bounds=opts.untargeted.max_min_bounds)
        self.structured_generator = ManipulationMethod(lr=opts.structured.lr,
                                                       num_iters=opts.structured.num_iters,
                                                       factors=opts.structured.factors,
                                                       beta_range=opts.structured.beta_range,
                                                       x_max_min_bounds=(
                                                           opts.max_bound, opts.min_bound)
                                                       )
        self.unstructured_generator = IterativeAttack(method=opts.unstructured.method,
                                                      eps=opts.unstructured.eps,
                                                      k=opts.unstructured.k,
                                                      num_iters=opts.unstructured.num_iters,
                                                      alpha=opts.unstructured.alpha,
                                                      measurement=opts.unstructured.measurement,
                                                      beta_growth=opts.unstructured.beta_growth,
                                                      x_bounds=opts.unstructured.x_bounds,
                                                      beta_range=opts.unstructured.beta_range)

        relex_method = RelEx(net,
                             shape=opts.relex.shape,
                             batch_size=opts.relex.batch_size,
                             lr=opts.relex.lr,
                             mtm=opts.relex.mtm,
                             x_std_level=opts.relex.x_std_level,
                             max_iters=opts.relex.max_iters,
                             lambda1=opts.relex.lambda1,
                             lambda2=opts.relex.lambda2,
                             mode=opts.relex.mode,
                             device=device)
        rt_sal_method = RealTimeSaliency(rts_net,
                                         model_confidence=opts.realtime_sal.model_confidence,
                                         device=device)
        smgrad_method = SmoothGrad(net,
                                   shape=opts.smoothgrad.shape,
                                   sample_size=opts.smoothgrad.sample_size,
                                   std_level=opts.smoothgrad.std_level,
                                   device=device)
        intgrad_method = IntegratedGradient(net,
                                            steps=opts.intgrad.steps,
                                            device=device)
        gradcam_method = GradCAM(net,
                                 target_layers=opts.gradcam.target_layers,
                                 resize=opts.gradcam.resize)
        self.sal_methods = [
            relex_method, rt_sal_method, smgrad_method, intgrad_method,
            gradcam_method
        ]

    def __call__(self, x):
        adv_x_sets = self.generate_adversarial(x)
        sal_sets = self.generate_saliency(x)
        return adv_x_sets, sal_sets

    def generate_adversarial(self, x):
        # Untargeted
        pgd_adv_x_sets = {}
        for pgd_eps in self.opts.untargeted.eps_sets:
            pgd_adv_x = PGD_generator(x.detach().clone(), pgd_eps)
            pgd_adv_x_sets[f'eps{pgd_eps}'] = pgd_adv_x.detach()

        # Targeted, Structured
        man_adv_x_sets = {}
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.structured.attack_methods:
                man_adv_x = structured_generator(
                    x.detach().clone(), self.target_x, sal_method)[0]
                man_adv_x_sets[sal_method_name] = man_adv_x.detach()

        # Targeted, Unstructured
        iter_att_adv_x_sets = collections.defaultdict({})
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.unstructured.attack_methods:
                for idx, iter_att_eps in enumerate(self.opts.unstructured.eps_sets):
                    scaled_iter_att_eps = self.opts.unstructured.scaled_eps_sets[idx]
                    iter_att_adv_x = unstructured_generator(
                        x.detach().clone(), scaled_iter_att_eps, sal_method)[0]
                    iter_att_adv_x_sets[sal_method_name][f'eps{iter_att_eps}'] = iter_att_adv_x.detach(
                    )

        return pgd_adv_x_sets, man_adv_x_sets, iter_att_adv_x_sets

    def generate_saliency(self, x):
        sal_sets = []
        for sal_method in self.sal_methods:
            sal = sal_method(x)[0]
            sal_sets.append(sal.detach())
        return sal_sets


'''
RobustGenerator generates Adversarial example and Saliency based on adversarially trained model
adversarially trained model is trained by PGD
'''


class RobustGenerator:
    def __init__(self, opts):
        net = load_network(opts.network, encoder=False, robust=True)
        if opts.gpu:
            net = DataParallel(net).to(0)

        self.opts = opts

        self.pgd_generator = ProjectedGradientDescent(net,
                                                      eps=opts.untargeted.eps,
                                                      a=opts.untargeted.a,
                                                      K=opts.untargeted.K,
                                                      norm=opts.untargeted.norm,
                                                      max_min_bounds=opts.untargeted.max_min_bounds)
        self.structured_generator = ManipulationMethod(lr=opts.structured.lr,
                                                       num_iters=opts.structured.num_iters,
                                                       factors=opts.structured.factors,
                                                       beta_range=opts.structured.beta_range,
                                                       bounds=opts.structured.bounds)
        self.unstructured_generator = IterativeAttack(method=opts.unstructured.method,
                                                      eps=opts.unstructured.eps,
                                                      k=opts.unstructured.k,
                                                      num_iters=opts.unstructured.num_iters,
                                                      alpha=opts.unstructured.alpha,
                                                      measurement=opts.unstructured.measurement,
                                                      beta_growth=opts.unstructured.beta_growth,
                                                      x_bounds=opts.unstructured.x_bounds,
                                                      beta_range=opts.unstructured.beta_range)

        relex_method = RelEx(net,
                             shape=opts.relex.shape,
                             batch_size=opts.relex.batch_size,
                             lr=opts.relex.lr,
                             mtm=opts.relex.mtm,
                             x_std_level=opts.relex.x_std_level,
                             max_iters=opts.relex.max_iters,
                             lambda1=opts.relex.lambda1,
                             lambda2=opts.relex.lambda2,
                             mode=opts.relex.mode,
                             device=device)
        smgrad_method = SmoothGrad(net,
                                   shape=opts.smoothgrad.shape,
                                   sample_size=opts.smoothgrad.sample_size,
                                   std_level=opts.smoothgrad.std_level,
                                   device=device)
        intgrad_method = IntegratedGradient(net,
                                            steps=opts.intgrad.steps,
                                            device=device)
        gradcam_method = GradCAM(net,
                                 target_layers=opts.gradcam.target_layers,
                                 resize=opts.gradcam.resize)
        self.sal_methods = [
            relex_method, smgrad_method, intgrad_method, gradcam_method
        ]

    def __call__(self, x):
        adv_x_sets = self.generate_adversarial(x)
        sal_sets = self.generate_saliency(x)
        return adv_x_sets, sal_sets

    def generate_adversarial(self, x):
        # Untargeted
        pgd_adv_x_sets = {}
        for pgd_eps in self.opts.untargeted.eps_sets:
            pgd_adv_x = PGD_generator(x.detach().clone(), pgd_eps)
            pgd_adv_x_sets[f'eps{pgd_eps}'] = pgd_adv_x.detach()

        # Targeted, Structured
        man_adv_x_sets = {}
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.structured.attack_methods:
                man_adv_x = structured_generator(
                    x.detach().clone(), self.target_x, sal_method)[0]
                man_adv_x_sets[sal_method_name] = man_adv_x.detach()

        # Targeted, Unstructured
        iter_att_adv_x_sets = collections.defaultdict({})
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.unstructured.attack_methods:
                for idx, iter_att_eps in enumerate(self.opts.unstructured.eps_sets):
                    scaled_iter_att_eps = self.opts.unstructured.scaled_eps_sets[idx]
                    iter_att_adv_x = unstructured_generator(
                        x.detach().clone(), scaled_iter_att_eps, sal_method)[0]
                    iter_att_adv_x_sets[sal_method_name][f'eps{iter_att_eps}'] = iter_att_adv_x.detach(
                    )

        return pgd_adv_x_sets, man_adv_x_sets, iter_att_adv_x_sets

    def generate_saliency(self, x):
        sal_sets = []
        for sal_method in self.sal_methods:
            sal = sal_method(x)[0]
            sal_sets.append(sal.detach())
        return sal_sets


if __name__ == '__main__':
    import os
    opts = get_opts()

    if opts.gpu:
        cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpus

    natural_generator = NaturalGenerator(opts)

    workspace_dir = os.path.dirname(__file__)
    img_name = 'ILSVRC2012_val_00023552.JPEG'
    img_full_dir = os.path.join(workspace_dir, 'data', img_name)
    x = load_image(img_full_dir)
    print(x.size())
    exit()

    results = natural_generator(x)
    adv_x_sets, sal_sets = results
