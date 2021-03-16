import torch

from .models.adversarial import ProjectedGradientDescent, ManipulationMethod, IterativeAttack
from .models.saliency import RelEx, RealTimeSaliency, GradCAM, SmoothGrad, IntegratedGradient
from .models.network import resnet50
from .options import get_opts

r'''
This script performs to generate Adversarial example and Saliency.
'''

'''
NaturalGenerator generates Adversarial example and Saliency based on pretrained-model
pretrained-model is torchvison.models.resnet50(pretrained=True)
'''


class NaturalGenerator:
    def __init__(self, adv_method_names, sal_method_names, opts):
        self.adv_method_names = adv_method_names
        self.sal_method_names = sal_method_names

        net = load_net(opts.network, encoder=False, robust=False)
        rts_net = load_net(opts.network, encoder=True, robust=False)

        self.opts = opts

        self.pgd_generator = ProjectedGradientDescent(net,
                                                      eps=opts.pgd.eps,
                                                      a=opts.pgd.a,
                                                      K=opts.pgd.K,
                                                      norm=opts.pgd.norm,
                                                      max_min_bounds=opts.pgd.max_min_bounds)
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
        return adv_x_sets + sal_sets

    def generate_adversarial(self, x):
        # TODO
        # fit sal_methods by attack method

        # Untargeted
        pgd_adv_x_sets = []
        for pgd_eps in self.opts.untargeted.eps_sets:
            pgd_adv_x = PGD_generator(x.detach().clone(), pgd_eps)
            pgd_adv_x_sets.append(pgd_adv_x.detach())

        # Targeted, Structured
        man_adv_x_sets = []
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.structured.attack_methods:
                man_adv_x = structured_generator(
                    x.detach().clone(), self.target_x, sal_method)
                man_adv_x_sets.append(man_adv_x)

        # Targeted, Unstructured
        iter_att_adv_x_sets = []
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.unstructured.attack_methods:
                for iter_att_eps in self.opts.unstructured.eps_sets:
                    iter_att_adv_x = unstructured_generator(
                        x.detach().clone(), sal_method)
                    iter_att_adv_x_sets.append(iter_att_adv_x.detach())

    def generate_saliency(self, x):
        sal_sets = []
        for sal_method in self.sal_methods:
            sal = sal_method(x)[0]
            sal_sets.append(sal.detach())


'''
RobustGenerator generates Adversarial example and Saliency based on adversarially trained model
adversarially trained model is trained by PGD
'''


class RobustGenerator:
    def __init__(self, adv_method_names, sal_method_names):
        self.adv_method_names = adv_method_names
        self.sal_method_names = sal_method_names

        self.adv_methods = []
        self.sal_methods = []

        net = load_net(opts.network, encoder=False, robust=True)

        for adv_method_name in self.adv_method_names:
            pass

        for sal_method_name in self.sal_method_names:
            pass

    def __call__(self, x):
        adv_x_sets = self.generate_adversarial(x)
        sal_sets = self.generate_saliency(x)
        return adv_x_sets + sal_sets

    def generate_adversarial(self, x):
        pass

    def generate_saliency(self, x):
        pass


if __name__ == '__main__':
    opts = get_opts()
    rts_net = resnet50(encoder=True, robust=False)
    net = resnet50(encoder=False, robust=False)
