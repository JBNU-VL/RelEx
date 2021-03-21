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


class NaturalGenerator:
    '''
    NaturalGenerator generates Adversarial example and Saliency based on pretrained-model
    pretrained-model is torchvison.models.resnet50(pretrained=True)
    '''

    def __init__(self, opts):
        net = load_network(opts.network, encoder=False, robust=False)
        encoder_net = load_network(opts.network, encoder=True, robust=False)

        if opts.gpu:
            device = torch.device('cuda')
            net = DataParallel(net).to(device)
            encoder_net = DataParallel(encoder_net).to(device)
        else:
            device = torch.device('cpu')

        self.opts = opts

        self.pgd_generator = ProjectedGradientDescent(net,
                                                      eps=opts.untargeted.eps,
                                                      a=opts.untargeted.a,
                                                      K=opts.untargeted.K,
                                                      norm=opts.untargeted.norm,
                                                      max_min_bounds=(
                                                          opts.max_bound.max().item(),
                                                          opts.min_bound.min().item()
                                                      ))
        self.structured_generator = ManipulationMethod(lr=opts.structured.lr,
                                                       num_iters=opts.structured.num_iters,
                                                       factors=opts.structured.factors,
                                                       beta_range=opts.structured.beta_range,
                                                       x_max_min_bounds=[
                                                           opts.max_bound,
                                                           opts.min_bound
                                                       ],
                                                       device=device)
        workspace_dir = os.path.dirname(__file__)
        img_name = self.opts.structured.target_x_name
        img_full_dir = os.path.join(workspace_dir, 'data', img_name)
        self.target_x = load_image(img_full_dir, gpu=opts.gpu)[0]

        self.unstructured_generator = IterativeAttack(method=opts.unstructured.method,
                                                      eps=opts.unstructured.eps,
                                                      k=opts.unstructured.k,
                                                      num_iters=opts.unstructured.num_iters,
                                                      alpha=opts.unstructured.alpha,
                                                      measurement=opts.unstructured.measurement,
                                                      beta_growth=opts.unstructured.beta_growth,
                                                      x_max_min_bounds=[
                                                          opts.max_bound,
                                                          opts.min_bound
                                                      ],
                                                      beta_range=opts.unstructured.beta_range,
                                                      device=device)

        relex_method = RelEx(net,
                             shape=opts.x_shape,
                             batch_size=opts.relex.batch_size,
                             lr=opts.relex.lr,
                             mtm=opts.relex.mtm,
                             x_std_level=opts.relex.x_std_level,
                             max_iters=opts.relex.max_iters,
                             lambda1=opts.relex.lambda1,
                             lambda2=opts.relex.lambda2,
                             mode=opts.relex.mode,
                             device=device)
        rt_sal_method = RealTimeSaliency(encoder_net,
                                         model_confidence=opts.realtime_sal.model_confidence,
                                         device=device)
        smgrad_method = SmoothGrad(net,
                                   shape=opts.x_shape,
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
        # generate adversarial and saliency about orig x
        orig_sal_sets = self.generate_sal(x)
        adv_x_sets = self.generate_adv(x)
        adv_sal_sets = self.generate_adv_sal(adv_x_sets)

        return adv_x_sets, orig_sal_sets, adv_sal_sets

    def generate_adv(self, x):
        '''
        Args: 
            x: orig x (orignal Image)

        Returns:
            pgd_adv_x_sets: Untargeted attack against x
            man_adv_x_sets: Structured attack against x
            iter_att_adv_x_sets: Unstructured attack against x
        '''
        print('generate adversarial')
        # Untargeted
        pgd_adv_x_sets = {}
        for pgd_eps in self.opts.untargeted.eps_sets:
            pgd_adv_x = self.pgd_generator(x.detach().clone(),
                                           eps=pgd_eps)
            pgd_adv_x_sets[f'eps{pgd_eps}'] = pgd_adv_x.detach()

        # Targeted, Structured
        man_adv_x_sets = {}
        for sal_method in self.sal_methods:
            # it will be moved to Class method
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.structured.attack_methods:
                man_adv_x = self.structured_generator(
                    x.detach().clone(),
                    target_x=self.target_x,
                    sal_method=sal_method)[0]
                man_adv_x_sets[sal_method_name] = man_adv_x.detach()

        # Targeted, Unstructured
        iter_att_adv_x_sets = collections.defaultdict(dict)
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.unstructured.attack_methods:
                for idx, iter_att_eps in enumerate(self.opts.unstructured.eps_sets):
                    scaled_iter_att_eps = self.opts.unstructured.scaled_eps_sets[idx]
                    iter_att_adv_x = self.unstructured_generator(
                        x.detach().clone(),
                        eps=scaled_iter_att_eps,
                        sal_method=sal_method)[0]
                    iter_att_adv_x_sets[sal_method_name][f'eps{iter_att_eps}'] = iter_att_adv_x.detach(
                    )

        return pgd_adv_x_sets, man_adv_x_sets, iter_att_adv_x_sets

    def generate_sal(self, x):
        '''
        Args:
        Returns:
        '''

        print('generate saliency')
        sal_sets = {}
        for sal_method in self.sal_methods:
            sal = sal_method(x)[0]
            sal_method_name = sal_method.__class__.__name__
            sal_sets[sal_method_name] = sal.detach()
        return sal_sets

    def generate_adv_sal(self, adv_x_sets):
        '''
        Args:
            adv_x_sets: Tuple that cosists of 3 elements.
                        Adversarial example by PGD, Structured and Unstructured

        Returns:
        '''
        print('generate adversarial saliency')
        pgd_adv_x_sets, man_adv_x_sets, iter_att_adv_x_sets = adv_x_sets

        # generate saliency based on adversarial example
        # Untargeted, PGD
        pgd_adv_sal_sets = {}
        for pgd_eps in self.opts.untargeted.eps_sets:
            pgd_adv_x = pgd_adv_x_sets[f'eps{pgd_eps}'].clone()
            pgd_adv_sal_set = self.generate_sal(pgd_adv_x)
            pgd_adv_sal_sets[f'eps{pgd_eps}'] = pgd_adv_sal_set

        # Targeted, Structured
        man_adv_sal_sets = {}
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.structured.attack_methods:
                man_adv_x = man_adv_x_sets[sal_method_name].clone()
                man_adv_sal_set = self.generate_sal(man_adv_x)
                man_adv_sal_sets[sal_method_name] = man_adv_sal_set

        # Targeted, Unstructured
        iter_att_adv_sal_sets = collections.defaultdict(dict)
        for sal_method in self.sal_methods:
            sal_method_name = sal_method.__class__.__name__

            if sal_method_name in self.opts.unstructured.attack_methods:
                for idx, iter_att_eps in enumerate(self.opts.unstructured.eps_sets):
                    scaled_iter_att_eps = self.opts.unstructured.scaled_eps_sets[idx]
                    iter_att_adv_x = iter_att_adv_x_sets[sal_method_name][f'eps{iter_att_eps}'].clone(
                    )
                    iter_att_adv_sal_set = self.generate_sal(iter_att_adv_x)
                    iter_att_adv_sal_sets[sal_method_name][f'eps{iter_att_eps}'] = iter_att_adv_sal_set

        return pgd_adv_sal_sets, man_adv_sal_sets, iter_att_adv_sal_sets


class RobustGenerator:
    '''
    RobustGenerator generates Adversarial example and Saliency based on adversarially 
    trained model adversarially trained model is trained by PGD
    '''

    def __init__(self, opts):
        net = load_network(opts.network, encoder=False, robust=True)
        if opts.gpu:
            device = torch.device("cuda")
            net = DataParallel(net).to(0)
        else:
            device = torch.device("cpu")

        self.opts = opts

        self.pgd_generator = ProjectedGradientDescent(net,
                                                      eps=opts.untargeted.eps,
                                                      a=opts.untargeted.a,
                                                      K=opts.untargeted.K,
                                                      norm=opts.untargeted.norm,
                                                      max_min_bounds=(
                                                          opts.max_bound.max().item(),
                                                          opts.min_bound.min().item()
                                                      ))
        self.structured_generator = ManipulationMethod(lr=opts.structured.lr,
                                                       num_iters=opts.structured.num_iters,
                                                       factors=opts.structured.factors,
                                                       beta_range=opts.structured.beta_range,
                                                       x_max_min_bounds=[
                                                           opts.max_bound,
                                                           opts.min_bound
                                                       ],
                                                       device=device)
        self.unstructured_generator = IterativeAttack(method=opts.unstructured.method,
                                                      eps=opts.unstructured.eps,
                                                      k=opts.unstructured.k,
                                                      num_iters=opts.unstructured.num_iters,
                                                      alpha=opts.unstructured.alpha,
                                                      measurement=opts.unstructured.measurement,
                                                      beta_growth=opts.unstructured.beta_growth,
                                                      x_max_min_bounds=[
                                                          opts.max_bound,
                                                          opts.min_bound
                                                      ],
                                                      beta_range=opts.unstructured.beta_range,
                                                      device=device)

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
        adv_x_sets = self.generate_adv(x)
        sal_sets = self.generate_sal(x)
        adv_sal_sets = self.generate_adv_sal(adv_x_sets)

        return adv_x_sets, sal_sets, adv_sal_sets

    def generate_adv(self, x):
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

    def generate_sal(self, x):
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
    x = load_image(img_full_dir, gpu=opts.gpu)[0]

    import time
    start = time.time()
    ############################################################################
    results = natural_generator(x)
    ############################################################################
    torch.cuda.synchronize()  # Waitting for GPU Tasks
    end = time.time() - start
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end))
    print(f'Generating elapsed Time | {elapsed_time}')

    adv_x_sets, sal_sets, adv_sal_sets = results

    print('Output Test')
    # Adversarial
    print('Adversarial')
    pgd_adv_x_sets, man_adv_x_sets, iter_att_adv_x_sets = adv_x_sets
    print('PGD')
    for eps_name, adv_x in pgd_adv_x_sets.items():
        print(eps_name, adv_x.size())
    print('Manipulation')
    for sal_method_name, adv_x in man_adv_x_sets.items():
        print(sal_method_name, adv_x.size())
    print('IterativeAttack')
    for sal_method_name, inner_dict in iter_att_adv_x_sets.items():
        for eps_name, adv_x in inner_dict.items():
            print(sal_method_name, eps_name, adv_x.size())

    # Saliency
    print('Saliency')
    for sal_method_name, sal in sal_sets.items():
        print(sal_method_name, sal.size())

    # Adversarial Saliency
    pgd_adv_sal_sets, man_adv_sal_sets, iter_att_adv_sal_sets = adv_sal_sets
    print('PGD Saliency')
    for eps_name, adv_sal_sets in pgd_adv_sal_sets.items():
        for defense_sal_method_name, adv_sal in adv_sal_sets.items():
            print(eps_name, defense_sal_method_name, adv_sal.size())

    print('Manipulation Saliency')
    for attack_sal_method_name, adv_sal_sets in man_adv_sal_sets.items():
        for defense_sal_method_name, adv_sal in adv_sal_sets.items():
            print(attack_sal_method_name, defense_sal_method_name, adv_sal.size())

    print('IterativeAttack Saliency')
    for att_sal_method_name, inner_dict in iter_att_adv_sal_sets.items():
        for eps_name, adv_sal_sets in inner_dict.items():
            for def_sal_method, adv_sal in adv_sal_sets.items():
                print(att_sal_method_name, def_sal_method,
                      eps_name, adv_sal.size())
