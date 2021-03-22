import os
import torch
from torch.utils.data import Dataset
import collections

from models.network import load_network
from utils import load_image
from utils.evaluations import Robustness, Fieldility
from utils.process import normalize

from options import get_opts


if __name__ == '__main__':
    '''

    '''
    opts = get_opts()
    if opts.robust:
        full_network_name = 'Robust-ResNet50'
    else:
        full_network_name = 'Natural-ResNet50'

    workspace_dir = os.path.dirname(__file__)
    data_root_dir = os.path.join(workspace_dir, 'data')
    results_root_dir = os.path.join(workspace_dir, 'data')

    orig_x_name = 'ILSVRC2012_val_00023552'
    orig_x_full_dir = os.path.join(data_root_dir, orig_x_name + '.JPEG')
    orig_x = load_image(orig_x_full_dir, gpu=opts.gpu)[0]

    net = load_network(opts.network, encoder=False, robust=opts.robust)
    retrieval_rate = RetrievalRate(net)
    robustness = Robustness(device='cuda')
    fieldility = Fieldility(net)

    '''
    Load Saliency and Adversarial
    '''
    ############################################################################
    # Original Saliency
    orig_sal_sets = {}
    orig_sal_list = []
    sal_root_dir = os.path.join(
        results_root_dir, 'saliency', full_network_name)
    sal_method_names = os.listdir(sal_root_dir)
    for sal_method_name in sal_method_names:
        sal_full_dir = os.path.join(sal_root_dir, sal_method_name,
                                    'original', orig_x_name + '.pt')
        orig_sal = torch.load(sal_full_dir)
        orig_sal_sets[sal_method_name] = orig_sal
        orig_sal_list.append(orig_sal)
    total_orig_sal = torch.cat(orig_sal_list)
    ############################################################################
    ############################################################################
    # Untargeted, PGD
    pgd_adv_x_list = []
    pgd_root_dir = os.path.join(
        results_root_dir, 'adversarial', full_network_name, 'PGD')
    for pgd_eps in opts.untargeted.eps_sets:
        pgd_adv_x_full_dir = os.path.join(
            pgd_root_dir, f'eps{pgd_eps}', orig_x_name + '.pt')
        pgd_adv_x = torch.load(pgd_adv_x_full_dir)
        pgd_adv_x_list.append(pgd_adv_x)
    total_pgd_adv_x = torch.cat(pgd_adv_x_list)

    # Targeted, Structured, ManipulationMethod
    structured_adv_x_set = {}
    structured_root_dir = os.path.join(
        results_root_dir, 'adversarial', full_network_name, 'Structured')
    structured_att_methods = os.listdir(structured_root_dir)
    for structured_att_method in structured_att_methods:
        structured_adv_x_full_dir = os.path.join(
            structured_root_dir, structured_att_method, orig_x_name + '.pt')
        structured_adv_x = torch.load(structured_adv_x_full_dir)
        structured_adv_x_set[structured_att_method] = structured_adv_x

    # Targeted, Unstructured, IterativeAttack
    # unstructured_adv_x_sets = {}
    unstructured_adv_x_sets = collections.defaultdict(dict)
    unstructured_root_dir = os.path.join(
        results_root_dir, 'adversarial', full_network_name, 'Unstructured')
    unstructured_att_methods = os.listdir(unstructured_root_dir)
    for unstructured_att_method in unstructured_att_methods:
        unstructured_adv_x_list = []

        for unstructured_eps in opts.unstructured.eps_sets:
            unstructured_adv_x_full_dir = os.path.join(
                unstructured_root_dir, unstructured_att_method,
                f'eps{unstructured_eps}', orig_x_name + '.pt')
            unstructured_adv_x = torch.load(unstructured_adv_x_full_dir)
            unstructured_adv_x_list.append(x)

        total_unstructured_adv_x = torch.cat(unstructured_adv_x_list)
        unstructured_adv_x_sets[opts.unstructured.method][unstructured_att_method] = total_unstructured_adv_x
    ############################################################################
    ############################################################################
    # Adversarial saliency
    adv_sal_sets = collections.defaultdict(dict)
    for def_sal_method_name in sal_method_names:
        # PGD
        pgd_adv_sal_list = []
        for pgd_eps in opts.untargeted.eps_sets:
            adv_sal_full_dir = os.path.join(
                sal_root_dir, def_sal_method_name, 'PGD', pgd_eps,
                orig_x_name + '.pt')
            pgd_adv_sal = torch.load(adv_sal_full_dir)
            pgd_adv_sal_list.append(pgd_adv_sal)
        total_pgd_adv_sal = torch.cat(pgd_adv_sal_list)
        adv_sal_sets[def_sal_method_name]['PGD'] = total_pgd_adv_sal

        # Structured
        structured_adv_sal_set = {}
        for att_sal_method_name in structured_att_methods:
            adv_sal_full_dir = os.path.join(
                sal_root_dir, def_sal_method_name, 'Structured',
                att_sal_method_name, orig_x_name + '.pt')
            structured_adv_sal = torch.load(adv_sal_full_dir)
            structured_adv_sal_set[att_sal_method_name] = structured_adv_sal
        adv_sal_sets[def_sal_method_name]['Structured'] = structured_adv_sal_set

        # Unstructured
        unstructured_adv_sal_sets = collections.defaultdict(dict)
        for att_sal_method_name in unstructured_att_methods:
            unstructured_adv_sal_list = []
            for unstructured_eps in opts.unstructured.eps_sets:
                adv_sal_full_dir = os.path.join(sal_root_dir, def_sal_method_name,
                                                'Unstructured',
                                                opts.unstructured.method,
                                                att_sal_method_name,
                                                unstructured_eps,
                                                orig_x_name + '.pt')
                unstructured_adv_sal = torch.load(adv_sal_full_dir)
                unstructured_adv_sal_list.append(unstructured_adv_sal)
            total_unstructured_adv_sal = torch.cat(unstructured_adv_sal_list)
            unstructured_adv_sal_sets['topk'][att_sal_method_name] = total_unstructured_adv_sal
        adv_sal_sets[def_sal_method_name]['Unstructured'] = unstructured_adv_sal_sets
    ############################################################################
    '''
    Retrieval Rate
    '''
    for sal_method_name, orig_sal in orig_sal_sets.items():
        print('PGD', sal_method_name)
        pgd_adv_sals = adv_sal_sets[sal_method_name]['PGD']

        orig_sal_norm = normalize(orig_sal.clone(), sal_method_name)
        pgd_adv_sals_norm = normalize(pgd_adv_sals.clone(), sal_method_name)

        retrieval_rate_outputs = retrieval_rate(
            orig_x, total_pgd_adv_x, orig_sal_norm, pgd_adv_sals_norm)
        for input_type, retrieval_rate_output in retrieval_rate_outputs.items():
            print(input_type, retrieval_rate_output)

        print('Structured', sal_method_name)
        structured_adv_sal_set = adv_sal_sets[sal_method_name]['Structured']
        for att_sal_method_name, structured_adv_x in structured_adv_x_set.items():
            structured_adv_sal = structured_adv_sal_set[att_sal_method_name]
            structured_adv_sal_norm = normalize(
                structured_adv_sal.clone(), sal_method_name)
            retrieval_rate_outputs = retrieval_rate(
                orig_x, structured_adv_x, orig_sal_norm, structured_adv_sal_norm)

            for input_type, retrieval_rate_output in retrieval_rate_outputs.items():
                print(input_type, retrieval_rate_output)

        print('Unstructured', sal_method_name)
        unstructured_adv_sal_sets = adv_sal_sets[sal_method_name]['Unstructured']
        for unstructured_method, unstructured_adv_sal_set in unstructured_adv_sal_sets.items():
            for att_sal_method_name, unstructured_adv_sals in unstructured_adv_sal_set.items():
                total_unstructured_adv_x = unstructured_adv_x_sets[
                    unstructured_method][att_sal_method_name]
                unstructured_adv_sals_norm = normalize(
                    unstructured_adv_sals.clone(), sal_method_name)
                retrieval_rate_outputs = retrieval_rate(orig_x, total_unstructured_adv_x,
                                                        orig_sal_norm, unstructured_adv_sals_norm)
                for input_type, retrieval_rate_output in retrieval_rate_outputs.items():
                    print(input_type, retrieval_rate_output)
    '''
    Robustness
    '''
    for sal_method_name, orig_sal in orig_sal_sets.items():
        print('PGD', sal_method_name)
        pgd_adv_sals = adv_sal_sets[sal_method_name]['PGD']
        robustness_outputs = robustness(orig_sal, pgd_adv_sals)
        print('top-k, intersection', robustness_outputs[0])
        print('spearman correlation', robustness_outputs[1])

    for sal_method_name, orig_sal in orig_sal_sets.items():
        print('Structured', sal_method_name)
        structured_adv_sal_set = adv_sal_sets[sal_method_name]['Structured']
        for att_method_name, structured_adv_sal in structured_adv_sal_set.items():
            robustness_outputs = robustness(orig_sal, structured_adv_sal)
            print('top-k, intersection', robustness_outputs[0])
            print('spearman correlation', robustness_outputs[1])

    for sal_method_name, orig_sal in orig_sal_sets.items():
        print('Unstructured', sal_method_name)
        unstructured_adv_sal_sets = adv_sal_sets[sal_method_name]['Unstructured']
        for unstructured_method, unstructured_adv_sal_set in unstructured_adv_sal_sets.items():
            for att_sal_method_name, unstructured_adv_sals in unstructured_adv_sal_set.items():
                robustness_outpust = robustness(
                    orig_sal, unstructured_adv_sals)
                print('top-k, intersection', robustness_outpust[0])
                print('spearman correlation', robustness_outpust[1])

    '''
    Fieldility
    '''
    for sal_method_name, orig_sal in orig_sal_sets.items():
        print('PGD', sal_method_name)
        pgd_adv_sals = adv_sal_sets[sal_method_name]['PGD']
        field_outpus = fieldility(orig_x, torch.cat([orig_sal, pgd_adv_sals]))
        print('Deletion', field_outpus[0])
        print('Preservation', field_outpus[1])
