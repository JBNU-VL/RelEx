import os
import torch

DATA_ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) + '/results'


def make_dir(dirname):
    try:
        if not(os.path.isdir(dirname)):
            os.makedirs(dirname)
    except OSError:
        print(f'Failed Create Your Directory : {dirname}')


def save_results(results):
    natural_robust_flags = list(results.keys())[:-2]
    results_values = list(results.values())
    network_name = results_values[-2]
    img_name = results_values[-1]

    for natural_robust_flag in natural_robust_flags:
        adv_sets = results[natural_robust_flag]['adversarial']
        sal_sets = results[natural_robust_flag]['saliency']
        adv_sal_sets = results[natural_robust_flag]['adversarial_saliency']
        full_network_name = natural_robust_flag + '-' + network_name

        save_adv(adv_sets, img_name, full_network_name)
        save_sal(sal_sets, img_name, full_network_name)
        save_adv_sal(adv_sal_sets, img_name, full_network_name)


def save_adv(adv_sets, img_name, network_name='Natural-ResNet50'):
    pgd_name, structured_name, unstructured_name = list(adv_sets.keys())

    pgd_set = adv_sets[pgd_name]
    structured_set = adv_sets[structured_name]
    unstructured_set = adv_sets[unstructured_name]

    for pgd_eps, pgd_adv_x in pgd_set.items():
        pgd_root_dir = os.path.join(
            DATA_ROOT_DIR, 'adversarial', network_name, pgd_name, pgd_eps)
        make_dir(pgd_root_dir)

        torch.save(pgd_adv_x.cpu(), os.path.join(
            pgd_root_dir, img_name + '.pt'))

    for sal_method_name, structured_adv_x in structured_set.items():
        structuread_root_dir = os.path.join(
            DATA_ROOT_DIR, 'adversarial', network_name, structured_name, sal_method_name)
        make_dir(structuread_root_dir)

        torch.save(structured_adv_x.cpu(), os.path.join(
            structuread_root_dir, img_name + '.pt'))

    for unstructured_method_name, _unstructured_set in unstructured_set.items():
        for sal_method_name, inner_dict in _unstructured_set.items():
            for unstructured_eps, unstructured_adv_x in inner_dict.items():
                structuread_root_dir = os.path.join(
                    DATA_ROOT_DIR, 'adversarial', network_name, unstructured_name,
                    unstructured_method_name, sal_method_name, unstructured_eps)
                make_dir(structuread_root_dir)

                torch.save(unstructured_adv_x.cpu(), os.path.join(
                    structuread_root_dir, img_name + '.pt'))


def save_sal(sal_sets, img_name, network_name='Natural-ResNet50'):
    for sal_method_name, sal in sal_sets.items():
        sal_method_root_dir = os.path.join(
            DATA_ROOT_DIR, 'saliency', network_name, sal_method_name, 'original')
        make_dir(sal_method_root_dir)

        torch.save(sal.cpu(), os.path.join(
            sal_method_root_dir, img_name + '.pt'))


def save_adv_sal(adv_sal_sets, img_name, network_name='Natural-ResNet50'):
    pgd_name, structured_name, unstructured_name = list(adv_sal_sets.keys())

    pgd_adv_sal_sets = adv_sal_sets[pgd_name]
    structured_adv_sal_sets = adv_sal_sets[structured_name]
    unstructured_adv_sal_sets = adv_sal_sets[unstructured_name]

    for pgd_eps, pgd_adv_sal_set in pgd_adv_sal_sets.items():
        for def_sal_method_name, pgd_adv_sal in pgd_adv_sal_set.items():
            adv_sal_method_root_dir = os.path.join(
                DATA_ROOT_DIR, 'saliency', network_name, def_sal_method_name,
                'PGD', pgd_eps)
            make_dir(adv_sal_method_root_dir)

            torch.save(pgd_adv_sal.cpu(), os.path.join(
                adv_sal_method_root_dir, img_name + '.pt'))

    for att_sal_method_name, structured_adv_sal_set in structured_adv_sal_sets.items():
        for def_sal_method_name, structured_adv_sal in structured_adv_sal_set.items():
            adv_sal_method_root_dir = os.path.join(
                DATA_ROOT_DIR, 'saliency', network_name, def_sal_method_name,
                'Structured', att_sal_method_name)
            make_dir(adv_sal_method_root_dir)

            torch.save(structured_adv_sal.cpu(), os.path.join(
                adv_sal_method_root_dir, img_name + '.pt'))

    for unstructured_method_name, _unstructured_adv_sal_sets in unstructured_adv_sal_sets.items():
        for att_sal_method_name, inner_dict in _unstructured_adv_sal_sets.items():
            for unstructured_eps, unstructured_adv_sal_sets in inner_dict.items():
                for def_sal_method_name, unstructured_adv_sal in unstructured_adv_sal_sets.items():
                    adv_sal_method_root_dir = os.path.join(
                        DATA_ROOT_DIR, 'saliency', network_name, def_sal_method_name,
                        'Unstructured', unstructured_method_name,
                        att_sal_method_name, unstructured_eps)
                    make_dir(adv_sal_method_root_dir)

                    torch.save(unstructured_adv_sal.cpu(), os.path.join(
                        adv_sal_method_root_dir, img_name + '.pt'))
