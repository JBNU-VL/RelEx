import torch
from torch.nn import functional as F
import numpy as np

'''
This script is used to provide funtion to evaluate Saliency against Adversarial example.
'''


class RetrievalRate:
    def __init__(self, net):
        self.net = net.to(0)

    def __call__(self, orig_x, adv_x, orig_sal, adv_sal):
        exp = orig_x * orig_sal
        recov_exp = adv_x * orig_sal
        adv_exp = adv_x * adv_sal
        recov_adv_exp = orig_x * adv_sal

        exp_top_accu1 = self.top1_accu(orig_x, exp)
        recov_exp_top_accu1 = self.top1_accu(orig_x, recov_exp)
        adv_exp_top1_accu1 = self.top1_accu(orig_x, adv_exp)
        recov_adv_exp_top1_accu1 = self.top1_accu(orig_x, recov_adv_exp)

        top_accu1_outputs = {
            'orig_x * orig_sal': exp_top_accu1,
            'adv_x * orig_sal': recov_exp_top_accu1,
            'adv_x * adv_sal': adv_exp_top1_accu1,
            'orig_x * adv_sal': recov_adv_exp_top1_accu1
        }

        return top_accu1_outputs

    def top1_accu(self, orig_x, input_x):
        target_indices = self.net(orig_x).max(1)[1]
        max_indices = self.net(input_x).max(1)[1]

        N = float(max_indices.size(0))
        match_sum = (target_indices == max_indices).sum()
        accu = match_sum / N
        return accu.item()


class Robustness:
    def __init__(self, k=1000, h=224, w=224, device=None):
        self.k = k
        self.hw = float(h * w)
        self.rank_values = torch.arange(
            self.hw, dtype=torch.float32, device=device) + 1

    def __call__(self, orig_sal_set, adv_sal_set):
        orig_sal_set_flatten = orig_sal_set.flatten(start_dim=1)
        adv_sal_set_flatten = adv_sal_set.flatten(start_dim=1)

        topk_metric = self.topk_intersecection(
            orig_sal_set_flatten, adv_sal_set_flatten)
        spearman_metric = self.spearman_correlation(
            orig_sal_set_flatten, adv_sal_set_flatten)
        return topk_metric, spearman_metric

    def topk_intersecection(self, orig_sal_set_flatten, adv_sal_set_flatten):
        orig_topk_indices = torch.topk(
            orig_sal_set_flatten, k=self.k, dim=-1)[1]
        adv_topk_indices = torch.topk(
            adv_sal_set_flatten, k=self.k, dim=-1)[1]

        adv_topk_compareview = adv_topk_indices.unsqueeze(
            -1).repeat(1, 1, self.k)
        adv_topk_compareview = adv_topk_compareview.permute(0, 2, 1)

        match_indices = (adv_topk_compareview ==
                         orig_topk_indices.unsqueeze(-1))  # batch by k by k
        adv_topk_match_indices = match_indices.permute(
            0, 2, 1).sum(dim=-1)  # batch by k

        return adv_topk_match_indices.sum(dim=-1) / float(self.k)

    def spearman_correlation(self, orig_sal_set_flatten, adv_sal_set_flatten):
        orig_orders = torch.argsort(orig_sal_set_flatten, dim=-1)
        adv_orders = torch.argsort(adv_sal_set_flatten, dim=-1)
        orig_sal_set_rank = torch.zeros_like(orig_sal_set_flatten)
        adv_sal_set_rank = torch.zeros_like(adv_sal_set_flatten)

        orig_sal_set_rank[0, orig_orders[0]] = self.rank_values
        for b_idx in range(adv_sal_set_flatten.size(0)):
            adv_sal_set_rank[b_idx, adv_orders[b_idx]] = self.rank_values

        d = orig_sal_set_rank - adv_sal_set_rank
        num = 6 * (d**2).sum(dim=-1)
        denom = self.hw * (self.hw**2 - 1)
        return 1 - num / denom


class Fieldility:
    '''
    '''

    def __init__(self, net, substrate_fn=torch.zeros_like, step=224):
        self.net = net
        self.step = step
        self.substrate_fn = substrate_fn

    # calculate auc for x
    # def calculate_game(self, x, sal_set, batch=False):
    #     scores = []
    #     for mode in ['deletion', 'preservation']:
    #         score = self._run(x, sal_set, mode=mode)
    #         # score = self._batch_run(x, exp_map, mode=mode)
    #         scores.append(score)

    #     del_auc = self.calculate_auc(scores[0])
    #     pres_auc = self.calculate_auc(scores[1])
    #     harm_auc = self.calculate_harmonic_avg(del_auc, pres_auc)

    #     return del_auc, pres_auc, harm_auc

    def __call__(self, x, sal_set):
        scores = []
        for mode in ['deletion', 'preservation']:
            score = self._run(x, sal_set, mode=mode)
            scores.append(score)

        del_auc = self.calculate_auc(scores[0])
        pres_auc = self.calculate_auc(scores[1])
        harm_auc = self.calculate_harmonic_avg(del_auc, pres_auc)

        return del_auc, pres_auc, harm_auc

    def calculate_auc(self, scores):
        '''
        calculate AUC under scores

        Args:
        Returns:
        '''
        assert len(scores.size()) == 2, 'scores shape is B by HW'
        return (scores.sum(dim=-1) - scores[:, 0] / 2 - scores[:, -1] / 2) / (scores.size(-1) - 1)

    def calculate_harmonic_avg(self, x1, x2):
        denom = 1 / x1 + 1 / x2
        return 2 / denom

    # @torch.no_grad()
    # def _single_run(self, x, exp_map, mode='preservation', target_idx=None):
    #     outputs = self.model(x)
    #     if target_idx == None:
    #         target_idx = torch.argmax(outputs.type(torch.float32))

    #     n_steps = (self.HW + self.step - 1) // self.step

    #     start = x.clone()
    #     finish = self.substrate_fn(x)

    #     scores = torch.zeros(n_steps + 1)

    #     # Coordinates of pixels in order of decreasing saliency
    #     if mode == 'preservation':
    #         salient_order = torch.argsort(
    #             exp_map.reshape(-1), descending=False)
    #     else:
    #         salient_order = torch.argsort(exp_map.reshape(-1), descending=True)

    #     for i in range(n_steps + 1):
    #         _score = F.softmax(self.net(start), 1)
    #         scores[i] = _score[0, target_idx]

    #         if i < n_steps:
    #             coords = salient_order[self.step * i: self.step * (i + 1)]
    #             start.view(
    #                 1, 3, -1)[0, :, coords] = finish.view(1, 3, -1)[0, :, coords]

        return scores

    @torch.no_grad()
    def _run(self, x, sal_set, mode='preservation'):
        outputs = self.net(x)
        target_indices = torch.argmax(outputs.type(torch.float32), dim=-1)

        hw = x.size(2) * x.size(3)
        n_steps = (hw + self.step - 1) // self.step
        batch_size = sal_set.size(0)
        batch_indices = torch.arange(batch_size)

        starts = x.detach().clone()
        finishes = self.substrate_fn(x)

        batch_scores = torch.zeros(batch_size, n_steps + 1)

        if mode == 'preservation':
            salient_orders = torch.argsort(sal_set.reshape(
                batch_size, -1), descending=False, dim=-1)
        else:
            salient_orders = torch.argsort(sal_set.reshape(
                batch_size, -1), descending=True, dim=-1)

        for i in range(n_steps + 1):
            _scores = F.softmax(self.net(starts), 1)
            batch_scores[:, i] = _scores[batch_indices, target_indices]

            if i < n_steps:
                batch_coords = salient_orders[batch_indices,
                                              self.step * i: self.step * (i + 1)]

                for b_idx, coords in enumerate(batch_coords):
                    starts.view(
                        batch_size, 3, -1)[b_idx, :, coords] = finishes.view(batch_size, 3, -1)[b_idx, :, coords]

        return batch_scores
