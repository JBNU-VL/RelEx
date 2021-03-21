import torch


class RobustMetric:
    def __init__(self, k=1000, h=224, w=224):
        self.k = k
        self.hw = float(h * w)
        self.rank_values = torch.arange(
            self.hw, dtype=torch.float32, device='cuda') + 1

    def robustness(self, org_exp_maps, adv_exp_maps):
        batch_size = org_exp_maps.size(0)
        org_exp_maps_flatten = org_exp_maps.reshape(batch_size, -1)
        adv_exp_maps_flatten = adv_exp_maps.reshape(batch_size, -1)

        topk_metrics = self.topk_intersec(org_exp_maps_flatten, adv_exp_maps_flatten,
                                          batch_size)
        spearman_metrics = self.spearman_correlation(org_exp_maps_flatten, adv_exp_maps_flatten,
                                                     batch_size)
        return topk_metrics, spearman_metrics

    '''
    https://github.com/amiratag/InterpretationFragility/blob/master/utils.py
    it is used to np.intersec1d above code
    '''

    def topk_intersec(self, org_exp_maps_flatten, adv_exp_maps_flatten, batch_size=32):
        ''' calculate recall '''
        org_exp_maps_flatten = org_exp_maps_flatten.clone()
        adv_exp_maps_flatten = adv_exp_maps_flatten.clone()

        ''' extract topk indices '''
        org_topk_indices = torch.topk(
            org_exp_maps_flatten, k=self.k, dim=-1)[1]
        adv_topk_indices = torch.topk(
            adv_exp_maps_flatten, k=self.k, dim=-1)[1]

        ''' 
        https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors 
        reference to above site..
        '''
        adv_topk_compareview = adv_topk_indices.unsqueeze(
            -1).repeat(1, 1, self.k)
        adv_topk_compareview = adv_topk_compareview.permute(0, 2, 1)

        '''
                   [[1,1,1,1],
        compare     [2,2,2,2], transpose  and [1,3,5,7]
                    [3,3,3,3],
                    [4,4,4,4]]
        '''
        match_indices = (adv_topk_compareview ==
                         org_topk_indices.unsqueeze(-1))  # batch by k by k
        adv_topk_match_indices = match_indices.permute(
            0, 2, 1).sum(dim=-1)  # batch by k

        return adv_topk_match_indices.sum(dim=-1) / float(self.k)

    def spearman_correlation(self, org_exp_maps_flatten, adv_exp_maps_flatten, batch_size=32):
        org_exp_maps_flatten = org_exp_maps_flatten.clone()
        adv_exp_maps_flatten = adv_exp_maps_flatten.clone()

        ''' sorting as ascending'''
        org_orders = torch.argsort(org_exp_maps_flatten, dim=-1)
        adv_orders = torch.argsort(adv_exp_maps_flatten, dim=-1)
        org_exp_maps_rank = torch.zeros_like(org_exp_maps_flatten)
        adv_exp_maps_rank = torch.zeros_like(adv_exp_maps_flatten)

        ''' insert rank values using orders '''
        for b_idx in range(batch_size):
            org_exp_maps_rank[b_idx, org_orders[b_idx]] = self.rank_values
            adv_exp_maps_rank[b_idx, adv_orders[b_idx]] = self.rank_values

        ''' calculate spearman's rank correlation coeffiefient'''
        d = org_exp_maps_rank - adv_exp_maps_rank
        num = 6 * (d**2).sum(dim=-1)
        denom = self.hw * (self.hw**2 - 1)

        return 1 - num / denom


class FieldMetric():
    def __init__(self, model, substrate_fn, step=224):
        self.model = model
        self.step = step
        self.HW = 224 * 224
        self.substrate_fn = substrate_fn

    # calculate auc for x
    def calc_game(self, x, exp_map, batch=False, field_mode=None, target_idx=None):
        if not batch and field_mode == 'ins' or field_mode == 'pres':
            score = self._single_run(
                x, exp_map, mode=field_mode, target_idx=target_idx)
            return score

        if batch:
            scores = []
            for mode in ['del', 'pres']:
                score = self._batch_run(x, exp_map, mode=mode)

                scores.append(score)

            # return scores, games_seqs
            return scores

        scores = []
        for mode in ['del', 'pres']:
            score = self._single_run(x, exp_map, mode=mode)
            scores.append(score)

        return scores

    def get_metric(self):
        pass

    def _calc_comb_avg(self, x, y):
        denom = 1 / x + 1 / y
        return 2 / denom

    @torch.no_grad()
    def _single_run(self, x, exp_map, mode='pres', target_idx=None):
        outputs = self.model(x)
        if target_idx == None:
            target_idx = torch.argmax(outputs.type(torch.float32))

        n_steps = (self.HW + self.step - 1) // self.step

        if mode == 'del' or mode == 'pres':
            start = x.clone()
            finish = self.substrate_fn(x)

        elif mode == 'ins':
            start = self.substrate_fn(x)
            finish = x.clone()

        scores = torch.zeros(n_steps + 1)

        # Coordinates of pixels in order of decreasing saliency
        if mode == 'pres':
            salient_order = torch.argsort(
                exp_map.reshape(-1), descending=False)
        else:
            salient_order = torch.argsort(exp_map.reshape(-1), descending=True)

        for i in range(n_steps + 1):
            pred = self.model(start)
            scores[i] = pred[0, target_idx]

            if i < n_steps:
                coords = salient_order[self.step * i: self.step * (i + 1)]
                start.view(
                    1, 3, -1)[0, :, coords] = finish.view(1, 3, -1)[0, :, coords]

        return scores

    @torch.no_grad()
    def _batch_run(self, x, exp_mapes, mode='pres'):
        outputs = self.model(x)
        target_indices = torch.argmax(outputs.type(torch.float32), dim=-1)
        n_steps = (self.HW + self.step - 1) // self.step
        batch_size = exp_mapes.size(0)
        batch_indices = torch.arange(batch_size)

        if mode == 'del' or mode == 'pres':
            starts = x.detach().clone()
            finishes = self.substrate_fn(x)

        elif mode == 'ins':
            starts = self.substrate_fn(x)
            finishes = x.detach().clone()

        batch_scores = torch.zeros(batch_size, n_steps + 1)

        # Coordinates of pixels in order of decreasing saliency
        if mode == 'pres':
            salient_orders = torch.argsort(exp_mapes.reshape(
                batch_size, -1), descending=False, dim=-1)
        else:
            salient_orders = torch.argsort(exp_mapes.reshape(
                batch_size, -1), descending=True, dim=-1)

        # game_seqs = []
        for i in range(n_steps + 1):
            preds = self.model(starts)
            batch_scores[:, i] = preds[batch_indices, target_indices]

            if i < n_steps:
                batch_coords = salient_orders[batch_indices,
                                              self.step * i: self.step * (i + 1)]

                for b_idx, coords in enumerate(batch_coords):
                    starts.view(
                        batch_size, 3, -1)[b_idx, :, coords] = finishes.view(batch_size, 3, -1)[b_idx, :, coords]

        return batch_scores
