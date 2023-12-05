"""Sparsely-Gated Mixture-of-Experts Layers and Task-routing Mixture-of-Graph-Experts Layers
Author: Xu Yao

MoE Paper: See "Outrageously Large Neural Networks"
https://arxiv.org/abs/1701.06538

This code is based on this torch implementation of MoE:
https://github.com/davidmrau/mixture-of-experts
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch_geometric.nn import global_mean_pool, SAGPooling
from modules.gnn import GINEConv
from modules.utils import reset_all_weights, cos_loss, cos_same_loss


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch_batch_index(self):
        return torch.split(self._batch_index, self._part_sizes, dim=0)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        combined = combined.log()
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Tanh()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.out(x) * 10.
        return x



class MoCE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4, task_routing=False,
                 task_routing_sizes=None, dropout=0.5, num_g_experts=16, sag_pool=True, kt=None,
                 iattvec_loss=False, expert_struct_mode='bottleneck', hierarchical=True,
                 hk=12):
        super(MoCE, self).__init__()
        self.noisy_gating = noisy_gating  # for load balance
        self.num_experts = num_experts
        self.num_g_experts = num_g_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.kt = kt if kt is not None else k
        self.hk = hk
        self.hierarchical = hierarchical

        self.iattvec_loss = iattvec_loss

        self.experts = nn.ModuleList(
            [MLP(self.input_size, self.output_size, self.hidden_size, dropout=dropout) for i in
             range(self.num_experts)])

        if expert_struct_mode == 'bottleneck':
            g_hid = int(self.hidden_size / 2)
        elif expert_struct_mode == 'expand':
            g_hid = self.hidden_size * 2
        else:
            g_hid = self.hidden_size
        self.experts_g = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(input_size, g_hid),
                                       nn.BatchNorm1d(g_hid),
                                       nn.ReLU(),
                                       nn.Linear(g_hid, hidden_size)),
                         edge_dim=self.hidden_size) for _ in range(self.num_g_experts)
            ]
        )

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.task_routing = task_routing

        if task_routing:
            self.h_w_gates = nn.Parameter(torch.zeros(task_routing_sizes, num_experts), requires_grad=True)
            self.h_w_noises = nn.Parameter(torch.zeros(task_routing_sizes, num_experts), requires_grad=True)

            self.g_gate_map = nn.Linear(task_routing_sizes, self.num_g_experts)
            nn.init.xavier_uniform_(self.g_gate_map.weight)
            self.g_gate_softmax = nn.Softmax(0)

        self.sag_pool = sag_pool
        if sag_pool:
            self.pool = SAGPooling(hidden_size, ratio=0.5)
            self.attn_vectors = nn.Parameter(torch.zeros(self.num_g_experts, hidden_size), requires_grad=True)
            nn.init.xavier_uniform_(self.attn_vectors)
        else:
            self.pool = global_mean_pool

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([2.0]) if self.hierarchical else torch.tensor([1.0]))
        assert (self.k <= self.num_experts and self.kt <= self.num_experts)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.w_gate)
        nn.init.zeros_(self.w_noise)
        if self.task_routing:
            nn.init.zeros_(self.h_w_gates)
            nn.init.zeros_(self.h_w_noises)
            self.g_gate_map.reset_parameters()
            nn.init.xavier_uniform_(self.g_gate_map.weight)
        if self.sag_pool:
            nn.init.xavier_uniform_(self.attn_vectors)
        reset_all_weights(self.experts)
        reset_all_weights(self.experts_g)

    def freeze_router(self):
        self.w_gate.requires_grad = False
        self.w_noise.requires_grad = False
        if self.task_routing:
            self.h_w_gates.requires_grad = False
            self.h_w_noises.requires_grad = False
            self.g_gate_map.requires_grad = False

    def task_routing_noisy_top_k_gating(self, x, train, task_routing_x, noise_epsilon=1e-2, dataset_idx=None):
        """Task Adjusted Noisy top-k gating.
          Source version see paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            task_routing_x: list of vector
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        if not self.hierarchical:
            # task_routing_x: [batch_size, task_routing_sizes]; x: [batch_all_node_size, emb_size];
            # self.h_w_gates: [task_routing_sizes, num_experts]; self.w_gate: [emb_size, num_experts]
            h_clean_logits = task_routing_x @ self.h_w_gates
            clean_logits = x @ self.w_gate

            if self.noisy_gating and train:
                raw_noise_stddev = x @ self.w_noise
                h_raw_noise_stddev = task_routing_x @ self.h_w_noises

                noise_stddev = ((self.softplus(raw_noise_stddev + h_raw_noise_stddev) + noise_epsilon))
                noisy_logits = clean_logits + h_clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                logits = noisy_logits
            else:
                logits = clean_logits + h_clean_logits

            k = self.kt if self.training else self.k

            # calculate topk + 1 that will be needed for the noisy gates
            top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
            top_k_logits = top_logits[:, :k]
            top_k_indices = top_indices[:, :k]
            top_k_gates = self.softmax(top_k_logits)

            zeros = torch.zeros_like(logits, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)

            if self.noisy_gating and k < self.num_experts and train:
                load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
            else:
                load = self._gates_to_load(gates)
            return gates, load
        else:
            k = self.kt if self.training else self.k
            h_k = self.hk

            h_clean_logits = task_routing_x @ self.h_w_gates
            clean_logits = x @ self.w_gate

            _, h_top_indices = h_clean_logits.topk(min(h_k + 1, self.num_experts), dim=1)
            h_clean_logits = h_clean_logits.scatter(1, h_top_indices[:, h_k:], h_clean_logits.detach().min().item())
            clean_logits = clean_logits.scatter(1, h_top_indices[:, h_k:], clean_logits.detach().min().item())

            if self.noisy_gating and train:
                raw_noise_stddev = x @ self.w_noise
                h_raw_noise_stddev = task_routing_x @ self.h_w_noises

                noise_stddev = ((self.softplus(raw_noise_stddev + h_raw_noise_stddev) + noise_epsilon))
                noisy_logits = clean_logits + h_clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                logits = noisy_logits
            else:
                logits = clean_logits + h_clean_logits

            # calculate topk + 1 that will be needed for the noisy gates
            top_logits, top_indices = logits.topk(min(k + 1, self.num_experts), dim=1)
            top_k_logits = top_logits[:, :k]
            top_k_indices = top_indices[:, :k]
            top_k_gates = self.softmax(top_k_logits)

            zeros = torch.zeros_like(logits, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)

            if self.noisy_gating and k < self.num_experts and train:
                load = (self._prob_in_top_k(clean_logits + h_clean_logits, noisy_logits, noise_stddev, top_logits)).sum(
                    0)
            else:
                load = self._gates_to_load(gates)
            return gates, load

    def gnns_forward(self, x, edge_index, edge_attr):
        out_x = torch.stack([self.experts_g[i](x, edge_index, edge_attr)
                             for i in range(self.num_g_experts)]).mean(0)
        return out_x

    def forward(self, x, edge_index, edge_attr, batch, loss_coef=1e-2, task_routing_x=None, dy=None, dataset_idx=None,
                return_gates=False):
        """Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        if self.sag_pool:
            sag_graphs = [
                self.pool(x, edge_index, edge_attr, batch=batch, attn=self.attn_vectors[i].repeat(x.size(0), 1))
                for i in range(self.num_g_experts)]
            agg_x_part = torch.stack([global_mean_pool(o[0], o[3]) for o in sag_graphs])
            agg_x = agg_x_part.mean(0)
        else:
            agg_x = self.pool(x, batch)

        if len(self.experts_g) == 1:
            out_x = self.experts_g[0](x, edge_index, edge_attr)
            out_x_part = out_x.unsqueeze(0)
        elif self.task_routing:
            out_x_part = torch.stack(
                [self.experts_g[i](x, edge_index, edge_attr) for i in range(self.num_g_experts)])

            assert isinstance(task_routing_x, torch.Tensor), f"task_routing_x should be Tensor but get {task_routing_x}"
            with torch.no_grad():
                _, repeats = batch.unique(return_counts=True)
                g_task_routing_x = task_routing_x.repeat_interleave(repeats, dim=0)

            g_gate = self.g_gate_softmax(self.g_gate_map(g_task_routing_x).T.unsqueeze(-1))

            out_x = (out_x_part * g_gate).sum(0)
        else:
            out_x_part = torch.stack([self.experts_g[i](x, edge_index, edge_attr)
                                      for i in range(self.num_g_experts)])
            out_x = out_x_part.mean(0)

        if self.task_routing:
            gates, load = self.task_routing_noisy_top_k_gating(agg_x, self.training, task_routing_x=task_routing_x,
                                                               dataset_idx=dataset_idx)
        else:
            gates, load = self.noisy_top_k_gating(agg_x, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(agg_x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        if dy is not None and self.training:
            experts_loss = 0
            expert_targets = dispatcher.dispatch(dy.unsqueeze(-1))
            for i, o in enumerate(expert_outputs):
                if o.size(0) > 0:
                    experts_loss += nn.BCELoss()(o.sigmoid().squeeze(), expert_targets[i].squeeze())

            experts_loss /= self.num_experts
            loss += experts_loss

        y = dispatcher.combine(expert_outputs)

        if return_gates:
            return y, loss, out_x, gates
        else:
            return y, loss, out_x, None

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
            x: a `Tensor`.
        Returns:
            a `Scalar`.
        """
        eps = 1e-10  # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
            clean_values: a `Tensor` of shape [batch, n].
            noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
              normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch, n], or None
            noisy_top_values: a `Tensor` of shape [batch, m].
               "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
