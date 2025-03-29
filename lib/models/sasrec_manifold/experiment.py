from functools import partial

import numpy as np
import torch

from .source import SASRec
from .sampler import packed_sequence_batch_sampler
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator


import torch.utils.checkpoint as checkpoint




def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    model = SASRecModel(config, n_items)
    model.fit(dataset.train, evaluator)


# def hyperbolic_dist(emb1, emb2):
#     """
#     Compute hyperbolic distance in the Poincaré ball model for batched inputs.
#     emb1: (batch_size, num_items, emb_dim)
#     emb2: (batch_size, num_items, emb_dim)
#     """
#     diff = emb1 - emb2  # (batch_size, num_items, emb_dim)
#     euclidean_dist_sq = torch.sum(diff ** 2, dim=-1)  # (batch_size, num_items)
    
#     u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
#     v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)

#     denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-5  # (batch_size, num_items)
#     x = 1 + 2 * euclidean_dist_sq / denom  # (batch_size, num_items)
#     x = torch.clamp(x, min=1 + 1e-5)  # Ensure valid input for acosh
#     hyperbolic_dist = torch.acosh(x)  # (batch_size, num_items)

#     return hyperbolic_dist


# def hyperbolic_dist(emb1, emb2):
#     """
#     Compute hyperbolic distance in the Poincaré ball model for batched inputs.
#     emb1, emb2: (batch_size, num_items, emb_dim)
#     """
#     # Compute norms without broadcasting the full difference tensor.
#     u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
#     v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)
    
#     # Compute inner product and then Euclidean distance squared
#     inner_prod = torch.sum(emb1 * emb2, dim=-1)  # (batch_size, num_items)
#     euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

#     # Denom and hyperbolic distance as before.
#     denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-5
#     x = 1 + 2 * euclidean_dist_sq / denom
#     x = torch.clamp(x, min=1 + 1e-5)
#     hyperbolic_dist = torch.acosh(x)

#     return hyperbolic_dist


def hyperbolic_dist(emb1, emb2, c=1.0):
    """
    Compute hyperbolic distance in the Poincaré ball model with curvature c.
    emb1, emb2: (batch_size, num_items, emb_dim)
    c: positive float (curvature)
    """
    # Compute norms of embeddings
    u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
    v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)
    
    # Compute inner product and then squared Euclidean distance
    inner_prod = torch.sum(emb1 * emb2, dim=-1)  # (batch_size, num_items)
    euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

    # Incorporate curvature into denominators and numerator
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq) + 1e-5
    x = 1 + 2 * c * euclidean_dist_sq / denom
    x = torch.clamp(x, min=1 + 1e-5)
    
    # Scale the result appropriately
    dist = (2 / torch.sqrt(torch.tensor(c, device=emb1.device))) * torch.acosh(x)
    return dist



class CurvatureEstimator:
    def __init__(self, dist_fn, num_samples=10000, epsilon=1e-5):
        self.dist_fn = dist_fn
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.delta_p = torch.log1p(torch.sqrt(torch.tensor(2.0)))  # ln(1 + sqrt(2))

    def sample_quadruples(self, emb):
        N = emb.size(0)
        idx = torch.randint(0, N, (self.num_samples, 4), device=emb.device)
        return idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

    def gromov_product(self, d_xy, d_xz, d_yz):
        return 0.5 * (d_xz + d_yz - d_xy)

    def estimate_delta(self, emb):
        x, y, z, w = self.sample_quadruples(emb)
        d = lambda a, b: self.dist_fn(emb[a], emb[b])

        gp1 = self.gromov_product(d(x, y), d(x, w), d(y, w))
        gp2 = self.gromov_product(d(y, z), d(y, x), d(z, x))
        gp3 = self.gromov_product(d(z, x), d(z, y), d(x, y))

        delta = torch.min(torch.stack([gp1, gp2, gp3]), dim=0).values
        return delta.max()

    def estimate_diameter(self, emb):
        # estimate diameter via random samples
        idx_a = torch.randint(0, emb.size(0), (self.num_samples,), device=emb.device)
        idx_b = torch.randint(0, emb.size(0), (self.num_samples,), device=emb.device)
        d_ab = self.dist_fn(emb[idx_a], emb[idx_b])
        return torch.quantile(d_ab, 0.95)  # robust diameter

    def curvature(self, emb):
        delta = self.estimate_delta(emb)
        diam = self.estimate_diameter(emb)
        delta_rel = 2 * delta / diam

        diam_P = 2 * torch.log(torch.Tensor([(1 + 2 * (1 - self.epsilon)) / self.epsilon])).to(delta_rel.device)
        delta_rel_p = 2 * self.delta_p / diam_P

        c_est = (delta_rel_p / delta_rel) ** 2
        return c_est.item()


# def hyperbolic_dist(emb1, emb2):
#     # Wrap the computation in a function suitable for checkpointing.
#     def compute_dist(emb1, emb2):
#         """
#         Compute hyperbolic distance in the Poincaré ball model for batched inputs.
#         emb1, emb2: (batch_size, num_items, emb_dim)
#         """
#         # Compute norms without broadcasting the full difference tensor.
#         u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
#         v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)
    
#         # Compute inner product and then Euclidean distance squared
#         inner_prod = torch.sum(emb1 * emb2, dim=-1)  # (batch_size, num_items)
#         euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

#         # Denom and hyperbolic distance as before.
#         denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-5
#         x = 1 + 2 * euclidean_dist_sq / denom
#         x = torch.clamp(x, min=1 + 1e-5)
#         return torch.acosh(x)
    
#     # Use checkpointing to save memory.
#     return checkpoint.checkpoint(compute_dist, emb1, emb2, use_reentrant=False)



def seq_euclidean_dist(eseq1, eseq2, single_dist):
    """
    Compute sequence-wise Euclidean distance using a distance function (batched).
    eseq1, eseq2: (batch_size, num_items, emb_dim)
    """
    dist_vector = single_dist(eseq1, eseq2)  # (batch_size, num_items)
    eseq_dist = torch.norm(dist_vector, dim=-1)  # (batch_size)
    return eseq_dist

# def pairseq_dist_affinity(seq, single_dist):
#     """
#     Compute the pairwise sequence distance affinity matrix efficiently.
#     seq: (batch_size, num_items, emb_dim)
#     single_dist: function to compute distance between two sequences
#     """
#     batch_size = seq.shape[0]
    
#     # Expand dimensions to compute pairwise distances efficiently
#     seq1 = seq.unsqueeze(1)  # (batch_size, 1, num_items, emb_dim)
#     seq2 = seq.unsqueeze(0)  # (1, batch_size, num_items, emb_dim)

#     pairwise_distances = seq_euclidean_dist(seq1, seq2, single_dist)  # (batch_size, batch_size)
    
#     affinity = torch.exp(-pairwise_distances ** 2)  # (batch_size, batch_size)
    
#     return affinity


def pairseq_dist_affinity(seq, single_dist, chunk_size=64):
    """
    Compute the pairwise sequence distance affinity matrix in chunks.
    seq: (batch_size, num_items, emb_dim)
    single_dist: function to compute distance between two sequences
    chunk_size: number of sequences to process in one chunk
    """
    batch_size = seq.shape[0]
    affinity = torch.empty(batch_size, batch_size, device=seq.device)
    
    for i in range(0, batch_size, chunk_size):
        seq_chunk = seq[i:i+chunk_size]  # (chunk_size, num_items, emb_dim)
        # Expand dimensions to compute pairwise distances against the whole batch
        seq_chunk_exp = seq_chunk.unsqueeze(1)   # (chunk_size, 1, num_items, emb_dim)
        seq_all = seq.unsqueeze(0)                 # (1, batch_size, num_items, emb_dim)
        
        # Compute distance: returns (chunk_size, batch_size)
        chunk_dist = seq_euclidean_dist(seq_chunk_exp, seq_all, single_dist)
        affinity[i:i+chunk_size] = torch.exp(-chunk_dist ** 2)
    
    return affinity

def aggregate_seq(seq):
    """
    Aggregate a sequence of vectors by taking the arithmetic mean.
    
    seq: (batch_size, num_items, emb_dim)
    Returns: (batch_size, emb_dim)
    """
    return seq.mean(dim=1)


def hyperbolic_pairwise_distance(agg, eps=1e-5):
    """
    Compute pairwise hyperbolic distance between aggregated vectors.
    
    agg: (batch_size, emb_dim)
    Returns: (batch_size, batch_size) distance matrix.
    """
    # Compute squared norms of each vector: shape (batch_size, 1)
    agg_norm_sq = torch.sum(agg ** 2, dim=-1, keepdim=True)  # (B, 1)
    
    # Compute squared Euclidean distance between all pairs:
    # Using the identity: ||u-v||^2 = ||u||^2 + ||v||^2 - 2 * u.v
    inner_prod = torch.mm(agg, agg.t())  # (B, B)
    d_sq = agg_norm_sq + agg_norm_sq.t() - 2 * inner_prod  # (B, B)
    
    # Denominators for each pair (u,v)
    denom = (1 - agg_norm_sq) * (1 - agg_norm_sq.t()) + eps  # (B, B)
    
    # Compute the argument for acosh
    x = 1 + 2 * d_sq / denom
    x = torch.clamp(x, min=1 + eps)  # ensure valid input for acosh
    
    return torch.acosh(x)


# def project_embeddings(embeddings, max_norm=1.0, eps=1e-5):
#     """
#     Project embeddings back into the Poincaré ball.
    
#     Args:
#         embeddings (torch.Tensor): Tensor of shape (..., emb_dim).
#         max_norm (float): Maximum allowed norm (typically 1.0 for the unit ball).
#         eps (float): Small epsilon to ensure numerical stability (to avoid exactly reaching the boundary).
    
#     Returns:
#         torch.Tensor: The projected embeddings.
#     """
#     norm = embeddings.norm(dim=-1, keepdim=True)
#     # Determine scaling factors: if norm > max_norm, scale down to (max_norm - eps)
#     scale = (max_norm - eps) / norm
#     # Only scale embeddings where norm > max_norm - eps; otherwise, keep as is.
#     scale = torch.where(norm > max_norm - eps, scale, torch.ones_like(scale))
#     return embeddings * scale


def project_embeddings(embeddings, c=1.0, eps=1e-5):
    """
    Project embeddings back into the Poincaré ball of curvature c.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (..., emb_dim).
        c (float): Curvature parameter. For a Poincaré ball of curvature -c, 
                   valid embeddings satisfy ||x|| < 1/sqrt(c).
        eps (float): Small epsilon to avoid exactly reaching the boundary.
    
    Returns:
        torch.Tensor: The projected embeddings.
    """
    # The radius of the ball is 1/sqrt(c)
    r = 1.0 / torch.sqrt(torch.tensor(c, device=embeddings.device))
    
    norm = embeddings.norm(dim=-1, keepdim=True)
    # Determine scaling factors: if norm > r, scale down to (r - eps)
    scale = (r - eps) / norm
    # Only scale embeddings where norm exceeds (r - eps); otherwise, keep as is.
    scale = torch.where(norm > (r - eps), scale, torch.ones_like(scale))
    return embeddings * scale


def fixed_sample_indices(num_items, sample_count):
    """
    Returns a tensor of sample indices uniformly spaced between 0 and num_items-1.
    """
    return torch.linspace(0, num_items - 1, steps=sample_count).long()

def approx_seq_hyperbolic_distance_fixed(seq1, seq2, sample_count=10, eps=1e-5):
    """
    Approximate the mean hyperbolic distance between two sequences by sampling the same fixed indices.
    
    seq1, seq2: (num_items, emb_dim)
    sample_count: number of vectors to sample (same indices for both sequences)
    Returns: A scalar mean distance.
    """
    num_items = seq1.shape[0]
    # Get fixed sample indices
    indices = fixed_sample_indices(num_items, sample_count).to(seq1.device)
    
    # Sample the same positions from both sequences
    sample1 = seq1[indices]  # (sample_count, emb_dim)
    sample2 = seq2[indices]  # (sample_count, emb_dim)
    
    # Compute pairwise hyperbolic distances between sampled vectors.
    # Note: this computes a (sample_count, sample_count) matrix.
    D = hyperbolic_distance_pairwise(sample1, sample2, eps)
    
    return D.mean()

def hyperbolic_distance_pairwise(u, v, eps=1e-5):
    """
    Compute pairwise hyperbolic distance between two sets of vectors.
    
    u: (n, d)
    v: (m, d)
    Returns: (n, m) distance matrix.
    """
    u_norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True)  # (n, 1)
    v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # (m, 1)
    
    d_sq = u_norm_sq + v_norm_sq.t() - 2 * torch.mm(u, v.t())
    
    denom = (1 - u_norm_sq) * (1 - v_norm_sq.t()) + eps
    x = 1 + 2 * d_sq / denom
    x = torch.clamp(x, min=1 + eps)
    
    return torch.acosh(x)

def approximate_batch_affinity_fixed(seq, sample_count=10):
    """
    Compute an approximate (batch_size, batch_size) affinity matrix,
    where each entry is the mean hyperbolic distance between sequences,
    approximated by sampling the same fixed indices.
    
    seq: (batch_size, num_items, emb_dim)
    sample_count: number of items to sample per sequence.
    """
    batch_size = seq.shape[0]
    affinity = torch.empty(batch_size, batch_size, device=seq.device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            affinity[i, j] = approx_seq_hyperbolic_distance_fixed(seq[i], seq[j], sample_count)
    
    return affinity

def approx_seq_hyperbolic_distance_fixed_batch(seq, sample_count=10, eps=1e-5):
    """
    Approximate the mean hyperbolic distance between all sequences in the batch using fixed sampled indices.
    
    seq: (batch_size, num_items, emb_dim)
    sample_count: number of sampled positions per sequence.
    
    Returns:
        A (batch_size, batch_size) matrix of distances.
    """
    batch_size, num_items, emb_dim = seq.shape
    device = seq.device
    
    # Get the fixed sample indices for all sequences
    indices = fixed_sample_indices(num_items, sample_count).to(device)  # (sample_count,)
    
    # Sample the same positions for all sequences
    sampled_seq = seq[:, indices, :]  # (batch_size, sample_count, emb_dim)

    # Compute pairwise hyperbolic distances in a vectorized manner
    hyperbolic_distances = hyperbolic_distance_batch(sampled_seq, eps)  # (batch_size, batch_size, sample_count)

    # Compute the mean distance across sampled positions
    return hyperbolic_distances.mean(dim=-1)  # (batch_size, batch_size)

# def hyperbolic_distance_batch(seq, eps=1e-5):
#     """
#     Compute batched hyperbolic distances for all sequences at once.
    
#     seq: (batch_size, sample_count, emb_dim)
#     Returns:
#         A (batch_size, batch_size, sample_count) tensor containing distances.
#     """
#     batch_size, sample_count, emb_dim = seq.shape

#     # Compute squared norms for all sequences
#     norm_sq = torch.sum(seq ** 2, dim=-1, keepdim=True)  # (batch_size, sample_count, 1)

#     # Compute squared Euclidean distance in a batch-wise manner
#     d_sq = norm_sq + norm_sq.transpose(0, 1) - 2 * torch.matmul(seq, seq.transpose(0, 1))

#     # Compute the denominator for hyperbolic distance
#     denom = (1 - norm_sq) * (1 - norm_sq.transpose(0, 1)) + eps

#     # Compute hyperbolic distance
#     x = 1 + 2 * d_sq / denom
#     x = torch.clamp(x, min=1 + eps)

#     return torch.acosh(x)  # (batch_size, batch_size, sample_count)

def hyperbolic_distance_batch(seq, eps=1e-5):
    """
    Compute batched hyperbolic distances for all sequences at once.
    
    seq: (batch_size, sample_count, emb_dim)
    Returns:
        A (batch_size, batch_size, sample_count) tensor containing distances.
    """
    batch_size, sample_count, emb_dim = seq.shape

    # Compute squared norms for all sequences
    norm_sq = torch.sum(seq ** 2, dim=-1, keepdim=True)  # (batch_size, sample_count, 1)

    # Expand tensors to enable broadcasting
    seq_exp_1 = seq.unsqueeze(1)  # (batch_size, 1, sample_count, emb_dim)
    seq_exp_2 = seq.unsqueeze(0)  # (1, batch_size, sample_count, emb_dim)

    # Compute squared Euclidean distances
    d_sq = torch.sum((seq_exp_1 - seq_exp_2) ** 2, dim=-1)  # (batch_size, batch_size, sample_count)

    # Compute the denominator for hyperbolic distance
    denom = (1 - norm_sq) * (1 - norm_sq.transpose(0, 1)) + eps  # (batch_size, batch_size, sample_count)

    # Compute hyperbolic distance
    x = 1 + 2 * d_sq / denom
    x = torch.clamp(x, min=1 + eps)

    return torch.acosh(x)  # (batch_size, batch_size, sample_count)

def approximate_batch_affinity_fixed_vectorized(seq, sample_count=10):
    """
    Compute an approximate (batch_size, batch_size) affinity matrix,
    using a fully vectorized implementation.
    
    seq: (batch_size, num_items, emb_dim)
    sample_count: number of items to sample per sequence.
    
    Returns:
        A (batch_size, batch_size) affinity matrix.
    """
    distances = approx_seq_hyperbolic_distance_fixed_batch(seq, sample_count)
    return torch.exp(-distances ** 2)  # Apply affinity transformation


class SASRecModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = SASRec(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
    
    @property
    def model(self):
        return self._model

    def fit(self, data: tuple, evaluator: Evaluator):
        indices, sizes = data
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.config['batch_size'],
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )
        self.n_batches = (len(sizes) - 1) // self.config['batch_size']
        trainer(self, evaluator)

    def train_epoch(self):
        model = self.model
        pad_token = model.pad_token
        criterion, optimizer, sampler, device, n_batches = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'sampler', 'device', 'n_batches']
        ]
        l2_emb = self.config['l2_emb']
        as_tensor = partial(torch.as_tensor, dtype=torch.int32, device=device)
        loss = 0
        model.train()

        single_dist = lambda u,v: hyperbolic_dist(u,v, c = self.config['geometry_c']) 

        #clean_dist = lambda u,v: hyperbolic_dist(u,v, c = 1.0) 

        #ce = CurvatureEstimator(dist_fn=clean_dist, num_samples=10000)


        for _ in range(n_batches):
            _, *seq_data = next(sampler)
            # convert batch data into torch tensors
            seq, pos, neg = [as_tensor(arr) for arr in seq_data]
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            indices = torch.where(pos != pad_token)
            batch_loss = criterion(pos_logits[indices], pos_labels[indices])
            batch_loss += criterion(neg_logits[indices], neg_labels[indices])
            
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2


            pos_lambda_man_reg = self.config['pos_lambda_reg']

            if pos_lambda_man_reg > 0:
                with torch.amp.autocast('cuda'):
                    idx_samples = torch.randint(0, self.config['maxlen'], (self.config['num_items_sampled'],), device=pos.device)
                    pos_sub = pos[:, idx_samples]
                    #pos_eseq = model.item_emb(pos)

                    pos_eseq = model.item_emb(pos_sub.detach())

                    pos_affinity = pairseq_dist_affinity(pos_eseq,single_dist)

                    #pos_affinity = approximate_batch_affinity_fixed_vectorized(pos_eseq)

                    pos_logits_dist = torch.cdist(pos_logits,pos_logits)**2

                    pos_lap = pos_affinity*pos_logits_dist

                    pos_man_reg = pos_lap.sum()

                # agg_pos = aggregate_seq(model.item_emb(pos.clone().detach()))
                # agg_pos_affinity = hyperbolic_pairwise_distance(agg_pos, eps=1e-5)
                # pos_logits_dist = torch.cdist(pos_logits,pos_logits)**2
                # agg_pos_lap = agg_pos_affinity*pos_logits_dist
                # pos_man_reg = agg_pos_lap.sum()
                    

            else:
                pos_man_reg=torch.tensor(0)

            neg_lambda_man_reg = self.config['neg_lambda_reg']

            if neg_lambda_man_reg > 0:
                with torch.amp.autocast('cuda'):

                    idx_samples = torch.randint(0, self.config['maxlen'], (self.config['num_items_sampled'],), device=pos.device)
                    neg_sub = neg[:, idx_samples]

                    #neg_eseq = model.item_emb(neg)

                    neg_eseq = model.item_emb(neg_sub.detach())

                    neg_affinity = pairseq_dist_affinity(neg_eseq,single_dist)

                    #neg_affinity = approximate_batch_affinity_fixed_vectorized(neg_eseq)

                    neg_logits_dist = torch.cdist(neg_logits,neg_logits)**2

                    neg_lap = neg_affinity*neg_logits_dist

                    neg_man_reg = neg_lap.sum()

                # agg_neg = aggregate_seq(model.item_emb(neg.clone().detach()))
                # agg_neg_affinity = hyperbolic_pairwise_distance(agg_neg, eps=1e-5)
                # neg_logits_dist = torch.cdist(neg_logits,neg_logits)**2
                # agg_neg_lap = agg_neg_affinity*neg_logits_dist
                # neg_man_reg = agg_neg_lap.sum()
                    

            else:
                neg_man_reg=torch.tensor(0)

            if pos_lambda_man_reg > 0:
                batch_loss += pos_lambda_man_reg*pos_man_reg
            if neg_lambda_man_reg > 0:
                batch_loss += neg_lambda_man_reg*neg_man_reg
            #print('emb pos norms = {:.4f} emb neg norms = {:.4f} loss = {:.4f} pos_reg = {:.4f} neg_reg = {:.4f}'.format(torch.mean(torch.linalg.norm(pos_eseq,dim=-1),dim=-1).mean(),torch.mean(torch.linalg.norm(neg_eseq,dim=-1),dim=-1).mean(),criterion(pos_logits[indices], pos_labels[indices])+criterion(neg_logits[indices], neg_labels[indices]),pos_lambda_man_reg*pos_man_reg,neg_lambda_man_reg*neg_man_reg))
            
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()




            for param in model.item_emb.parameters():
                param.data = project_embeddings(param.data, c=self.config['geometry_c'], eps=1e-5)

            loss += batch_loss.item()

        # with torch.no_grad():
        #     emb = model.item_emb.weight.data  # or any embedding you want
        #     # c = ce.curvature(emb)
        #     # print("Estimated curvature c =", c)

        #     emb1 = emb.unsqueeze(1)

        #     emb2 = emb.unsqueeze(0)

        #     D = hyperbolic_dist(emb1, emb2, c=0.08226903980753697)

        #     K = torch.exp(-D**2)

        #     U, S, V = torch.svd(K)

        #     ratio = S[1] / S[0]

        #     print("Estimated curvature SVD c =", ratio.item())

        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs.unsqueeze(0))
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits.cpu().numpy().squeeze()

    def predict_sequential(self, target_seq, seen_seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        n_seen = len(seen_seq)
        n_targets = len(target_seq)
        seq = np.concatenate([seen_seq, target_seq])

        with torch.no_grad():
            pad_seq = torch.as_tensor(
                np.pad(
                    seq, (max(0, maxlen-n_seen), 0),
                    mode = 'constant',
                    constant_values = model.pad_token
                ),
                dtype = torch.int64,
                device = device
            )
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1))
            log_feats = model.log2feats(log_seqs)
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = final_feat.matmul(item_embs.T)
        return logits.cpu().numpy()
        
