import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO

from pyro.contrib.gp.kernels import Matern52
from pyro.contrib.gp.util import conditional

class SpatialIndianBuffetProcess(PyroModule):

    def __init__(self, n_latent_factors: int = 20, device: str = None, low_rank_approximation_dimension: int = 20, length_scale: int = 100):
        # identify device
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = device

        # Setup as a tensor, makes life easier, we coerce to a float for ease.
        self.n_latent_factors = torch.tensor(n_latent_factors).float().to(device)

        # Low rank dimension of MVN approximmation sampling.
        self.low_rank_approximation_dimension = low_rank_approximation_dimension

        # Determines the degree of correlation between factors due purely based on their distance
        # TODO: Make some heuristic, probably based on co-occurence within some random sasmple.
        self.length_scale = length_scale
    
    def model(self, coordinates, count_matrix, group_assignments):
        N, D = count_matrix.shape
        K = int(self.n_latent_factors.item())
        G = int(torch.max(group_assignments).item() + 1)

        size_factor = torch.log(count_matrix.sum(axis=1) / count_matrix.sum(axis=1).mean())

        torch.tensor(np.log((count_matrix.sum(axis=1) / adata.X.sum(axis=1).mean())))
        
        # scaling mu is the strongest impact. 0 cuts a good balance, higher causes K to scale rapidly
        mu = pyro.param("mu", torch.tensor(0.0, device=self.device, dtype=dtype))
        tau = pyro.param("tau", torch.tensor(1.0, device=self.device, dtype=dtype), constraint=constraints.positive)
        phi = pyro.param("phi", torch.tensor(self.length_scale, device=self.device, dtype=dtype), constraint=constraints.positive)

        # Setting up GP (MVN draw).
        kernel = Matern52(input_dim=2, lengthscale=phi)
        cov_matrix = kernel(coordinates)
        scaled_cov_matrix = (1.0 / tau) * cov_matrix
        mean_vec = mu * torch.ones(N, device=self.device, dtype=dtype)

        # K latent factor GP Sampling.
        with pyro.plate("latent_features", K):
            # Low Rank Adaptation to avoid destroying my device.
            cov_factor = pyro.param("cov_factor", torch.randn(K, N, self.low_rank_approximation, device=self.device)).to(self.device)
            cov_diag = pyro.param("cov_diag", torch.ones(K, N, device=self.device) * 1e-2, constraint=dist.constraints.positive).to(device)

            u_k = pyro.sample(
                "u_k",
                dist.LowRankMultivariateNormal(
                    loc=mean_vec.expand(K, N),       
                    cov_factor=cov_factor,           
                    cov_diag=cov_diag                
                )
            )
        u_k_T = u_k.transpose(0, 1)

        # Latent feature stick-breaking
        with pyro.plate("features", K):
            sigmoid_u_k = torch.sigmoid(u_k)
            pi_k = torch.cumprod(sigmoid_u_k, dim=0)
        pi_expand = pi_k.transpose(0, 1)
        

        # FIXME:
        # I don't think this is needed, but keeping around until I'm sure
        # with pyro.plate("observations", N):
        #     z = pyro.sample("z", dist.Bernoulli(probs=pi_expand).to_event(1))  # [N, K]

        W = pyro.sample(
            "W",
            dist.Normal(torch.zeros(K, D, device=self.device, dtype=dtype),
                        torch.ones(K, D, device=self.device, dtype=dtype)).to_event(2)
        )

        # Setting up NB draw
        # Multiple folders (when that becomes a problem)
        folder_logit = pyro.param(
            "folder_logit",
            torch.zeros(G, D, device=self.device, dtype=dtype)
        )
        r = pyro.sample(
            "r",
            dist.Gamma(torch.full((D,), 0.0, device=self.device, dtype=dtype), # 0 makes more sense, right?
                    torch.full((D,), 1.0, device=self.device, dtype=dtype)).to_event(1)
        )

        # Compute latent feature contribution to expression.
        # features = z * sigmoid_to_interval(u_k_T) # FIXME: is this interval bounding needed, testing without.
        features = z * u_k_T
        logits = features @ W 
        logits = logits + size_factor.reshape(-1, 1)
        logits = logits + folder_logit[group_assignments.long()]
        # logits = torch.clamp(logits, -15, 15) # FIXME: I don't think clamping is needed here.

        with pyro.plate("data", N):
            # Sample NB.
            pyro.sample("count_matrix", dist.NegativeBinomial(total_count=r, logits=logits).to_event(1), obs=count_matrix)

    def guide(self, coordinates, count_matrix, group_assignments):
        N, D = count_matrix.shape
        K = int(self.n_latent_factors.item())
        size_factor = torch.log(X.sum(axis=1) / X.sum(axis=1).mean())

        # Variational distribution for u_k (latent GPs): mean-field Gaussian
        u_loc = pyro.param(
            "u_loc",
            torch.zeros(K, N, device=self.device),
        )
        u_scale = pyro.param(
            "u_scale",
            0.1 * torch.ones(K, N, device=self.device),
            constraint=constraints.positive
        )
        with pyro.plate("latent_features", K):
            pyro.sample("u_k", dist.Normal(u_loc, u_scale).to_event(1))

        # Variational distribution for stick-breaking IBP: Beta
        v_alpha_q = pyro.param(
            "v_alpha_q",
            torch.ones(K, device=self.device),
            constraint=constraints.positive
        )
        v_beta_q = pyro.param(
            "v_beta_q",
            torch.ones(K, device=self.device),
            constraint=constraints.positive
        )
        with pyro.plate("features", K):
            pyro.sample("v_k", dist.Beta(v_alpha_q, v_beta_q))

        # Variational distribution for W: mean-field Normal
        W_loc = pyro.param(
            "W_loc",
            torch.zeros(K, D, device=self.device)-5,
        )
        W_scale = pyro.param(
            "W_scale",
            0.1 * torch.ones(K, D, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample("W", dist.Normal(W_loc, W_scale).to_event(2))

        # Variational distribution for r: Gamma
        r_alpha_q = pyro.param(
            "r_alpha_q",
            torch.full((D,), 2.0, device=self.device),
            constraint=constraints.positive
        )
        r_beta_q = pyro.param(
            "r_beta_q",
            torch.full((D,), 1.0, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample("r", dist.Gamma(r_alpha_q, r_beta_q).to_event(1))

        # Distribution for tau: Gamma
        tau_alpha_q = pyro.param(
            "tau_alpha_q",
            torch.full((), 2.0, device=self.device),
            constraint=constraints.positive
        )
        tau_beta_q = pyro.param(
            "tau_beta_q",
            torch.full((), 1.0, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample("tau", dist.Gamma(tau_alpha_q, tau_beta_q))

    def fit(coordinates, count_matrix, group_assignments, num_steps=300_000, lr=0.01, clear_param_store = True):
        if clear_param_store == True:
            pyro.clear_param_store()

        optimizer = ClippedAdam({"lr": lr, "clip_norm": 5.0})

        svi = SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=JitTrace_ELBO(num_particles=1),
        )

        for step in range(num_steps):
            loss = svi.step(
                coordinates=coordinates,
                count_matrix=count_matrix,
                group_assignments=group_assignments,
            )
            wandb.log({
                "loss": loss,
                "mean_logit": pyro.get_param_store()["u_loc"].mean().item(),
                "feature_sparsity": (pyro.get_param_store()["u_loc"] > 0).float().mean().item(),
            })

            if step % 100 == 0 or step == num_steps - 1:
                print(f"[{step:04d}] ELBO loss: {loss:.2f}")

class IndianBuffetProcess(PyroModule):
    def __init__(self, n_latent_factors: int = 20, device: str = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.dtype = torch.float
        self.n_latent_factors = torch.tensor(n_latent_factors).float().to(self.device)

    def model(self, X, size_factor, group_assignments, alpha=1.0):
        N, D = X.shape
        K = int(self.n_latent_factors.item())
        G = int(torch.max(group_assignments).item() + 1)

        # IBP stick-breaking process
        alpha_param = pyro.param(
            "alpha",
            torch.tensor(alpha, device=self.device, dtype=self.dtype),
            constraint=constraints.positive
        )
        beta_param = pyro.param(
            "beta",
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
            constraint=constraints.positive
        )

        with pyro.plate("ibp_features", K):
            v = pyro.sample("v", dist.Beta(alpha_param, beta_param))

        pi = torch.cumprod(v, dim=0)

        W = pyro.sample(
            "W",
            dist.Normal(torch.zeros(K, D, device=self.device),
                        torch.ones(K, D, device=self.device)).to_event(2)
        )

        r = pyro.sample(
            "r",
            dist.Gamma(torch.full((D,), 2.0, device=self.device),
                       torch.full((D,), 1.0, device=self.device)).to_event(1)
        )

        folder_logit = pyro.param(
            "folder_logit",
            torch.zeros(G, D, device=self.device)
        )

        with pyro.plate("observations", N):
            z = pyro.sample("Z", dist.Bernoulli(pi).to_event(1))  # [N, K]
            logits = z @ W + size_factor.reshape(-1, 1)
            logits = logits + folder_logit[group_assignments.long()]
            logits = torch.clamp(logits, -15, 15)
            pyro.sample("X", dist.NegativeBinomial(total_count=r, logits=logits).to_event(1), obs=X)

    def guide(self, X, size_factor, group_assignments, alpha=1.0):
        N, D = X.shape
        K = int(self.n_latent_factors.item())

        with pyro.plate("ibp_features", K):
            qv_alpha = pyro.param("qv_alpha", torch.ones(K, device=self.device),
                                  constraint=constraints.greater_than(1e-2))
            qv_beta = pyro.param("qv_beta", torch.ones(K, device=self.device),
                                 constraint=constraints.greater_than(1e-2))
            pyro.sample("v", dist.Beta(qv_alpha, qv_beta))

        with pyro.plate("observations", N):
            qz_logits = pyro.param("qz_logits", torch.randn(N, K, device=self.device) * 0.01 - 2.0)
            pyro.sample("Z", dist.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(0.5, device=self.device),
                logits=qz_logits
            ).to_event(1))

        qW_loc = pyro.param("qW_loc", torch.randn(K, D, device=self.device))
        qW_scale = pyro.param("qW_scale", torch.ones(K, D, device=self.device), constraint=constraints.positive)
        pyro.sample("W", dist.Normal(qW_loc, qW_scale).to_event(2))

        qr_alpha = pyro.param("qr_alpha", torch.ones(D, device=self.device), constraint=constraints.positive)
        qr_beta = pyro.param("qr_beta", torch.ones(D, device=self.device), constraint=constraints.positive)
        pyro.sample("r", dist.Gamma(qr_alpha, qr_beta).to_event(1))

    def fit(self, X, size_factor, group_assignments, num_steps=300_000, lr=0.01, clear_param_store=True):
        if clear_param_store:
            pyro.clear_param_store()

        optimizer = ClippedAdam({"lr": lr, "clip_norm": 5.0})
        svi = SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=JitTrace_ELBO()
        )

        for step in range(num_steps):
            loss = svi.step(X=X, size_factor=size_factor, group_assignments=group_assignments)
            if step % 100 == 0 or step == num_steps - 1:
                print(f"[{step:05d}] ELBO loss: {loss:.2f}")
