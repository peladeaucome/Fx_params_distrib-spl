import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from ..utils import safe_log
from typing import Literal


class SoftplusMax(nn.Softplus):
    def __init__(self, dim: int = 1, beta: int = 1, threshold: int = 20) -> None:
        super().__init__(beta, threshold)
        self.dim = dim

    def forward(self, x: Tensor):
        out = super().forward(x)
        out = out / out.sum(self.dim, keepdim=True)
        return out


class Distribution(nn.Module):
    """
    Distribution object.
    It computes distribution parameters based on an input embedding,
    and then can compute the entropy and sample from the distribution.
    The entropy can be computed either directly (closed form or lower bound)
    or with the Monte Carlo method.
    """

    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__()
        self.params_dim = params_dim
        self.embedding_dim = embedding_dim
        self.base_entropy = base_entropy

    def sample(self, embedding: Tensor) -> Tensor:
        raise NotImplementedError()

    def entropy(self, embedding: Tensor) -> Tensor:
        raise NotImplementedError()

    def log_pdf(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        z = self.sample(embedding)
        H = self.entropy(embedding)
        return z, H

    def sample_entropy_mixing(self, embedding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, H = self.sample_and_entropy(embedding)
        bs = z.size(0)
        z = z.unsqueeze(1)
        mix = torch.ones(bs, 1, device=z.device)
        return z, H, mix


class Dirac(Distribution):
    def __init__(self, params_dim: int, embedding_dim: int):
        self.get_z = torch.nn.Linear(embedding_dim, params_dim)

    def sample(self, embedding: Tensor) -> Tensor:
        z = self.get_z(embedding)
        return z

    def entropy(self, embedding: Tensor) -> Tensor:
        bs = embedding.size(0)
        return torch.zeros(bs, device=embedding.device)


class Uniform(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)

        self.get_mu = torch.nn.Linear(embedding_dim, params_dim)
        self.get_sigma = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, params_dim), torch.nn.Softplus()
        )

    def log_pdf(self, sample, mu, sigma):
        return safe_log(sigma / 4).sum(1)

    def sample(self, embedding: Tensor) -> Tensor:
        mu = self.get_mu(embedding)
        sigma = -self.get_sigma(embedding) + 2

        eps = torch.rand_like(mu) * 4 - 2
        z = mu + eps * sigma
        return z

    def entropy(self, embedding: Tensor) -> Tensor:
        sigma = -self.get_sigma(embedding) + 2

        H = safe_log(sigma).sum(1) + 1.3863 * self.params_dim
        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        mu = self.get_mu(embedding)
        sigma = -self.get_sigma(embedding) + 2

        eps = torch.rand_like(mu) * 4 - 2
        z = mu + eps * sigma

        H = safe_log(sigma).sum(1) + 1.3863 * self.params_dim
        return z, H


class Gaussian(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)

        self.entropy_constant: float = params_dim / 2 * (1 + np.log(2 * np.pi))

        self.get_mu = nn.Linear(embedding_dim, params_dim)
        self.get_sigma = nn.Sequential(
            torch.nn.Linear(embedding_dim, params_dim), nn.Softplus()
        )

    def log_pdf(self, x: Tensor, mu: Tensor, sigma2: Tensor):
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        out = -((x - mu).square() / sigma2).sum(1) * 0.5
        out = out - 0.5 * safe_log(
            torch.pow(2 * torch.ones_like(out) * np.pi, d) * torch.prod(sigma2, axis=1)
        )
        return out

    def sample(self, embedding: Tensor) -> Tensor:
        mu = self.get_mu(embedding)
        sigma = self.get_sigma(embedding)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu
        return z

    def entropy(self, embedding: Tensor) -> Tensor:
        sigma = self.get_sigma(embedding)
        H = self.entropy_constant + safe_log(sigma).sum(
            1
        )  # Not divided by two as it is the log of squared values of sigma
        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        mu: Tensor = self.get_mu(embedding)
        sigma: Tensor = self.get_sigma(embedding)
        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        if self.base_entropy == "MC":
            H = -self.log_pdf(x=z, mu=mu, sigma2=sigma.square())
        else:
            H = self.entropy_constant + safe_log(sigma).sum(
                1
            )  # Not divided by two as it is the log of squared values of sigma
        return z, H


class Gaussian_LogSigma(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)

        self.entropy_constant: float = params_dim / 2 * (1 + np.log(2 * np.pi))

        self.get_mu = nn.Linear(embedding_dim, params_dim)
        self.get_logsigma = nn.Sequential(
            torch.nn.Linear(embedding_dim, params_dim),
        )

    def log_pdf(self, x: Tensor, mu: Tensor, sigma2: Tensor):
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        out = -((x - mu).square() / sigma2).sum(1) * 0.5
        out = out - 0.5 * safe_log(
            torch.pow(2 * torch.ones_like(out) * np.pi, d) * torch.prod(sigma2, axis=1),
            eps=0,
        )
        return out

    def get_params(self, embedding) -> tuple[Tensor]:
        mu: Tensor = self.get_mu(embedding)
        sigma: Tensor = self.get_logsigma(embedding)

        min_s = -4
        max_s = 2
        sigma = torch.sigmoid(sigma)
        sigma = sigma*(max_s-min_s)+min_s
        # sigma = -torch.nn.functional.softplus(sigma) + 2
        return mu, sigma

    def sample(self, embedding: Tensor) -> Tensor:
        mu, log_sigma = self.get_params(embedding)

        eps = torch.randn_like(mu)
        z = eps * log_sigma.exp() + mu
        return z

    def entropy(self, embedding: Tensor) -> Tensor:
        mu, log_sigma = self.get_params(embedding)

        H = self.entropy_constant + log_sigma.sum(
            1
        )  # Not divided by two as it is the log of squared values of sigma
        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        mu, log_sigma = self.get_params(embedding)

        eps = torch.randn_like(mu)
        z = eps * log_sigma.exp() + mu

        if self.base_entropy == "MC":
            H = -self.log_pdf(x=z, mu=mu, sigma2=log_sigma.exp().square())
        else:
            H = self.entropy_constant + log_sigma.sum(
                1
            )  # Not divided by two as it is the log of squared values of sigma
        return z, H


class Normal(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)

        self.entropy_constant: Tensor = (
            params_dim / 2 * (1 + np.log(2 * np.pi)) * torch.ones(1, 1)
        )

    def sample(self, embedding: Tensor) -> Tensor:
        bs = embedding.size(0)
        dim = self.params_dim
        z = torch.randn(bs, dim, device=embedding.device)
        return z

    def entropy(self, embedding: Tensor) -> Tensor:
        bs = embedding.size(0)
        dim = self.params_dim

        H = self.entropy_constant.to(embedding.device).expand(bs, dim)
        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        bs = embedding.size(0)
        dim = self.params_dim
        z = torch.randn(bs, dim, device=embedding.device)

        H = self.entropy_constant.to(embedding.device).expand(bs, dim)
        return z, H


class GMMUniform(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        num_mixtures: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)
        self.num_mixtures = num_mixtures

        self.get_mu = nn.Sequential(
            nn.Linear(embedding_dim, params_dim * num_mixtures),
            nn.Unflatten(1, (num_mixtures, params_dim)),
        )
        self.get_sigma = nn.Sequential(
            nn.Linear(embedding_dim, params_dim * num_mixtures),
            # nn.Softplus(),
            # SoftplusWithEps(eps=1e-4),
            nn.Unflatten(1, (num_mixtures, params_dim)),
        )
    
    def get_params(self, embedding):
        mu_list: Tensor = self.get_mu(embedding)
        sigma_list: Tensor = self.get_sigma(embedding)

        sigma_list = torch.sigmoid(sigma_list)*3+1e-4
        return mu_list, sigma_list

    def sample(self, embedding: Tensor) -> Tensor:
        device = embedding.device
        mu_list, sigma_list = self.get_params(embedding)

        bs = embedding.size(0)
        eps_list = torch.randn_like(mu_list)
        z_list = eps_list * sigma_list + mu_list

        choice = torch.randint(
            high=self.num_mixtures, size=(bs, 1, 1), device=device
        ).expand(size=(bs, 1, self.params_dim))

        z = z_list.gather(1, choice).squeeze(1)
        # sigma = sigma_list.gather(1, choice).squeeze(1)

        return z

    def gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor):
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution
        with diagonal covariance matrix.
        """
        d = self.params_dim
        out = torch.exp(
            -torch.sum((x - mu).square() / sigma2, axis=3, keepdim=True) * 0.5
        )
        out = out / torch.sqrt(
            np.power(2 * np.pi, d) * torch.prod(sigma2, axis=3, keepdim=True)
        )
        return out

    def log_gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor) -> Tensor:
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        log_num = -torch.sum((x - mu).square() / sigma2, axis=3, keepdim=True) * 0.5
        log_den = torch.sum(safe_log(sigma2), axis=3, keepdim=True)
        log_den = log_den + safe_log(2 * np.pi * torch.ones_like(log_den)) * d
        log_den = log_den / 2
        out = log_num - log_den
        return out

    def compute_entropy_direct(self, mu_list: Tensor, sigma_list: Tensor):
        mu_list1 = mu_list.unsqueeze(2)
        mu_list2 = mu_list.unsqueeze(1)
        sigma_list1 = sigma_list.unsqueeze(2)
        sigma_list2 = sigma_list.unsqueeze(1)

        sigma2 = sigma_list1.square() + sigma_list2.square()

        H = self.gaussian(mu_list1, mu_list2, sigma2).squeeze(3)
        H = -safe_log(H.mean(1)).mean(1)
        return H

    def compute_entropy_sample(self, z: Tensor, mu_list: Tensor, sigma_list: Tensor):

        H = self.log_gaussian(
            z.unsqueeze(1).unsqueeze(2),
            mu_list.unsqueeze(1),
            sigma_list.square().unsqueeze(1),
        ).squeeze(3)
        H = torch.exp(H)
        H = -safe_log(H.mean(2))
        H = H.mean(1)
        return H

    def entropy(self, embedding: Tensor) -> Tensor:
        mu_list, sigma_list = self.get_params(embedding)

        H = self.compute_entropy_direct(mu_list, sigma_list)

        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        device = embedding.device
        mu_list, sigma_list = self.get_params(embedding)

        bs = embedding.size(0)

        eps_list = torch.randn_like(mu_list)
        z_list = eps_list * sigma_list + mu_list

        choice = torch.randint(
            high=self.num_mixtures, size=(bs, 1, 1), device=device
        ).expand(size=(bs, 1, self.params_dim))

        z = z_list.gather(1, choice).squeeze(1)

        if self.training and self.base_entropy == "direct":
            H = self.compute_entropy_direct(mu_list, sigma_list)
        else:
            H = self.compute_entropy_sample(z, mu_list, sigma_list)

        return z, H


class GMMFull(Distribution):
    def __init__(
        self,
        params_dim: int,
        embedding_dim: int,
        num_mixtures: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(params_dim, embedding_dim, base_entropy)
        self.num_mixtures = num_mixtures

        self.get_mu = nn.Sequential(
            nn.Linear(embedding_dim, params_dim * num_mixtures),
            nn.Unflatten(1, (num_mixtures, params_dim)),
        )

        self.get_sigma = nn.Sequential(
            nn.Linear(embedding_dim, params_dim * num_mixtures),
            SoftplusWithEps(eps=1e-2),
            nn.Unflatten(1, (num_mixtures, params_dim)),
        )

        self.get_mix = nn.Sequential(
            nn.Linear(embedding_dim, num_mixtures),
            nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

        self.eps = 1e-6

    def get_params(self, embedding):
        mu_list: Tensor = self.get_mu(embedding)
        sigma_list: Tensor = self.get_sigma(embedding)
        mix: Tensor = self.get_mix(embedding)

        sigma_list = torch.sigmoid(sigma_list)*3+1e-4
        return mu_list, sigma_list, mix

    def sample(self, embedding: Tensor) -> Tensor:
        mu_list, sigma_list, mix = self.get_params(embedding)

        bs = embedding.size(0)
        choice = (
            torch.multinomial(mix, num_samples=1)
            .unsqueeze(1)
            .expand(size=(bs, 1, self.params_dim))
        )

        mu = mu_list.gather(1, choice).squeeze(1)
        sigma = sigma_list.gather(1, choice).squeeze(1)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        return z

    def gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor) -> Tensor:
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        out = (-0.5 * (x - mu).square() / sigma2).sum(axis=3, keepdim=True).exp()
        out = out / (np.power(2 * np.pi, d) * (sigma2.prod(axis=3, keepdim=True))).sqrt()

        return out

    def log_gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor) -> Tensor:
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        log_num = -torch.sum((x - mu).square() / sigma2, axis=3, keepdim=True) * 0.5
        log_den = torch.sum(safe_log(sigma2, eps=self.eps), axis=3, keepdim=True)
        log_den = (
            log_den + safe_log(2 * np.pi * torch.ones_like(log_den), eps=self.eps) * d
        )
        log_den = log_den / 2
        out = log_num - log_den
        return out

    def compute_entropy_direct(self, mu_list: Tensor, sigma_list: Tensor, mix: Tensor):
        bs = mu_list.size(0)
        num_mix = self.num_mixtures
        dim = self.params_dim

        mu_list1 = mu_list.view(bs, num_mix, 1, dim)
        mu_list2 = mu_list.view(bs, 1, num_mix, dim)
        sigma_list1 = sigma_list.view(bs, num_mix, 1, dim)
        sigma_list2 = sigma_list.view(bs, 1, num_mix, dim)
        sigma2 = sigma_list1.square() + sigma_list2.square()
        mix1 = mix.view(bs, 1, num_mix)
        mix2 = mix.view(bs, num_mix)

        H = self.gaussian(mu_list1, mu_list2, sigma2).squeeze(3)
        H = -(safe_log((H * mix1).sum(2), eps=1e-6) * mix2).sum(1)
        return H

    def entropy(self, embedding: Tensor) -> Tensor:
        mu_list, sigma_list, mix = self.get_params(embedding)

        H = self.compute_entropy_direct(mu_list, sigma_list, mix)

        return H

    def sample_and_entropy(self, embedding: Tensor) -> tuple[Tensor]:
        mu_list, sigma_list, mix = self.get_params(embedding)

        bs = embedding.size(0)

        choice = (
            torch.multinomial(mix, num_samples=1)
            .unsqueeze(1)
            .expand(size=(bs, 1, self.params_dim))
        )

        mu = torch.gather(mu_list, 1, choice).squeeze(1)
        sigma = torch.gather(sigma_list, 1, choice).squeeze(1)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        if self.training and self.base_entropy == "direct":
            H = self.compute_entropy_direct(mu_list, sigma_list, mix)
        else:
            H = self.log_gaussian(
                z.unsqueeze(1).unsqueeze(2),
                mu_list.unsqueeze(1),
                sigma_list.square().unsqueeze(1),
            ).squeeze(3)
            H = torch.exp(H)
            H = -safe_log((H * (mix.unsqueeze(1))).sum(2), eps=1e-6)
            H = (H * mix).sum(1)

        return z, H

    def sample_entropy_mixing(self, embedding: Tensor):
        mu_list, sigma_list, mix = self.get_params(embedding)

        eps = torch.randn_like(mu_list)
        z = eps * sigma_list + mu_list

        if self.training and self.base_entropy == "direct":
            H = self.compute_entropy_direct(mu_list, sigma_list, mix)
        else:
            H = self.log_gaussian(
                z.unsqueeze(2),
                mu_list.unsqueeze(1),
                sigma_list.square().unsqueeze(1),
            ).squeeze(3)
            H = torch.exp(H)
            H = -safe_log((H * (mix.unsqueeze(1))).sum(2))
            H = (H * mix).sum(1)

        return z, H, mix



class ConstantDistribution(nn.Module):
    """
    Distribution object.
    It computes distribution parameters based on an input embedding,
    and then can compute the entropy and sample from the distribution.
    The entropy can be computed either directly (closed form or lower bound)
    or with the Monte Carlo method.
    """

    def __init__(
        self,
        dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__()
        self.dim = dim
        self.base_entropy = base_entropy

    def sample(self, bs: int = 1) -> Tensor:
        raise NotImplementedError()

    def entropy(self, bs: int = 1) -> Tensor:
        raise NotImplementedError()

    def log_pdf(self, bs: int = 1) -> Tensor:
        raise NotImplementedError()

    def sample_and_entropy(self, bs: int = 1) -> tuple[Tensor, Tensor]:
        z = self.sample()
        H = self.entropy()
        return z, H

    def sample_entropy_mixing(self, bs: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        z, H = self.sample_and_entropy()
        bs = z.size(0)
        z = z.unsqueeze(1)
        mix = torch.ones(bs, 1, device=z.device)
        return z, H, mix


class ConstantGaussian(ConstantDistribution):
    def __init__(
        self,
        dim: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(dim, base_entropy)

        self.entropy_constant: float = dim / 2 * (1 + np.log(2 * np.pi))

        self.mu = nn.Parameter(torch.zeros(1, dim))
        self.sigma = nn.Parameter(torch.ones(1, dim))

    def get_sigma(self, bs: int = 1):
        return nn.functional.softplus(self.sigma).expand(bs, self.dim)

    def get_mu(self, bs: int = 1):
        return self.mu.expand(bs, self.dim)

    def log_pdf(self, x: Tensor, mu: Tensor, sigma2: Tensor):
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.params_dim
        out = -((x - mu).square() / sigma2).sum(1) * 0.5
        out = out - 0.5 * safe_log(
            torch.pow(2 * torch.ones_like(out) * np.pi, d) * torch.prod(sigma2, axis=1)
        )
        return out

    def sample(self, bs: int = 1) -> Tensor:
        mu = self.get_mu(bs)
        sigma = self.get_sigma(bs)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu
        return z

    def entropy(self, bs: int = 1) -> Tensor:
        sigma = self.get_sigma(bs)
        H = self.entropy_constant + safe_log(sigma).sum(
            1
        )  # Not divided by two as it is the log of squared values of sigma
        return H

    def sample_and_entropy(self, bs: int = 1) -> tuple[Tensor]:
        mu: Tensor = self.get_mu(bs)
        sigma: Tensor = self.get_sigma(bs)
        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        if self.base_entropy == "direct":
            H = self.entropy_constant + safe_log(sigma).sum(
                1
            )  # Not divided by two as it is the log of squared values of sigma
        elif self.base_entropy == "MC":
            H = -self.log_pdf(x=z, mu=mu, sigma2=sigma.square())
        return z, H


class ConstantGMMFull(ConstantDistribution):
    def __init__(
        self,
        dim: int,
        num_mixtures: int,
        base_entropy: Literal["direct", "MC"] = "direct",
    ):
        super().__init__(dim, base_entropy)
        self.num_mixtures = num_mixtures

        # (1, num_mixtures, params_dim)

        self.mu = nn.Parameter(torch.randn(1, num_mixtures, dim) * 2)
        self.sigma = nn.Parameter(torch.ones(1, num_mixtures, dim))
        self.mix = nn.Parameter(torch.zeros(1, num_mixtures))

        self.spm = SoftplusMax(dim=1)

    def get_mu(self, bs: int = 1):
        num_mix = self.num_mixtures
        dim = self.dim
        return self.mu.expand(bs, num_mix, dim)

    def get_sigma(self, bs: int = 1):
        num_mix = self.num_mixtures
        dim = self.dim
        return nn.functional.softplus(self.sigma).expand(bs, num_mix, dim)

    def get_mix(self, bs: int = 1):
        return self.spm(self.mix).expand(bs, self.num_mixtures)

    def sample(self, bs: int = 1) -> Tensor:
        mu_list: Tensor = self.get_mu(bs)
        sigma_list: Tensor = self.get_sigma(bs)
        mix: Tensor = self.get_mix(bs)

        choice = (
            torch.multinomial(mix, num_samples=1)
            .unsqueeze(1)
            .expand(size=(bs, 1, self.params_dim))
        )

        mu = mu_list.gather(1, choice).squeeze(1)
        sigma = sigma_list.gather(1, choice).squeeze(1)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        return z

    def gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor) -> Tensor:
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.dim
        out = torch.exp(
            -torch.sum((x - mu).square() / sigma2, axis=3, keepdim=True) * 0.5
        )
        out = out / torch.sqrt(
            np.power(2 * np.pi, d) * torch.prod(sigma2, axis=3, keepdim=True)
        )
        return out

    def log_gaussian(self, x: Tensor, mu: Tensor, sigma2: Tensor) -> Tensor:
        """
        Computes N(x; mu, sigma), N being a multivariate gaussian distribution with diagonal covariance matrix.
        """
        d = self.dim
        log_num = -torch.sum((x - mu).square() / sigma2, axis=3, keepdim=True) * 0.5
        log_den = torch.sum(safe_log(sigma2), axis=3, keepdim=True)
        log_den = log_den + safe_log(2 * np.pi * torch.ones_like(log_den)) * d
        log_den = log_den / 2
        out = log_num - log_den
        return out

    def compute_entropy_direct(self, mu_list: Tensor, sigma_list: Tensor, mix: Tensor):
        num_mix = self.num_mixtures
        dim = self.dim

        bs = mu_list.size(0)

        mu_list1 = mu_list.view(bs, num_mix, 1, dim)
        mu_list2 = mu_list.view(bs, 1, num_mix, dim)
        sigma_list1 = sigma_list.view(bs, num_mix, 1, dim)
        sigma_list2 = sigma_list.view(bs, 1, num_mix, dim)
        sigma2 = sigma_list1.square() + sigma_list2.square()
        mix1 = mix.view(bs, 1, num_mix)
        mix2 = mix.view(bs, num_mix)

        H = self.gaussian(mu_list1, mu_list2, sigma2).squeeze(3)
        H = -(safe_log((H * mix1).sum(2)) * mix2).sum(1)
        return H

    def entropy(self, bs: int = 1) -> Tensor:
        mu_list: Tensor = self.get_mu(bs)
        sigma_list: Tensor = self.get_sigma(bs)
        mix: Tensor = self.get_mix(bs)

        H = self.compute_entropy_direct(mu_list, sigma_list, mix)

        return H

    def sample_and_entropy(self, bs: int = 1) -> tuple[Tensor]:
        mu_list: Tensor = self.get_mu(bs)
        sigma_list: Tensor = self.get_sigma(bs)
        mix: Tensor = self.get_mix(bs)

        choice = (
            torch.multinomial(mix, num_samples=1)
            .unsqueeze(1)
            .expand(size=(bs, 1, self.dim))
        )

        mu = torch.gather(mu_list, 1, choice).squeeze(1)
        sigma = torch.gather(sigma_list, 1, choice).squeeze(1)

        eps = torch.randn_like(mu)
        z = eps * sigma + mu

        if self.base_entropy == "direct":
            H = self.compute_entropy_direct(mu_list, sigma_list, mix)
        elif self.base_entropy == "MC":
            H = self.log_gaussian(
                z.unsqueeze(1).unsqueeze(2),
                mu_list.unsqueeze(1),
                sigma_list.square().unsqueeze(1),
            ).squeeze(3)
            H = torch.exp(H)
            H = -safe_log((H * (mix.unsqueeze(1))).sum(2))
            H = (H * mix).sum(1)

        return z, H

    def sample_entropy_mixing(self, bs: int = 1):
        mu_list: Tensor = self.get_mu(bs)
        sigma_list: Tensor = self.get_sigma(bs)
        mix: Tensor = self.get_mix(bs)

        eps = torch.randn_like(mu_list)
        z = eps * sigma_list + mu_list

        if self.base_entropy == "direct":
            H = self.compute_entropy_direct(mu_list, sigma_list, mix)
        elif self.base_entropy == "MC":
            H = self.log_gaussian(
                z.unsqueeze(2),
                mu_list.unsqueeze(1),
                sigma_list.square().unsqueeze(1),
            ).squeeze(3)
            H = torch.exp(H)
            H = -safe_log((H * (mix.unsqueeze(1))).sum(2))
            H = (H * mix).sum(1)

        return z, H, mix


class SoftplusWithEps(nn.Softplus):
    def __init__(self, beta=1, threshold=20, eps=1e-3):
        super().__init__(beta, threshold)
        self.eps = eps

    def forward(self, input):
        return super().forward(input) + self.eps
