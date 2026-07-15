import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class PositiveLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            input,
            torch.clamp(self.weight, min=1e-7),
            self.bias,
        )


class MCQRNNModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dense_features: int,
    ):
        super().__init__()

        self.x_feature_extractor = nn.Sequential(
            nn.Linear(in_features, dense_features),
            nn.Sigmoid(),
            nn.Linear(dense_features, out_features),
            nn.Sigmoid(),
        )

        self.tau_embedding = PositiveLinear(1, out_features)
        self.monotone_hidden = PositiveLinear(out_features, dense_features)
        self.output_dense = PositiveLinear(dense_features, 1)
        for m in self.modules():
            if isinstance(m, PositiveLinear):
                nn.init.uniform_(m.weight, 0.01, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        tau = tau.reshape(-1, 1)

        h_x = self.x_feature_extractor(x)  # (n,out)
        h_tau = self.tau_embedding(tau)  # (r,out)
        h_x = h_x.unsqueeze(1)  # (n,1,out)
        h_tau = h_tau.unsqueeze(0)  # (1,r,out)
        merged = h_x + h_tau  # (n,r,out)
        hidden = self.activation(self.monotone_hidden(merged))
        out = self.output_dense(hidden)

        return out.squeeze(-1)  # (n,r)


class MCQRNNRegressor:
    def __init__(
        self,
        tau: NDArray[np.float64],
        out_features: int = 64,
        dense_features: int = 32,
        lr: float = 0.01,
        epochs: int = 5000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.tau = torch.tensor(tau, dtype=torch.float32, device=self.device)
        self.out_features = out_features
        self.dense_features = dense_features
        self.lr = lr
        self.epochs = epochs
        self.model: MCQRNNModel | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(
            y,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)

        self.model = MCQRNNModel(
            in_features=X.shape[1],
            out_features=self.out_features,
            dense_features=self.dense_features,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor, self.tau)
            diff = y_tensor - pred
            loss = torch.maximum(self.tau * diff, (self.tau - 1.0) * diff).mean()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] Loss={loss.item():.4f}")

        return self

    @torch.no_grad()
    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        if self.model is None:
            raise ValueError("Please call .fit() first.")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        return self.model(X_tensor, self.tau).cpu().numpy()
