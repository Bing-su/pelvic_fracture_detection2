from typing import Optional, Union

import torch


class AUCMLoss(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function:
        a novel loss function to directly optimize AUROC

    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e.,
                 the ratio of number of postive samples to number of total samples
    outputs:
        loss value

    Reference:
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T.,
        Large-scale Robust Deep AUC Maximization:
            A New Surrogate Loss and Empirical Studies on Medical Image Classification.
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """

    def __init__(
        self,
        margin: float = 1.0,
        imratio: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.margin = margin
        self.p = imratio
        # https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
        self.a = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        ).to(self.device)
        self.b = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        ).to(self.device)
        self.alpha = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        ).to(self.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]

        y_pred = y_pred.reshape(-1, 1)  # be carefull about these shapes
        y_true = y_true.reshape(-1, 1)
        loss = (
            (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float())
            + self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float())
            + 2
            * self.alpha
            * (
                self.p * (1 - self.p) * self.margin
                + torch.mean(
                    self.p * y_pred * (0 == y_true).float()
                    - (1 - self.p) * y_pred * (1 == y_true).float()
                )
            )
            - self.p * (1 - self.p) * self.alpha**2
        )
        return loss
