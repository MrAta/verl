# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.nn import Module

IGNORE_INDEX = -100


class PerTokenDistillationLoss(ABC, Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def per_token_loss(self, probs, teacher_probs, inf_mask):
        pass

    @staticmethod
    def _shift_and_mask(
        gt_token_ids,
        logits_or_hidden_states,
        teacher_logits_hidden_states,
        ignore_index,
        is_shifted,
    ):
        # shift logits or hidden/_states if needed or just shift label with mask check
        shifted_labels = gt_token_ids[:, 1:].contiguous()  # label always shift
        loss_mask = (shifted_labels != ignore_index).int()
        # If all tokens are ignored, skip the loss computation,
        # Otherwise loss will be NaN
        if torch.all(1 - loss_mask):
            return torch.tensor(0.0, requires_grad=True)
        shifted_logits_or_hidden_states = (
            logits_or_hidden_states if is_shifted else logits_or_hidden_states[:, :-1, :].contiguous()
        )
        shifted_teacher_logits_or_hidden_states = (
            teacher_logits_hidden_states if is_shifted else teacher_logits_hidden_states[:, :-1, :].contiguous()
        )
        return shifted_labels, shifted_logits_or_hidden_states, shifted_teacher_logits_or_hidden_states, loss_mask

    def forward(
        self,
        gt_token_ids,
        logits_or_hidden_states,
        teacher_logits_or_hidden_states,
        logits_or_hidden_states_shifted=False,
        ignore_index=IGNORE_INDEX,
        temperature=1.0,
        **kwargs,
    ):
        """
        Forward pass for the distillation loss computation.

        Args:
            gt_token_ids (torch.Tensor): Ground truth token IDs. Shape: (batch_size, sequence_length)
            logits_or_hidden_states (torch.Tensor): Logits from the student model.
                Shape: (batch_size, sequence_length, vocab_size)
            teacher_logits_or_hidden_states (torch.Tensor): Logits from the teacher model.
                Shape: (batch_size, sequence_length, vocab_size)
            logits_or_hidden_states_shifted (bool, optional): Whether the logits are already shifted.
                Defaults to False.
            ignore_index (int, optional): Index to ignore in the loss computation. Defaults to -100.
            temperature (float, optional): Temperature for distillation. Defaults to 1.0.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Computed distillation loss. Shape: scalar
        """
        # If the teacher and student token size is different, pad student logits to match the teacher's.
        # This only applies to cases where they share exactly the same vocab and tokenizer just
        # that teacher logit is padded for some training efficiency such as
        # https://huggingface.co/Qwen/Qwen1.5-72B-Chat/discussions/1#662883f568adf59b07b176d2
        if torch.all(gt_token_ids == -100):
            return torch.tensor(0.0, requires_grad=True, device=logits_or_hidden_states.device)

        logits = logits_or_hidden_states
        teacher_logits = teacher_logits_or_hidden_states
        logits_shifted = logits_or_hidden_states_shifted
        if teacher_logits.shape[-1] > logits.shape[-1]:
            pad_size = teacher_logits.shape[-1] - logits.shape[-1]
            pad_tensor = torch.zeros((*logits.shape[:-1], pad_size), dtype=logits.dtype, device=logits.device)
            logits = torch.cat([logits, pad_tensor], dim=-1)

        if temperature != 1.0:
            logits = logits / temperature
            teacher_logits = teacher_logits / temperature
        shifted_teacher_logits = teacher_logits if logits_shifted else teacher_logits[:, :-1, :].contiguous()
        shift_labels, shifted_logits, shifted_teacher_logits, loss_mask = self._shift_and_mask(
            gt_token_ids, logits, teacher_logits, ignore_index, logits_shifted
        )
        inf_mask = torch.isinf(shifted_logits)  # do we need this? (maybe won't have inf logits)

        teacher_probs = F.softmax(shifted_teacher_logits, dim=-1, dtype=torch.float32)
        probs = F.softmax(shifted_logits, dim=-1, dtype=torch.float32)
        per_token_loss = self.per_token_loss(probs, teacher_probs, inf_mask)  # [B * T,]
        distill_loss = torch.sum(per_token_loss * loss_mask) / torch.sum(loss_mask)
        # Perform temperature scaling on the loss based on Hinton's 2015 paper
        # Mathematically we should perform temperature T^2 scaling on the
        # loss to compensate for the scaling of the logits during the
        # gradient computation.
        # https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py#L167
        # does not perform such temperature scaling
        return distill_loss * (temperature**2)


class ForwardKLDiv(PerTokenDistillationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _per_token_loss(probs, teacher_probs, inf_mask):
        prod_probs = torch.masked_fill(-teacher_probs * torch.log(probs), inf_mask, 0)
        return torch.sum(prod_probs, dim=-1)

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        return self._per_token_loss(probs, teacher_probs, inf_mask)


DISTILL_LOSS_MAP = {
    "forward_kl": ForwardKLDiv,
}
