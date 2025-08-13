# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from verl import DataProto
from verl.trainer.distillation.losses import DISTILL_LOSS_MAP


class DistillLanguageModel:
    def __init__(self, model, optimizer, loss_fn, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        assert loss_fn in DISTILL_LOSS_MAP, f"Invalid distillation loss function: {loss_fn}"
        self.loss_fn = DISTILL_LOSS_MAP[loss_fn]

    def _forward_batch(self, batch):
        student_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # batch contains the teacher logits.
        distillation_loss = self.loss_fn(batch, student_outputs.logits)

        # NLL loss
        train_loss = student_outputs.loss

        total_loss = (
            student_outputs.loss * (1 - self.config.distillation_loss_ratio)
            + distillation_loss * self.config.distillation_loss_ratio
        )
        return distillation_loss, train_loss, total_loss

    def update_student_model(self, batch: DataProto):
        self.model.train()
        distillation_loss, train_loss, total_loss = self._forward_batch(batch)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return {
            "distillation_loss": distillation_loss.item(),
            "train_loss": train_loss.item(),
            "total_loss": total_loss.item(),
        }
