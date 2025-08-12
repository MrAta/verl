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

from .utils import get_distillation_loss_fn


class DistillLanguageModel:
    def __init__(self, model, optimizer, loss_fn, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = get_distillation_loss_fn(loss_fn)

    def _forward_batch(self, batch, teacher_logits):
        student_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        distillation_loss = self.loss_fn(batch, student_outputs.logits, teacher_logits)

        # NLL loss
        train_loss = student_outputs.loss

        total_loss = (
            student_outputs.loss * (1 - self.config.distillation_loss_ratio)
            + distillation_loss * self.config.distillation_loss_ratio
        )
        return distillation_loss, train_loss, total_loss
