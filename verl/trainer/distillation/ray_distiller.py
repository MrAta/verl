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

from typing import Optional

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType


class RayDistiller:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_sampler: Optional[Sampler] = None,
        collate_fn=None,
        device_name: Optional[str] = None,
    ):
        """
        Initialize the RayDistiller with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping: Mapping from roles to worker classes.
            resource_pool_manager: Manager for Ray resource pools.
            ray_worker_group_cls: Class for Ray worker groups.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        self.collate_fn = collate_fn
        self.device_name = device_name

        self._create_dataloader()

    def _create_dataloader(self):
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("batch_size", 1),
            num_workers=self.config.data.get("dataloader_num_workers", 1),
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.get("batch_size", 1),
            num_workers=self.config.data.get("dataloader_num_workers", 1),
            collate_fn=self.collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train datalaoder is empty!"
        assert len(self.val_dataloader) >= 1, "Validation datalaoder is empty!"

        self.total_training_steps = len(self.train_dataloader) * self.config.num_epochs

    def init_workers(self):
        """
        Initializes the distributed training workers using Ray backend.

        Creates:
        - Ray resource pools from the configuration
        - Worker groups for the student and the teacher
        """

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        student_resource_pool = self.resource_pool_manager.get_resource_pool(Role.DistillationStudent)
        student_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.DistillationStudent],
            conf=self.config,
            generates_sequences=False,
            role="student",
        )

        self.resource_pool_to_cls[student_resource_pool]["student"] = student_cls

        teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.DistillationTeacher)
        teacher_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.DistillationTeacher],
            conf=self.config,
            generates_sequences=True,
            role="teacher",
        )

        self.resource_pool_to_cls[student_resource_pool]["teacher"] = teacher_cls

        wg_kwargs = {}
        wg_kwargs["device_name"] = self.device_name

        self.student_wg = self.ray_worker_group_cls(
            resource_pool=student_resource_pool, ray_cls_with_init=student_cls, name_prefix="student_" ** wg_kwargs
        )

        self.teacher_wg = self.ray_worker_group_cls(
            resource_pool=teacher_resource_pool, ray_cls_with_init=teacher_cls, name_prefix="teacher_", **wg_kwargs
        )

        self.student_wg.init_model()
        self.teacher_wg.init_model()
