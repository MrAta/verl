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


import os
import socket

import hydra
import ray

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.distillation.ray_distiller import RayDistiller
from verl.trainer.ppo.ray_trainer import Role
from verl.workers.fsdp_workers import DistillationWorker


@hydra.main(config_path="config", config_name="distill_trainer", version_base=None)
def main(config):
    """
    Main entry point for distillation training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing distillation parameters.
    """
    run_distill(config)


def run_distill(config) -> None:
    """
    Run distillation training process.

    Args:
        config: Distillation configuration object containing all necessary parameters
                for distributed distillation training including Ray initialization settings,
                model paths, and distillation hyperparameters.
    """
    if not ray.is_initialized():
        ray.init(runtime_env=get_ppo_ray_runtime_env(), num_cpus=config.ray_init.num_cpus)

    # TODO (MrAta): instantiate the task runner with profiling options.
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    # TODO (MrAta): add timeline trace file for performance analysis.


@ray.remote(num_cpus=1)
class TaskRunner:
    """
    Ray remote class for executing distributed distillation tasks.
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.n_gpus_per_node] * config.nnodes,
        }
        self.mapping[Role.DistllationStudent] = global_pool_id
        self.mapping[Role.DistllationTeacher] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def run(self, config):
        """
        Execute the main distillation workflow.
        This method sets up the distributed environment, initializes the student and teacher models,
        and starts the distillation process.

        Args:
            config: Distillation configuration object containing all necessary parameters
                    for distributed distillation including Ray initialization settings,
                    model paths, and distillation hyperparameters.
        """
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        student_local_path = copy_to_local(config.student.model_path, use_shm=config.get("use_shm", False))

        from verl.utils import hf_tokenizer

        trust_remote_code = config.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(student_local_path, trust_remote_code=trust_remote_code)

        self.role_worker_mapping[Role.DistillationStudent] = ray.remote(DistillationWorker)
        self.role_worker_mapping[Role.DistillationTeacher] = ray.remote(DistillationWorker)

        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_distillation_dataset(config)
        val_dataset = create_distillation_dataset(config)
        train_sampler = create_distillation_sampler(config)

        distiller = RayDistiller(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_sampler=train_sampler,
        )
        distiller.init_workers()
        distiller.fit()


def create_distillation_dataset(config):
    """Create distillation dataloader."""
    pass


def create_distillation_sampler(config):
    """Create distillation sampler."""
    pass


if __name__ == "__main___":
    main()
