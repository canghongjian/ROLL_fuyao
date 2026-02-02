'''
auto pipeline: convert megatron checkpoint to huggingface checkpoint --> deepinsight eval
'''

import json
from fuyao_patch.convert_tools import convert_mca_to_hf, ConvertArguments
import shutil
import types
from fuyao_patch.fuyao_cli import deploy_job
import torch.distributed as dist
import os
import subprocess
from roll.utils.logging import get_logger


logger = get_logger()
def apply_megatron_patch_roll(target_class):
    """
    应用 Patch 的入口函数
    target_class: 例如 MegatronTrainStrategy
    """
    # 保存原始方法，避免无限递归
    original_save_func = target_class.save_checkpoint
    def save_checkpoint_wrapper(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        """
        这就是你的新函数，它只负责 '拦截' 和 '触发'
        """
        
        # step 1: 完整执行原本的保存逻辑 (Megatron 自己的代码)
        # 所有的 optimizer保存、barrier、upload、timer 都在这里面完成
        metrics = original_save_func(self, save_dir, global_step, ckpt_id, tag, local_state_path, **kwargs)

        # step 2: 执行你的新增逻辑 (Hook)
        # 只在 Rank 0 且配置开启时执行
        # logger.info(f"dist.is_initialized():{dist.is_initialized()}, dist.get_rank():{dist.get_rank()}")
        if dist.get_rank() == 0:
            trigger_remote_pipeline(self, save_dir, ckpt_id)
        
        return metrics

    # 替换类的方法
    target_class.save_checkpoint = save_checkpoint_wrapper
    logger.info(f"Successfully patched {target_class.__name__}.save_checkpoint")


def trigger_remote_pipeline(self, save_dir, ckpt_id):
    """
    生成配置并启动独立的编排进程
    """
    ckpt_config = self.worker_config.checkpoint_config
    di_args = ckpt_config.get('deepinsight_args', {})
    
    assert di_args, "deepinsight_args must be setted in checkpoint_config!"
    megatron_path = os.path.join(self.checkpoint_manager.uploader.output_dir, ckpt_id)
    hf_path = os.path.join(self.checkpoint_manager.uploader.output_dir, 'converted_hf', ckpt_id)

    # 读取 base eval config 用于填充默认值
    base_eval_config = {}
    base_config_path = ckpt_config.get('base_eval_config_path')
    if base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            base_eval_config = json.load(f)
    # 区分llm or mllm
    if ckpt_config.get('eval_type', 'llm') == 'llm':
        deepinsight_file_path = base_eval_config['start_file_path']
        deepinsight_task_name = f"deepinsight_convert_then_eval_{ckpt_id}"
        deepinsight_eval_datasets = self.worker_config.checkpoint_config.get('deepinsight_args').get('eval_datasets')
        deepinsight_system_envs = self.worker_config.checkpoint_config.get('deepinsight_args').get('system_envs')
        deepinsight_launch_args = self.worker_config.checkpoint_config.get('deepinsight_args').get('launch_args')

        model_dir = os.path.join(self.checkpoint_manager.uploader.output_dir, 'converted_hf')
        eval_args_str = f"--task_name {deepinsight_task_name}"

        eval_args_str += f" --site_gpu {base_eval_config.get('site_gpu', 'fuyao_b1')}"
        eval_args_str += f" --queue_gpu {base_eval_config.get('queue_gpu', 'rc-llmrl-a100')}"
        eval_args_str += f" --site_cpu {base_eval_config.get('site_cpu', 'fuyao_b1')}"
        eval_args_str += f" --queue_cpu {base_eval_config.get('queue_cpu', 'rc-cpu')}"

        eval_args_str += f" {deepinsight_launch_args} --model-dir {model_dir} {ckpt_id} {deepinsight_eval_datasets}"

        launch_eval_script = f"cd /eval_client && {deepinsight_system_envs} DEEPINSIGHT_EVAL_ROOT=/eval_client /bin/bash {deepinsight_file_path} {eval_args_str}"
    
    elif ckpt_config.get('eval_type', 'llm') == 'mllm':
        deepinsight_task_name = f"deepinsight_convert_then_eval_{ckpt_id}"
        deepinsight_file_path = base_eval_config['start_file_path']
        deepinsight_cluster_name = f" --site_gpu {base_eval_config.get('site_gpu', 'fuyao_b1')}"
        deepinsight_eval_datasets = self.worker_config.checkpoint_config.get('deepinsight_args').get('eval_datasets')
        
        deepinsight_launch_args = self.worker_config.checkpoint_config.get('deepinsight_args').get('launch_args')

        launch_eval_script = f"cd /eval_client && /bin/bash {deepinsight_file_path} --task-name {deepinsight_task_name} --cluster-name {deepinsight_cluster_name} --test-model-path {hf_path} --test-model-name {ckpt_id} --datasets {deepinsight_eval_datasets} {deepinsight_launch_args}"
    else:
        logger.info(f"error eval type:{ckpt_config.get('eval_type', 'llm')}")
        return
    bifrost_job_name = os.environ.get("BIFROST_JOB_NAME", "empty")

    # 1. 准备 Pipeline 配置数据
    pipeline_config = {
        "ckpt_id": ckpt_id,
        "experiment": base_eval_config.get('experiment', 'zhangjh37/llm_rl'),
        "convert_then_eval": self.worker_config.checkpoint_config.get('convert_then_eval', True),
        "megatron_path": megatron_path,
        "hf_path": hf_path,
        "keep_megatron_files": self.worker_config.checkpoint_config.get('keep_megatron_files', False),

        # Convert 任务配置
        "convert_job_args": 
        {
            "docker_image": base_eval_config.get('convert_image', 'infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-251218-0432'),
            "site": base_eval_config.get('site_gpu', 'fuyao_b1'),
            "partition": base_eval_config.get('queue_gpu', 'rc-llmrl-a100'),
            "node_count": base_eval_config.get('node_count', 1),
            "gpus_per_node": base_eval_config.get('gpus_per_node', 8),
            "experiment": base_eval_config.get('experiment', 'zhangjh37/llm_rl'),
            "start_command": f"bash fuyao_examples/convert_megatron.sh --ckpt-path {megatron_path} --output-path {hf_path}",
            "artifact_path": "/code", # to be verified
            "label": f"{bifrost_job_name}_convert_{ckpt_id}",
            # "cpus_per_node": 32,
            # "gibs_per_node": 256,
            "device_type": base_eval_config.get('gpu_type', 'a100')
        },
        
        # Eval 任务配置
        "eval_job_args": {
            "docker_image": base_eval_config.get('docker_image', 'infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:xpeng-rl-llm-eval-torch280-v4.7'),
            "site": base_eval_config.get('site_gpu', 'fuyao_b1'),
            "partition": base_eval_config.get('queue_gpu', 'rc-llmrl-a100'),
            "node_count": base_eval_config.get('node_count', 2),
            "gpus_per_node": base_eval_config.get('gpus_per_node', 8),
            "gpu_type": base_eval_config.get('gpu_type', 'a100'),
            "experiment": base_eval_config.get('experiment', 'zhangjh37/llm_rl'),
            "start_command": launch_eval_script,
            "artifact_path": None,
            "label": f"{bifrost_job_name}_eval_{ckpt_id}",
            # "cpus_per_node": 32,
            # "gibs_per_node": 256,
            "device_type": base_eval_config.get('gpu_type', 'a100')
        }
    }

    # 2. 将配置写入临时文件
    # 建议放在 output_dir 下的 logs 目录，方便排查
    pipeline_config_dir = os.path.join(self.checkpoint_manager.uploader.output_dir, "pipeline_configs")
    os.makedirs(pipeline_config_dir, exist_ok=True)
    config_file_path = os.path.join(pipeline_config_dir, f"autopipeline_{ckpt_id}.json")
    
    with open(config_file_path, 'w') as f:
        json.dump(pipeline_config, f, indent=4)
    
    logger.info(f"[AutoPipeline] Pipeline config saved to {config_file_path}")

    # 3. 启动编排脚本 (Fire and Forget)
    # 使用 nohup 确保即使训练进程退出，编排脚本也能继续运行
    # 假设 orchestrate_pipeline.py 在当前工作目录或 PYTHONPATH 中
    script_path = "fuyao_patch/orchestrate_pipeline.py" 
    
    log_file = os.path.join(pipeline_config_dir, f"pipeline_{ckpt_id}.log")
    
    cmd = [
        "nohup", "python3", script_path,
        "--config", config_file_path
    ]
    
    with open(log_file, "w") as out:
        subprocess.Popen(
            cmd,
            stdout=out,
            stderr=out,
            preexec_fn=os.setpgrp # 这里的关键是脱离当前进程组
        )
    
    logger.info(f"[AutoPipeline] Orchestrator launched! Log: {log_file}")

