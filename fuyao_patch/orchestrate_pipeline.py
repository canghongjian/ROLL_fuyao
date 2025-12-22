#!/usr/bin/env python3
import os
import time
import argparse
import json
import logging
import sys
import types
from xbigdata.fuyao.sdkv2.etl import model, constant
import fuyao
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Pipeline] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class FuyaoClient:
    def __init__(self):
        if not os.getenv("AUTH_USER"):
            # 如果环境变量没设置，尝试从系统用户获取
            os.environ["AUTH_USER"] = os.environ.get("USER", "unknown_user")
        
        self.user_name = os.getenv("AUTH_USER")
        # 初始化连接 (复用你的代码)
        fuyao.etl.init(
            fuyao_api_uri="http://fuyao-v2-api.xiaopeng.link",
            fuyao_api_key="abf55cb029834a389ec944c6b1e5f06b",
            user_name=self.user_name
        )

    def deploy_job(self, args_dict):
        """提交任务"""
        # 将字典转换为 Namespace 以兼容你的 deploy_job 逻辑
        args = types.SimpleNamespace(**args_dict)
        
        deploy_run_args = model.DeployRunArgs(
            docker_image=args.docker_image,
            site=args.site,
            partition=args.partition,
            node_count=args.node_count,
            gpus_per_node=args.gpus_per_node,
            experiment=args.experiment,
            start_command=args.start_command,
            artifact_path=args.artifact_path,
            label=args.label,
            # 兼容可选参数
            cpus_per_node=getattr(args, 'cpus_per_node', 0),
            gibs_per_node=getattr(args, 'gibs_per_node', 0),
            device_type=getattr(args, 'device_type', 'a100'),
            envs={"enable_prometheus_metrics": "true"}
        )

        try:
            result = fuyao.etl.deploy_run(deploy_run_args)
            logger.info(f"Job Deployed: {result.data.run_name} (ID: {result.data.run_id})")
            return result.data.run_name
        except Exception as e:
            logger.error(f"Failed to deploy job: {e}")
            raise e

    def get_job_state(self, job_name):
        """查询任务状态"""
        try:
            run_info = fuyao.etl.get_run_by_name(job_name=job_name)
            # 注意：这里假设 fuyao_job.state 返回的是标准状态字符串
            # 常见的状态: 'JOB_PENDING', 'JOB_RUNNING', 'JOB_COMPLETE', 'JOB_FAILED', 'JOB_CANCELLED'
            state = run_info.fuyao_job.state
            return state
        except Exception as e:
            logger.error(f"Error querying job {job_name}: {e}")
            return "UNKNOWN"

def run_pipeline(config_path):
    """主流水线逻辑"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    client = FuyaoClient()
    ckpt_id = config['ckpt_id']
    logger.info(f"Starting pipeline for Checkpoint: {ckpt_id}")

    # ==========================================
    # Phase 1: Submit Convert Job
    # ==========================================
    logger.info(">>> Phase 1: Submitting Convert Job")
    

    convert_job_name = client.deploy_job(config['convert_job_args'])

    # ==========================================
    # Phase 2: Monitor Convert Job
    # ==========================================
    logger.info(f">>> Phase 2: Monitoring Convert Job [{convert_job_name}]")

    max_time = 18000 # 最长等待5小时
    cur_time = 0
    while cur_time <= max_time:
        state = client.get_job_state(convert_job_name)
        logger.info(f"Convert Job State: {state}")
        
        if state in ["JOB_COMPLETE"]:
            logger.info("Convert Job Finished Successfully!")
            if not config['keep_megatron_file'] and config.get('megatron_path'):
                # 删除原megatron文件
                shutil.rmtree(config['megatron_path'], ignore_errors=True)
                logger.info(f"remove original megatron path:{config['megatron_path']}")
            break
        elif state in ["JOB_FAILED", "JOB_CANCELLED"]:
            logger.error(f"Convert Job Failed with state: {state}. Pipeline Aborted.")
            return # 退出流水线
        
        time.sleep(30) # 每 30 秒轮询一次
        cur_time += 30

    if config.get('convert_then_eval', False):
        # ==========================================
        # Phase 3: Submit Eval Job
        # ==========================================
        logger.info(">>> Phase 3: Submitting DeepInsight Eval Job")
        
        # 构造评测命令 (直接复用你之前的逻辑)
        eval_job_name = client.deploy_job(config['eval_job_args'])
        logger.info(f"Eval Job Submitted: {eval_job_name}. Pipeline Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to pipeline config json')
    args = parser.parse_args()
    
    run_pipeline(args.config)