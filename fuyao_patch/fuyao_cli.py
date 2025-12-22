#!/usr/bin/env python3
import argparse
import os
import fuyao
from xbigdata.fuyao.sdkv2.etl import model
from xbigdata.fuyao.sdkv2.etl import constant

def init_fuyao():
    """Initialize FuYao connection"""
    if not os.getenv("AUTH_USER"):
        raise ValueError("AUTH_USER is not set")
    user_name = os.getenv("AUTH_USER")
    fuyao.etl.init(
        fuyao_api_uri="http://fuyao-v2-api.xiaopeng.link",
        fuyao_api_key="abf55cb029834a389ec944c6b1e5f06b",
        user_name=user_name
    )

def deploy_job(args):
    """Deploy a job to FuYao"""
    init_fuyao()

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
        cpus_per_node=args.cpus_per_node if hasattr(args, 'cpus_per_node') and args.cpus_per_node else 0,
        gibs_per_node=args.gibs_per_node if hasattr(args, 'gibs_per_node') and args.gibs_per_node else 0,
        device_type=args.device_type if hasattr(args, 'device_type') and args.device_type else 'A100',
        envs={"enable_prometheus_metrics": "true"},
    )

    job_name = None
    try:
        result = fuyao.etl.deploy_run(deploy_run_args)
        print(
            f"run_name: {result.data.run_name}\n"
            f"msg: {result.msg}\n"
            f"deploy_code: {result.code}\n"
        )
        job_name = result.data.run_name
    except Exception as e:
        print(f"Error deploying job: {e}")

    return job_name

def query_job(args):
    """Query job status by name"""
    init_fuyao()
    
    info = {}
    try:
        run_info = fuyao.etl.get_run_by_name(job_name=args.job_name)
        fuyao_job = run_info.fuyao_job
        print(
            f"gpu_type: {fuyao_job.gpu_type}\n"
            f"state: {fuyao_job.state}\n"
            f"site: {fuyao_job.site}\n"
            f"run_id: {run_info.info.run_id}"
        )
        pod_info = fuyao.etl.search_run_pods(run_name=args.job_name, site=fuyao_job.site)
        print(f"host_ip: {pod_info.pods[0].host_ip}")
        info = {
            "gpu_type": fuyao_job.gpu_type,
            "state": fuyao_job.state,
            "site": fuyao_job.site,
            "run_id": run_info.info.run_id,
            "host_ip": pod_info.pods[0].host_ip
        }
    except Exception as e:
        print(f"Error querying job: {e}")
    return info

def cancel_job(args):
    """Cancel a job by run_id"""
    init_fuyao()
    
    run_info = fuyao.etl.get_run_by_name(job_name=args.job_name)
    cancel_reason = args.reason if args.reason else "early stop"
    
    try:
        cancel_run_args = model.CancelRunArgs(
            run_id=run_info.info.run_id,
            cancel_option=constant.CancelOptions.OTHERS,
            cancel_reason=cancel_reason,
        )
        result = fuyao.etl.cancel_run(cancel_run_args)
        print(f"msg: {result.msg}")
    except Exception as e:
        print(f"Error cancelling job: {e}")

def main():
    parser = argparse.ArgumentParser(description="FuYao CLI Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a job')
    deploy_parser.add_argument('--docker_image', required=True, help='Docker image')
    deploy_parser.add_argument('--site', required=True, help='Site name')
    deploy_parser.add_argument('--partition', required=True, help='Partition/queue')
    deploy_parser.add_argument('--node_count', type=int, required=True, help='Number of nodes')
    deploy_parser.add_argument('--gpus_per_node', type=int, required=True, help='GPUs per node')
    deploy_parser.add_argument('--experiment', required=True, help='Experiment name')
    deploy_parser.add_argument('--start_command', required=True, help='Start command')
    deploy_parser.add_argument('--artifact_path', required=False, default=os.getcwd(), help='Artifact path')
    deploy_parser.add_argument('--label', required=False, help='test label')
    deploy_parser.add_argument('--cpus_per_node', type=int, required=False, help='CPUs per node')
    deploy_parser.add_argument('--gibs_per_node', type=int, required=False, help='Memory per node in GiB')
    deploy_parser.add_argument('--device_type', required=False, help='GPU device type (default: A100)')
    # Query command
    query_parser = subparsers.add_parser('query', help='Query job status')
    query_parser.add_argument('--job_name', required=True, help='Job name to query')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('--job_name', required=True, help='Job name to cancel')
    cancel_parser.add_argument('--reason', required=False, help='Cancel reason (default: early stop)')
    
    args = parser.parse_args()

    if args.command == 'deploy':
        deploy_job(args)
    elif args.command == 'query':
        query_job(args)
    elif args.command == 'cancel':
        cancel_job(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
