该目录适用于fuyao系统的roll框架提交

参数配置、启动指引等可见参考文档：https://xiaopeng.feishu.cn/wiki/IlbYwgD6LiQDVnktl5ccZ6Ccnlc


目前问题：
1. math rule reward worker还有超时5秒的情况
2. 自动megatron转hf并评测


转化脚本：
```bash
fuyao deploy --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:zhangjh37-251204-2351     --project=rc-perception --gpu-type a100     --gpus-per-node 8 --node=1 --label=roll-convert     --site=fuyao_b1 --queue=rc-llmrl-a100 bash fuyao_examples/convert_megatron.sh --ckpt-path /dataset_rc_mm/roll_output/bifrost-2025120915240600-zhangjh37_qwen3-8B-rlvr-deepmath/roll_ckpt/20251209-081019/checkpoint-100 --output-path /dataset_rc_mm/roll_output_converted/bifrost-2025120915240600-zhangjh37_qwen3-8B-rlvr-deepmath/roll_ckpt/20251209-081019/checkpoint-100
```