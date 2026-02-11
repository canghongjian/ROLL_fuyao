from roll.utils.tracking import SwanlabTracker, tracker_registry
from roll.utils.logging import get_logger


logger = get_logger()

class DeepInsightSwanlabTracker(SwanlabTracker):
    """
    继承自 SwanlabTracker，仅拦截 log 方法用于重命名指标，
    其他逻辑（init, finish）完全复用父类。
    """
    def log(self, values: dict, step: int = None, **kwargs):
        # 定义指标映射关系： "原始指标名": "新指标名"
        metric_mapping = {
            # infra指标
            "time/reference/compute_log_probs/total": "deepinsight_infra/ref_logp_time",
            "time/actor_train/compute_log_probs/total": "deepinsight_infra/logp_time",
            "time/actor_train/train_step/total": "deepinsight_infra/backward_step_time",
            "time/step_generate": "deepinsight_infra/rollout_step_time",
            "time/actor_train/model_update/total": "deepinsight_infra/sync_weight_time",
            "time/step_total": "deepinsight_infra/step_time",
            "system/tps_gpu": "deepinsight_infra/throughput",

            # 算法指标
            "critic/score/mean": "deepinsight_algorithm/reward",
            "critic/entropy/mean": "deepinsight_algorithm/entropy",
            "actor/pg_loss": "deepinsight_algorithm/policy_loss",
            "actor/kl_loss": "deepinsight_algorithm/kl_loss",
            "actor/ppo_ratio_clipfrac": "deepinsight_algorithm/clip_ratio",
            "token/response_length/mean": "deepinsight_algorithm/response_length",
            "actor_train/grad_norm": "deepinsight_algorithm/grad_norm",
        }

        # 遍历映射表，如果存在则复制并重命名
        # 使用 list(values.keys()) 避免在迭代字典时修改字典大小导致报错（虽然这里只是添加应该没事，但保险起见）
        for key in list(values.keys()):
            if key in metric_mapping:
                new_key = metric_mapping[key]
                values[new_key] = values[key]

        # 调用父类（原 SwanlabTracker）的逻辑执行实际上传
        super().log(values, step, **kwargs)

def apply_tracker_patch():
    """
    执行 Patch 动作：将自定义 Tracker 注册到 roll 的全局注册表中
    """
    # 注册到 registry，这样 yaml 配置里就能用 'deepinsight_swanlab' 了
    tracker_registry["deepinsight_swanlab"] = DeepInsightSwanlabTracker
    logger.info("[Patch] Successfully registered 'deepinsight_swanlab' tracker.")