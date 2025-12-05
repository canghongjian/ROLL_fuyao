import logging

# WARNING:latex2sympy2_extended.math_normalization:equations is deprecated, as it handled by the parser now
logging.getLogger('latex2sympy2_extended.math_normalization').setLevel(logging.ERROR)

from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from math_verify import parse, verify
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing
from roll.pipeline.rlvr.rewards.math_utils import grade_answer_verl

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider
from roll.utils.context_managers import state_offload_manger

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def _extract_after_last_end_think(response: str) -> str:
    """
    提取字符串中最后一个 "</think>" 标签之后的所有文本。

    校验逻辑：
    - 如果字符串中包含开标签 "<think>"，直接返回空字符串。
    - 如果字符串中包含超过一个的闭标签 "</think>"，也直接返回空字符串。

    如果校验通过，则执行原有逻辑：
    1. 优先按最后一个 '</think>' 分割。
    2. 如果找不到，则回退到按最后一个双换行符 '\n\n' 分割。
    3. 如果都找不到，则返回空字符串。

    Args:
        response (str): 输入的完整文本。

    Returns:
        str: 提取出的文本块（已去除首尾空格），或空字符串。
    """
    # 如果检测到 "<think>" 或超过一个 "</think>"，直接返回空字符串
    if "<think>" in response or response.count('</think>') > 1:
        return ""
    
    # 1. 优先尝试按 '</think>' 分割
    _before_think, sep_think, after_think = response.rpartition('</think>')

    if sep_think:
        # 如果找到了 '</think>'，则返回它后面的部分，并清理首尾空格
        return after_think.strip()
    else:
        # 2. 如果没找到 '</think>'，则尝试按最后一个 '\n\n' 分割
        _before_newline, sep_newline, after_newline = response.rpartition('\n\n')
        if sep_newline:
            # 如果找到了 '\n\n'，返回它后面的部分，并清理首尾空格
            return after_newline.strip()
        else:
            # 3. 如果连 '\n\n' 都没找到，则返回空字符串
            return ""

def _hf_verify_math_sample(response, answer, result):
    try:
        # 在解析之前，先对模型的原始输出进行预处理
        cleaned_response = _extract_after_last_end_think(response)
        """
        --- `parse` 函数完整参数介绍与使用建议 ---
        `parse` 函数用于从文本中提取并解析数学答案，其主要参数如下：
        
        1. `pred` (位置参数): 需要被解析的输入字符串。
           => 建议：传入净化后的文本（如 cleaned_response），可以显著提高准确率。
        
        2. `extraction_config` (关键字参数): 定义要寻找的答案类型。
           => 默认值: [LatexExtractionConfig(), ExprExtractionConfig()] (寻找LaTeX和纯数字)
           => 建议：对于数学计算题，保持默认即可。
        
        3. `fallback_mode` (关键字参数): 定义当找到答案文本但无法成功解析时怎么办。
           => 默认值: "first_match" (返回原始匹配的字符串)
           => 强烈建议: 设为 "no_fallback"，这样在解析失败时会返回空列表[]，避免输出垃圾内容。
        
        4. `extraction_mode` (关键字参数): 定义搜寻答案的范围。
           => 默认值: "any_match" (搜寻全文，找到第一个能成功解析的答案)
           => 建议：保持默认值，因为它更可能在复杂文本中找到正确答案。
        
        5. `parsing_timeout` (关键字参数): 解析单个表达式的超时时间（秒）。
           => 默认值: 5
           => 建议：保留默认值，作为防止程序卡死的安全保护。
        
        6. `raise_on_error` (关键字参数): 遇到内部程序错误时是否抛出异常。
           => 默认值: False (不抛出异常，返回空列表)
           => 建议：保持默认值，确保程序的健壮性，不会因单个样本出错而中断。
        """
        parsed_answers = parse(cleaned_response, fallback_mode="no_fallback")
        print(f"cleaned_response:{cleaned_response}, parsed_answers:{parsed_answers}")
        # 如果解析结果为空，则认为提取失败
        if not parsed_answers:
            exect_answer = None
        else:
            # 通常我们只关心第一个解析出的结果
            exect_answer = parsed_answers[0]

        gold_answer = parse(answer)

        if gold_answer is None or exect_answer is None:
            result.append((False, "", ""))
        else:
            # 假设 verify 函数可以处理 parse 返回的对象
            ans = verify(gold_answer[0], exect_answer)
            result.append((ans, str(gold_answer[0]), str(exect_answer)))
            
    except Exception as e:
        print('exception:', e)
        # 捕获任何潜在的异常，确保进程不会崩溃
        result.append((False, "", ""))


def hf_verify_math_sample(answer_a, answer_b, timeout_sec=5.0):
    with multiprocessing.Manager() as manager:
        result = manager.list()
        
        p = multiprocessing.Process(
            target=_hf_verify_math_sample,
            args=(answer_a, answer_b, result)
        )
        
        p.start()
        try:
            max_timeout = min(timeout_sec + 1, 10)
            p.join(timeout=max_timeout)
        except Exception as e:
            pass
        finally:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
            p.join(timeout=2)
        if not result:
            return False, "", ""
        return result[0]

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    def repetition_penalty_reward(response, **kwargs) -> float:
        if response == "" or len(response.split()) < ngram_size:
            return 0.0
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1
        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        return reward
    return repetition_penalty_reward

def long_block_penalty_reward_fn(text: str, max_length: int = 100) -> float:
    max_block_len = max([len(i) for i in text.split(" ")])
    reward = -float(max_block_len > max_length)
    return reward

def format_reward_fn(text: str, pattern: Optional[str] = r"^<think>.*?</think>.*?<answer>.*?</answer>$"):
    if pattern is None:
        pattern: str = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matche = re.match(pattern, text, re.DOTALL | re.MULTILINE)
    reward = 0 if matche else -1
    return reward


class MathRuleRewardWorker(Worker):
    """
    (x)Reward Model 使用 AutoModelForSequenceClassification 协议
    面向math的rule reward model
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.1)
        self.format_pattern = getattr(self.worker_config, "format_pattern", None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        verify_answer = []
        repetition_penalty_rewards = []
        long_block_penalty_rewards = []
        response_length_rewards = []
        format_rewards = []
        
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        for response, answer in zip(response_text_list, data.non_tensor_batch["ground_truth"]):
            response = response.replace("<|endoftext|>", "").replace("<pad>", "")
            
            try:
                with timeout(5):
                    # correct, extracted_ground_truth, extracted_response = hf_verify_math_sample(
                    #     response, f"${answer}$"
                    # )
                    correct, extracted_ground_truth, extracted_response = grade_answer_verl(response, answer)
            
                log_data = {
                    "response": response,
                    "extracted_response": extracted_response,
                    "answer": answer,
                    "extracted_ground_truth": extracted_ground_truth,
                    "correct": correct,
                }
                # self.logger.info(json.dumps(log_data, ensure_ascii=False))

            except Exception as e:
                self.logger.error(f"timeout or error during hf_verify_math_sample. answer: {answer}, response: {response}")
                correct = False
                extracted_response = ""
                extracted_ground_truth = ""
            
            if correct:
                verify_answer.append(1)
            else:
                verify_answer.append(0)
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(response))
            format_rewards.append(format_reward_fn(response, self.format_pattern))
            long_block_penalty_rewards.append(long_block_penalty_reward_fn(response))
            response_length_rewards.append(len(response) / 20000)
            
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_length_rewards = torch.tensor(response_length_rewards, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        long_block_penalty_rewards = torch.tensor(long_block_penalty_rewards, dtype=torch.float16)
        format_rewards = torch.tensor(format_rewards, dtype=torch.float16)
        scores = torch.tensor(verify_answer, dtype=torch.float16)
        response_level_rewards = torch.tensor(verify_answer, dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
            }
        )

        self.logger.debug(f"reward output: {output}, response_level_rewards: {response_level_rewards}")
        return output


if __name__ == "__main__":
    data = {"response": "<think>\nOkay, so I need to figure out what 9 divided by 2 is as a decimal. Let me think. Hmm, I remember that dividing by 2 is the same as splitting something into two equal parts. But how does that translate to decimals?\n\nWait, maybe I should just do the division. Let me write it out. 9 divided by 2. Let me recall how long division works. So, 2 goes into 9 how many times? Well, 2 times 4 is 8, and 2 times 5 is 10. But 10 is too big, so it's 4 times. So 4 times 2 is 8. Subtract 8 from 9, and I get a remainder of 1. Then I add a decimal point and a zero, making it 10. Now, 2 goes into 10 exactly 5 times. So that would be 4.5? Let me check that again.\n\nAlternatively, maybe I can think of 9/2 as a fraction. Since 9 divided by 2 is the same as 4 and 1/2. Because 2 times 4 is 8, and then there's 1 left over, which is 1/2. And 1/2 as a decimal is 0.5. So adding that to 4 gives me 4.5. That seems right.\n\nWait, let me verify with another method. If I multiply 4.5 by 2, I should get 9. Let me do that. 4.5 times 2. 4 times 2 is 8, and 0.5 times 2 is 1. So 8 + 1 is 9. Perfect, that checks out. So 9 divided by 2 is indeed 4.5.\n\nIs there another way to think about this? Maybe converting the fraction to a decimal by expanding the denominator to 10? Let me try. If I have 9/2, I can multiply numerator and denominator by 5 to get the denominator to 10. So 9 times 5 is 45, and 2 times 5 is 10. So that becomes 45/10, which is 4.5. Yep, same answer.\n\nI think that's solid. All methods point to 4.5. I don't see any mistakes in my reasoning. So the decimal form of 9/2 is 4.5.\n\n**Final Answer**\nThe decimal form of $\\frac{9}{2}$ is \\boxed{4.5}.\n</think>\n\nTo express the fraction $\\frac{9}{2}$ as a decimal, we can approach the problem through division or by converting the fraction to a decimal directly.\n\n---\n\n### Step 1: Perform the Division\n\nWe divide the numerator (9) by the denominator (2):\n\n$$\n9 \\div 2\n$$\n\n- 2 goes into 9 **4 times** (since $2 \\times 4 = 8$).\n- Subtract 8 from 9, leaving a remainder of **1**.\n- Bring down a zero (to continue the division), making it **10**.\n- 2 goes into 10 **5 times** (since $2 \\times 5 = 10$).\n\nSo, the result of the division is:\n\n$$\n4.5\n$$\n\n---\n\n### Step 2: Confirm with Fraction Conversion\n\nWe can also convert the fraction $\\frac{9}{2}$ to a mixed number:\n\n$$\n\\frac{9}{2} = 4 \\frac{1}{2}\n$$\n\nNow, convert the fractional part $\\frac{1}{2}$ to a decimal:\n\n$$\n\\frac{1}{2} = 0.5\n$$\n\nAdding this to the whole number part:\n\n$$\n4 + 0.5 = 4.5\n$$\n\n---\n\n### Step 3: Verification\n\nTo ensure accuracy, we can reverse the process by multiplying the decimal by the denominator:\n\n$$\n4.5 \\times 2 = 9\n$$\n\nThis confirms that the decimal representation is correct.\n\n---\n\n### Final Answer\n\n$$\n\\boxed{4.5}\n$$<|im_end|>", "extracted_response": "", "answer": "4.5", "extracted_ground_truth": "", "correct": False}
    arr = []
    print(grade_answer_verl(data['response'], '4.5'))