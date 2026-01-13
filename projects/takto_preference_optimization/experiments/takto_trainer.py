"""
TAKTO: Token-Level Adaptive Kahneman-Tversky Optimization

A novel preference optimization method combining:
1. Token-level prospect theory (loss aversion at token granularity)
2. Adaptive loss aversion parameter (curriculum-based scheduling)
3. Reference-free formulation (SimPO-style average log-probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig
import numpy as np
import math


@dataclass
class TAKTOConfig:
    beta: float = 0.1
    lambda_init: float = 1.0
    lambda_final: float = 2.0
    lambda_schedule: str = "linear"
    alpha: float = 0.88
    gamma_margin: float = 0.5
    use_reference_free: bool = True
    token_weighting: str = "contrast"
    temperature: float = 1.0
    label_smoothing: float = 0.0
    max_length: int = 512
    max_prompt_length: int = 256


class ProspectTheoreticValueFunction(nn.Module):
    def __init__(self, alpha: float = 0.88, lambda_loss: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_loss = lambda_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive_mask = x >= 0
        negative_mask = x < 0
        
        result = torch.zeros_like(x)
        result[positive_mask] = torch.pow(x[positive_mask] + 1e-8, self.alpha)
        result[negative_mask] = -self.lambda_loss * torch.pow(-x[negative_mask] + 1e-8, self.alpha)
        
        return result

    def update_lambda(self, new_lambda: float):
        self.lambda_loss = new_lambda


class TokenImportanceEstimator(nn.Module):
    def __init__(self, method: str = "contrast"):
        super().__init__()
        self.method = method

    def forward(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.method == "uniform":
            return torch.ones_like(policy_logprobs)
        
        elif self.method == "contrast":
            if ref_logprobs is None:
                entropy = -policy_logprobs * torch.exp(policy_logprobs)
                importance = 1.0 / (entropy + 1e-8)
            else:
                prob_diff = torch.abs(
                    torch.exp(policy_logprobs) - torch.exp(ref_logprobs)
                )
                importance = prob_diff / (prob_diff.mean(dim=-1, keepdim=True) + 1e-8)
            
            return importance
        
        elif self.method == "gradient":
            importance = torch.abs(policy_logprobs.grad) if policy_logprobs.grad is not None else torch.ones_like(policy_logprobs)
            return importance
        
        else:
            return torch.ones_like(policy_logprobs)


class AdaptiveLambdaScheduler:
    def __init__(
        self,
        lambda_init: float = 1.0,
        lambda_final: float = 2.0,
        schedule: str = "linear",
        total_steps: int = 10000,
        warmup_steps: int = 1000
    ):
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.schedule = schedule
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lambda(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.lambda_init
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule == "linear":
            return self.lambda_init + progress * (self.lambda_final - self.lambda_init)
        
        elif self.schedule == "cosine":
            return self.lambda_init + 0.5 * (self.lambda_final - self.lambda_init) * (1 - math.cos(math.pi * progress))
        
        elif self.schedule == "exponential":
            return self.lambda_init * (self.lambda_final / self.lambda_init) ** progress
        
        else:
            return self.lambda_init

    def step(self):
        self.current_step += 1
        return self.get_lambda()


class TAKTOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel] = None,
        tokenizer: PreTrainedTokenizer = None,
        config: TAKTOConfig = None,
        optimizer: torch.optim.Optimizer = None,
        total_steps: int = 10000
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or TAKTOConfig()
        self.optimizer = optimizer
        
        self.value_function = ProspectTheoreticValueFunction(
            alpha=self.config.alpha,
            lambda_loss=self.config.lambda_init
        )
        
        self.token_importance = TokenImportanceEstimator(
            method=self.config.token_weighting
        )
        
        self.lambda_scheduler = AdaptiveLambdaScheduler(
            lambda_init=self.config.lambda_init,
            lambda_final=self.config.lambda_final,
            schedule=self.config.lambda_schedule,
            total_steps=total_steps
        )
        
        self.step_count = 0
        self.metrics = {
            "loss": [],
            "lambda": [],
            "desirable_reward": [],
            "undesirable_reward": []
        }

    def compute_token_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_logprobs

    def compute_implicit_reward(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        use_reference_free: bool = True
    ) -> torch.Tensor:
        if use_reference_free:
            seq_lengths = attention_mask[:, 1:].sum(dim=-1)
            avg_logprob = (policy_logprobs * attention_mask[:, 1:]).sum(dim=-1) / seq_lengths
            return avg_logprob
        else:
            if ref_logprobs is None:
                raise ValueError("Reference logprobs required for reference-based reward")
            return self.config.beta * (policy_logprobs.sum(dim=-1) - ref_logprobs.sum(dim=-1))

    def compute_token_level_loss(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        is_desirable: bool,
        reference_point: float = 0.0
    ) -> torch.Tensor:
        token_weights = self.token_importance(
            policy_logprobs,
            ref_logprobs,
            attention_mask[:, 1:]
        )
        
        token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)
        token_weights = token_weights * attention_mask[:, 1:]
        
        if self.config.use_reference_free:
            token_rewards = policy_logprobs
        else:
            token_rewards = self.config.beta * (policy_logprobs - ref_logprobs)
        
        relative_rewards = token_rewards - reference_point
        
        token_values = self.value_function(relative_rewards)
        
        weighted_values = (token_weights * token_values).sum(dim=-1)
        
        if is_desirable:
            loss = -weighted_values.mean()
        else:
            loss = weighted_values.mean()
        
        return loss

    def compute_takto_loss(
        self,
        policy_desirable_logprobs: torch.Tensor,
        policy_undesirable_logprobs: torch.Tensor,
        ref_desirable_logprobs: Optional[torch.Tensor],
        ref_undesirable_logprobs: Optional[torch.Tensor],
        desirable_mask: torch.Tensor,
        undesirable_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        desirable_reward = self.compute_implicit_reward(
            policy_desirable_logprobs,
            ref_desirable_logprobs,
            desirable_mask,
            self.config.use_reference_free
        )
        
        undesirable_reward = self.compute_implicit_reward(
            policy_undesirable_logprobs,
            ref_undesirable_logprobs,
            undesirable_mask,
            self.config.use_reference_free
        )
        
        reference_point = (desirable_reward.mean() + undesirable_reward.mean()) / 2
        
        desirable_loss = self.compute_token_level_loss(
            policy_desirable_logprobs,
            ref_desirable_logprobs,
            desirable_mask,
            is_desirable=True,
            reference_point=reference_point.item()
        )
        
        undesirable_loss = self.compute_token_level_loss(
            policy_undesirable_logprobs,
            ref_undesirable_logprobs,
            undesirable_mask,
            is_desirable=False,
            reference_point=reference_point.item()
        )
        
        margin_loss = F.relu(
            self.config.gamma_margin - (desirable_reward - undesirable_reward)
        ).mean()
        
        total_loss = desirable_loss + undesirable_loss + margin_loss
        
        metrics = {
            "desirable_loss": desirable_loss.item(),
            "undesirable_loss": undesirable_loss.item(),
            "margin_loss": margin_loss.item(),
            "desirable_reward": desirable_reward.mean().item(),
            "undesirable_reward": undesirable_reward.mean().item(),
            "reward_margin": (desirable_reward - undesirable_reward).mean().item(),
            "current_lambda": self.value_function.lambda_loss
        }
        
        return total_loss, metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        self.model.train()
        
        desirable_input_ids = batch["desirable_input_ids"]
        desirable_attention_mask = batch["desirable_attention_mask"]
        undesirable_input_ids = batch["undesirable_input_ids"]
        undesirable_attention_mask = batch["undesirable_attention_mask"]
        
        policy_desirable_logprobs = self.compute_token_logprobs(
            self.model,
            desirable_input_ids,
            desirable_attention_mask,
            desirable_input_ids
        )
        
        policy_undesirable_logprobs = self.compute_token_logprobs(
            self.model,
            undesirable_input_ids,
            undesirable_attention_mask,
            undesirable_input_ids
        )
        
        ref_desirable_logprobs = None
        ref_undesirable_logprobs = None
        
        if self.ref_model is not None and not self.config.use_reference_free:
            with torch.no_grad():
                ref_desirable_logprobs = self.compute_token_logprobs(
                    self.ref_model,
                    desirable_input_ids,
                    desirable_attention_mask,
                    desirable_input_ids
                )
                ref_undesirable_logprobs = self.compute_token_logprobs(
                    self.ref_model,
                    undesirable_input_ids,
                    undesirable_attention_mask,
                    undesirable_input_ids
                )
        
        loss, metrics = self.compute_takto_loss(
            policy_desirable_logprobs,
            policy_undesirable_logprobs,
            ref_desirable_logprobs,
            ref_undesirable_logprobs,
            desirable_attention_mask,
            undesirable_attention_mask
        )
        
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        new_lambda = self.lambda_scheduler.step()
        self.value_function.update_lambda(new_lambda)
        
        self.step_count += 1
        metrics["loss"] = loss.item()
        metrics["step"] = self.step_count
        
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return metrics

    def get_training_stats(self) -> Dict[str, float]:
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values[-100:])
                stats[f"{key}_std"] = np.std(values[-100:])
        return stats


def create_takto_trainer(
    model_name: str,
    config: Optional[TAKTOConfig] = None,
    learning_rate: float = 1e-6,
    total_steps: int = 10000
) -> TAKTOTrainer:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    config = config or TAKTOConfig()
    
    trainer = TAKTOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        total_steps=total_steps
    )
    
    return trainer


if __name__ == "__main__":
    print("TAKTO Trainer Module")
    print("=" * 50)
    
    config = TAKTOConfig(
        beta=0.1,
        lambda_init=1.0,
        lambda_final=2.0,
        lambda_schedule="linear",
        use_reference_free=True
    )
    
    print(f"Config: {config}")
    
    value_fn = ProspectTheoreticValueFunction(alpha=0.88, lambda_loss=1.5)
    test_values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    transformed = value_fn(test_values)
    print(f"\nProspect Theory Transformation:")
    print(f"Input:  {test_values.tolist()}")
    print(f"Output: {transformed.tolist()}")
    
    scheduler = AdaptiveLambdaScheduler(
        lambda_init=1.0,
        lambda_final=2.0,
        schedule="linear",
        total_steps=100
    )
    
    print(f"\nLambda Schedule (linear):")
    for step in [0, 25, 50, 75, 100]:
        scheduler.current_step = step
        print(f"  Step {step}: λ = {scheduler.get_lambda():.3f}")
    
    print("\n✅ TAKTO module ready for training")
