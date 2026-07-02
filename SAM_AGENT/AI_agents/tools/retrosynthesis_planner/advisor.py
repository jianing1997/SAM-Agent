from AI_agents.tools.retrosynthesis_planner.retro_star.api import RSPlanner
from typing import List
import logging

# 尝试初始化规划器，如果遇到 CUDA 错误则回退到 CPU
planner = None
try:
    # 首先尝试使用 GPU
    planner = RSPlanner(
        gpu=0,  # 尝试使用 GPU 0
        use_value_fn=True,
        iterations=60,
        expansion_topk=50
    )
    print("✓ 逆合成规划器已使用 GPU 初始化")
except Exception as e:
    # 如果 GPU 初始化失败，回退到 CPU
    if "cuda" in str(e).lower() or "gpu" in str(e).lower():
        logging.warning(f"GPU 初始化失败，回退到 CPU: {e}")
        print("⚠ GPU 初始化失败，逆合成规划器将使用 CPU")
        try:
            planner = RSPlanner(
                gpu=-1,  # 使用 CPU
                use_value_fn=True,
                iterations=60,
                expansion_topk=50
            )
            print("✓ 逆合成规划器已使用 CPU 初始化")
        except Exception as e2:
            logging.error(f"逆合成规划器初始化失败: {e2}")
            print(f"✗ 逆合成规划器初始化失败: {e2}")
    else:
        raise

def plan_and_print(smiles: str):
    result = planner.plan(smiles)
    if result:
        return result
    else:
        return "No valid plan found."
    
