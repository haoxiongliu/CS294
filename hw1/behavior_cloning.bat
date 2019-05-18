FOR %%E IN (Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2) ^
DO python run_cloning.py expert_data/%%E.pkl %%E --render --num_rollouts=20
