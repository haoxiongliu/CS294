FOR %%E IN (Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2) ^
DO python run_expert.py experts/%%E.pkl %%E --render --num_rollouts=20
