from pose.test import Tester
import torch

model = 3

tester = Tester(None)

files = tester.get_ply_files()
points = files.get(model)

R1 = [1, 0, 0,
    0, 1, 0,
    0, 0, 1]

R2 = [-1, 0, 0,
    0, 1, 0,
    0, 0, 1]

R1 = torch.tensor(R1, dtype=torch.float).reshape(3,3)

R2 = torch.tensor(R2, dtype=torch.float).reshape(3,3)

p1 = torch.tensor([1, 2, 3], dtype=torch.float)
p2 = torch.tensor([1, 2, 3], dtype=torch.float)

add = tester.compute_ADD(points, R_gt=R1, t_gt=p1, R_pred=R2, t_pred=p2)

print(add)
