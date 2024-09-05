import torch
import time

# Define function using first operation
def operation1(value, scale, zero_point, min_val, max_val):
    return value - value.div(scale).sub(zero_point).round().clamp(min_val, max_val)

# Define function using second operation
def operation2(value, scale, zero_point, min_val, max_val):
    return value - torch.clamp(
        torch.round(
            value / scale - zero_point
        ), 
        min=min_val, max=max_val
    )

# Generate a decent size tensor
tensor_size = (1000, 1000)  # 1 million elements tensor
value = torch.rand(tensor_size).cuda()
scale = torch.tensor(0.1).cuda()
zero_point = torch.tensor(0.5).cuda()
min_val = torch.tensor(-10.0).cuda()
max_val = torch.tensor(10.0).cuda()

# Number of loops for testing
n_loops = 100000

# Measure performance for operation1
start_time_op1 = time.time()
for _ in range(n_loops):
    result_op1 = operation1(value, scale, zero_point, min_val, max_val)
end_time_op1 = time.time()
time_op1 = end_time_op1 - start_time_op1

# Measure performance for operation2
start_time_op2 = time.time()
for _ in range(n_loops):
    result_op2 = operation2(value, scale, zero_point, min_val, max_val)
end_time_op2 = time.time()
time_op2 = end_time_op2 - start_time_op2

print(time_op1, time_op2)