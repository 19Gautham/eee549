import numpy as np

A = np.array([[1, 2, 3], [2, 0, 4], [1, 2, 4]])
b = np.array([[1], [-2], [1]])
c = np.array([[1], [1], [1]])

# print(f"A shape: {A.shape}")
# print(f"b shape: {b.shape}")
# print(f"c shape: {c.shape}")

print(f"A:\n{A}\n\n")
print(f"b:\n{b}\n\n")
print(f"c:\n{c}\n\n")

# 6a solution
print("Solution 6a\n")
a_inv = np.linalg.inv(A)
print(f"(A)^-1:\n{a_inv}\n\n")

# 6b solution
print("Solution 6b\n")
a_inv_dot_b = np.dot(a_inv, b)
print(f'(A)^-1 dot b:\n{a_inv_dot_b}\n\n')

# 6c solution
print("Solution 6c\n")
a_dot_c = np.dot(A, c)
print(f'A dot c:\n{a_dot_c}\n\n')