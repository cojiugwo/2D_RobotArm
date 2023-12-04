import numpy as np

theta = [0.5, 05., 0.2]

def get_eep(x):
    x = 400 # self.width // 2
    y = 300 # self.height // 2
    for angle in theta:
        x += 100 * np.cos(angle)
        y += 100 * np.sin(angle)
    # return np.array([x, y], dtype=np.float32)

    return np.array([x, y])
print(f'first method: {get_eep(theta)}')


theta = [-0.2, 0.3, 0.5]
def get_eep1(x):
    x = 400 + 100 * np.cos(theta[0]) + 100 * np.cos(theta[0]+theta[1]) + 100 * np.cos(theta[0]+theta[1]+theta[2])
    y = 300 + 100 * np.sin(theta[0]) + 100 * np.sin(theta[0]+theta[1]) + 100 * np.sin(theta[0]+theta[1]+theta[2])
    return np.array([x, y])
print(f'second method:{get_eep1(theta)}')

def get_eep2(theta):
    x, y = 400, 300
    for i in range(len(theta)):
        x += 100 * np.cos(np.sum(theta[:i+1]))
        y += 100 * np.sin(np.sum(theta[:i+1]))
    return np.array([x, y])

result = get_eep2(theta)
print(f'third mehtod: {result}')