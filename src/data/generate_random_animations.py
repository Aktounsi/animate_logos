import numpy as np
import random


def random_animation_vector(nr_animations, frac_animations=0.5, frac_animation_type=[1/6] * 6, seed=73):
    random.seed(seed)
    np.random.seed(seed)
    animation_list = []
    for _ in range(nr_animations):
        animate = np.random.choice(a=[False, True], p=[1 - frac_animations, frac_animations])
        if not animate:
            animation_list.append(np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1]))
        else:
            vec = np.zeros(11)
            animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=frac_animation_type)
            vec[animation_type] = 1
            for i in range(6, 11):
                vec[i] = random.uniform(0, 1)
            animation_list.append(vec)
    return np.array(animation_list)


if __name__ == '__main__':
    a = random_animation_vector(3, seed=75)
    b = random_animation_vector(3, seed=75)
    print(a)
    print(b)
