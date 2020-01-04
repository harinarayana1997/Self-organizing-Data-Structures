import numpy as np
import pandas as pd
import random as rand
from tqdm import tqdm
from collections import Counter


# ======================================================================================================================
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None
        self.height = 1
        self.counter = 0


class Tree:
    def __init__(self, root):
        self.root = root
        self.nil = None


def insert(T, root, node):
    if node.data < root.data:
        if root.left:
            insert(T, root.left, node)
        else:
            node.parent = root
            root.left = node
    else:
        if root.right:
            insert(T, root.right, node)
        else:
            node.parent = root
            root.right = node
    root.height = max(root.left.height if root.left else 0,
                      root.right.height if root.right else 0) + 1


# ======================================================================================================================
def generate_access_sequence():
    keys = np.array(list(range(1, 1001)))
    rand.shuffle(keys)
    a = np.array([rand.randint(1, 100) for _ in range(1000)])
    c = np.append(np.array([0]), np.cumsum(a))
    s = []
    A = c[-1]

    for _ in range(10000):
        j = rand.randint(1, A)
        for i in range(1, len(c)):
            if c[i - 1] < j <= c[i]:
                s.append(keys[i - 1])

    _keys, _ = zip(*sorted(list(Counter(s).items()), key=lambda x: x[0]))

    _keys, _probabilities = zip(
        *sorted(list(Counter(s).items()) + [(key, 0) for key in list(set(keys) - set(_keys))], key=lambda x: x[0]))

    _probabilities = list(np.array(_probabilities, dtype=float))

    res = {
        'keys': keys,
        'a': a,
        'c': c,
        's': s,
        'p': _probabilities,
        'A': A,
        'n': 1000,
    }

    return res


# ======================================================================================================================
def left_rotate(T, x, cost):
    cost['rotations'] += 1

    y = x.right
    x.right = y.left
    if y.left != T.nil:
        y.left.parent = x
    y.parent = x.parent

    if x.parent is T.nil:
        T.root = y
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y

    x.height = max(x.left.height if x.left else 0,
                   x.right.height if x.right else 0) + 1
    y.height = max(y.left.height if y.left else 0,
                   y.right.height if y.right else 0) + 1


def right_rotate(T, x, cost):
    cost['rotations'] += 1

    y = x.left
    x.left = y.right

    if y.right != T.nil:
        y.right.parent = x

    y.parent = x.parent

    if x.parent == T.nil:
        T.root = y
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y

    y.right = x
    x.parent = y

    x.height = max(x.left.height if x.left else 0,
                   x.right.height if x.right else 0) + 1
    y.height = max(y.left.height if y.left else 0,
                   y.right.height if y.right else 0) + 1


# splay.
# ======================================================================================================================
def splay(T, x, cost):
    while x.parent is not None:
        if x.parent.parent is None:
            # L case
            if x == x.parent.left:
                right_rotate(T, x.parent, cost)
            # R case
            elif x == x.parent.right:
                left_rotate(T, x.parent, cost)
        # LL
        elif x == x.parent.left and x.parent == x.parent.parent.left:
            right_rotate(T, x.parent.parent, cost)
            right_rotate(T, x.parent, cost)
        # RR
        elif x == x.parent.right and x.parent == x.parent.parent.right:
            left_rotate(T, x.parent.parent, cost)
            left_rotate(T, x.parent, cost)
        # LR
        elif x == x.parent.right and x.parent == x.parent.parent.left:
            left_rotate(T, x.parent, cost)
            right_rotate(T, x.parent, cost)
        # RL
        else:
            right_rotate(T, x.parent, cost)
            left_rotate(T, x.parent, cost)


def splay_search(T, root, data, cost):
    if root:
        cost['depth'] += 1
        if root.data == data:
            splay(T, root, cost)
            return
        if data < root.data:
            splay_search(T, root.left, data, cost)
        else:
            splay_search(T, root.right, data, cost)


# obst.
# ======================================================================================================================
def obst(p, q, n):
    p = np.array(p)
    q = np.array(q)
    e = np.zeros((n + 2, n + 1))
    w = np.zeros((n + 2, n + 1))
    root = np.zeros((n + 1, n + 1))

    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    for l in tqdm(range(1, n + 1), total=n, desc="OBST "):
        for i in range(1, n - l + 2):
            j = i + l - 1
            e[i][j] = float('inf')
            w[i][j] = w[i][j - 1] + p[j] + q[j]

            for r in range(i, j + 1):
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e, root


# Move to root.
# ======================================================================================================================
def mtr(T, x, cost):
    while x.parent is not None:
        # L case
        if x == x.parent.left:
            right_rotate(T, x.parent, cost)
        # R case
        elif x == x.parent.right:
            left_rotate(T, x.parent, cost)


def mtr_search(T, root, data, cost):
    if root:
        cost['depth'] += 1
        if root.data == data:
            mtr(T, root, cost)
            return
        if data < root.data:
            mtr_search(T, root.left, data, cost)
        else:
            mtr_search(T, root.right, data, cost)


# dynamic monotone.
# ======================================================================================================================
def counter(x):
    if x:
        return x.counter
    else:
        return 0


def dm(T, x, cost):
    while x.parent is not None:
        # L case
        if x == x.parent.left and counter(x.parent) < counter(x):
            right_rotate(T, x.parent, cost)
        # R case
        elif x == x.parent.right and counter(x.parent) < counter(x):
            left_rotate(T, x.parent, cost)
        else:
            break


def dm_search(T, root, data, cost):
    if root:
        cost['depth'] += 1
        if root.data == data:
            root.counter += 1
            dm(T, root, cost)
            return
        if data < root.data:
            dm_search(T, root.left, data, cost)
        else:
            dm_search(T, root.right, data, cost)


# ======================================================================================================================
def main():
    all_costs = []
    runs = 10
    for run in range(runs):
        print("=" * 25, " RUN {} ".format(run + 1), "=" * 25)

        res = generate_access_sequence()

        T_splay = Tree(Node(res['keys'][0]))
        T_mtr = Tree(Node(res['keys'][0]))
        T_dm = Tree(Node(res['keys'][0]))

        for i in range(1, len(res['keys'])):
            insert(T_splay, T_splay.root, Node(res['keys'][i]))
            insert(T_mtr, T_mtr.root, Node(res['keys'][i]))
            insert(T_dm, T_dm.root, Node(res['keys'][i]))

        # OBST
        p = res['p']
        p.insert(0, 0)
        e, root = obst(p, [0 for _ in range(1000 + 1)], res['n'])

        # Splay.
        splay_cost = {'depth': [], 'rotations': []}
        cost = {'rotations': 0, 'depth': 0}
        for i in tqdm(res['s'], total=len(res['s']), desc='SPLAY'):
            splay_search(T_splay, T_splay.root, i, cost)
            splay_cost['depth'].append(cost['depth'])
            splay_cost['rotations'].append(cost['rotations'])
            cost = {'rotations': 0, 'depth': 0}

        # Move to root.
        mtr_cost = {'depth': [], 'rotations': []}
        cost = {'rotations': 0, 'depth': 0}
        for i in tqdm(res['s'], total=len(res['s']), desc="MTR  "):
            mtr_search(T_mtr, T_mtr.root, i, cost)
            mtr_cost['depth'].append(cost['depth'])
            mtr_cost['rotations'].append(cost['rotations'])
            cost = {'rotations': 0, 'depth': 0}

        # Dynamic monotone.
        dm_cost = {'depth': [], 'rotations': []}
        cost = {'rotations': 0, 'depth': 0}
        for i in tqdm(res['s'], total=len(res['s']), desc="DM   "):
            dm_search(T_dm, T_dm.root, i, cost)
            dm_cost['depth'].append(cost['depth'])
            dm_cost['rotations'].append(cost['rotations'])
            cost = {'rotations': 0, 'depth': 0}

        print("Static optimal BST cost = {}.".format(e[1][1000]))
        print("Splay cost              = {}.".format(
            sum(splay_cost['depth'] + splay_cost['rotations'])))
        print("Move to root cost       = {}.".format(
            sum(mtr_cost['depth'] + mtr_cost['rotations'])))
        print("Dynamic monotone cost   = {}.".format(
            sum(dm_cost['depth'] + dm_cost['rotations'])))

        all_costs.append([
            e[1][1000],
            sum(splay_cost['depth'] + splay_cost['rotations']),
            sum(mtr_cost['depth'] + mtr_cost['rotations']),
            sum(dm_cost['depth'] + dm_cost['rotations'])
        ])

        all_costs.append([
            "S.C.R",
            sum(splay_cost['depth'] + splay_cost['rotations']) / e[1][1000],
            sum(mtr_cost['depth'] + mtr_cost['rotations']) / e[1][1000],
            sum(dm_cost['depth'] + dm_cost['rotations']) / e[1][1000]
        ])

    pd.DataFrame(all_costs, columns=['OBST', 'Splay', 'MTR', 'DM']).to_csv(
        'all_costs_02.csv', index=False)


if __name__ == '__main__':
    main()
