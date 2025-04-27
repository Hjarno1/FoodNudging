import numpy as np

class LinUCB:
    def __init__(self, n_arms: int, d: int, alpha: float = 0.1):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        # One (A, b) per arm
        self.A = [np.identity(d) for _ in range(n_arms)]
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)]

    def select_arm(self, contexts: list) -> int:
        p_vals = []
        for arm, x in enumerate(contexts):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            mean = float(theta.T.dot(x))
            var = float(x.T.dot(A_inv).dot(x))
            bonus = self.alpha * np.sqrt(var)
            p_vals.append(mean + bonus)
        return int(np.argmax(p_vals))

    def update(self, arm: int, x: np.ndarray, reward: float):
        self.A[arm] += x.dot(x.T)
        self.b[arm] += reward * x

    def reset(self):
        self.A = [np.identity(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.n_arms)]

    @classmethod
    def from_state(cls, A_list, b_list, alpha=0.1):
        n_arms = len(A_list)
        d      = A_list[0].shape[0]
        inst   = cls(n_arms=n_arms, d=d, alpha=alpha)
        inst.A = A_list
        inst.b = b_list
        return inst

    def ucb_scores(self, contexts):
        import numpy as np
        scores = []
        for arm, x in enumerate(contexts):
            A_inv = np.linalg.inv(self.A[arm])
            θ     = A_inv.dot(self.b[arm])
            mean  = float(θ.T.dot(x))
            var   = float(x.T.dot(A_inv).dot(x))
            bonus = self.alpha * np.sqrt(var)
            scores.append(mean + bonus)
        return scores