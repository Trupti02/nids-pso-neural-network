import numpy as np

class PSO:
    def __init__(self, fitness_fn, lower, upper, n_particles=8, iterations=10,
                w=0.7, c1=1.5, c2=1.5, seed=0):
        self.fitness_fn = fitness_fn
        self.lb, self.ub = np.array(lower, float), np.array(upper, float)
        self.n, self.iters = n_particles, iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = np.random.RandomState(seed)

    def run(self):
        dim = len(self.lb)
        x = self.rng.uniform(self.lb, self.ub, (self.n, dim))
        v = np.zeros_like(x)
        pbest = x.copy()
        pfit = np.array([self.fitness_fn(p) for p in x])
        gbest, gfit = pbest[pfit.argmax()].copy(), pfit.max()
        history = [float(gfit)]

        for t in range(self.iters):
            r1, r2 = self.rng.rand(self.n, dim), self.rng.rand(self.n, dim)
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = np.clip(x + v, self.lb, self.ub)
            for i in range(self.n):
                f = self.fitness_fn(x[i])
                if f > pfit[i]:
                    pfit[i], pbest[i] = f, x[i].copy()
            if pfit.max() > gfit:
                gfit, gbest = pfit.max(), pbest[pfit.argmax()].copy()
            history.append(float(gfit))
            print(f"Iteration {t + 1}/{self.iters} | best validation accuracy = {gfit:.4f}")
        return gbest, float(gfit), history