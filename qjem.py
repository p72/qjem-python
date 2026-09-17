"""Q-JEM (2019 version) solver in pure Python.

Replicates what the BOJ EViews package does:
  * solve_qjem / change_sym : swap endogenous <-> exogenous variables
    (e.g. fix FXYEN on a target path and back out the shock V_FXYEN)
  * DM decomposition        : split the system into recursive blocks
  * EViews dynamic solve    : period-by-period, using solved lagged values
"""
import re
import sys

import numpy as np
from scipy.optimize import root
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from eviews_expr import Parser, collect_vars, compile_equation
from wf1reader import read_wf1

sys.setrecursionlimit(10000)
np.seterr(all="ignore")


def quarter_index(label, start="0001Q1"):
    y, q = map(int, label.split("Q"))
    y0, q0 = map(int, start.split("Q"))
    return (y - y0) * 4 + (q - q0)


class QJEM:
    def __init__(self, model_txt, wf1_path, start="0001Q1"):
        self.start = start
        self.series, self.coefs = read_wf1(wf1_path)
        self.names = sorted(self.series)
        self.vidx = {v: i for i, v in enumerate(self.names)}
        self.T = len(next(iter(self.series.values())))
        self.base = np.vstack([self.series[v] for v in self.names])

        self.eq_names, self.funcs, self.contemp, self.src = [], [], [], []
        for line in open(model_txt):
            line = re.sub(r"\s", "", line.split("'")[0]).upper()
            if not line:
                continue
            name, eq = line.split(":", 1)
            lhs, rhs = eq.split("=", 1)
            lhs_ast = Parser(lhs, self.coefs).parse()
            rhs_ast = Parser(rhs, self.coefs).parse()
            used = set()
            collect_vars(lhs_ast, used)
            collect_vars(rhs_ast, used)
            missing = used - set(self.vidx)
            if missing:
                raise KeyError(f"{name}: series not in workfile: {missing}")
            fn, contemp, src = compile_equation(lhs_ast, rhs_ast, self.vidx)
            self.eq_names.append(name)
            self.funcs.append(fn)
            self.contemp.append(contemp)
            self.src.append(src)
        self.endo = set(self.eq_names)

    # ------------------------------------------------------------ structure
    def block_structure(self, endo2exog=(), exog2endo=(), fixed=()):
        """fixed: endogenous variables whose own equation is dropped and whose
        value is taken as given from X (e.g. expectations pinned by forward
        guidance). Unlike endo2exog, no exogenous variable is freed in return."""
        endo2exog = {v.upper() for v in endo2exog}
        exog2endo = {v.upper() for v in exog2endo}
        fixed = {v.upper() for v in fixed}
        assert endo2exog <= self.endo, endo2exog - self.endo
        assert not (exog2endo & self.endo), exog2endo & self.endo
        assert len(endo2exog) == len(exog2endo)
        assert fixed <= set(self.eq_names) and not (fixed & endo2exog), fixed
        unknowns = sorted((self.endo - endo2exog - fixed) | exog2endo)
        uidx = {self.vidx[v]: j for j, v in enumerate(unknowns)}
        active = [i for i, name in enumerate(self.eq_names) if name not in fixed]
        n = len(active)

        # incidence: active equation k -- contemporaneous unknown j
        rows, cols = [], []
        for k, i in enumerate(active):
            for v in self.contemp[i]:
                if v in uidx:
                    rows.append(k)
                    cols.append(uidx[v])
        inc = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        match = maximum_bipartite_matching(inc, perm_type="column")  # eq -> unknown
        if (match < 0).any():
            bad = [self.eq_names[active[k]] for k in np.where(match < 0)[0]]
            raise ValueError(f"structurally singular system; unmatched equations: {bad[:10]}")
        eq_of_unknown = np.empty(n, int)
        eq_of_unknown[match] = np.arange(n)

        # dependency graph eq k -> eq that determines each unknown in eq k
        adj = [[eq_of_unknown[c] for c in inc.indices[inc.indptr[k]:inc.indptr[k + 1]]
                if eq_of_unknown[c] != k] for k in range(n)]
        blocks = []
        for comp in tarjan_scc(adj):  # dependencies come first
            comp = sorted(comp)
            eqs = [active[k] for k in comp]
            vars_ = [self.vidx[unknowns[match[k]]] for k in comp]
            blocks.append((eqs, vars_))
        return blocks

    # ------------------------------------------------------------ solving
    def solve(self, X, t_from, t_to, blocks, tol=1e-10, maxit=100):
        for t in range(t_from, t_to + 1):
            for eqs, vars_ in blocks:
                if len(eqs) == 1:
                    self._newton1(X, t, self.funcs[eqs[0]], vars_[0], tol, maxit)
                else:
                    self._newton_block(X, t, [self.funcs[i] for i in eqs], vars_, tol, maxit)
        return X

    @staticmethod
    def _newton1(X, t, f, j, tol, maxit):
        x = X[j, t]
        for _ in range(maxit):
            r = f(X, t)
            if abs(r) < tol * max(1.0, abs(x)):
                return
            h = 1e-7 * max(1.0, abs(x))
            X[j, t] = x + h
            dr = (f(X, t) - r) / h
            if dr == 0 or not np.isfinite(dr):
                X[j, t] = x
                raise RuntimeError(f"zero derivative at t={t}, var {j}")
            step = r / dr
            x_new = x - step
            # damp if the step produced an invalid point (e.g. log of negative)
            for _ in range(30):
                X[j, t] = x_new
                if np.isfinite(f(X, t)):
                    break
                step /= 2
                x_new = x - step
            if abs(x_new - x) < tol * max(1.0, abs(x)):
                return
            x = x_new
        raise RuntimeError(f"no convergence t={t}, var {j}")

    @staticmethod
    def _newton_block(X, t, fs, vars_, tol, maxit):
        def F(x):
            X[vars_, t] = x
            return np.array([f(X, t) for f in fs])

        sol = root(F, X[vars_, t].copy(), method="hybr", options={"xtol": tol})
        # hybr may report "not making good progress" once it hits machine
        # precision, so judge convergence by the residuals themselves
        resid = np.max(np.abs(F(sol.x)))
        if not resid < 1e-7:
            raise RuntimeError(f"block of size {len(fs)} failed at t={t}: "
                               f"max|resid|={resid:.2e} ({sol.message})")
        X[vars_, t] = sol.x

    # ------------------------------------------------------------ helpers
    def residuals(self, X, t):
        return np.array([f(X, t) for f in self.funcs])

    def series_of(self, X, name):
        return X[self.vidx[name.upper()]]

    def t(self, label):
        return quarter_index(label, self.start)

    def label(self, t):
        y0, q0 = map(int, self.start.split("Q"))
        k = (q0 - 1) + t
        return f"{y0 + k // 4:04d}Q{k % 4 + 1}"


def tarjan_scc(adj):
    """Iterative Tarjan; returns SCCs in reverse topological order
    (a component is emitted only after everything it points to)."""
    n = len(adj)
    index, low, on, stack, out = [None] * n, [0] * n, [False] * n, [], []
    counter = 0
    for s in range(n):
        if index[s] is not None:
            continue
        work = [(s, 0)]
        while work:
            v, pi = work.pop()
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for k in range(pi, len(adj[v])):
                w = adj[v][k]
                if index[w] is None:
                    work.append((v, k + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                elif on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return out
