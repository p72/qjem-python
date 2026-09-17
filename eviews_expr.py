"""Translate EViews model equations into Python residual functions.

An equation "NAME:LHS=RHS" becomes  f(X, t) = LHS - RHS,  where X is a
(n_vars x n_periods) array. Lag operators and EViews time-series functions
(D, DLOG, @MOVAV, @MOVSUM) are expanded into explicit time offsets at
compile time, so X[i, t-k] references are resolved once, not per call.
"""
import math
import re

TOKEN_RE = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+)|(@?[A-Za-z_]\w*)|(.))")
FUNCS = {"D", "LOG", "DLOG", "EXP", "ABS", "@ABS", "@MOVAV", "@MOVSUM"}


# ---------------------------------------------------------------- parsing
def tokenize(s):
    toks = []
    for num, ident, op in TOKEN_RE.findall(s):
        if num:
            toks.append(("num", float(num)))
        elif ident:
            toks.append(("id", ident.upper()))
        elif op.strip():
            toks.append(("op", op))
    return toks


class Parser:
    def __init__(self, text, coefs):
        self.toks = tokenize(text)
        self.i = 0
        self.coefs = coefs

    def peek(self, kind=None, val=None):
        if self.i >= len(self.toks):
            return False
        k, v = self.toks[self.i]
        return (kind is None or k == kind) and (val is None or v == val)

    def take(self, kind=None, val=None):
        if not self.peek(kind, val):
            raise SyntaxError(f"expected {kind} {val} at token {self.i}: {self.toks[self.i:self.i+3]}")
        tok = self.toks[self.i]
        self.i += 1
        return tok[1]

    def parse(self):
        node = self.expr()
        if self.i != len(self.toks):
            raise SyntaxError(f"trailing tokens {self.toks[self.i:]}")
        return node

    def expr(self):
        node = self.term()
        while self.peek("op", "+") or self.peek("op", "-"):
            node = ("bin", self.take(), node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek("op", "*") or self.peek("op", "/"):
            node = ("bin", self.take(), node, self.unary())
        return node

    def unary(self):
        if self.peek("op", "-") or self.peek("op", "+"):
            op = self.take()
            node = self.unary()
            return ("neg", node) if op == "-" else node
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek("op", "^"):
            self.take()
            return ("bin", "^", base, self.unary())
        return base

    def atom(self):
        if self.peek("num"):
            return ("num", self.take())
        if self.peek("op", "("):
            self.take()
            node = self.expr()
            self.take("op", ")")
            return node
        name = self.take("id")
        if not self.peek("op", "("):
            return ("var", name, 0)
        self.take()
        if name in FUNCS:
            args = [self.expr()]
            while self.peek("op", ","):
                self.take()
                args.append(self.expr())
            self.take("op", ")")
            return ("fn", name.lstrip("@"), args)
        # coefficient C_XXX(k) or lagged variable X(-k)
        sign = -1 if self.peek("op", "-") and self.take() else 1
        if self.peek("op", "+"):
            self.take()
        k = int(self.take("num")) * sign
        self.take("op", ")")
        if name.startswith("C_"):
            return ("num", float(self.coefs[name][k - 1]))
        return ("var", name, k)


# ---------------------------------------------------------------- codegen
def gen(node, shift, vidx, contemp):
    """Emit Python source for node evaluated at time t+shift."""
    kind = node[0]
    if kind == "num":
        return repr(node[1])
    if kind == "var":
        lag = node[2] + shift
        if lag == 0:
            contemp.add(vidx[node[1]])
        return f"X[{vidx[node[1]]},t{lag:+d}]"
    if kind == "neg":
        return f"(-{gen(node[1], shift, vidx, contemp)})"
    if kind == "bin":
        op = "**" if node[1] == "^" else node[1]
        return f"({gen(node[2], shift, vidx, contemp)}{op}{gen(node[3], shift, vidx, contemp)})"
    name, args = node[1], node[2]
    g = lambda s: gen(args[0], s, vidx, contemp)
    if name == "D":
        return f"({g(shift)}-{g(shift - 1)})"
    if name == "LOG":
        return f"log({g(shift)})"
    if name == "EXP":
        return f"exp({g(shift)})"
    if name == "ABS":
        return f"abs({g(shift)})"
    if name == "DLOG":
        return f"(log({g(shift)})-log({g(shift - 1)}))"
    if name in ("MOVAV", "MOVSUM"):
        n = int(args[1][1])
        s = "+".join(g(shift - k) for k in range(n))
        return f"(({s})/{n})" if name == "MOVAV" else f"({s})"
    raise ValueError(name)


def collect_vars(node, out):
    if node[0] == "var":
        out.add(node[1])
    elif node[0] in ("neg",):
        collect_vars(node[1], out)
    elif node[0] == "bin":
        collect_vars(node[2], out)
        collect_vars(node[3], out)
    elif node[0] == "fn":
        for a in node[2]:
            collect_vars(a, out)


def safe_log(x):
    return math.log(x) if x > 0 else math.nan


def compile_equation(lhs_ast, rhs_ast, vidx):
    contemp = set()
    src = f"lambda X,t: {gen(lhs_ast, 0, vidx, contemp)}-({gen(rhs_ast, 0, vidx, contemp)})"
    fn = eval(src, {"log": safe_log, "exp": math.exp, "abs": abs})
    return fn, contemp, src
