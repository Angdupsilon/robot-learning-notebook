# Oct 29, 2026
import math

#  visualize neural tree
import networkx as nx
import matplotlib.pyplot as plt

def trace(root):
    # collect every node and edge reachable from root
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_nx(root, filename="micrograd_graph.png"):
    nodes, edges = trace(root)          # reuses your existing trace()
    G = nx.DiGraph()
    labels = {}

    # value nodes + op nodes, same structure as draw_dot
    for n in nodes:
        uid = id(n)
        G.add_node(uid, kind="value")
        labels[uid] = f"{n.label}\ndata {n.data:.4f}\ngrad {n.grad:.4f}"
        if n._op:
            op_id = (uid, n._op)
            G.add_node(op_id, kind="op")
            labels[op_id] = n._op
            G.add_edge(op_id, uid)

    for n1, n2 in edges:
        G.add_edge(id(n1), (id(n2), n2._op))

    # left-to-right layout: each column is one topological "generation"
    for layer, gen in enumerate(nx.topological_generations(G)):
        for node in gen:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer")

    value_nodes = [k for k, d in G.nodes(data=True) if d["kind"] == "value"]
    op_nodes    = [k for k, d in G.nodes(data=True) if d["kind"] == "op"]

    fig, ax = plt.subplots(figsize=(14, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=15, node_size=2500)
    nx.draw_networkx_labels(G, pos, labels={k: labels[k] for k in value_nodes},
                            ax=ax, font_size=8,
                            bbox=dict(boxstyle="round", fc="lightyellow", ec="black"))
    nx.draw_networkx_labels(G, pos, labels={k: labels[k] for k in op_nodes},
                            ax=ax, font_size=10,
                            bbox=dict(boxstyle="circle", fc="lightblue", ec="black"))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

# start of micrograd
class Value:
    # initialize children processes and operations to back-track
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children) # way to get previous functions
        self._op = _op
        self.label=label

    # represent instead of giving back obj
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2*self.data)-1)/(math.exp(2*self.data)+1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad +=  (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self): # backpropagation through each element
        topo = []
        visited = set()
        # topological order backwards, lists out the children and then use recursion to go deeper
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0 # initiate last element's gradient

        for node in reversed(topo): # backpropagation through each element
            node._backward()






a = Value(-1, label="a")
b = Value(2, label="b")
c = Value(3, label="c")

d = a*b; d.label = "d"
e = d + c; e.label = "e"

f = Value(-2, label="f")

g = e*f; g.label = "g"
h = g.tanh(); h.label="h"

'''
# because value start at 0, we need to initialize this
h.grad = 1.0
h._backward()
g._backward()
e._backward()
d._backward()

previously we needed to do this, but because backward pass in already defined, we can just call: h.backward()
'''
h.backward()

print(h)
draw_nx(h)