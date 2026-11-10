import math


class Value:
    def __init__(self, data, _prev = ()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -(self - other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Unsupported exponent type (only accept int/float)."
        out = Value(self.data ** other, (self, ))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def exp(self):
        out =  Value(math.exp(self.data), (self, ))
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self, ))

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self, ))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        out = Value(1.0/(1+math.exp(-self.data)), (self, ))

        def _backward():
            self.grad += (out.data * (1.0-out.data)) * out.grad
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    
    def backward(self):
        topo: list[Value] = []
        vis: set[Value] = set()

        def build(v: Value):
            if v not in vis:
                vis.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
