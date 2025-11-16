@startuml
skinparam classAttributeIconSize 0
left to right direction

' ===== BASE CORE =====
class Tensor {
    +data
    +shape: tuple
    +rows: int
    +cols: int
    +__init__(data)
    +_convert(data)
    +_get_shape(data)
    +__add__(other)
    +__sub__(other)
    +__mul__(other)
    +__truediv__(other)
    +dot(other)
    +transpose()
    +apply(func)
    +sum(axis)
    +get_shape()
    {static} +random(rows, cols, low, high)
    +scalar_multiply(scalar)
}

class MathUtils {
    {static} +exp(x, terms)
    {static} +log(x, base, max_iter, tol)
    {static} +abs(x)
    {static} +clip(x, min_val, max_val)
    {static} +sigmoid(x)
    {static} +tanh(x)
    {static} +relu(x)
    {static} +leaky_relu(x, alpha)
    {static} +softmax(values)
}

' ===== LAYERS =====
abstract class Layer {
    {abstract} +forward(x)
    {abstract} +backward(grad_output)
    +get_params()
    +set_params(params)
}

class Dense {
    -W: Tensor
    -b: Tensor
    -dW: Tensor
    -db: Tensor
    -in_features: int
    -out_features: int
    -use_bias: bool
    -x: Tensor
    +__init__(in_features, out_features, use_bias)
    +forward(x): Tensor
    +backward(grad_output): Tensor
    +get_params(): list
    +set_params(params)
}

class Sequential {
    -layers: list[Layer]
    +__init__(*layers)
    +forward(x): Tensor
    +backward(grad_output): Tensor
    +get_params(): list
    +set_params(params)
}

note right of Dense
    Linear is an alias
    for Dense
end note

Layer <|-- Dense
Layer <|-- Sequential

Sequential o--> "0..*" Layer

' ===== ACTIVATIONS =====
class ReLU {
    -x: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class Sigmoid {
    -out: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class Tanh {
    -out: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class LeakyReLU {
    -alpha: float
    -x: Tensor
    +__init__(alpha)
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class ELU {
    -alpha: float
    -x: Tensor
    +__init__(alpha)
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class Swish {
    -x: Tensor
    -sigmoid_x: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class GELU {
    -x: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

class Softmax {
    -x: Tensor
    -out: Tensor
    +forward(x): Tensor
    +backward(grad_output): Tensor
}

Layer <|-- ReLU
Layer <|-- Sigmoid
Layer <|-- Tanh
Layer <|-- LeakyReLU
Layer <|-- ELU
Layer <|-- Swish
Layer <|-- GELU
Layer <|-- Softmax

ReLU ..> Tensor: uses
Sigmoid ..> Tensor: uses
Sigmoid ..> MathUtils: uses
Tanh ..> Tensor: uses
Tanh ..> MathUtils: uses
LeakyReLU ..> Tensor: uses
ELU ..> Tensor: uses
ELU ..> MathUtils: uses
Swish ..> Tensor: uses
Swish ..> MathUtils: uses
GELU ..> Tensor: uses
GELU ..> MathUtils: uses
Softmax ..> Tensor: uses
Softmax ..> MathUtils: uses

' ===== MODEL =====
class Model {
    -network: Sequential
    +__init__(layers)
    +forward(x): Tensor
    +backward(grad): Tensor
    +get_params(): list
    +set_params(params)
}

Model *--> Sequential
Model ..> Layer: validates

' ===== LOSSES =====
abstract class _Loss {
    #_validate_inputs(y_pred, y_true)
    {abstract} +forward(y_pred, y_true): float
    {abstract} +backward(): Tensor
}

class MSELoss {
    -y_pred: Tensor
    -y_true: Tensor
    +forward(y_pred, y_true): float
    +backward(): Tensor
}

class MAELoss {
    -y_pred: Tensor
    -y_true: Tensor
    +forward(y_pred, y_true): float
    +backward(): Tensor
}

class BinaryCrossEntropyLoss {
    -y_pred: Tensor
    -y_true: Tensor
    +forward(y_pred, y_true): float
    +backward(): Tensor
}

class CrossEntropyLoss {
    -y_pred: Tensor
    -y_true: Tensor
    +forward(y_pred, y_true): float
    +backward(): Tensor
}

_Loss <|-- MSELoss
_Loss <|-- MAELoss
_Loss <|-- BinaryCrossEntropyLoss
_Loss <|-- CrossEntropyLoss

MSELoss ..> Tensor: uses
MAELoss ..> Tensor: uses
MAELoss ..> MathUtils: uses
BinaryCrossEntropyLoss ..> Tensor: uses
BinaryCrossEntropyLoss ..> MathUtils: uses
CrossEntropyLoss ..> Tensor: uses
CrossEntropyLoss ..> MathUtils: uses

' ===== OPTIMIZERS =====
abstract class Optimizer {
    #lr: float
    +__init__(lr)
    {abstract} +step(params): list
    +zero_grad()
}

class SGD {
    +__init__(lr)
    +step(params): list
}

class Momentum {
    -momentum: float
    -velocity: dict
    +__init__(lr, momentum)
    +step(params): list
}

class RMSprop {
    -rho: float
    -epsilon: float
    -cache: dict
    +__init__(lr, rho, epsilon)
    +step(params): list
}

class Adam {
    -beta1: float
    -beta2: float
    -epsilon: float
    -m: dict
    -v: dict
    -t: int
    +__init__(lr, beta1, beta2, epsilon)
    +step(params): list
    +zero_grad()
}

Optimizer <|-- SGD
Optimizer <|-- Momentum
Optimizer <|-- RMSprop
Optimizer <|-- Adam

Optimizer ..> Tensor: operates on
SGD ..> Tensor: uses
Momentum ..> Tensor: uses
RMSprop ..> Tensor: uses
Adam ..> Tensor: uses

' ===== TRAINER =====
class Trainer {
    -model: Model
    -loss_fn: _Loss
    -optimizer: Optimizer
    -epochs: int
    +__init__(model, loss_fn, optimizer, epochs)
    -_validate_dataset(X, Y)
    +fit(X, Y, verbose): list
    +predict(X): list
    +evaluate(X, Y): float
}

Trainer --> Model
Trainer --> _Loss
Trainer --> Optimizer

' ===== METRICS =====
class Accuracy {
    +score(y_pred, y_true): int
}

class Precision {
    +score(y_pred, y_true): tuple
}

class Recall {
    +score(y_pred, y_true): tuple
}

class F1Score {
    +score(y_pred, y_true): float
}

Accuracy ..> Tensor: uses
Precision ..> Tensor: uses
Recall ..> Tensor: uses
F1Score ..> Tensor: uses

note bottom of Trainer
    Returns loss_history
    from fit() method
end note
@enduml