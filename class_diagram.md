## NNF (Neural Network Framework) – Class Diagram

```mermaid
classDiagram
    %% ================= Core =================
    class Tensor {
      - data
      - shape
      + dot(other)
      + reshape(...)
      + softmax(values)
    }

    class MathUtils {
      + exp(x, terms=20)
      + log(x, base=None)
      + abs(x)
    }

    MathUtils <.. Tensor : "helper math ops"

    %% ================= Layers & Activations =================
    class Layer {
      <<abstract>>
      + forward(x)
      + backward(grad_output, lr=0.01)
    }

    class Linear {
      - W
      - b
      - x
      + forward(x)
      + backward(grad_output, lr=0.01)
    }

    class ReLU
    class Sigmoid
    class Tanh
    class LeakyReLU
    class ELU
    class Swish
    class GELU

    Layer <|-- Linear
    Layer <|-- ReLU
    Layer <|-- Sigmoid
    Layer <|-- Tanh
    Layer <|-- LeakyReLU
    Layer <|-- ELU
    Layer <|-- Swish
    Layer <|-- GELU

    %% ================= Model & Sequential =================
    class Sequential {
      - layers : List[Layer]
      + forward(x)
      + backward(grad, lr)
      + add(layer)
    }

    class Model {
      - network : Sequential
      + forward(x)
      + backward(grad, lr)
    }

    Sequential *-- Layer : "contains"
    Model o-- Sequential : "wraps"

    %% ================= Losses =================
    class Loss {
      <<abstract>>
      + forward(y_pred, y_true)
      + backward()
    }

    class MSELoss
    class MAELoss
    class BinaryCrossEntropyLoss
    class CrossEntropyLoss

    Loss <|-- MSELoss
    Loss <|-- MAELoss
    Loss <|-- BinaryCrossEntropyLoss
    Loss <|-- CrossEntropyLoss

    %% ================= Trainer =================
    class Trainer {
      - model : Model
      - loss_fn : Loss
      - lr : float
      - epochs : int
      + fit(X, Y, verbose=True)
      + predict(X)
    }

    Trainer o-- Model : "owns & trains"
    Trainer ..> Loss : "uses for error"

    %% ================= Metrics =================
    class Metric {
      <<interface>>
      + score(y_pred, y_true)
    }

    class Accuracy
    class Precision
    class Recall
    class F1Score

    Accuracy ..|> Metric
    Precision ..|> Metric
    Recall ..|> Metric
    F1Score ..|> Metric

    Trainer --> Metric : "evaluates performance"

    %% ================= Workflow arrows =================
    Trainer --> Model : "forward()"
    Model --> Sequential : "forward()"
    Sequential --> Layer : "forward()"
    Layer --> Tensor : "data transform"

    Trainer --> Loss : "forward() compute loss"

    Loss --> Model : "backward() gradients"
    Model --> Sequential : "backward()"
    Sequential --> Layer : "backward()"
```