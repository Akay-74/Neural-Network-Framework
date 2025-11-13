```mermaid
sequenceDiagram
    autonumber

    participant User
    participant Model
    participant Layer
    participant Trainer
    participant Optimizer
    participant Loss
    participant Data

    User ->> Model: create model
    activate Model
    User ->> Model: add layers
    Model ->> Layer: init layers
    deactivate Model

    User ->> Model: compile model
    activate Model
    Model ->> Loss: set loss
    Model ->> Optimizer: set optimizer
    Model ->> Trainer: pass model
    deactivate Model

    User ->> Trainer: start training
    activate Trainer

    loop each epoch
        Trainer ->> Data: load batch
        activate Data
        Data -->> Trainer: batch ready
        deactivate Data

        Trainer ->> Model: forward
        activate Model
        Model ->> Layer: forward layers
        activate Layer
        Layer -->> Model: outputs
        deactivate Layer
        Model -->> Trainer: predictions
        deactivate Model

        Trainer ->> Loss: compute loss
        activate Loss
        Loss -->> Trainer: loss value
        deactivate Loss

        Trainer ->> Model: backward
        activate Model
        Model ->> Layer: backward layers
        activate Layer
        Layer -->> Model: gradients
        deactivate Layer
        Model -->> Trainer: gradients ready
        deactivate Model

        Trainer ->> Optimizer: update weights
        activate Optimizer
        Optimizer ->> Model: apply updates
        activate Model
        Model -->> Optimizer: updated
        deactivate Model
        deactivate Optimizer
    end

    Trainer -->> User: training done
    deactivate Trainer

    User ->> Trainer: evaluate
    activate Trainer
    Trainer ->> Data: load test data
    activate Data
    Data -->> Trainer: test batch
    deactivate Data

    Trainer ->> Model: forward test
    activate Model
    Model -->> Trainer: predictions
    deactivate Model

    Trainer -->> User: evaluation metrics
    deactivate Trainer

    User ->> Model: save model
    activate Model
    deactivate Model

    User ->> Model: load model
    activate Model
    deactivate Model

    User ->> Model: predict input
    activate Model
    Model ->> Layer: forward
    activate Layer
    Layer -->> Model: output
    deactivate Layer
    Model -->> User: prediction result
    deactivate Model
```