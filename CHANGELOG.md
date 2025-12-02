# v0.4.0
* Parameterize average pooling size in classification with MLP head
* Let user supply class label names to classifier
* Classification visual in tensorboard: Show bar plot of softmax
* Separation of perception, rule and head model in BasicNCAModel for improved modularity
* Classification: Average pooling between NCA backbone and classification head
* Dependency updates
* Moved training/inference step parameter from Trainer to BasicNCAModel

# v0.3.2
* Fix use of max() reduction in uncertainty estimator

# v0.3.1
* Add uncertainty estimation feature

# v0.3.0
* Rename VisualMultiImageClassification --> VisualRGBImageClassification
* Deprecate VisualMultiImageClassification
* Add record() member to BasicNCAModel to record sequences of predictions
* Adapt Animator class to new prediction API
* Use Animator class in growing emoji example task
* Fix evaluation script for MNIST example task
* Fix dimensionality error in "classify" method
* Add evaluation scripts for MedMNIST example tasks
* Fix visualization of normalized data
* Add (optional) MLP classification head in classification model
* Slightly adjust training parameters (learning rate and adam betas)
