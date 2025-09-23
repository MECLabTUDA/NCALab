# v0.3.3
* Parameterize average pooling size in classification with MLP head
* Let user supply class label names to classifier
* Classification visual in tensorboard: Show bar plot of softmax

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
