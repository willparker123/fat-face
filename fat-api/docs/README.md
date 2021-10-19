# FAT-API

## Python AI Explainability module 

#### Designed to be more granular and abstract than others, allowing black-box predict functions to be passed to explainability methods without relying on constructs from other Python modules which encapsulate detail and otherwise variable parameters.


### Overview

This package is designed to give users complete freedom over the explainability methods included in this module and their parameters, and arguments can also be passed into explain() methods so that the same object can be used for different applications of the same method.

Objects and methods are completely transparent, and have as much extendability as possible to allow users to create their own explainability methods, or adapt those included in the package. Internal functions of objects can also be overwritten, without having to instantiate a new object.



### Requirements

The only requirements for this package are Python3 and Numpy, but to use existing machine learning models users will need to use those from modules (e.g. sklearn, scipy, tensorflow) or their own models (blackboxes with predict(), predict_proba() functions - see the class documentation for method-specific requirements)

| Module      | Version |
| ----------- | ----------- |
| Python      | 3.6 +      |
| Numpy   | 1.19.0        |



### Methods

| Method      | Source |
| ----------- | ----------- |
| FACE | [Feasible and Actionable Counterfactual Explanations](https://arxiv.org/abs/1909.09369) |
| CEM | [Contrastive Explanations with Pertient Negatives](https://papers.nips.cc/paper/2018/file/c5ff2543b53f4cc0ad3819a36752467b-Paper.pdf) |
| ALE | [Accumulated Local Effects](https://christophm.github.io/interpretable-ml-book/ale.html) |
| PDP | [Partial Dependence Plot](https://christophm.github.io/interpretable-ml-book/pdp.html) |