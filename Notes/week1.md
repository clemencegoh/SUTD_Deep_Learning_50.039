# Professor in charge: Alexander Binder
Library used: Pytorch
---

# todo: review loss, empirical risk.


Huge assumption made:
- Data follows a certain probability rate
    - i.e. data follows a probability distribution

Objective:
- Reduce Loss


Assumption in this class:
- supervised learning
- define Loss function(prediction, ground truth) => loss

Prediction:
- Classification is good for final, not for process, since it is binary (0 or 1)
- Regression

---
2 kinds of errors
- Bounding box
    - refer to slide 6 of lesson 2
    - Predicted bounding box vs ground truth bounding box
    - Intersection over Union = (area of gnd n pred) / (area of gnd u pred)
- ?


---
What about measuring loss in a video? (Computer vision)
- Language is dynamic, multiple sentences can mean the same thing
- look at (keywords?)

---
Notion of Good mapping (perform well) <=> Notion of low loss

To do this:
- try to make Expectation of loss is minimal

**Recap:**

Expectation:
- If countable: expectation = Sum over discrete sets
- If uncountable: expectation = integral of distribution

Terms:
- Empirical Risk, R(h) => Expectation of Loss function


Approximation:
- By law of large numbers, approx becomes better as n increases

---
Slide 10:

Data generated based on equation where:
- N is gaussian noise

Why do we need disjoint test set?
- review slides 12-13

---
Occam's razor:
- if there are 2 hypothesis, simpler one should always be taken


---
Splitting training data:
- Split in such a way that **property/characteristics** 
frequency/ratio are preserved
    - One possible way is to split such that the histogram of each set are preserved
    - Another way is to split such that there is no spacial bias
    
**NEVER OPTIMIZE HYPERPARAMETERS IN TEST SET**
- (? find out why)

If we already know the test distribution, no learning required
- function can straight away be given    

