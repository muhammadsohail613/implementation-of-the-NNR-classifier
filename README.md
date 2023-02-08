# implementation-of-the-NNR-classifier

A nearest neighbors in radius (NNR) classifier is a variation of the KNN classifier we saw in the lecture, where instead of looking for the majority vote within K nearest neighbors of an instance, it inspects all instance’s neighbors in a given radius and assigns it a label according the majority vote in the radius.

Consider the below example: the new instance will be assigned the red label due to the majority of red instances in the dashed-line radius around it.

# Comments:
(1) In the provided datasets: all columns but the last are features, the last column (class) is labels.
(2) Your code should work seamlessly on any dataset of the given structure (and split into three parts): do not hardcode any dataset-specific details (e.g., file or column names, the number of classes). As a concrete example, the code should work on both datasets with the single change of names in the config.json file. The submission will be graded on different (unseen) dataset(s).
(3) Note that you are required to implement your own NNR version (and not invoke the RadiusNeighborsClassifier() classifier from sklearn). You can have a look at its implementation, but you may not find it very helpful. However, you can try it out on your data and compare the results.
(4) Implementation obtaining roughly 96% and 58% accuracy on the two datasets: students_loan (two classes) and body_performance (four classes), respectively, can be considered a good achievement. However, the task will be graded on additional datasets (not limited to these two).
(5) Make sure your code runtime doesn’t exceed 5 minutes (invocationàresults are printed). (6) The classifier should be implemented in PyCharm (similarly to assignment #1).
