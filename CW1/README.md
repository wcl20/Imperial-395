# CO395 - Decision Tree Coursework

This is a coursework of Introduction to Machine Learning. This assignment require us to implement a decision tree algorithm and use it to
determine one of the indoor locations based on WIFI signal strengths collected from a mobile phone.

## Running the tests

To run the tests, python3 is required. Run the command

```
python3 main.py <path to data> <pruning option>
```


For example:
```
python3 main.py "wifi_db/clean_dataset.txt"
```
This will run a cross validation on the given dataset, and then it prints out
1) The confusion Matrix
2) The Average Recall for each class
3) The Average Precision for each class
4) The Average F1 Score for each class
5) The Average Classification rate
6) The Unweighted Average Recall
7) Depth of tree generated from entire dataset
8) Number of nodes of tree generated from entire dataset

There is also a pruning option (-p).
For example:
```
python3 main.py "wifi_db/clean_dataset.txt" -p
```
This will run a cross validation on the given dataset with pruning.


## Authors

* **Alvis Lee**
* **Chun Ki Chan**
* **Ivan Mang**
* **Matthew Cheung**
