# Seizure Detection
### In this document, we will discuss the development of a seizure detection system, which involves loading an EEG signal dataset and training a neural network model to identify seizures.

## Overview
In this project, we aim to design a system for detecting seizures in epilepsy patients. To achieve this, we will begin by creating a dataset and implementing a neural network for seizure detection. Subsequently, we will train the neural network on our dataset and assess its performance across various scenarios, presenting the results in the final report.

## Preparing Dataset
In this section, we prepare the dataset required for our learning process. The dataset, accessible via [this link](https://physionet.org/content/chbmit/1.0.0/), originates from `Boston Children's Hospital` and comprises EEG data from patients with epilepsy experiencing intractable seizures, which cannot be completely controlled with medication. The dataset consists of recordings from 23 cases (22 individuals, including one duplicate), comprising 5 boys and 17 girls, with ages ranging from 1.5 to 22 years.

This dataset encompasses both normal and seizure states, with samples recorded at a rate of 256. It contains a total of 664 files, including 129 seizure data files. Most recordings have a duration of 1 hour, although some extend to 2 or 4 hours due to specific cases. Additionally, due to sampling device constraints, data is sampled at a rate of 10 samples per second, and in some cases, a Dummy variable with a negative sign is used for data convenience. Now that we're familiar with the dataset, let's proceed to outline and thoroughly review the necessary steps in this section.

1. **Reading Seizure and Non-Seizure Data**
The data for each Subject Test, along with their types and seizure duration (if applicable), is stored in a specific format within a file named `chb-Summary.txt`.
 By reading each of these Summary files, we can access the contents of each individual's files to select their data and ranges.
Therefore, in the code implementation section, we extract the normal and seizure data by reading each summary file, and subsequently, we utilize this information.

2. **Channel Selection**
The data in the dataset is recorded at various frequencies, resulting in numerous available data channels. However, in this project, we exclusively utilize two channels: `FZ-CZ` and `CZ-PZ`. These channels are then extracted based on different labels obtained from the signals.

3. **Interval Selection**
When selecting data, there are various intervals to consider. Seizures have specific states and ranges, but choosing non-seizure data from different intervals can significantly impact the outcome. These intervals contain interesting sections. For better variation, we divide the selection into the following categories and focus on selecting data from these specified sections.
		-- Intervals with Seizures.
		-- Pre-seizure and initial intervals, indicating that a seizure is about to start.
		-- End intervals and post-seizure intervals, indicating that the seizure is over.
		-- Intervals between two seizures with normal data. 

4. **Eliminate noise**
To remove noise from the data, we use the following noise types and remove them. All these steps are included in the code, and there is the ability to choose whether to remove the specific type of noise or not. We use this ability to generate different datasets for testing and comparing the results.
		 -- 5% of the beginning and end of each data is recommended not to be used due to the lack of stability. For this reason, the duration of each data from Summary is obtained, and if the selected range is located in these sections, they will not be used.
		-- Existence of Gap in the data: This issue is mentioned in the dataset description section and is recognized by empty data located among the data, which will be deleted.
		-- Existence of Dummy data: These data have been added for the convenience of reading the signal, and they will be deleted by removing the negative sign.
		-- Existence of stray data and noise in the signal: This data is removed with a filter.

5. **Breaking Intervals with Windows**
After identifying the interesting intervals in the dataset, the next step is to segment them into equally-sized intervals, creating a clean dataset with uniform data sizes. These intervals can be divided into specified window sizes. Due to the sampling rate being 256, the length of each dataset data will be equal to `Window_Size × 256`.
Additionally, in this section, augmentation can be performed by shifting the window with a specific stride, generating different numbers of windows. The choice of whether to include this section is flexible. If you intend to create a relatively large dataset, this approach can be beneficial as it introduces similarity between consecutive data segments, enhancing the learning process. However, the stride size should be considered carefully; if it's too small, the data may become very similar, and if it's too large, not enough data will be generated.

6. **Attention to Balance and Sufficiency**
The quantity of data in the dataset and its balance can significantly impact the results. Neural networks require a substantial amount of data for effective model implementation; thus, the dataset should be relatively large. When working with limited data, data augmentation, which was discussed in a previous step, can be employed to increase the dataset's size.
Additionally, dataset balance is crucial for effective model learning. The model relies on diverse data types for learning, so presenting the data in a balanced format greatly aids in proper model training.

7. **Data Storage**
In this final step, the data is divided into equal windows and saved in Pickle files for use during the model's learning phase. We save different types of datasets in separate folders, and when using them, an information file specifies the data choices made during their creation.
To ensure comprehensive coverage across all cases, we carefully select the main data. Although our dataset is vast, we avoid relying solely on one Subject Test's data to maintain data completeness. The model's effectiveness is enhanced by incorporating data from multiple individuals, reflecting real-world diversity.

- Types of Generated Datasets
	To better illustrate the initial selections for each dataset's production, we employ various modes. The quality of each mode is evaluated in the results section using different criteria, including:
		-- Varying the window size (with values of 5, 10, and 20).
		-- Data generation with and without noise.
		-- Data generation with and without augmentation.
		-- Data generation with and without balanced labels.
		-- Exploring both limited and abundant data scenarios.
		-- Data generation with and without comprehensiveness.

- Additional Applications and System Enhancement
To facilitate different applications and enhance the proposed system, we consider various situations by adjusting the number of data classes:
		-- Separating pre-seizure and post-seizure sections as distinct classes for predicting seizures and preventing excessive medication injection if seizures are expected to end soon.
		-- Creating datasets related to gender for personalizing the system.
		-- Creating age-related datasets to personalize the system further.
This approach increases the system's adaptability and effectiveness in real-world scenarios.

## Classification

The next step is divided into data preparation and model design and implementation:

1. **Data Preparation:** 
In this section, we first extract the desired data from the Pickle files. Then, we filter out very high or very low frequencies and add the features from the previous phases to the dataset. The data is also normalized. We apply feature extraction to both channels and save the suitable features. Both the primary dataset and the feature dataset, in which feature extraction has been applied, are saved for use in the next step.

2. **Model Design and Implementation:** 
For model design and implementing the neural network, we experiment with changing the number of layers and filter sizes to achieve a state where the model's accuracy reaches a stable and high value. By altering these values, we aim to find the best model for the dataset. (The results of varying the number of layers and filter sizes will be presented in the results section.)
Before presenting the model, the data needs to be divided into training, validation, and test sets. Note that this division can also reduce the number of data entered into the learning section and therefore The number of data should be suitable for the network.
It's important to note that in the model, we need to implement a state where we can use the dataset with its features extracted. We add this state just before the final layer (without back propagation). The output neurons should match the number of classes, based on the features extracted by the CNN network and activated by our chosen algorithm. Thus, we need to include the capability to add the extracted features.
To achieve this, we feed the extracted features into the input of the second-to-last dense layer and concatenate this data with the output from the preceding layer and the one before it. Then, we feed this concatenated data into the final dense layer to obtain the result.
It's important to highlight that the layers before the feature addition layer continue to optimize weights using back propagation to predict the final labels, while the feature layer doesn't attempt to improve the weights.
The empirically determined best network architecture is as follows:
		-- Convolution layer with a filter size of 60, a kernel size of 4, ReLU activation, and a pooling layer with a size of 2 using the main data input.
		-- Convolution layer with a filter size of 60, a kernel size of 4, ReLU activation, and a pooling layer with a size of 2 using the input from the previous stage model.
		-- Flatten layer with the input from the previous step.
		-- Dense layer with 100 units using the input from the previous step.
		-- Dense layer with 20 units using the input from the previous step.
		-- Dense layer with the number of extracted features using the input from the feature extraction.
		-- Dense layer with 2 units using the concatenated input from the previous two steps.
Finally, we compile the model with the `Adam` optimizer and a learning rate of 0.001. We fit it to the training and validation data with a batch size of 20 and 8 epochs. The results are passed to the next step.

Your text is informative and detailed, but it could benefit from some minor punctuation and formatting improvements. Here's a revised version for better readability:

## Presentation of Results

In this section, we present the results by fitting and evaluating different types of datasets and observing their outcomes. We explore various aspects to assess the model's performance.

- **Various Metrics Checked in Previous Phases**
We assess the model's performance using a range of metrics including Precision, Recall, F1 Score, and the Confusion Matrix. To evaluate the results, we first obtain the Prediction Labels from the final model and then apply the aforementioned functions to check their values.

- **Finding the Accuracy Result on the Seizure Interval**
To determine the model's performance on seizure intervals, we use data that the model has not encountered before. This data is extracted from the last person in the dataset, ensuring that the model has not seen any Seizure or Non-Seizure data of this individual. The data is segmented into specific windows, and the results are evaluated as test data. We determine the accuracy value by examining whether more than half of the windows in this interval are detected as Seizure or Not. This analysis provides insights into the model's real-world performance over the long term of seizures.

- **Finding the False Alarm Value on the Non-Seizure Interval**
Similar to the previous step, we extract non-seizure data that has not been seen by the model. Using False Alarm analysis, we determine how many windows in this period are considered as seizures even though they are entirely within the normal range. The False Alarm value can be calculated using the False Positive Rate (FPR) or the False Discovery Rate (FDR). 
$$\text{False Positive Rate} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
$$\text{False Discovery Rate} = \frac{\text{FP}}{\text{FP} + \text{TP}}$$
FPR is a criterion that can be unknown in practice and its result in the learning process can be misleading, but compared to that, FDR has more applications in real models and establishes a more meaningful relationship. Therefore, we use FDR for reporting this section. Additionally, we create histograms of TP (True Positives), TN (True Negatives), FP (False Positives), and FN (False Negatives).

- **Plotting Accuracy and Loss from Different Epochs**
By plotting the curve, accuracy and loss graphs over different epochs, we can assess the model's performance with examining the model in the Train and Validation sections and identify potential `Overfitting`, `Underfitting` or `Goodfitting`. If there is a significant difference between the Validation and Training sections, it indicates underfitting because the model is not able to bring the results of these two parts together. Overfitting occurs when the Training and Validation graphs diverge. If the graphs do not fall into either of these categories, the model is considered to have goodfitting.

- **Plotting Accuracy Values from Evaluating the Test Data**
To demonstrate the stability of the model's results, we evaluate and consecutively assess the accuracy values. By plotting the accuracy chart for consecutive test data, we can observe the stability of the accuracy range. A flatter line indicates a more stable accuracy range.
 
	Now, we will examine the results in various different situations.
- **Various Metrics Checked in Previous Phases**
	To assess these metrics, we rely on the test data, and the values of these metrics are evaluated using the test data. The values of these metrics were displayed in the image of the epochs, so it is sufficient to check the confusion matrix.
	![](https://i.ibb.co/MZr1tPg/2-1.png)

- **Finding the Accuracy Result on the Seizure Interval**
In this model, the accuracy value on the Seizure interval that has not been visited by the model is equal to 0.8484.

- **Finding the False Alarm Value on the Non-Seizure Interval**
![enter image description here](https://i.ibb.co/svkVgqF/2-2.png)

- **Plotting Accuracy and Loss from Different Epochs**
![enter image description here](https://i.ibb.co/qkKCDbb/2-3.png)

- **Plotting Accuracy Values from Evaluating the Test Data**
![enter image description here](https://i.ibb.co/yXR9xKT/2-4.png)

- **Check in case of layer reduction**
78% accuracy when we remove the second conv1d layer and the result is as follows:
![enter image description here](https://i.ibb.co/M7g5cBd/2-5.png)

- **Check in case of layer addition**
In this case, as we add a convolutional layer, the accuracy gradually decreases, eventually leading to a divergence in the curve diagram.
![enter image description here](https://i.ibb.co/MsnBhs6/2-6.png)

## System Suggestion
Our system, through the analysis of EEG signals, segments the data into 5-second intervals and employs predictive modeling to determine the optimal timing for medication administration.
Given our comprehensive dataset that encompasses various modes, we can achieve promising results. However, to further enhance the system's performance, we propose several refinements.
First, we suggest considering the initial and final intervals of a Seizure event as distinct classes. By detecting signals that approach a Seizure event before it begins, we can make earlier predictions. Additionally, implementing a two-step evaluation process for Seizure occurrence can provide a higher level of confidence before administering medication.
To avoid unnecessary medication administration, we recommend introducing a class at the end of the Seizure interval. This allows for a two-stage confirmation of a Seizure event, ensuring medication is only administered when necessary.
For a more personalized approach, we can categorize individuals based on age and gender. This personalized data can be leveraged to improve Seizure diagnosis, taking into account the unique characteristics of each patient.
It's important to note that conducting comprehensive and personalized analyses, as described above, necessitates a substantial dataset. This dataset should include data collected at various times of the day to ensure both comprehensiveness and personalization in our approach.
