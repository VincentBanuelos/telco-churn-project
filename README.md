# <a name="top"></a>TELCO CHURN PREDICTION - CLASSIFICATION PROJECT
![]()

by: Vincent Banuelos

***
[[Project Description/Goals](#project_description_goals)]
[[Initial Questions](#initial_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
The purpose of this project is to create a model that will predict churn using data acquired from
a provided database, and using that model to make predictions from a group of customers.

Goal 1: Find drivers for customer churn at Telco. Why are customers churning?

Goal 2: Construct a ML classification model that accurately predicts customer churn.

Goal 3: Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?

[[Back to top](#top)]


## <a name="initial_questions"></a>Initial Questions:

- Do number of add-ons affect rate of churn?
- Do customers with partners and dependents churn more than those with no dependents?
- Are Senior citizens more or less likely to churn than non Senior Citizens?
- Does tenure affect churn?

[[Back to top](#top)]


## <a name="planning"></a>Planning:

- Create README.md with data dictionary, project and business goals, and come up with initial hypotheses.
- Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the function.
- Clearly define at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train four different classification models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best and evaluate that single model on the test dataset.
- Create CSV file with the measurement id, the probability of the target values, and the model's prediction for each observation in my test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| churn | 1 if the customer has churned | int |
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| customer_id | Unique id for each customer| string |
| senior_citizen| 1 if customer is a senior citizen | int |
| tenure | Months of tenure as a customer| int |
| monthly_charges| The customer's monthly bill| float |
| total_charges| The customer's total bills since they have been a customer| float|
| is_male | 1 if the customer is male | int |
| partner | 1 if the customer has a partner  | int |
| dependents | 1 if the customer has dependents | int |
| phone | 1 if the customer has phone service | int |
| paperless_billing | 1 if the customer has paperliess billing | int |
| multiple_lines_yes | 1 if the customer has multiple phone lines | int |
| online_security_no | 1 if the customer has internet but no online security | int |
| online_security_yes | 1 if the customer has online security add-on | int |
| online_backup_no | 1 if the customer has internet but no online backup | int |
| online_backup_yes | 1 if the customer has online backup | int |
| device_protection_no | 1 if the customer has internet but no device protection | int |
| device_protection_yes | 1 if the customer has device protection | int |
| tech_support_no | 1 if the customer has internet but no tech support | int |
| tech_support_yes | 1 if the customer has tech_support | int |
| streaming_tv_no | 1 if the customer has internet but no streaming tv | int |
| streaming_tv_yes | 1 if the customer has streaming tv | int |
| streaming_movies_no | 1 if the customer has internet but no streaming movies | int |
| streaming_movies_yes | 1 if the customer has streaming movies | int |
| contract_type_month-to-month | 1 if the customer has a month-to-month contract | int |
| contract_type_one_year | 1 if the customer has a one year contract  | int |
| contract_type_two_year | 1 if the customer has a two year contract | int |
| payment_type_bank_transfer_auto | 1 if the customer pays by automatic bank transfer | int
| payment_type_credit_card_auto | 1 if the customer pays automatically by credit card | int
| payment_type_electronic_check | 1 if the customer pays manually by electronic check | int
| payment_type_mailed_check | 1 if the customer pays manually by mailed check | int
| internet_type_dsl  | 1 if the customer has DSL internet service |  int
| internet_type_fiber_optic | 1 if the customer has fiber optic internet service | int
| internet_type_none | 1 if the customer has no internet | int
| num_addons | sum of how many internet service add-ons the customer has | int
---

## <a name="reproduce"></a>Reproduction Requirements:

You will need your own env.py file with database credentials then follow the steps below:

  - Download the acquire.py, prepare.py, explore.py, and final_report.ipynb files
  - Add your own env.py file to the directory (user, host, password)
  - Run the final_repot.ipynb notebook

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion and Next Steps:

- Exploration of Data showed that the following items showed the following features have at least some relationship to churn.
    - Dependents
    - Partners
    - Whether a customer is a senior citizen
    - Number of add-ons a customer has
- Takeaways
    - Customers who have dependents churned at a much lower rate than those who did not
    - Customers with partners also churn at a lower rate than those who did not.
    - In addition to customers with either a dependent or partner churning less, customers with both a dependent and a partner churn more so than those who have either or.
    - Senior citizens are at a more likely rate to churn than non-senior citizens.
    - The more add-ons that a customer has included in their plan the less likely they are to churn.
    
- Recommendations
    - Seeing that customers that have both dependents and a partner churn at a lower rate, perhaps we could implement promotions that encourage clients to add partners and dependents to account with the bonus that the more people they add the cheaper the overall plan will be in comparison to have multiple individual plans.
    - A promotion that could also be implemented could include a plan that offers different tiers of add-ons for a lower price when bundled. Thus making a customer more dependent on those add-ons, i.e online_security, tech_support, etc.

- Model
    - The model's performance can be summarized as follows
        - Accuracy of 75% on in-sample (train), 77% on out-of-sample data (validate), and an accuracy of 76% on the test data.

- Next Steps
    - Following this initial project, I would like to create a more refined model. Whether it is through running different groups of features through my current model, tuning the hyperparameters or both.
    - As well as creating more python files that store functions so as to make the final report less filled by code and easier to read.
    
    
[[Back to top](#top)]
