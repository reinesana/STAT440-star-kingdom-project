## Purpose

We need to build a model that estimates the `Cost` incurred when a pipe leaks, using train. The data used to train this model is train_predict_cost.csv.

We enter a pipe ID into the trained model. The trained model uses the pipe ID to obtain that pipe's location, length, and replacement timing, and feeds this information into the trained model.

The model calculates and returns the predicted cost of replacing that pipe.

- The dataset contains **14,107 rows**, after excluding 6 records from train with missing values (all missing installation dates).
- No records have been removed or capped as outliers.
- The period is January 6, 2019 through December 31, 2026.
- The train data was merged with pipes.csv by matching their pipe IDs.
- After merging the data, Takito created and added features that he felt could be useful for model training.

## Data Preprocessing

| Step | Details |
|---|---|
| Data merge | Attributes from `pipes.csv` were added by matching `Pipe ID` in `train.csv`. All 14,113 train records matched. |
| Missing-row removal | After the merge, rows with a missing value in any of the original 11 columns were removed. **Six rows with missing installation dates were excluded, leaving 14,107 rows**. |
| Years of use | **The number of days between the installation date and the leak date is divided by 365.25**. This approximates the number of days per year while accounting for leap years. Since installation times are unknown, the calculation uses dates only, and fractional years are retained without rounding. |
| Row ordering | Saved in **ascending `pipe_id` order**. |

## Meaning and Purpose of Each Column

| Column | Content / Calculation | Purpose / Role in Model Training |
|---|---|---|
| `pipe_id` | Original Pipe ID | Used for joining other data and tracking predictions. Not a basic predictor. |
| `date` | Leak date (YYYY-MM-DD) | Used to calculate pipe age, season, and calendar year, and for year-by-year validation. Do not input the date string directly. |
| `time` | Leak time (HH:MM) | Preserves the original record. Not used in the basic model; future scenarios do not require a time of day. |
| `cost` | Recorded cost | **Target variable y**. Do not include it in predictors X. The currency and cost breakdown cannot be established from the CSV alone. |
| `lay_date` | Installation date of the pipe before replacement | Source information for calculating pipe age. The date string itself is not a basic input. |
| `material` | Material before replacement, as a string | Used to learn cost differences by material. For models that support categorical inputs. |
| `surface` | Surface category, as a string | A candidate for differences in construction conditions and costs by surface. For models that support categorical inputs. |
| `gps_x1` | Start point x coordinate | Preserves the original geometry and is used to calculate length and midpoint. The basic model uses midpoint coordinates. |
| `gps_x2` | End point x coordinate | Same as above. |
| `gps_y1` | Start point y coordinate | Same as above. |
| `gps_y2` | End point y coordinate | Same as above. |
| `pipe_length` | sqrt((x2-x1)^2+(y2-y1)^2) | Represents pipe size. This is the straight-line distance between endpoints, not the actual length of a curved pipe. |
| `log1p_pipe_length` | ln(1+pipe_length) | A candidate for testing a nonlinear relationship with length. Select or compare it with the original length. |
| `years_used_at_leak` | (Days between leak date and installation date)/365.25 | Pipe age at the time of the leak. Calculated at date-level precision; fractional years are retained without rounding. |
| `log1p_years_used` | ln(1+years_used_at_leak) | A candidate for testing a nonlinear relationship between pipe age and cost. |
| `midpoint_x` | (x1+x2)/2 | Representative pipe location. A candidate for regional differences not captured by material and surface alone. |
| `midpoint_y` | (y1+y2)/2 | Same as above. |
| `event_decimal_year` | Leak year + elapsed days in that year / number of days in that year | A candidate for cost changes across years. It is not a price index itself. |
| `event_year` | Leak year as an integer, such as 2019 or 2020 | Used for annual cost differences, yearly summaries, and identifying years for validation. `event_decimal_year` is also retained so the representation can be selected for the task. |
| `event_month` | Leak month (1–12) | A candidate for learning month-specific cost differences. Calculate it from the scenario date for future predictions. Numeric differences do not correctly represent seasonal distance, so the seasonal sine/cosine features are also retained. |
| `season_sin` | sin(2π × elapsed fraction of the year) | Represents seasonality and the proximity of the end and beginning of a year. |
| `season_cos` | cos(2π × elapsed fraction of the year) | Used together with sine to distinguish positions within the seasonal cycle. |
| `material_code` | Fixed integer code for material (see below) | For models that specify categories using integers. The values do not represent order or distance. |
| `surface_code` | Fixed integer code for surface (see below) | Same as above. |
| `material_brass` | 1 for brass, otherwise 0 | Material flag for brass. |
| `material_cast_iron` | 1 for cast iron, otherwise 0 | Flag for the material label cast iron. |
| `material_copper` | 1 for copper, otherwise 0 | Material flag for copper. |
| `material_gray_iron` | 1 for gray iron, otherwise 0 | Flag for the material label gray iron. |
| `material_wrought_iron` | 1 for wrought iron, otherwise 0 | Material flag for wrought iron. |
| `surface_farmland` | 1 for farmland, otherwise 0 | Surface flag for farmland. |
| `surface_grassland` | 1 for grassland, otherwise 0 | Surface flag for grassland. |
| `surface_road` | 1 for road, otherwise 0 | Surface flag for roads. |
| `surface_structure` | 1 for structure, otherwise 0 | Flag for the structure surface category. |
| `surface_swamp` | 1 for swamp, otherwise 0 | Surface flag for swamp. |
| `surface_water` | 1 for water, otherwise 0 | Surface flag for water. |

Models cannot handle language, so they do not understand a surface described as water.
One idea was to create a column that assigns 1 to water and 2 to swamp. However, the model would then interpret swamp as something numerically greater than water, recognizing an order as it would for height. Therefore, columns such as surface_swamp and surface_water were created for each categorical variable, with values such as 1 for water and 0 otherwise.

Several log(variable) columns were also created. For example, with samples [1,2,1000], the model could be pulled toward 1000 during training. These columns apply a process called normalization to prevent that failure.

Columns named season_cos and season_sin were also added. January and February can be treated as adjacent numbers, which behaves well because their temperature and humidity do not differ. However, January and December also have similar temperatures and humidity, yet 12-1 puts them 11 months apart. This would cause the computer to categorize January and December as completely different seasons. Representing the values on a circle with sine and cosine solves this problem.

## Category Codes

| material | material_code |
|---|---:|
| brass | 0 |
| cast iron | 1 |
| copper | 2 |
| gray iron | 3 |
| wrought iron | 4 |

| surface | surface_code |
|---|---:|
| farmland | 0 |
| grassland | 1 |
| road | 2 |
| structure | 3 |
| swamp | 4 |
| water | 5 |

The flag columns were explained above, but category-code columns were also added in case someone would like to use them.

## Notes

When training a model, including many similar columns can cause overfitting to those columns. For example, if this project's training data contains 10 time-related columns but only one length-related column, the model could end up predicting costs using only time.

Therefore, when adding features in the future or using the features I created, please consult ChatGPT or a similar tool about which features to use.

## Validation Plan

Each person will build a model to predict costs.
We will compare the models using the mean and variability of RMSE across 5 folds.

In 5-fold validation, the train data is divided into five parts. Four parts are used as the actual training data, and the remaining part is used as test data. The trained model predicts costs, and RMSE is obtained by taking the differences from the actual costs in the test data, squaring them, and taking the square root. This metric indicates how far predictions are from the actual values.

The advantage of RMSE is that 1^2=1 but 2^2=4, so larger differences between predicted and actual values contribute more to the error.
Also, if costs are originally in CAD, RMSE is also expressed in CAD, making it easy to understand.

Use parts 1, 2, 3, and 4 as train and part 5 as test.
Use parts 1, 2, 3, and 5 as train and part 4 as test.
Repeat this for five rounds to examine how accurately the model predicts the test data.

This produces five RMSE values. Their average indicates the model's predictive accuracy. A larger value means the predictions deviate more from the actual values.

## Shared Setting

Please use **440** as the random seed.
