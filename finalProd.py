import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import snowflake.snowpark as snowpark

# Column Types
categorical = ['COMPANY_CODE_ID', 'VENDOR_ID', 'PLANT_ID', "POSTAL_CD"]
numerical = ['PLANNED_DELIVERY_DAYS']
date = ['CREATE_DATE', 'DELIVERY_DATE', 'REQUESTED_DELIVERY_DATE', 'FIRST_GR_POSTING_DATE']
drop = ['PURCHASE_DOCUMENT_ID', "RELEASE_DATE", "PURCHASE_DOCUMENT_ITEM_ID",
        "SUB_COMMODITY_DESC", "MRP_TYPE_ID", "MRP_TYPE_DESC_E", "SHORT_TEXT", "POR_DELIVERY_DATE",
        "INBOUND_DELIVERY_ID", "INBOUND_DELIVERY_ITEM_ID", 'BI_LAST_UPDATED_PURCHASE_DOCUMENT_ITEM'
        , 'MATERIAL_ID']


def process_batch(batch_df):
    # Process batch-wise transformations
    # Replace 0 with null and remove
    batch_df[date].replace(0, pd.NA)
    batch_df.dropna(subset=date, inplace=True, axis=0)

    # replace dates with days since 1/1/2000
    reference_date = pd.Timestamp('2000-01-01')
    for col in date:
        # Convert to string, format as a date, find days since reference date
        batch_df[col] = batch_df[col].astype(str)
        batch_df[col] = batch_df[col].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8])
        batch_df[col] = pd.to_datetime(batch_df[col], format='%Y-%m-%d', errors='coerce')
        batch_df[col] = (batch_df[col] - reference_date).dt.days

    # For each categorical feature, convert from labels to numbers and replace in the DF
    for cat in categorical:
        catties = batch_df[cat]
        enc = LabelEncoder()
        encoded = enc.fit_transform(catties)
        batch_df[cat] = encoded

    # Calculate target as told
    batch_df['target'] = batch_df['DELIVERY_DATE'] - batch_df['FIRST_GR_POSTING_DATE']
    batch_df.dropna(subset=['target'], inplace=True, axis=0)

    # Remove these columns as to not break the golden rule
    batch_df.drop('DELIVERY_DATE', axis=1, inplace=True)
    batch_df.drop('FIRST_GR_POSTING_DATE', axis=1, inplace=True)

    return batch_df


def main(session: snowpark.Session):
    # Your code goes here, inside the "main" handler.

    tableName = 'procurement_on_time_delivery.purchase_order_history'

    dataframe = session.table(tableName)
    # Drop Columns specified
    for column in drop:
        dataframe = dataframe.drop(column)

    # Convert to pandas
    df = dataframe.to_pandas()
    del dataframe

    # Define batch size
    batch_size = 10000
    num_rows = len(df)

    # Process data in batches
    for batch_start in range(0, num_rows, batch_size):
        batch_end = min(batch_start + batch_size, num_rows)
        batch_df = df.iloc[batch_start:batch_end]

        # Process batch
        processed_batch = process_batch(batch_df)

        # Train/test split
        processed_batch.dropna(subset=processed_batch.columns, inplace=True, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(processed_batch.drop('target', axis=1),
                                                            processed_batch['target'],
                                                            test_size=0.3, random_state=7)

        # Select the numerical features to scale
        numerical_features = ['PLANNED_DELIVERY_DAYS', 'CREATE_DATE', 'REQUESTED_DELIVERY_DATE']

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler to the training data and transform the numerical features
        x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])

        # Transform the test data using the scaler fitted on the training data
        x_test[numerical_features] = scaler.transform(x_test[numerical_features])

        # Implement Random Forest Regression
        gb = GradientBoostingRegressor(random_state=7)

    param_grid = {
    'n_estimators': [ 300],
    'max_depth': [25],
    'learning_rate': [0.05],
    'subsample': [0.9],
    'min_samples_split': [7],
    'min_samples_leaf': [2],
    'max_features': ['sqrt'],
    'max_leaf_nodes': [35]
}


    
    # Perform randomized search with cross-validation
    random_search_gb = GridSearchCV(gb, param_grid=param_grid, cv=3, n_jobs=-1)
    random_search_gb.fit(x_train, y_train)
    
    # Get best parameters and score
    best_params_gb = random_search_gb.best_params_
    best_score_gb = random_search_gb.best_score_
    
    # Predict on test set and calculate mean squared error
    y_pred_gb = random_search_gb.predict(x_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    
    print("Best Parameters (Gradient Boosting):", best_params_gb)
    print("Best Grid Search Score (Gradient Boosting):", best_score_gb)
    print("Mean Squared Error on Test Set (Gradient Boosting):", mse_gb)
    
    

    return session.create_dataframe(df)
