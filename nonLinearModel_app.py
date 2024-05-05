import streamlit as st
import pandas as pd
import joblib


def load_model():
    # Load the pre-trained model
    return joblib.load('nonlinear_best_mlp_model.joblib')  # Adjust the path as necessary


def predict(features):
    model = load_model()
    return model.predict(features)


def main():
    # Main area for outputs
    st.title('Renewable Electricity Net Supply Prediction App')
    st.subheader('Purpose:')
    st.markdown(
        "This tool predicts the :blue[net supply of renewable (solar and wind) electricity] for a fictional jurisdiction. A predicted positive net supply indicates that supply of renewable electricity is greater than demand for same in the specified period. A negative net supply indicates the opposite. This application was hastily developed and has not undergone testing or validation. It is solely intended for demonstration purposes only.")

    st.subheader('Instructions:')
    st.markdown(
        """
        From the sidebar, input your desired parameters. These will then be passed to the trained :blue[MLP Neural Network machine learning model] in the backend for analysis and prediction. The machine learning model was developed using Python, utilizing the Scikit-learn package for implementation.
        """
    )

    st.subheader('Credit:')
    st.markdown(
        "This tool was developed by :blue[Dr. Yakubu Abdul-Salam], Associate Professor of Energy Economics at the University of Aberdeen, UK.")

    # User inputs via sidebar
    st.sidebar.header('User Input Parameters')

    # Define season to month mapping
    season_month_mapping = {
        'Spring': ['March', 'April', 'May'],
        'Summer': ['June', 'July', 'August'],
        'Autumn': ['September', 'October', 'November'],
        'Winter': ['December', 'January', 'February']
    }

    # Season selection
    season = st.sidebar.selectbox('Season', list(season_month_mapping.keys()))

    # Month selection based on chosen season
    months = season_month_mapping[season]
    month = st.sidebar.selectbox('Month', months)

    # Other inputs
    day = st.sidebar.selectbox('Day',
                               ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
    hour = st.sidebar.selectbox('Hour', list(range(1, 24)))  # 1 to 23 for hours
    holiday = st.sidebar.selectbox('Holiday', ('No', 'Yes'))

    Temperature = st.sidebar.slider('Temperature', -20.0, 40.0, 20.0)
    Sunlight_Hours = st.sidebar.slider('Sunlight Hours', 0.0, 24.0, 12.0)
    Economic_Indicator = st.sidebar.number_input('Economic Indicator', min_value=0, value=100)

    Temperature_sqrd = Temperature ** 2
    # Create features based on input
    feature_order = ['Temperature', 'Temperature_sqrd', 'Sunlight_Hours', 'Economic_Indicator', 'Month_August', 'Month_December',
                     'Month_February', 'Month_January', 'Month_July', 'Month_June', 'Month_March', 'Month_May',
                     'Month_November', 'Month_October', 'Month_September', 'Day_Monday', 'Day_Saturday',
                     'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday', 'Hour_1', 'Hour_2', 'Hour_3',
                     'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
                     'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20',
                     'Hour_21', 'Hour_22', 'Hour_23', 'Season_spring', 'Season_summer', 'Season_winter', 'Holiday_Yes']

    data = {'Temperature': Temperature, 'Temperature_sqrd': Temperature_sqrd, 'Sunlight_Hours': Sunlight_Hours, 'Economic_Indicator': Economic_Indicator}
    for feature in feature_order[3:]:  # Start from 'Month_August' to skip continuous features already in 'data'
        if 'Month' in feature:
            data[feature] = 1 if month == feature.split('_')[1] else 0
        elif 'Day' in feature:
            data[feature] = 1 if day == feature.split('_')[1] else 0
        elif 'Hour' in feature:
            data[feature] = 1 if int(hour) == int(feature.split('_')[1]) else 0
        elif 'Season' in feature:
            data[feature] = 1 if season == feature.split('_')[1] else 0
        elif 'Holiday' in feature:
            data[feature] = 1 if holiday == 'Yes' else 0

    features = pd.DataFrame([data], columns=feature_order)

    if st.button(':white[Predict Net Supply]', key='calculate_button', type='primary', use_container_width=1, help='Click to predict net supply'):
        result = predict(features)
        formatted_result = f"{result[0]:.3f} MW"  # Format the result to 3 decimal places and add 'MW'
        st.success(f'Predicted Net Supply is {formatted_result}')

    aboutThisApp = '''
    Title: Energy Demand and Supply Prediction App
Developer: Dr. Yakubu Abdul-Salam, Associate Professor of Economics, University of Aberdeen, UK

Purpose: 
This tool predicts the net supply of electricity for a fictional jurisdiction. 

Instructions:
From the sidebar, input your desired input parameters. These will then be passed to the trained MLP Neural Network model in the background for analysis and prediction.

Description:
As data was not readily available in the timespace I had for this project, fictional hourly electricity demand and supply, along with relevant temporal, weather and economic indicators, were generated for 4 years for a fictional example jurisdiction, culminating into 8760*4 = 35,040 observations. Reasonable assumptions were made in the generation of the fictional dataset, including for example that;

1. Demand will be higher during daytime and peak in summer due to cooling needs.
2. Supply from solar will be higher during sunny hours and in summer.
3. Supply from wind will vary and is not directly correlated with temperature.
4. Temperature will follow a typical seasonal pattern with added random noise.

The generated fictional data was then used to train, validate and test several supervised machine learning models using the following regression specification;

net_supply(t) = beta0 + beta1*temperature(t) + beta2*temperature_squared(t) + beta3*sunlight_hours(t) + beta4*economic_indicator(t) + beta_february*month_february(t) + ... + beta_december*month_december(t) + beta_tuesday*day_tuesday(t) + ... + beta_sunday*day_sunday(t) + beta_hour2*Hour_2(t) + ... + beta_hour24*Hour_24(t) + beta_summer*season_summer(t) + beta_autumn*season_autumn(t) + beta_winter*season_winter(t) + beta_Holiday*Holiday(t)

In the above model specification;
1. Net supply is the dependent variable, and is the difference between supply and demand for electricity
2. Temperature is the temperature in degrees celsius. An additional term for temperature is incorporated to account for non-linearities in the effect of temperature on net supply. This is well established in the literature.
3. Sunlight hours reflects the number of daylight hours, which peaks in summer. It influences both the demand (less heating required during more sunlight hours) and solar electricity supply.
4. Economic indicator captures macroeconomic influences on electricity demand and supply. A good proxy is industrial production index for example.
5. Temporal variables 'Month', 'Day', 'Hour', meteorological 'Seasons' and 'Holiday' are dummy variables taking values of 0 and 1 accordingly. Reference temporal variables (e.g. month_january) have been dropped from the model to prevent perfect collinearity.

The best model was found to be the MLP Neural Network estimator. This was then further refined using a grid search hyperparameter tuning approach. The final model is what is being used for prediction in this web application. When a user specifies input parameters on the left, the parameters are passed to the MLP Neural Network estimator for prediction.
    '''
    st.download_button('Download further details about this app', aboutThisApp,
                       type="primary", file_name="About this app.txt", help="Download the text file showing model specification and other details about this app.")

if __name__ == '__main__':
    main()
