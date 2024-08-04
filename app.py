# app.py
import streamlit as st
from streamlit_option_menu import option_menu # pip install streamlit-option-menu
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from backend import load_data, preprocess_data, encode_data,encode_data2, load_model, preprocess_input, make_prediction, scale_data,split_data

data_path = "https://raw.githubusercontent.com/UM1412/Data-Set/main/UberDataset.csv"
data = load_data(data_path)
data = preprocess_data(data)
data_03 = data.copy()
data_01 = encode_data(data)
data_02 = encode_data2(data)

X = data_02.iloc[:,2:-1]
y = data_02.iloc[:,-1]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
summary = model.summary()

category_counts = data['CATEGORY'].value_counts().reset_index()
category_counts.columns = ['CATEGORY', 'COUNT']
fig_01 = px.bar(
    category_counts,
    x='CATEGORY',
    y='COUNT',
    title='Category Counts',
    labels={'CATEGORY': 'Category', 'COUNT': 'Count'},
    color='CATEGORY',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='COUNT'
)

fig_01.update_layout(
    xaxis_title='Category',
    yaxis_title='Count',
    xaxis=dict(tickangle=-45),
    template='plotly_dark',
    showlegend=False
)

fig_02 = px.pie(
    category_counts,
    names='CATEGORY',
    values='COUNT',
    title='Category Distribution',
    color='CATEGORY',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hole=0.4
)

fig_02.update_layout(
    title_text='Category Distribution',
    title_font_size=24,
    template='plotly_dark',
    legend_title='Category',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

PURPOSE_counts = data['PURPOSE'].value_counts().reset_index()
PURPOSE_counts.columns = ['PURPOSE', 'COUNT']
fig_03 = px.bar(
    PURPOSE_counts,
    x='PURPOSE',
    y='COUNT',
    title='PURPOSE Counts',
    labels={'PURPOSE': 'PURPOSE', 'COUNT': 'Count'},
    color='PURPOSE',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='COUNT'
)

fig_03.update_layout(
    xaxis_title='PURPOSE',
    yaxis_title='Count',
    xaxis=dict(tickangle=-45),
    template='plotly_dark',
    showlegend=False
)

fig_04 = px.pie(
    PURPOSE_counts,
    names='PURPOSE',
    values='COUNT',
    title='Purpose Distribution',
    color='PURPOSE',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hole=0.4
)

fig_04.update_layout(
    title_text='Purpose Distribution',
    title_font_size=24,
    template='plotly_dark',
    legend_title='Purpose',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

SHIFT_counts = data['SHIFT'].value_counts().reset_index()
SHIFT_counts.columns = ['SHIFT', 'COUNT']

fig_05 = px.bar(
    SHIFT_counts,
    x='SHIFT',
    y='COUNT',
    title='Shift Counts',
    labels={'SHIFT': 'Shift', 'COUNT': 'Count'},
    color='SHIFT',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='COUNT'
)

fig_05.update_layout(
    xaxis_title='Shift',
    yaxis_title='Count',
    xaxis=dict(tickangle=-45),
    template='plotly_dark',
    showlegend=False
)

fig_06 = px.pie(
    SHIFT_counts,
    names='SHIFT',
    values='COUNT',
    title='Shift Distribution',
    color='SHIFT',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hole=0.4
)

fig_06.update_layout(
    title_text='Shift Distribution',
    title_font_size=24,
    template='plotly_dark',
    legend_title='Shift',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)


fig_07 = px.bar(
    data,
    x='PURPOSE',
    color='CATEGORY',
    title='Purpose Distribution by Category',
    labels={'PURPOSE': 'Purpose', 'count': 'Count'},
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='PURPOSE'
)

fig_07.update_layout(
    title_text='Purpose Distribution by Category',
    title_font_size=24,
    xaxis_title='Purpose',
    yaxis_title='Count',
    xaxis=dict(
        tickangle=-45,
        tickmode='array'
    ),
    template='plotly_dark',
    barmode='stack',
    legend_title='Category',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    margin=dict(l=0, r=0, t=50, b=100)
)


fig_07.update_traces(texttemplate='%{text}', textposition='outside')

# Plot Correlation Matrix Heatmap
numeric_dataset = data_01.select_dtypes(include=[np.number])
corr_matrix = numeric_dataset.corr()

fig_08 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='BrBG',
    zmin=-1, zmax=1,
    colorbar=dict(
        title='Correlation',
        tickvals=[-1, 0, 1],
        ticktext=['-1', '0', '1']
    ),
    text=corr_matrix.values,
    texttemplate='%{text:.2f}',
    textfont=dict(size=12),
    showscale=True
))

fig_08.update_layout(
    title='Correlation Matrix Heatmap',
    title_font_size=24,
    xaxis_title='Variables',
    yaxis_title='Variables',
    xaxis=dict(
        title='Variables',
        tickangle=-45,
        side='bottom'
    ),
    yaxis=dict(
        title='Variables',
        tickangle=0,
        scaleanchor='x'
    ),
    template='plotly_dark',
    margin=dict(l=100, r=20, t=50, b=100)
)

data['MONTH'] = pd.DatetimeIndex(data['START_DATE']).month
month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'April',
               5: 'May', 6: 'June', 7: 'July', 8: 'Aug',
               9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
data['MONTH'] = data['MONTH'].map(month_label)

month_counts = data['MONTH'].value_counts(sort=False)
month_max_miles = data.groupby('MONTH', sort=False)['MILES'].mean()
month_max_eta = data.groupby('MONTH', sort=False)['ETA'].mean()

df = pd.DataFrame({
'MONTHS': list(month_counts.index),
    'TOTAL COUNT': month_counts.values,
    'MEAN MILES': month_max_miles.values,
    'MEAN ETA': month_max_eta.values
}).reset_index()

df['MONTHS'] = pd.Categorical(df['MONTHS'], categories=month_label.values(), ordered=True)
df = df.sort_values('MONTHS')

fig_09 = px.line(df, x='MONTHS', y=['TOTAL COUNT', 'MEAN MILES','MEAN ETA'], title="Distribution of Month's Total Count,Mean Coount and Maen ETA")
fig_09.update_traces(mode='lines+markers', marker=dict(size=10, symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
                  line=dict(width=2))
fig_09.update_layout(
    xaxis=dict(title='Months', showgrid=True, showline=True, linecolor='black', mirror=True),
    yaxis=dict(title='Value Count', showgrid=True, showline=True, linecolor='black', mirror=True),
    plot_bgcolor='rgba(240,240,240,0.8)',
    paper_bgcolor='rgba(240,240,240,0.8)',
    font=dict(family='Arial', size=12, color='black'),
    title=dict(x=0.5, xanchor='center', y=0.95, yanchor='top'),
    template='plotly_dark',
    margin=dict(l=50, r=50, t=50, b=50)
)

data['DAY'] = data['START_DATE'].dt.weekday
day_label = {
    0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
data['DAY'] = data['DAY'].map(day_label)

day_counts = data['DAY'].value_counts(sort=False)
day_max_miles = data.groupby('DAY', sort=False)['MILES'].mean()
month_max_eta = data.groupby('DAY', sort=False)['ETA'].mean()

df = pd.DataFrame({
    'DAYS': list(day_counts.index),
    'TOTAL COUNT': day_counts.values,
    'MEAN MILES': day_max_miles.values,
    'MEAN ETA': month_max_eta.values
})

df = df.sort_values('DAYS', key=lambda x: pd.Categorical(x, categories=day_label.values(), ordered=True))


fig_10 = px.line(df, x='DAYS', y=['TOTAL COUNT', 'MEAN MILES','MEAN ETA'],
              title="Distribution of Day's Total Count,Mean Coount and Maen ETA")
fig_10.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle', line=dict(width=2)),
                  line=dict(width=2))
fig_10.update_layout(
    xaxis=dict(title='Days', showgrid=True, showline=True, linecolor='black', mirror=True,
               categoryorder='array', categoryarray=list(day_label.values())),
    yaxis=dict(title='Count', showgrid=True, showline=True, linecolor='black', mirror=True),
    plot_bgcolor='rgba(240,240,240,0.8)',
    paper_bgcolor='rgba(240,240,240,0.8)',
    font=dict(family='Arial', size=12, color='black'),
    title=dict(x=0.5, xanchor='center', y=0.95, yanchor='top'),
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.5)'),
    template='plotly_dark',
    margin=dict(l=50, r=50, t=50, b=50)
)

fig_11 = px.box(data[data['MILES']<50], y=['MILES'], title='Box Plot of Less than 50 Miles')
fig_11.update_traces(marker=dict(color='royalblue'))
fig_11.update_layout(
    xaxis=dict(title='Miles'),
    yaxis=dict(title='Values'),
    plot_bgcolor='rgba(240,240,240,0.8)',
    paper_bgcolor='rgba(240,240,240,0.8)',
    font=dict(family='Arial', size=12, color='black'),
    title=dict(x=0.5, xanchor='center', y=0.95, yanchor='top'),
    margin=dict(l=50, r=50, t=50, b=50)
)

coefficients = [3.0945, 0.0014, 0.0231, 0.8587, 0.5852, 0.0062, 0.4282, 0.2620]
labels = ['CATEGORY', 'START', 'STOP', 'MILES', 'PURPOSE', "DATE", "TIME", "SHIFT"]

total = sum(coefficients)
percentages = [(coef / total) * 100 for coef in coefficients]
data_011 = {'labels': labels, 'percentages': percentages}
df = pd.DataFrame(data_011)

# Plot pie chart
fig_12 = px.pie(
    df,
    names='labels',  # Column with the names
    values='percentages',  # Column with the values
    title='Parameter Distribution',
    color='labels',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    hole=0.4
)
fig_12.update_layout(
    title_text='Parameter Distribution',
    title_font_size=24,
    template='plotly_dark',
    legend_title='Category',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)


def cool_title(title_text):
    st.markdown(
        f"""
        <div style="background-color:#f63366;padding:10px;border-radius:10px;font-family: 'Times New Roman', Times, serif;">
        <h1 style="color:white;text-align:center;">{title_text}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )



def main():
    cool_title("Uber Estimated Time of Arrival")
    #st.header("Uber Data Analysis")
    selected = option_menu(menu_title = "Menu",
                options = ["Analysis","Prediction","Parameter Estimation"],
                #icons = ["File text fill","File text fill"],
                menu_icon = "Emoji sunglasses fill",
                default_index = 1,
                                orientation = "horizontal"
                )

    if selected == "Prediction":
        #st.write(data_03)
        # Prediction Section
        st.subheader('Predict ETA')

        # Input fields for user to enter new trip details
        category = st.selectbox('Category', data_03['CATEGORY'].unique())
        start_location = st.selectbox('Start Location', data_03['START'].unique())
        stop_location = st.selectbox('Stop Location', data_03['STOP'].unique())
        miles = st.number_input('Miles', min_value=0.1, value=1.0)
        purpose = st.selectbox('Purpose', data_03['PURPOSE'].unique())
        date = st.date_input('Start Date')
        time = st.time_input('Start Time')
        shift = st.selectbox('Shift', data_03['SHIFT'].unique())


        if st.button('Predict'):
            # Encode the new input data
            new_data = pd.DataFrame([[category, start_location, stop_location, miles,purpose,date,time,shift]],
                            columns=['CATEGORY', 'START', 'STOP', 'MILES','PURPOSE','DATE','TIME','SHIFT'])
            #new_data = encode_data(new_data)
            #X_train, X_test, y_train, y_test = split_data(data_encoded)
            #X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
            new_data_encoded = encode_data2(new_data)

            # Load the model
            model_path = "deep_eta_model.h5"
            model = load_model(model_path)

            # Make prediction
            prediction = make_prediction(model, new_data_encoded)
            st.write(f'Estimated Time of Arrival (ETA): {prediction[0][0]:.2f} minutes')



    # Second tab - Hindi translation
    if selected == "Analysis":

        st.subheader('Data Visualization')

        st.markdown('''***:orange[Time Series Distribution of Month's Total Ride Count ,Mean Mile Count and Maen ETA Time].***''')
        st.plotly_chart(fig_09)
        st.markdown("***Statistical Inference***.")
        st.write("""
            1. **Total Count**:
            - The total count of activities is highest on Fridays, peaking at around 85.
            - The lowest total count is on Wednesdays, with a value slightly above 50.
            - The overall trend shows an increase in total count towards the end of the week, with a significant peak on Friday, and a slight decline over the weekend.

            2. **Mean Miles**:
            - The mean miles remain relatively stable throughout the week, fluctuating between approximately 10 to 15.
            - There are slight increases on Wednesdays and Sundays, but the changes are not substantial.

            3. **Mean ETA**:
            - The mean ETA (Estimated Time of Arrival) is also relatively stable, varying between approximately 15 to 20.
            - There is a minor peak on Fridays, coinciding with the peak in total count, suggesting that the increased number of activities on Fridays might slightly affect the mean ETA.
            """)

        st.markdown('''***:orange[Time Series Distribution of Day's Total Ride Count ,Mean Mile Count and Maen ETA Time].***''')
        st.plotly_chart(fig_10)
        st.markdown("***Statistical Inference***.")
        st.write("""
            1. **Total Count**:
            - The total count of activities varies significantly across months.
            - The highest total count is observed in October, peaking at around 65.
            - The lowest total counts are seen in January and September, both with values close to 20.
            - There is a noticeable peak in April and a smaller peak in July, with significant drops in the subsequent months.

            2. **Mean Miles**:
            - The mean miles exhibit fluctuations throughout the year, with the highest value in May at around 30.
            - The lowest mean miles are observed in January and September, both close to 10.
            - There are minor peaks in March and November, indicating some variability in travel distances across the months.

            3. **Mean ETA**:
            - The mean ETA shows notable peaks in April and May, aligning with the high mean miles in May.
            - The lowest mean ETA is observed in February, closely followed by September.
            - The overall trend indicates that certain months with higher total counts and mean miles tend to have higher mean ETA as well.

            """)


        st.divider()

        st.markdown('''***:red[Distribution of Category Attribut].***''')

        col1, col2 = st.columns([1,1])

        with col1:
            st.plotly_chart(fig_01)
            st.write("""
            The "Business" category has a significantly higher count compared to the "Personal" category. The exact values are:\n
                • Business: 400\n
                • Personal: 13\n
            The "Business" category is far more prevalent, with a count that is visually much larger than the "Personal" category
            """)
        with col2:
            st.plotly_chart(fig_02)
            st.write("""
            The "Business" category comprises 96.9% of the data, while the "Personal" category accounts for 3.1% of the data.
            The distribution is highly skewed towards the "Business" category, indicating a lack of balance between the two categories.
            """)
        st.divider()


        st.markdown('''***:red[Distribution of Purpose Attribute].***''')
        col1, col2 = st.columns([1,1])

        with col1:
            st.plotly_chart(fig_03)
            st.write("""
            The category "Unknown" has the highest count, indicating that the purpose of a significant portion of trips is unidentified.
            The categories "Meeting" and "Meal/Entertainment" follow closely behind "Unknown" in terms of frequency,
            categories like "Customer Visit," "Errand/Supplies," "Temporary Site," and "Between Offices" have considerably lower counts.
            """)
        with col2:
            st.plotly_chart(fig_04)
            st.write("""
            Unknown" is the most frequent purpose, accounting for 36.5% of all trips. This suggests a significant portion of trip data lacks detailed purpose information.
            "Meeting" and "Meal/Entertainment" follow closely behind "Unknown," together constituting over 50% of the total trips.
            Also the remaining categories, including "Customer Visit," "Errand/Supplies," "Temporary Site," and "Between Offices," have relatively smaller proportions, indicating they are less frequent trip purposes.
            """)
        st.divider()


        st.markdown('''***:red[Distribution of Shift Attribute].***''')
        col1, col2 = st.columns([1,1])

        with col1:
            st.plotly_chart(fig_05)
            st.write("""
            The "Afternoon" shift has a significantly higher count compared to the other shift's. The exact values are:\n
                • Morning: 63\n
                • Afternoon: 142\n
                • Evening: 137\n
                • Night: 71\n
            The "Afternoon" and "Evening" shifts have significantly higher counts compared to the "Night" and "Morning" shifts.
            """)
        with col2:
            st.plotly_chart(fig_06)
            st.write("""
            The distribution of shift is relatively balanced, with no shift dominating significantly,but the "Afternoon" and "Evening" shifts have slightly higher percentages compared to the "Night" and "Morning" shifts.\n
                • Morning: 15.3%\n
                • Afternoon: 33.2%\n
                • Evening: 34.4%\n
                • Night: 17.2%\n
            """)
        st.divider()

        st.markdown('''***:red[Purpose Allocation by Category].***''')
        st.plotly_chart(fig_07)
        st.markdown("***Statistical Inference of Correlation Matrix Heatmap Interpretation***.")
        st.write("""
            1. **Unknown Purpose**:- Largest category (~160), mostly business.
            2. **Meeting**:- Significant activities (~80), predominantly business.
            3. **Meal/Entertain**:- High count (~60), mainly business.
            4. **Customer Visit**:- Frequent (~50), primarily business.
            5. **Errand/Supplies**:- Moderate count (~40), mainly business.
            6. **Between Offices and Temporary Site**:- Lowest counts (<20), primarily business.

            ***General Observations***
            - **Business vs. Personal**:- Majority are business activities.- Personal activities are rare and outnumbered.

            - *Activity Purposes*:- Most common purposes: "Unknown," "Meeting," "Meal/Entertain."- Business activities well-documented; many remain unspecified under "Unknown."

            """)
        st.divider()

        st.markdown('''***:red[Correlation Plot].***''')
        st.plotly_chart(fig_08)
        st.markdown("***Statistical Inference of Correlation Matrix Heatmap***.")
        st.write("""
            1. **Strong Correlations**:
            - *Miles and Time (0.75)*: There is a strong positive correlation between miles traveled and time taken, which is expected as longer distances typically require more time.
            - *Business and Meeting (0.23)*: Business activities have a moderate positive correlation with meetings.
            - *Business and Unknown Purpose (0.23)*: There is a moderate positive correlation between business activities and activities with an unknown purpose.
            2. **Moderate Correlations**:
            - *ETA and Time (0.22)*: There is a moderate positive correlation between ETA and time, suggesting that longer times might slightly influence ETA.
            - *Miles and Business (0.16)*: There is a moderate positive correlation between miles traveled and business activities.
            3. **Weak or Negligible Correlations**:
            - Most other variables show weak or negligible correlations with each other, indicating that there are no strong relationships among these variables in the dataset.
            """)

        st.divider()

        st.markdown('''***:red[Box Plot of Less than 50 Miles].***''')
        st.plotly_chart(fig_11)
        st.markdown("***Statistical Inference of Box Plot***.")
        st.write("""
            1. The data is skewed to the right, as indicated by the longer whisker and outliers extending towards higher values.
            2. The median distance traveled is approximately 20 miles.
            3. The IQR, represented by the box's height, appears to be around 10 miles, suggesting that the middle 50% of the data is relatively compact.
            """)
        st.divider()

    if selected == "Parameter Estimation":
        st.write(summary)
        st.markdown("""
            :blue-background[R-squared] : :red[0.571] :red[Approximately 57.1% of the total variation in ETA is accounted for by the model. This suggests a moderate fit, indicating that the model explains more than half of the variance in the dependent variable.]
            :blue-background[Adjusted R-squared] : :red[0.563] :red[An adjusted R-squared of 0.563 means that after adjusting for the number of predictors, about 56.3% of the variance in ETA is explained by the model.]



            :blue-background[Overall Model Significance]


            **Overall Model Hypothesis:**

            - ***Null Hypothesis (H0):*** The model is statistically not significant.<br>
                    **V/S**
            - ***Alternative Hypothesis (H1):*** The model is statistically significant.

            **F-statistic:** :red[67.32]
            **Prob (F-statistic):** :red[1.28e-69]

            ***Decision:*** The p-value is significantly less than 0.05, so we reject the null hypothesis.i.e., The model is statistically significant.

            :blue-background[Durbin-Watson]


            **Durbin-Watson Hypothesis:**

            - ***Null Hypothesis (H0):*** There is no first-order autocorrelation in the residuals of the regression model (i.e., the residuals are independent).<br>
                 **V/S**
            - ***Alternative Hypothesis (H1):*** There is first-order autocorrelation in the residuals of the regression model (i.e., the residuals are correlated).

            **Durbin-Watson statistic** : :red[2.382]

            ***Decision:*** The computed value is between the critical values (often close to 2), we do not reject the null hypothesis,i.e., there is no significant first-order autocorrelation in the residuals of the regression model(residual are independent).

            """,unsafe_allow_html=True)

        st.divider()
        st.markdown(''':red[Parameter Distribution].''')
        st.plotly_chart(fig_12)
        st.write("""
            The "CATEGORY" parameter is the most frequent, accounting for 58.8% of the explination of "ETA".
            Also the parameters like "MILES" and "PURPOSE" follow closely behind "CATEGORY," together constituting over 27% of the total explination.
            The remaining parameters, including "TIME," "SHIFT," "STOP," "DATE," and "START," have relatively smaller proportions, indicating they are weak regressor frequent for "ETA".
            """)
        st.divider()


if __name__ == "__main__":
    main()
