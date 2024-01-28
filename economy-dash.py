import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import ipywidgets as widgets
import folium
from folium.plugins import HeatMap
import streamlit_option_menu
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

file_c1 = "gdp_data.csv"
file_c2 = "gdp_capita_data.csv"
file_c3 = "gdp_growth_data.csv"
file_c4 = "inflation_data.csv"
file_c5_1 = "unemployment_data.csv"
file_c5_2 = "unemployment_fe_data.csv"
file_c5_3 = "unemployment_ma_data.csv"
file_c6 = "stocks_data.csv"
file_c7 = "stocks_pc_data.csv"

def stemGraph(country, file, ylabel, title):
    df = pd.read_csv(file, on_bad_lines='skip')

    CountriesSearch = df[df["Country Name"] == country]
    df3 = CountriesSearch.dropna(axis=1)  # columns

    list_labels = [str(i) for i in df3.columns if i.isnumeric()]
    list_values = [float(df3[i]) for i in list_labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot markers for stem points
    ax.plot(list_labels, list_values, 'bo', label='Stem')

    # Add lines connecting the stems
    for x, y in zip(list_labels, list_values):
        ax.plot([x, x], [0, y], 'b-', linewidth=1)

    # Set labels and title
    ax.set(xlabel='Years', ylabel=ylabel, title=title)
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

def stackplotGraph(country, file, ylabel, title):
    df = pd.read_csv(file, on_bad_lines='skip')

    CountriesSearch = df[df["Country Name"] == country]
    df3 = CountriesSearch.dropna(axis=1)  # columns

    list_labels = [str(i) for i in df3.columns if i.isnumeric()]
    list_values = [float(df3[i]) for i in list_labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the stackplot
    ax.stackplot(list_labels, list_values, labels=[ylabel])

    # Set labels and title
    ax.set(xlabel='Years', ylabel=ylabel, title=title)
    ax.legend()

    # Display the chart using Streamlit
    st.pyplot(fig)

def plotGraph(country, file, startYear, endYear, xlabel, ylabel, title):
    df = pd.read_csv(file, on_bad_lines='skip')

    CountriesSearch = df[df["Country Name"] == country]
    df3 = CountriesSearch.dropna(axis=1)  # columns

    list_labels = [int(i) for i in df3.columns if i.isnumeric()]  # Convert years to integers
    list_values = [float(df3[str(i)]) for i in list_labels]

    fig, ax = plt.subplots(figsize=(13, 10))

    # Plot the line chart
    ax.plot(list_labels, list_values, label=ylabel)

    # Set labels and title
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()

    # Set x-axis range
    ax.set_xlim(startYear, endYear)

    # Display the chart using Streamlit
    st.pyplot(fig)

def layout(country):
    c2, c3 = st.columns(2)
    c1, *_ = st.columns(1)
    c4, c5 = st.columns(2)
    c6, c7 = st.columns(2)

    with c1:
        forecast(file_c1, chosen_year,country)

    with c2:
        stackplotGraph(country, "gdp_capita_data.csv", "US$", "GDP - per capita")

    with c3:
        stemGraph(country, file_c3, "per cent", "GDP - growth")

    with c4:
        plotGraph(country, file_c4, 1991, 2022, "1991-2022", "rate", "Inflation")

    with c5:
        df = pd.read_csv("unemployment_data.csv", on_bad_lines='skip')

        countries = df["Country Name"]
        CountriesSearch = df[df["Country Name"] == country]
        df3 = CountriesSearch.dropna(axis=1)  # columns

        list_labels = [str(i) for i in df3.columns if i.isnumeric()]
        list_values = [float(df3[i]) for i in list_labels]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(list_labels, list_values)
        ax.set(xlabel='1991 - 2022', ylabel='Unemployment rate', title='Unemployment rates')

        # Display the chart using Streamlit
        st.pyplot(fig)

    with c6:
        stackplotGraph(country, file_c6, "billions US$", "Stocks Value")

    with c7:
        stemGraph(country, file_c7, "per cent", "Stocks % of GDP")

# Define a function to handle dropdown value changes
def on_dropdown_change(change):
    fig = plt.figure(figsize=(20, 10))
    selected_value = change['new']
    CountriesSearch = file_c1[file_c1["Country Name"] == selected_value]
    df3 = CountriesSearch.dropna(axis=1)  # this cleans the data
    list_labels = [str(i) for i in df3.columns if i.isnumeric()]
    list_values = [float(df3[i]) for i in list_labels]
    plt.plot(list_labels, list_values)
    plt.show()
    df3

def forecast(df, year,country):
    historical_data = pd.read_csv(file_c1, on_bad_lines='skip')
    historical_data_romania = historical_data[historical_data["Country Name"] == country].dropna(axis=1)

    historical_years = [int(col) for col in historical_data_romania.columns if col.isnumeric()]
    historical_values = [float(historical_data_romania[str(col)]) for col in historical_years]

    model = LinearRegression()
    model.fit(np.array(historical_years).reshape(-1, 1), historical_values)

    forecast_years = list(range(2022, year + 1))
    forecast_values = model.predict(np.array(forecast_years).reshape(-1, 1))

    sine_wave = 5 * np.sin(np.linspace(0, 4 * np.pi, len(forecast_values)))
    forecast_values_with_sine = forecast_values + sine_wave

    combined_years = historical_years + forecast_years
    combined_values = historical_values + forecast_values_with_sine.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(combined_years, combined_values, label='Historical', color='blue')
    ax.plot(forecast_years, forecast_values_with_sine, label='Forecast with Sine Wave', color='red')

    ax.set(xlabel='Year', ylabel='GDP', title=f'GDP Forecast (2022 - {year})')
    ax.legend()

    st.pyplot(fig)
    st.write("")

#with st.sidebar:
    # selected = option_menu(
    #     menu_title="Economic Indicators Dashboard",
    #     options=["Romania", "Other Countries", "RO Inflation Forecast", "Help"],
    #     menu_icon="cast",
    #     default_index=0,
    # )
# Add this code to directly select the country in the main content
countries =  pd.read_csv(file_c4, on_bad_lines='skip')
st.sidebar.subheader("Choose a country to see its economic indicators")
selected_country = st.sidebar.selectbox("", countries["Country Name"].unique())
st.sidebar.subheader(" ")
st.sidebar.subheader("Choose a year to see the GDP forecast")
chosen_year = st.sidebar.number_input("", value=2022, step=1)
st.sidebar.subheader("Each chart you can make fullscreen by clicking on the arrows in its corner")

# Main content
st.header(f'Economic Indicators Dashboard for {selected_country}')

layout(selected_country)
# if selected_country == "Albania":
#     layout("Romania")
#     # st.write("For years from 1991 to 2022 you can find the GDP in billions of US$, the GDP per capita and GDP growth(%)")
#     # st.write("The inflation and unemployment rate")
#     # st.write("The total value of stocks in US$ and the stocks % of GDP")

# elif selected_country == "Other Countries":
#     list_of_countries = []
#     df = pd.read_csv("unemployment_data.csv", on_bad_lines='skip')
#     countries = df["Country Name"]

#     for i in range(len(countries)):
#         list_of_countries.append(countries[i])

#     selected_country_other = st.selectbox('Select a Country:', list_of_countries)
#     layout(selected_country_other)

# if selected == "RO Inflation Forecast":
#     st.subheader(f"**Choose a Year for the {selected}**")
#     chosen_year = st.number_input("", value=2022, step=1)
#     forecast(file_c4, chosen_year)

# if selected == "Help":
#     st.subheader(f"**Navigation instructions**")
#     st.write(
#         "This is a web application with a dashboard with economic indicators for Romania as home page, but also for other chosen country. You can also find a forecast of the Inflation in Romania, where you can choose the year till you want the forecast.")
