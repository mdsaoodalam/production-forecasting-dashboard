import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def scatterplot(dataset, x, y, hue=None, title="Scatter Plot"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=dataset, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)  
