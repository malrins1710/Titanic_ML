import streamlit as st
from Predict import predict_from_model
from EDA import Explore_Data_Analysis

def main():
    with st.sidebar:
        add_ratio = st.radio("FUNCTION (CHỨC NĂNG)", ("PHÂN TÍCH DỮ LIỆU", "DỰ ĐOÁN KHẢ NĂNG SỐNG SÓT"))
    
    if add_ratio == "PHÂN TÍCH DỮ LIỆU":
        Explore_Data_Analysis()
    else:
        predict_from_model()
    
if __name__ == "__main__":
    main()