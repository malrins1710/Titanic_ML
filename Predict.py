import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests

Embark = {
    "Southampton": {"Embarked_C": 0, "Embarked_Q": 0, "Embarked_S": 1},
    "Cherbourg": {"Embarked_C": 1, "Embarked_Q": 0, "Embarked_S": 0},
    "Queenstown": {"Embarked_C": 0, "Embarked_Q": 1, "Embarked_S": 0}
}

Family = {
    1: {"Family_Cate_Single": {"Family_Cate_Single": 1, "Family_Cate_Multiplayer": 0, "Family_Cate_SuperMultiplayer": 0}},
    **{i: {"Family_Cate_Multiplayer": {"Family_Cate_Single": 0, "Family_Cate_Multiplayer": 1, "Family_Cate_SuperMultiplayer": 0}}  for i in range(2, 4)},
    **{i: {"Family_Cate_SuperMultiplayer": {"Family_Cate_Single": 0, "Family_Cate_Multiplayer": 0, "Family_Cate_SuperMultiplayer": 1}} for i in range(4, 11)}
}

Sex = {
    "Male": 1,
    "Female": 0
}

Pclass = {
    "Hạng nhất": 1,
    "Hạng thương gia": 2,
    "Hạng thường": 3
}

Position = {
    "Mr": {"Title_Master": 0, "Title_Miss": 0, "Title_Mr": 1, "Title_Mrs": 0, "Title_Others": 0},
    "Mrs": {"Title_Master": 0, "Title_Miss": 0, "Title_Mr": 0, "Title_Mrs": 1, "Title_Others": 0},
    "Miss": {"Title_Master": 0, "Title_Miss": 1, "Title_Mr": 0, "Title_Mrs": 0, "Title_Others": 0},
    "Master": {"Title_Master": 1, "Title_Miss": 0, "Title_Mr": 0, "Title_Mrs": 0, "Title_Others": 0},
    "Other": {"Title_Master": 0, "Title_Miss": 0, "Title_Mr": 0, "Title_Mrs": 0, "Title_Others": 1}
}

def load_model():
    with open(r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Models\model_RF.pkl", 'rb') as f:
        data = pickle.load(f)
    return data

def create_dataframe_from_dict(family_dict):
    df = pd.DataFrame.from_dict(family_dict, orient='index')
    return df

def predict_from_model():
    st.title("DỰ ĐOÁN KHẢ NĂNG SỐNG SÓT CỦA HÀNH KHÁCH TRÊN TÀU TITANIC")
    
    # Embarked
    selected_embark = st.selectbox("Cảng đón", list(Embark.keys()))
    if selected_embark in Embark:
        embarked_values = Embark[selected_embark]
        embarked_df = pd.DataFrame(embarked_values, index= [0])
        st.write(embarked_df)
    
    # SibSp, Parch
    selected_number = st.number_input("Nhập số từ 1 đến 11: ", min_value= 1, max_value= 11, value= 1, step= 1)
    if selected_number in Family:
        family_values = Family[selected_number]
        family_df = pd.DataFrame.from_dict(family_values, orient= 'index')
        family_df.index = [0]
        st.write(family_df)
    else:
        st.write("Số không hợp lệ")
    
    # Sex
    selected_sex = st.selectbox("Giới tính", Sex)
    if selected_sex in Sex.keys():
        sex_values = Sex[selected_sex]
        sex_df = pd.DataFrame({"Sex": [sex_values]})
        st.write(sex_df)
    
    # Pclass
    selected_Pclass = st.selectbox("Hạng vé tàu", Pclass)
    if selected_Pclass in Pclass:
        pclass_values = Pclass[selected_Pclass]
        pclass_df = pd.DataFrame({"Pclass": [pclass_values]})
        st.write(pclass_df)
    
    # Fare
    selected_Fare = st.number_input("Nhập giá vé của hành khách", min_value= 8, max_value= 500, value= 8, step= 1)
    fare_df = pd.DataFrame({"Fare": [selected_Fare]})
    with open(r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Models\Scaler\farescale.pkl", 'rb') as f:
        fare_scaler = pickle.load(f)
    
    fare_scaled = fare_scaler.transform([[selected_Fare]])
    fare_scaled = fare_scaled.ravel()
    scaled_fare_df = pd.DataFrame({"Fare_scaled" : fare_scaled})
    fare_df = pd.concat([fare_df, scaled_fare_df], axis= 1)
    st.write(fare_df)
    
    # Age
    age_input_type = st.radio("Phương thức nhập thuộc tính Age", ("Thanh kéo", " Nhập số"))
    if age_input_type == "Thanh kéo":
        selected_Age = st.slider("Độ tuổi của hành khách", 0., 100.)
    else:
        selected_Age = st.number_input("Nhập độ tuổi của hành khách đó: ", min_value=0., max_value= 100., value= 1., step= 1.)
    
    with open(r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Models\Scaler\agescale.pkl", 'rb') as f:
        age_scaler = pickle.load(f)
        
    age_scaled = age_scaler.transform([[selected_Age]])
    age_scaled = age_scaled.ravel()
    scaled_age_df = pd.DataFrame({"Age_scaled": age_scaled})
    age_df = pd.DataFrame({"Age": [selected_Age]})
    age_df = pd.concat([age_df, scaled_age_df], axis= 1)
    st.write(age_df)
    
    # Title
    selected_Position = st.selectbox("Chức vụ", Position)
    position_values = Position[selected_Position]
    title_df = pd.DataFrame(position_values, index= [0])
    st.write(title_df)
    
    # Gộp các dataFrame lại với nhau tạo thành 1 dataFrame gốc
    df = []
    values = [title_df, sex_df, pclass_df, family_df, embarked_df, scaled_age_df, scaled_fare_df]
    for value in values:
        df.append(value)
    
    st.subheader("Tổng hợp dữ liệu")
    merged_df = pd.concat(df, axis=1)
    original_columns = ["Title_Mr", "Title_Mrs", "Title_Miss", "Title_Master", "Title_Others", "Sex", "Pclass", "Family_Cate_Single", "Family_Cate_Multiplayer", "Family_Cate_SuperMultiplayer", "Embarked_Q", "Embarked_S", "Embarked_C", "Age_scaled", "Fare_scaled"]
    merged_df =merged_df.reindex(columns= original_columns)
    st.write(merged_df)
    
    # tạo button để dự đoán
    predict = st.button("Dự đoán")

    if predict:
        X = merged_df.values
        st.write(X)
        print(X)
        data = load_model()
        model = data["model"]
        prediction = model.predict(X)
        if prediction == 1:
            st.subheader(f"Khả năng là Sống sót!")
        else:
            st.subheader("Khả năng là Chết!")

        
    
        
    
    