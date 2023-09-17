import streamlit as st
import pickle
import pandas as pd
import numpy as np

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
    with open(r'C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Models\model_RF.pkl', 'rb') as f:
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
    st.write(fare_df)
    
    # Age
    age_input_type = st.radio("Phương thức nhập thuộc tính Age", ("Thanh kéo", " Nhập số"))
    if age_input_type == "Thanh kéo":
        selected_Age = st.slider("Độ tuổi của hành khách", 0., 100.)
    else:
        selected_Age = st.number_input("Nhập độ tuổi của hành khách đó: ", min_value=0., max_value= 100., value= 1., step= 1.)
    age_df = pd.DataFrame({"Age": [selected_Age]})
    st.write(age_df)
    
    # Title
    selected_Position = st.selectbox("Chức vụ", Position)
    if selected_Position in Position:
        position_values = Position[selected_Position]
        title_df = pd.DataFrame(position_values, index= [0])
        st.write(title_df)
    
    # tạo button để dự đoán
    predict = st.button("Dự đoán", key= 'centered-button')

    if predict:
        X = np.array([[embarked_df, family_df, title_df, pclass_df, sex_df]])
        print(X)
        pass
        
    
        
    
    