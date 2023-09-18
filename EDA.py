import streamlit as st
import pandas as pd
from PIL import Image


@st.cache_data
def load_data():
    url_1 = "https://raw.githubusercontent.com/malrins1710/Titanic_ML/main/Datasets/train.csv"
    df = pd.read_csv(url_1, index_col= "PassengerId")

    url_2 = "https://raw.githubusercontent.com/malrins1710/Titanic_ML/main/Datasets/New_Data/best_data.csv"
    train_df_preprocessed = pd.read_csv(url_2, index_col= "PassengerId")
    
    url_3 = "https://raw.githubusercontent.com/malrins1710/Titanic_ML/main/Models/best_model_kfold.csv"
    best_model_kfold = pd.read_csv(url_3)
    url_4 = "https://raw.githubusercontent.com/malrins1710/Titanic_ML/main/Models/best_model_std.csv"
    best_model_std = pd.read_csv(url_4)
    url_5 = "https://raw.githubusercontent.com/malrins1710/Titanic_ML/main/Models/best_score_model.csv"
    best_score_model = pd.read_csv(url_5)
    return df, train_df_preprocessed, best_model_kfold, best_model_std, best_score_model

@st.cache_resource
def load_images():
    EDA_1 = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\EDA_1.png"
    EDA_2 = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\EDA_2.png"
    EDA_3 = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\EDA_3.png"
    EDA_4 = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\EDA_4.png"
    EDA_5 = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\EDA_5.png"
    corrmap = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\CorrelationMap.png"
    corrm = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\Correlation Matrix.png"
    cfm = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\ConfusionMatrix.png"
    accmodel = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\Accuracy_model.png"
    fea_importances = r"C:\Users\trang\OneDrive\Máy tính\Đồ Án Môn\Chuyên Ngành Trí Tuệ Nhân Tạo\Báo cáo\Images\features_importances.png"
    return EDA_1, EDA_2, EDA_3, EDA_4, EDA_5, corrm, cfm, accmodel, corrmap, fea_importances
    
# Chuyển đổi dữ liệu để thực hiện download data
@st.cache_data
def convert_df(df):
    return df.to_csv(index= False).encode('utf-8')


def Explore_Data_Analysis():
    st.title("TRỰC QUAN HÓA DỮ LIỆU HÀNH KHÁCH TRÊN TÀU TITANIC")
    st.write("### <center>Mô tả thuộc tính bên trong dữ liệu</center>", unsafe_allow_html= True)
    # data = {
    #     'Survived' : "Khả năng sống sót của hành khách trên tàu ***0*** là die ***1*** là alive",
    #     'Pclass' : "Hạng vé lên tàu ***1*** là Cao cấp, ***2*** là thương gia, ***3*** là vé thường",
    #     'Name' : "Tên hành khách lên chuyến tàu Titanic",
    #     'Sex' : "Giới tính của từng người ***male*** là nam, ***female*** là nữ",
    #     'Age' : "Độ tuổi của từng người (Dữ liệu số liên tục)",
    #     'SibSp' : "Số lượng anh chị em, vợ/ chồng đi chung",
    #     'Parch' : "Số lượng hành khách là Cha mẹ / Con cái đi cùng",
    #     'Ticket' : "Mã vé tàu",
    #     'Fare' : "Giá vé (Dữ liệu số rời rạc)",
    #     'Cabin' : "Mã toa tàu của từng hành khách trên tàu",
    #     'Embarked' : "Cảng đón hành khách ***S*** là Southampton (Anh), ***C*** là Cherbourg (Pháp), ***Q*** là Queenstown (Ireland)",
    # }
    # for feature in data:
    #     st.write(f"**{feature}**", data[feature], unsafe_allow_html= True)
        
    data = {
        "Thuộc tính" : ["Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        "Mô tả" : ["Khả năng sống sót của hành khách trên tàu 0 là die 1 là alive",
                   "Hạng vé lên tàu 1 là Cao cấp, 2 là thương gia, 3 là vé thường",
                   "Tên hành khách lên chuyến tàu Titanic", "Giới tính của từng người male là nam, female là nữ",
                   "Độ tuổi của từng người (Dữ liệu số liên tục)", "Số lượng anh chị em, vợ/ chồng đi chung",
                   "Số lượng hành khách là Cha mẹ / Con cái đi cùng",
                   "Mã vé tàu", "Giá vé (Dữ liệu số rời rạc)", "Mã toa tàu của từng hành khách trên tàu",
                   "Cảng đón hành khách S là Southampton (Anh), C là Cherbourg (Pháp), Q là Queenstown (Ireland)"]
    }
    
    data = pd.DataFrame(data)
    st.dataframe(data, hide_index= True)
    
    # Load data + images
    df, train_df_processed, best_model_kfold, best_model_std, best_score_model = load_data()
    EDA_1, EDA_2, EDA_3, EDA_4, EDA_5, corrm, cfm, acc_model, corrmap, fea_importances = load_images()
    
    # Dữ liệu gốc
    st.write(f"### Dữ liệu gốc với ***{df.shape[1]}*** đặc trưng và ***{df.shape[0]}*** mẫu")
    cols_drop = st.multiselect("Drop features", df.columns)
    main_df = df.drop(cols_drop, axis= 1)
    st.write(main_df.head())
    
    # Nút download dữ liệu
    csv = convert_df(df)
    st.download_button(
        "Tải Database", csv,
        "train_df.csv",
        "text/csv",
        key= 'download-csv'
    )
    
    # Explore Data Analysis
    st.write("### <center> Explore Data Analysis </center>", unsafe_allow_html= True)
    st.write("#### Categories")
    eda_1 = Image.open(EDA_1)
    st.image(eda_1)
    eda_2 = Image.open(EDA_2)
    st.image(eda_2)
    eda_3 = Image.open(EDA_3)
    
    st.write("#### Numeric")
    st.image(eda_3)
    eda_4 = Image.open(EDA_4)
    st.image(eda_4, width= 720)
    eda_5 = Image.open(EDA_5)
    st.image(eda_5, width= 720)
    st.write("""
             **Lưu ý**: Đường ***màu đỏ*** đại diện cho chỉ số ***mode***, 
             màu ***xanh biển*** đại diện cho ***median***, 
             màu ***xanh lá cây*** đại diện cho ***mean***""")
    corrm = Image.open(corrm)
    st.image(corrm)
    st.write("**Lưu ý**: Mức quan hệ nằm trong khoảng **-1** đến **1**. Nếu mối tương quan dần về 0 tức là mối tương quan của hai đặc trưng đó không có quan hệ với nhau hoặc quan hệ kém.")
    corrmap = Image.open(corrmap)
    st.image(corrmap)
    
    st.write(f"### Dữ liệu sau khi đã được tiền xử lý với ***{train_df_processed.shape[1]}*** đặc trưng và ***{train_df_processed.shape[0]}*** mẫu")
    cols_drop_2 = st.multiselect("Drop features", train_df_processed.columns)
    df_processed = train_df_processed.drop(cols_drop_2, axis= 1)
    st.write(df_processed)
    
    # Nút download dữ liệu
    csv_2 = convert_df(df_processed)
    st.download_button(
        "Tải Preprocessed Data", csv_2,
        "best_data.csv",
        "text/csv",
        key= 'download_2-csv'
    )
    
    acc_model = Image.open(acc_model)
    st.image(acc_model)
    col1, col2 = st.columns(2)
    best_model_kfold = best_model_kfold.nlargest(10, 'accuracy_score')
    st.write(best_model_kfold.drop(best_model_kfold.columns[0], axis= 1))
    col1.dataframe(best_model_std, hide_index= True)
    col2.dataframe(best_score_model.drop(best_score_model.columns[0], axis= 1), hide_index= True)
    st.write("**Kết Luận**: Bài toán này sẽ lựa chọn **Random Forest** là mô hình tốt nhất sẽ được đem fine-tuning")
    cfm = Image.open(cfm)
    st.image(cfm)
    st.write("**Kết Luận**: Có 55 nhãn dự đoán đúng với nhãn 0 (Chết) và 44 nhãn dự đoán đúng với nhãn 1 (Sống sót)")
    fea_importances = Image.open(fea_importances)
    st.image(fea_importances)
    st.write("**Kết Luận**: Mô hình phụ thuộc vào 2 đặc trưng chính là **Title_Mr** và **Fare**")
    
    
    
    
    
    