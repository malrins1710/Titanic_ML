# Phân tích và dự báo khả năng sống sót của du khách trên tàu Titanic
## Introduction
Từ khi sự kiện thảm họa đắm tàu 🚢[RMS Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic) vào tháng 04 năm 1912, đã có rất nhiều phỏng đoán về nguyên nhân xảy ra đắm tàu, các cuộc khảo sát số người tử vong và sống sót là bao nhiêu?
Nhiệm vụ của bài toán là dự đoán khả năng sống sót dựa trên lí lịch của du khách đó.
## Exploratory Data Analysis (EDA)
### Description:
- **survived**: Khả năng sống sót của hành khách trên tàu 0 là die 1 là alive <br/>
- **pclass**: Hạng vé lên tàu 1 là Cao cấp, 2 là thương gia, 3 là vé thường <br/>
- **Name**: Tên hành khách lên chuyến tàu Titanic <br/>
- **Sex**: Giới tính của từng người male là nam, female là nữ <br/>
- **Age**: Độ tuổi của từng người (Dữ liệu số liên tục) <br/>
- **SibSp**: Số lượng anh chị em, vợ/ chồng đi chung <br/>
- **Parch**: Số lượng hành khách là Cha mẹ / Con cái đi cùng <br/>
- **Ticket**: Mã vé tàu <br/>
- **Fare**: Giá vé (Dữ liệu số rời rạc) <br/>
- **Cabin**: Mã toa tàu của từng hành khách trên tàu <br/>
- **Embarked**: Cảng đón hành khách S là Southampton (Anh), C là Cherbourg (Pháp), Q là Queenstown (Ireland) <br/>
### Các thuộc tính dữ liệu của từng biến (features):
Dữ liệu Category (Nominal - Ordinal): Name, Sex, Embarked, Pclass (ordinal), Survived
Dữ liệu Numeric (Continuouns - Discrete - TimeSeries): Age (continuouns), Fare (continuouns), SibSp, Parch
Dữ liệu Mix types: Ticket, Cabin
### Pie Chart
Đại diện cho tỷ lệ cho thuộc tính con của Pclass, Sex và Embarked
![image](https://github.com/user-attachments/assets/fd4883ae-f2be-452f-9577-d42e54f7c881)
### Bar Graphs
So sánh giữa biến phụ thuộc so với các biến độc lập khác
![image](https://github.com/user-attachments/assets/394989c1-8e38-449d-813d-15d32b474b8e)
![image](https://github.com/user-attachments/assets/35a9a01f-9c7a-4137-8988-95a9d7d04faf)
### Histograms
Đại diện cho mật độ phân bố của độ tuổi của hành khách và giá vé tàu <br/>
<div align="center">
  <a href="https://github.com/user-attachments/assets/2c0a17e9-aa0c-4862-9f01-b223e92ad2e5">
  <img height=300 src="https://github.com/user-attachments/assets/2c0a17e9-aa0c-4862-9f01-b223e92ad2e5" />
</a>
  <a href="https://github.com/user-attachments/assets/37be7c4a-1c27-4319-bacc-0c2b14b78499">
    <img height=300 src="https://github.com/user-attachments/assets/37be7c4a-1c27-4319-bacc-0c2b14b78499" />
  </a>
</div>

## Modeling and Evaluation
Sau các bước xử lý dữ liệu, chia dữ liệu theo phương pháp cross-validation ta được:
![image](https://github.com/user-attachments/assets/f076d850-7845-4020-86b3-cd048db279aa)

| Model  | Mean | Standard Deviation |
| ------------- | ------------- | ------------- |
| XGBClassifier | 0.862452 | 0.021383 |
| RandomForestClassifier | 0.855188 | 0.019878 |
| AdaBoostClassifier | 0.849675	| 0.026192 |
| DecisionTreeClassifier | 0.819658 | 0.033527 |
* **Lựa chọn mô hình RandomForestClassifier và ta có confusion matrix đại diện**
![image](https://github.com/user-attachments/assets/6c140cc9-b4d0-4a72-9189-aef90732366c)
![image](https://github.com/user-attachments/assets/8d8ce1b1-dab9-41ed-aa72-090f58edf7fa)

## Result
| Sample | PassengerId | Nhãn Dự Đoán | Nhãn Thực Tế |
| ------------- | ------------- | ------------- | ------------- |
| 0 |	892 |	0 |	0 |
| 1	| 893 |	0 |	1 |
| 2	| 894 |	0 |	0 |
| 3	| 895 |	0 |	0 |
| 4	| 896 |	1 |	1 |
| ... | ... | ... | ... |
| 416 |	1308 |	0 |	0 |
| 417 | 1309 |	1	| 0 |
418 mẫu
Trong đó có 99 mẫu **dự đoán sai**

[Triển khai mô hình sử dụng Streamlit](https://drive.google.com/file/d/1X3q9Ne4P7tJhKXRQ6Ypt0AYVMr34c4-F/view?usp=drive_link)
