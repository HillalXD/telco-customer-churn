import streamlit as st
import pandas as pd
import keras


model = keras.models.load_model('annclass.h5')

def preprocessing(data):

    df = pd.DataFrame(data, index=[0])

    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
  
    for col in yes_no_columns:
        df[col].replace({'Yes': 1,'No': 0},inplace=True)

    df["gender"].replace({'Female':1, 'Male':0}, inplace=True)
    df["InternetService"] = pd.factorize(df['InternetService'])[0] + 1
    df["Contract"] = pd.factorize(df['Contract'])[0] + 1
    df["PaymentMethod"] = pd.factorize(df['PaymentMethod'])[0] + 1

    return df

def predict_churn(data):

    prepro = preprocessing(data)
    prediction = model.predict(prepro)
    return prediction

def main():
    st.title("Internet Customer Churn Prediction")
    st.write("Enter the customer details to predict churn:")
    
    # Create input fields for customer details
    inputs = {
    'gender': st.selectbox("Select gender", ['Female', 'Male']),
    'SeniorCitizen': st.selectbox("Select SC", [0, 1]),
    'Partner': st.selectbox("partner status", ["Yes", "No"]),
    'Dependents': st.selectbox("Dependents", ["Yes", "No"]),
    'tenure': st.number_input("Tenure"),
    'PhoneService': st.selectbox("Phone service", ["Yes", "No"]),
    'MultipleLines': st.selectbox("multiple lines", ["Yes", "No"]),
    'InternetService': st.selectbox("Internet Service", ["DSL", "Fiber optic","No"]),
    'OnlineSecurity': st.selectbox("Online security", ["Yes", "No"]),
    'OnlineBackup': st.selectbox("Online backup", ["Yes", "No"]),
    'DeviceProtection': st.selectbox("Device protection", ["Yes", "No"]),
    'TechSupport': st.selectbox("Tech support", ["Yes", "No"]),
    'StreamingTV': st.selectbox("TV Stream", ["Yes", "No"]),
    'StreamingMovies': st.selectbox("Movies Stream", ["Yes", "No"]),
    'Contract': st.selectbox("Contract length", ["Month-to-month", "One year", "Two year"]),
    'PaperlessBilling': st.selectbox("Paperless billing", ["Yes", "No"]),
    'PaymentMethod': st.selectbox("Payment method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                    'Credit card (automatic)']),
    'MonthlyCharges': st.number_input("Monthly Charges")
    }
    
    

    # Create a dataframe with the customer details

    
    if st.button("Predict"):
        prediction = predict_churn(inputs)
        churn_status = 'Churn' if prediction[0] > .5 else 'Not Churn'
        st.success("Churn Prediction: {}".format(churn_status))

if __name__ == "__main__":
    main()
