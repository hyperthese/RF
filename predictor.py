import os
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型路径
model_path = os.path.join(current_dir, "rf.pkl")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

#加载模型
model = joblib.load('rf.pkl')

#定义选项
H1_options = {
    0:'None(0)',
    1:'Mild(1)',
    2:'Moderate(2)',
    3:'Severe(3)',
    4:'Extremely severs(4)'
}
H2_options = {
    0:'None(0)',
    1:'Mild(1)',
    2:'Moderate(2)',
    3:'Severe(3)',
    4:'Extremely severs(4)'
}
H6_options = {
    0:'None(0)',
    1:'Mild(1)',
    2:'Moderate(2)',
    3:'Severe(3)',
    4:'Extremely severs(4)'
}
B3_options = {
    0:'None(0)',
    1:'Mild, not very distressing(1)',
    2:'Moderate, uncomfortable but still tolerable(2)',
    3:'Severe, barely tolerable(3)'
}
B5_options = {
    0:'None(0)',
    1:'Mild, not very distressing(1)',
    2:'Moderate, uncomfortable but still tolerable(2)',
    3:'Severe, barely tolerable(3)'
}
G4_options = {
    0:'Not at all(0)',
    1:'Seversal days(1)',
    2:'More than half the days(2)',
    3:'nearly every day(3)'
}
G5_options = {
    0:'Not at all(0)',
    1:'Seversal days(1)',
    2:'More than half the days(2)',
    3:'nearly every day(3)'
}
G5_options = {
    0:'Not at all(0)',
    1:'Seversal days(1)',
    2:'More than half the days(2)',
    3:'nearly every day(3)'
}
P1_options = {
    0:'Not at all(0)',
    1:'Seversal days(1)',
    2:'More than a week (2)',
    3:'nearly every day(3)'
}
P3_options = {
    0:'Not at all(0)',
    1:'Seversal days(1)',
    2:'More than a week (2)',
    3:'nearly every day(3)'
}

#定义变量名
feature_names = ['H1','H2','H6','B3','B5','G4','G5','P1','P3','CPSI_LQ']

#streanlit 用户界面
st.title("Anxiety Disorder Predictor")

CPSI_LQ = st.number_input("The impact of the above symptoms on quality of life: ", min_value = 0, max_value = 12)

H1 = st.selectbox("Feeling anxious or worried most of the time", options=list(H1_options.keys()), format_func=lambda x: H1_options[x])
H2 = st.selectbox("Feeling tense, restless, or unable to relax", options=list(H2_options.keys()), format_func=lambda x: H2_options[x])
H6 = st.selectbox("Losting of interest, lack of pleasure in hobbies", options=list(H6_options.keys()), format_func=lambda x: H6_options[x])
B3 = st.selectbox("Feeling involuntary tremors in the legs", options=list(B3_options.keys()), format_func=lambda x: B3_options[x])
B5 = st.selectbox("Afraid of something bad happening", options=list(B5_options.keys()), format_func=lambda x: B5_options[x])
G4 = st.selectbox("Finding it hard to relax", options=list(G4_options.keys()), format_func=lambda x: G4_options[x])
G5 = st.selectbox("Becoming wasily annoyed or irritable", options=list(G5_options.keys()), format_func=lambda x: G5_options[x])
P1 = st.selectbox("Little interest or pleasure in doing things", options=list(P1_options.keys()), format_func=lambda x: P1_options[x])
P3 = st.selectbox("Difficulty falling asleep, restless sleep, or excessive sleep", options=list(P3_options.keys()), format_func=lambda x: P3_options[x])

#进程输入和预测
feature_values = [H1,H2,H6,B3,B5,G4,G5,P1,P3,CPSI_LQ]
features=np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
# Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
# Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
           f"According to our model, you have a high risk of anxiety disorder."
           f"The model predicts that your probability of having anxiety disorder is {probability:.1f}%. "
           "While this is just an estimate, it suggests that you may be at significant risk. "
           "I recommend that you consult a psychologist as soon as possible for further evaluation and "
           "to ensure you receive an accurate diagnosis and necessary treatment."
)
    else:
        advice = (
           f"According to our model, you have a low risk of anxiety disorder. "
           f"The model predicts that your probability of not having anxiety disorder is {probability:.1f}%. "
           "However, maintaining a healthy lifestyle is still very important. "
           "I recommend regular check-ups to monitor your heart health, "
           "and to seek medical advice promptly if you experience any symptoms." )

    st.write(advice)
#SHAP解释
    st.subheader("SHAP Force Plot Explanation")
# Calculate SHAP values and display force plot
    explainer_shap = shap.TreeExplainer(model)
#计算SHAP值用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

#根据预测类别显示SHAP强制图
#期望值
#解释类别 1 的shap值
#特征值数据
#使用matplotlib函数绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_value[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib = True)
#解释类别为  0 的shap值
#特征值数据
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_value[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib = True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
