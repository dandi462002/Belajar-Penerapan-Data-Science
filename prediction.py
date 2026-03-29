import pandas as pd
import joblib

# Load pipeline model
pipeline = joblib.load("model_attrition.pkl")

print("=== Input Data Karyawan ===")

# Input numerik
age = int(input("Age: "))
daily_rate = int(input("Daily Rate: "))
distance = int(input("Distance From Home: "))
hourly_rate = int(input("Hourly Rate: "))
monthly_income = int(input("Monthly Income: "))
monthly_rate = int(input("Monthly Rate: "))
num_companies = int(input("Num Companies Worked (0-9): "))
salary_hike = int(input("Percent Salary Hike: "))
total_years = int(input("Total Working Years: "))
training = int(input("Training Times Last Year (0-6): "))
years_company = int(input("Years At Company: "))
years_role = int(input("Years In Current Role: "))
years_promo = int(input("Years Since Last Promotion: "))
years_manager = int(input("Years With Current Manager: "))

education = int(input("Education (1-5): "))
env_sat = int(input("Environment Satisfaction (1-4): "))
job_inv = int(input("Job Involvement (1-4): "))
job_level = int(input("Job Level (1-5): "))
job_sat = int(input("Job Satisfaction (1-4): "))
performance = int(input("Performance Rating (1-4): "))
relation = int(input("Relationship Satisfaction (1-4): "))
stock = int(input("Stock Option Level (0-3): "))
worklife = int(input("Work Life Balance (1-4): "))

# Mapping pilihan
business_travel_map = {
    "1": "Non-Travel",
    "2": "Travel_Frequently",
    "3": "Travel_Rarely"
}

gender_map = {
    "1": "Male",
    "2": "Female"
}

marital_map = {
    "1": "Single",
    "2": "Married",
    "3": "Divorced"
}

overtime_map = {
    "1": "Yes",
    "2": "No"
}

job_role_map = {
    "1":"Healthcare Representative",
    "2":"Human Resources",
    "3":"Laboratory Technician",
    "4":"Manager",
    "5":"Manufacturing Director",
    "6":"Research Director",
    "7":"Research Scientist",
    "8":"Sales Executive",
    "9":"Sales Representative"
}
education_field_map ={
    "1":"Human Resources",
    "2":"Life Sciences",
    "3":"Marketing",
    "4":"Medical",
    "5":"Other",
    "6":"Technical Degree"
}
department_map = {
    "1":"Human Resources",
    "2":"Research & Development",
    "3":"Sales"
}

print("\n=== Input Kategori (pakai angka) ===")

# Input + konversi
bt_input = input("Business Travel (1=Non-Travel, 2=Travel_Frequently, 3=Travel_Rarely): ")
business_travel = business_travel_map.get(bt_input)

gender_input = input("Gender (1=Male, 2=Female): ")
gender = gender_map.get(gender_input)

marital_input = input("Marital Status (1=Single, 2=Married, 3=Divorced): ")
marital = marital_map.get(marital_input)

overtime_input = input("OverTime (1=Yes, 2=No): ")
overtime = overtime_map.get(overtime_input)

department_input = input("Department (1=Human Resources, 2=Research & Development,3=Sales) ")
department = department_map.get(department_input)

education_field_input = input("Education Field (1=Human Resources, 2=Life Sciences, 3=Marketing, =4Medical, 5=Other, 6=Technical Degree)")
education_field = education_field_map.get(education_field_input)

job_role_input = input("Job Role (1=Healthcare Representative, 2=Human Resources, 3=Laboratory Technician, 4=Manager, 5=Manufacturing Director, 6=Research Director, 7=Research Scientist, 8=Sales Executive, 9=Sales Representative)")
job_role = job_role_map.get(job_role_input)

# Gabungkan ke DataFrame
data = {
    'Age': [age],
    'DailyRate': [daily_rate],
    'DistanceFromHome': [distance],
    'HourlyRate': [hourly_rate],
    'MonthlyIncome': [monthly_income],
    'MonthlyRate': [monthly_rate],
    'NumCompaniesWorked': [num_companies],
    'PercentSalaryHike': [salary_hike],
    'TotalWorkingYears': [total_years],
    'TrainingTimesLastYear': [training],
    'YearsAtCompany': [years_company],
    'YearsInCurrentRole': [years_role],
    'YearsSinceLastPromotion': [years_promo],
    'YearsWithCurrManager': [years_manager],

    'Education': [education],
    'EnvironmentSatisfaction': [env_sat],
    'JobInvolvement': [job_inv],
    'JobLevel': [job_level],
    'JobSatisfaction': [job_sat],
    'PerformanceRating': [performance],
    'RelationshipSatisfaction': [relation],
    'StockOptionLevel': [stock],
    'WorkLifeBalance': [worklife],

    'BusinessTravel': [business_travel],
    'Department': [department],
    'EducationField': [education_field],
    'Gender': [gender],
    'JobRole': [job_role],
    'MaritalStatus': [marital],
    'OverTime': [overtime]
}

new_data = pd.DataFrame(data)

# Prediksi
prediction = pipeline.predict(new_data)[0]

print("\n=== HASIL ===")
if prediction == 1:
    print("⚠️ Karyawan berisiko resign.")
else:
    print("✅ Karyawan diprediksi tidak resign.")