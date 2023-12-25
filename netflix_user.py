# 플랫폼 사용자와 관련된 주요 행동과 패턴 식별 
# 인구통계학적 분석, 기기 사용 분석, 구독 습관 분석 

import pandas as pd 
data = pd.read_csv("Netflix Userbase.csv")
data.head(5)

# 지도 시각화 
import plotly.express as px 

# User ID 특징은 count 함수로 집계, Monthly Revenue 특징은 sum 함수로 집계한다 
country_data = data.groupby('Country').agg({'User ID' : 'count', 'Monthly Revenue' : 'sum'}).reset_index()  
country_data.head(5)

# locationmode 속성의 country names 중 locations 속성의 나라를 일치시켜 시각화시킬수 있다 
fig1 = px.choropleth(country_data, locations = 'Country', locationmode= 'country names', color = 'User ID'
, title = "Number of Netflix Users by Country", hover_name = 'Country', color_continuous_scale = 'Plasma')
fig1.show()

# hover_name 속성으로 지도를 선택했을 때 나라명이 뜨게 한다 
fig2 = px.choropleth(country_data, locations = 'Country', locationmode = 'country names', color = 'Monthly Revenue',
                     title = 'Total Netflix Revenue by Country',hover_name='Country', color_continuous_scale = 'Plasma')
fig2.show()

# 넷플릭스의 유저는 누구인가?

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() # 그림 기본테마 설정
figs, axes = plt.subplots(3, 1, figsize= (10, 15)) #3행 1열 그림판 생성 

# ax 속성으로 그림 순서를 정한다 
sns.histplot(data = data, x="Age",color = 'skyblue', binwidth = 10, ax = axes[0])
axes[0].set_title("Age Distribution")

sns.countplot(data = data, x="Gender", palette = 'pastel', ax = axes[1])
axes[1].set_title("Gender Distribution")

sns.countplot(data= data, y = "Country", palette = 'pastel', ax = axes[2])
axes[2].set_title("Country Distribution")

plt.tight_layout()
plt.show()


# 넷플릭스의 주 연령대는 30~40대지만 비교적 다양한 연령층이 분포하고 있다 
# 넷플릭스의 성별 분포는 남,여가 비슷하게 보인다 
# 이용자가 가장 많은 국가는 스페인, 미국 ,캐나다 순이다 

# 넷플릭스 유저의 콘텐츠 소비 방식은?
plt.figure(figsize = (10, 6))

sns.countplot(data = data, y = 'Device', palette = 'pastel' )
plt.title('Device User Distribution')
plt.show()

# 넷플릭스 이용자의 상당수가 모바일 기기에서 콘텐츠 소비를 선호하며, 이는 사용자가 이동 중에도 콘텐츠를 소비할 수 있는
# 유연성과 편리성의 장점이 있기 때문으로 보인다 

# 유저의 구독 습관?
fig, axes = plt.subplots(2, 1, figsize = (10,10))

sns.countplot(data = data, x ="Subscription Type", palette = "pastel", ax = axes[0])
axes[0].set_title("Subscription Type Distribution")

sns.countplot(data= data, x= "Plan Duration", palette = "pastel", ax= axes[1])
axes[1].set_title("Plan Duration Distribution")

plt.tight_layout()
plt.show()

# 넷플릭스의 이용자 대부분은 월정액 요금제에 가입한 것으로 보인다

# 사용자로부터의 수익은?

fig, axes = plt.subplots(3, 1, figsize =(10, 15))

sns.boxplot(data = data, x = "Subscription Type", y= "Monthly Revenue", palette = "pastel", ax= axes[0])
axes[0].set_title("Revenue Distribution by Subscription Type")

sns.boxplot(data = data, x= "Monthly Revenue", y = "Country", palette = "pastel", ax= axes[1])
axes[1].set_title("Montly Revenue by Country")

sns.boxplot(data = data ,  x = "Device", y = "Monthly Revenue", palette = "pastel", ax= axes[2])
axes[2].set_title("Monthly Revenue by Device")

plt.tight_layout()
plt.show()

# 이탈률 파악 
from datetime import datetime

data['Join Date'] = pd.to_datetime(data['Join Date'], format = "%d-%m-%y")
data['Last Payment Date'] = pd.to_datetime(data['Last Payment Date'], format = "%d-%m-%y")


# 마지막 결제일과 가입일의 차이가 30일보다 작다면 이탈하였다고 판단한다(구독을 많이 할수록 마지막 결제일이 늦춰지기 때문이다)

data['Days Active'] = (data['Last Payment Date'] - data['Join Date']).dt.days

# True이면 1, False이면 0이므로 평균을 구하면 확률이 된다 (베르누이 분포의 평균은 확률이 된다)
churn_rate = (data['Days Active'] < 30).mean()
churn_rate
