import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 해당 코드를 통해 matplotlib로 생성된 그림을 표시할 수 있다
%matplotlib inline
sns.set_style("white", {'ytick.major.size' : 10.0}) # seaborn으로 생성된 그림의 배경을 흰 배경으로 설정하며 눈금의 길이를 10으로 설정한다
sns.set_context("poster", font_scale=0.5) # 폰트의 크기를 조절하는 코드로 paper, notebook, talk, poster 종류로 구성된다

# 파일 encoding 형식 파악 
pip install chardet
import chardet

with open('/content/train_users_2.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

# EDA 고려사항
## 데이터에 잘못된 부분이 있나요?(결측치 등)
## 특이한 데이터가 있나요? (NaN 데이터가 아닌 다른 이상한 데이터는 NaN로 교체해야한다)
## 데이터를 수정하거나 삭제해야 하나요? (필요한 특징만 추출)
## 날짜 데이터가 datetime 타입인가요?

# 1. 데이터 가져오기
## 'utf-8' codec can't decode byte 0x8a in position 12: invalid start byte 에러 메시지가 나오면 encoding = "cp494" 또는 ignore를 한다
## Error tokenizing data. C error: Expected 1 fields in line 5, saw 3 에러 메시지가 나오면 sep = "|" 라는 구분자를 추가한다
train_users = pd.read_csv("/content/train_users_2.csv")
test_users = pd.read_csv("/content/test_users.csv")

# 2. 가져온 데이터 살펴보기
train_users.head(5)

# 2. 가져온 데이터 설명
## shape[0]은 행의 수, shape[1]은 열의 수
print("We have", train_users.shape[0],"users in the training set and", test_users.shape[0], "in the test set.")
print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")


users = pd.concat((train_users, test_users), axis = 0, ignore_index = True) # 행 결합
users.drop('id', axis = 1, inplace = True) # id 특성 제거
users.head(5)


# 데이터에 결측치(NaN)가 있는 걸 확인하였다

# gender 열에 unknown이 있으므로 이를 NaN로 교체해야 한다 
users.gender.replace('-unknown-', np.nan, inplace = True)


# 각 특징별 NaN 비율 파악 

users_nan = (users.isnull().sum()/users.shape[0]) * 100
users_nan[users_nan >0].drop('country_destination')


# age와 gender에서 많은 결측치를 가지고 있어 분류모델의 성능 하락의 원인이 될것이다

print("Just for the sake of curiosity; we have", int((train_users.date_first_booking.isnull().sum() / train_users.shape[0]) * 100)
, "% of missing values at date_first_booking in the training data")

# age 데이터를 면밀히 보자
users.age.describe()

# age 데이터에 이상값이 존재한다 
print(sum(users.age > 122)) # 122세 이상?
print(sum(users.age < 18)) # 18세 이하 

users[users.age > 122]['age'].describe()

# 이상치를 처리하기 위해 특정 범위 이외 데이터는 NaN로 처리한다 
users.loc[users.age>95, 'age'] = np.nan # loc[특정행, 열] => 나이가 95보다 많고, age열을 가진 데이터를 NaN으로 교체 
users.loc[users.age <13, 'age'] = np.nan



# 중요한 특징 카테고리화 

categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'                    
]

# 해당 특징의 데이터 타입을 category로 변환
for categorical_feature in categorical_features:
  users[categorical_feature] = users[categorical_feature].astype('category')


# 날짜 데이터를 datetime 타입으로 변환
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format = '%Y%m%d')

# 시각화

users.gender.value_counts(dropna = False).plot(kind='bar', rot = 0) # rot = 0는 x축 값 글자 회전, dropna = False를 통해 결측치도 포함 
plt.xlabel('Gender')
sns.despine() # 상단과 오른쪽의 plot 선이 지워진다


# 성별에 따른 선호도 파악

women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

# value_counts()는 NaN 미포함, 각 값별로 나온 횟수 표현
female_destinations = (users.loc[users['gender']== 'FEMALE', 'country_destination'].value_counts() / women) * 100 
male_destinations = (users.loc[users['gender']== 'MALE', 'country_destination'].value_counts() / men) * 100 


# 시각화2 

# position을 통해 두 막대그래프를 한눈에 파악할 수 있다  
male_destinations.plot(kind='bar', width = 0.4, label = 'Male', rot =0, position = 0 ) 
female_destinations.plot(kind = 'bar', width = 0.4, label = 'Female', rot = 0, color ='#FFA35D', position = 1 )
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()

# 성별 차이는 없어 보인다 

destination_percentage = (users.country_destination.value_counts() / users.shape[0]) * 100
destination_percentage.plot(kind = 'bar', rot = 0)
plt.xlabel("Destination Country")
plt.ylabel("percentage")

# 예약을 하지 않은 사람은 45% 이상이고, 미국에서 예약을 많이 하는 것이 보인다 

# 모국어를 사용하는 나라로 여행가는 편이지 않을까?
print((sum(users.language == 'en') / users.shape[0])* 100)


sns.distplot(users.age.dropna(False), color = "#FD5C64")
plt.xlabel('Age')
sns.despine()

# 25세에서 40세 사이가 주요 고객이다 

# 젊은 집단과 늙은 집단별로 예약한 나라
younger = sum(users.loc[users['age'] < 45, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > 45, 'country_destination'].value_counts())

younger_destinations = ((users.loc[users['age'] < 45, 'country_destination'].value_counts()) / younger ) * 100
older_destinations = ((users.loc[users['age'] > 45, 'country_destination'].value_counts()) / older) * 100

younger_destinations.plot(kind = 'bar', width =0.4, color = '#63EA55', position = 0, label = 'Youngers', rot = 0)
older_destinations.plot(kind = 'bar', width = 0.4, color = '#4DD3C9', position = 1, label = 'Olders', rot = 0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()

# 시간별로 생성된 유저 계정 수 
sns.set_style("whitegrid", {'axes.edgecolor' : '0'})
sns.set_context("poster", font_scale = 0.5)
users.date_account_created.value_counts().plot(kind = 'line', linewidth = 1.2, color = "#FD5C64")

# 유저가 처음으로 활동한 날짜와 상관관계가 있을까?
users.date_first_active.value_counts().plot(kind='line', linewidth = 1.2, color = '#FD5C64')


# 급격히 상승한 시점인 2013년과 2014년의 데이터를 살펴보자 

users_2013 = users[users['date_first_active'] > pd.to_datetime(20130101, format = "%Y%m%d")]
users_2013 = users_2013[users_2013['date_first_active'] < pd.to_datetime(20140101, format = "%Y%m%d")]
users_2013.date_first_active.value_counts().plot(kind='line', linewidth = 2, color = '#FD5C64')
plt.show()

# 일자별로 예약횟수를 분석해보자 
weekdays = []
for date in users.date_account_created:
  weekdays.append(date.weekday()) # weekday 함수는 월 0, 화 1  순으로 숫자로 표현
weekdays = pd.Series(weekdays)

sns.barplot(x= weekdays.value_counts().index, y = weekdays.value_counts().values)
plt.xlabel('Week day')
sns.despine()

# 2014년 이전과 이후의 예약 이용률은 어떻게 되었을까? 
date = pd.to_datetime(20140101, format = "%Y%m%d")

before = sum(users.loc[users['date_first_active'] < date, 'country_destination']. value_counts())
after = sum(users.loc[users['date_first_active'] > date, 'country_destination'].value_counts())
before_destinations = (users.loc[users['date_first_active'] < date, 'country_destination'].value_counts() / before)*100
after_destinations = (users.loc[users['date_first_active'] >date, 'country_destination'].value_counts()/ after) * 100
before_destinations.plot(kind = 'bar', width =0.4 ,color = '#63EA55', position = 0, label = 'Before 2014', rot = 0)
after_destinations.plot(kind='bar', width = 0.4, color = '#4DD3C9', position =1, label = 'After 2014', rot = 0 )
plt.legend()
sns.despine()
plt.show()

# 2014년 이후에 가입자 수는 늘었지만 예약율은 줄었다 => 서비스 개선이 필요하다 

