from matplotlib import markers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import mplcyberpunk

list_fifas=[15,16,17,18,19,20,21]
list_files=[r"players_15.csv",r"players_16.csv",r"players_17.csv",r"players_18.csv",r"players_19.csv",r"players_20.csv",r"players_21.csv"]

def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

for l in range(len(list_files)):

    df=pd.read_csv(list_files[l])

    # top 5 leagues 
    df=df.loc[(df['league_name']=='Spain Primera Division')|(df['league_name']=='German 1. Bundesliga')|(df['league_name']=='French Ligue 1')|(df['league_name']=='English Premier League')|(df['league_name']=='Italian Serie A')]
    df=df.drop(columns=["sofifa_id","player_url","dob"])


    #tags-----------------------------------------------------------------------------------------
    df_tags= df.dropna(subset=['player_tags'])#removing any columns that have na in the column tags in this dataframe.
    df_tags=df_tags.drop(df_tags[df_tags["player_positions"]=='SUB'].index)
    df_tags=df_tags.drop(df_tags[df_tags["team_position"]=='SUB'].index)
    #traits-----------------------------------------------------------------------------------------------
    df_traits= df.dropna(subset=['player_traits']) #same thing with this dataframe
    #defenders---------------------------------------------------
    #the defenders' positions all contain "B", so i can create a df based on this.
    defenders_df=df_tags.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    defenders_df=defenders_df.reset_index()
    defenders_df=defenders_df.drop(columns=["index"])
    number_rows=len(defenders_df)

#these for loops are made to extract the unique tags after slicing the strings of each cell.

    defender_player_tags=[] #unique player tag
    for i in range(number_rows):#iterate on all players
        wow=defenders_df.iloc[i,22]
        list=wow.split(",")
        for list1 in list:#iterates on all player tags in each player
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("#","")
                    for i in defender_player_tags:#iterates on all unique tags
                        if(i==list1):#if unique player tag==one of the total break
                            check=False
                            break
                    if(check):
                        defender_player_tags.append(list1) #if none are equal, append


#these for loops  are made to look at how much each unique tag was repeated so i can take the top 3
    size_of_list=len(defender_player_tags)
    defender_tags_counter=np.zeros(size_of_list,dtype='int32')
    list_tag_row_defenders=[]
    for i in range(number_rows):#iterate on all player
        tag_row=defenders_df.iloc[i,22]
        list_tag_row_defenders=tag_row.split(",")
        for list1 in list_tag_row_defenders: #iterate on all tags of each player and slices the string
            list1=list1.replace(" ","")
            list1=list1.replace("#","")
            for j in range(len(defender_player_tags)): #iterates the unique tags and finds 
                if(defender_player_tags[j]==list1): #add 1 when the tag found in player equals one of the unique tags
                    defender_tags_counter[j]+=1
                    break


#puts the two lists unique tag and counter of unique tags in list of tuples then sorts descendingly
    defender_tags=[]
    for i in range(len(defender_player_tags)):
        defender_tags.append((defender_player_tags[i],defender_tags_counter[i]))
    def sortSecond(val):
        return val[1] 
    defender_tags.sort(key = sortSecond,reverse=True) 
    defender_tags_names= [lis[0] for lis in defender_tags]

 #-----------------------------------------------------------------------------------------------------
 #same thing but with traits
    defenders_df=df_traits.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    defenders_df=defenders_df.reset_index()
    defenders_df=defenders_df.drop(columns=["index"])
    number_rows=len(defenders_df)

#find unique traits
    defender_player_traits=[]
    for i in range(number_rows):#iterate on players
        wow=defenders_df.iloc[i,42]
        list=wow.split(",")
        for list1 in list:#iterates on player traits
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("(AI)","")
                    for i in defender_player_traits:#iterates on all traits
                        if(i==list1):#if player traits==one of the total break
                            check=False
                            break
                    if(check):
                     defender_player_traits.append(list1)

    size_of_list=len(defender_player_traits)
    defender_traits_counter=np.zeros(size_of_list,dtype='int32')
    list_trait_row_defenders=[]
#find how much each unique trait was repeated
    for i in range(number_rows):
        tag_row=defenders_df.iloc[i,42]
        list_trait_row_defenders=tag_row.split(",")
        for list1 in list_trait_row_defenders:
            list1=list1.replace(" ","")
            list1=list1.replace("(AI)","")
            for j in range(len(defender_player_traits)):
                if(defender_player_traits[j]==list1):
                    defender_traits_counter[j]+=1
    
    #put the trait and counter in a tuple of traits and counter
    defender_traits=[]

    for i in range(len(defender_player_traits)):
        defender_traits.append((defender_player_traits[i],defender_traits_counter[i]))
    def sortSecond(val):
        return val[1] 
    defender_traits.sort(key = sortSecond,reverse=True) 
    defender_traits_names= [lis[0] for lis in defender_traits]

    #----------------------------------------------------------------------------------
    #taking avg of pace shooting etc
    avg_defender_df=df.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    avg_defender_df=avg_defender_df.describe()
    avg_defender_df=avg_defender_df.iloc[1,15:21]
    avg_defender_df=avg_defender_df.astype(int)

#avg height
    def_height=df.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    def_height=def_height.describe()
    def_height=int(def_height.iloc[1,1])

#avg weight

    def_weight=df.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    def_weight=def_weight.describe()
    def_weight=int(def_weight.iloc[1,2])

#avg rating

    def_rating=df.loc[(df["player_positions"].str.contains("B"))|(df["team_position"].str.contains("B"))]
    def_rating=def_rating.describe()
    def_rating=int(def_rating.iloc[1,4])


    print(f'------------------------------------------------FIFA {list_fifas[l]} -----------------------------------------------------')
    print(f"""To be a Defender in the top 5 Leagues:
            Have a total rating of {def_rating}
            You need to have these 3 player tags: {defender_tags_names[0]}, {defender_tags_names[1]} ,and {defender_tags_names[2]}
            You need to have these Player traits: {defender_traits_names[0]}, {defender_traits_names[1]} ,and {defender_traits_names[2]}
            Your stats need to be on average:
                        Pace:     {avg_defender_df[0]}    Dribbling: {avg_defender_df[3]}
                        Shooting: {avg_defender_df[1]}    Defending: {avg_defender_df[4]}
                        Passing:  {avg_defender_df[2]}    Physical:  {avg_defender_df[5]}
            Your height and weight should be {def_height} cm and {def_weight} kg on average.""")
    print("----------------------------------------------------------------------------------------------------------------------------")
    #----------------------------------------------------------------------------


    #mid-------------------------------------------------------------------------

    #EVERYTHING IS THE SAME WITH ONLY CHANGES IN VARIABLES.
    midfielder_df=df_tags.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    midfielder_df=midfielder_df.reset_index()
    midfielder_df=midfielder_df.drop(columns=["index"])
    number_rows=len(midfielder_df)



    midfielder_player_tags=[]
    for i in range(number_rows):#iterate on players
        wow=midfielder_df.iloc[i,22]
        list=wow.split(",")
        for list1 in list:#iterates on player tags
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("#","")
                    for i in midfielder_player_tags:#iterates on all tags
                        if(i==list1):#if player tag==one of the total break
                            check=False
                            break
                    if(check):
                        midfielder_player_tags.append(list1)
    size_of_list=len(midfielder_player_tags)
    midfielder_tags_counter=np.zeros(size_of_list,dtype='int32')
    list_tag_row_midfielder=[]
    for i in range(number_rows):
        tag_row=midfielder_df.iloc[i,22]
        list_tag_row_midfielder=tag_row.split(",")
        for list1 in list_tag_row_midfielder:
            list1=list1.replace(" ","")
            list1=list1.replace("#","")
            for j in range(len(midfielder_player_tags)):
                if(midfielder_player_tags[j]==list1):
                    midfielder_tags_counter[j]+=1
    midfielder_tags=[]
    for i in range(len(midfielder_player_tags)):
        midfielder_tags.append((midfielder_player_tags[i],midfielder_tags_counter[i]))
    midfielder_tags.sort(key = sortSecond,reverse=True) 
    midfielder_tags_names= [lis[0] for lis in midfielder_tags]
    #----------------------------------------------------------------------------
    midfielders_df=df_traits.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    midfielders_df=midfielders_df.reset_index()
    midfielders_df=midfielders_df.drop(columns=["index"])
    number_rows=len(midfielders_df)

    midfielders_player_traits=[]
    for i in range(number_rows):#iterate on players
        wow=midfielders_df.iloc[i,42]
        list=wow.split(",")
        for list1 in list:#iterates on player traits
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("(AI)","")
                    for i in midfielders_player_traits:#iterates on all traits
                        if(i==list1):#if player traits==one of the total break
                            check=False
                            break
                    if(check):
                     midfielders_player_traits.append(list1)

    size_of_list=len(midfielders_player_traits)
    midfielders_traits_counter=np.zeros(size_of_list,dtype='int32')
    list_trait_row_midfielders=[]

    for i in range(number_rows):
        tag_row=midfielders_df.iloc[i,42]
        list_trait_row_midfielders=tag_row.split(",")
        for list1 in list_trait_row_midfielders:
            list1=list1.replace(" ","")
            list1=list1.replace("(AI)","")
            for j in range(len(midfielders_player_traits)):
                if(midfielders_player_traits[j]==list1):
                    midfielders_traits_counter[j]+=1
    midfielders_traits=[]
    for i in range(len(midfielders_player_traits)):
        midfielders_traits.append((midfielders_player_traits[i],midfielders_traits_counter[i]))
    def sortSecond(val):
        return val[1] 
    midfielders_traits.sort(key = sortSecond,reverse=True) 
    midfielders_traits_names= [lis[0] for lis in midfielders_traits]



    #----------------------------------------------------------------------------
    avg_midfielder_df=df.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    avg_midfielder_df=avg_midfielder_df.describe()
    avg_midfielder_df=avg_midfielder_df.iloc[1,15:21]
    avg_midfielder_df=avg_midfielder_df.astype(int)

    mid_height=df.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    mid_height=mid_height.describe()
    mid_height=int(mid_height.iloc[1,1])


    midfielder_weight=df.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    midfielder_weight=midfielder_weight.describe()
    midfielder_weight=int(midfielder_weight.iloc[1,2])

    midfielder_rating=df.loc[(df["player_positions"].str.contains("M"))|(df["team_position"].str.contains("M"))]
    midfielder_rating=midfielder_rating.describe()
    midfielder_rating=int(midfielder_rating.iloc[1,4])
    print(f"""To be a Midfielder in the top 5 Leagues:
            Have a total rating of {midfielder_rating}
            You need to have these 3 player tags, {midfielder_tags_names[0]}, {midfielder_tags_names[1]} ,and {midfielder_tags_names[2]}
            You need to have these Player traits: {midfielders_traits_names[0]}, {midfielders_traits_names[1]}, {midfielders_traits_names[2]}
            Your stats need to be on average:
                        Pace:     {avg_midfielder_df[0]}    Dribbling: {avg_midfielder_df[3]}
                        Shooting: {avg_midfielder_df[1]}    Defending: {avg_midfielder_df[4]}
                        Passing:  {avg_midfielder_df[2]}    Physical:  {avg_midfielder_df[5]}
            Your height and weight should be {mid_height} cm and {midfielder_weight} kg on average.""")
    print("----------------------------------------------------------------------------------------------------------------------------")
    

    #attack----------------------------------------------------------------------
    attacker_df=df_tags.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    attacker_df=attacker_df.reset_index()
    attacker_df=attacker_df.drop(columns=["index"])
    number_rows=len(attacker_df)
    attacker_player_tags=[]
    for i in range(number_rows):#iterate on players
        wow=attacker_df.iloc[i,22]
        list=wow.split(",")
        for list1 in list:#iterates on player tags
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("#","")
                    for i in attacker_player_tags:#iterates on all tags
                        if(i==list1):#if player tag==one of the total break
                            check=False
                            break
                    if(check):
                        attacker_player_tags.append(list1)

    size_of_list=len(attacker_player_tags)
    attacker_tags_counter=np.zeros(size_of_list,dtype='int32')
    list_tag_row_attacker=[]
    for i in range(number_rows):
        tag_row=attacker_df.iloc[i,22]
        list_tag_row_attacker=tag_row.split(",")
        for list1 in list_tag_row_attacker:
            list1=list1.replace(" ","")
            list1=list1.replace("#","")
            for j in range(len(attacker_player_tags)):
                if(attacker_player_tags[j]==list1):
                    attacker_tags_counter[j]+=1
    attacker_tags=[]
    for i in range(len(attacker_player_tags)):
        attacker_tags.append((attacker_player_tags[i],attacker_tags_counter[i]))
    attacker_tags.sort(key = sortSecond,reverse=True) 
    attacker_tags_names= [lis[0] for lis in attacker_tags]
    #----------------------------------------------------------------------------
    attacker_df=df_traits.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    attacker_df=attacker_df.reset_index()
    attacker_df=attacker_df.drop(columns=["index"])
    number_rows=len(attacker_df)

    attacker_player_traits=[]
    for i in range(number_rows):#iterate on players
        wow=attacker_df.iloc[i,42]
        list=wow.split(",")
        for list1 in list:#iterates on player traits
                    check=True
                    list1=list1.replace(" ","")
                    list1=list1.replace("(AI)","")
                    for i in attacker_player_traits:#iterates on all traits
                        if(i==list1):#if player traits==one of the total break
                            check=False
                            break
                    if(check):
                     attacker_player_traits.append(list1)

    size_of_list=len(attacker_player_traits)
    attacker_traits_counter=np.zeros(size_of_list,dtype='int32')
    list_trait_row_attacker=[]

    for i in range(number_rows):
        tag_row=attacker_df.iloc[i,42]
        list_trait_row_attacker=tag_row.split(",")
        for list1 in list_trait_row_attacker:
            list1=list1.replace(" ","")
            list1=list1.replace("(AI)","")
            for j in range(len(attacker_player_traits)):
                if(attacker_player_traits[j]==list1):
                    attacker_traits_counter[j]+=1
    attacker_traits=[]
    for i in range(len(attacker_player_traits)):
        attacker_traits.append((attacker_player_traits[i],attacker_traits_counter[i]))
    def sortSecond(val):
        return val[1] 
    attacker_traits.sort(key = sortSecond,reverse=True) 
    attacker_traits_names= [lis[0] for lis in attacker_traits]



    #----------------------------------------------------------------------------
    avg_attacker_df=df.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    avg_attacker_df=avg_attacker_df.describe()
    avg_attacker_df=avg_attacker_df.iloc[1,15:21]
    avg_attacker_df=avg_attacker_df.astype(int)

    attacker_height=df.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    attacker_height=attacker_height.describe()
    attacker_height=int(attacker_height.iloc[1,1])


    attacker_weight=df.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    attacker_weight=attacker_weight.describe()
    attacker_weight=int(attacker_weight.iloc[1,2])

    attacker_rating=df.loc[(df["player_positions"].str.contains("F"))|(df["player_positions"].str.contains("T"))|(df["player_positions"].str.contains("W"))|(df["team_position"].str.contains("W"))|(df["team_position"].str.contains("F"))|(df["team_position"].str.contains("T"))]
    attacker_rating=attacker_rating.describe()
    attacker_rating=int(attacker_rating.iloc[1,4])
    print(f"""To be an Attacker in the top 5 Leagues:
            Have a total rating of {attacker_rating}
            You need to have these 3 player tags, {attacker_tags_names[0]}, {attacker_tags_names[1]} ,and {attacker_tags_names[2]}
            You need to have these Player traits: {attacker_traits_names[0]}, {attacker_traits_names[1]}, {attacker_traits_names[2]}
            Your stats need to be on average:
                        Pace:     {avg_attacker_df[0]}    Dribbling: {avg_attacker_df[3]}
                        Shooting: {avg_attacker_df[1]}    Defending: {avg_attacker_df[4]}
                        Passing:  {avg_attacker_df[2]}    Physical:  {avg_attacker_df[5]}
            Your height and weight should be {attacker_height} cm and {attacker_weight} kg on average.""")
    print("----------------------------------------------------------------------------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------------------------
#Logistic Regression-----------------------------------------------------------------------------------------------------------------
#extract all info for messi from all years
# counter=0
# for j in list_files:
#     fifa_df=pd.read_csv(j,index_col=False)
#     GOAT=fifa_df.loc[fifa_df['short_name']=="L. Messi"]
#     if counter ==0:
#         GOAT.to_csv(r"C:\Users\karee\Desktop\GOAT.csv",mode='w',index=False,header=True)
#     else:
#         GOAT.to_csv(r"C:\Users\karee\Desktop\GOAT.csv",mode='a',index=False,header=False)
#     counter+=1
    

df=pd.read_csv(r'GOAT.csv')

#these columns have string equations like 56+3 so i need to use eval
#iloc the columns doesnt work with eval i tried.
#alot of data cleaning happened in excel such as deleting a column that didnt have any values

columns=['ls','st','rs','lw','lf','cf','cam','lam','rw','rf','rb','rcb','cb','lcb','lb','rwb','rdm','cdm','ldm','lwb','rm','rcm','cm','lcm','lm','ram']
for j in columns:
    total=[]
    for i in df[j]:
        total.append(eval(i))
    df[j]=total



# ----split data------
X=df.iloc[:,1:].values #rest
Y=df.iloc[:,0].values #overall
# -----------------------
X_train,X_test,Y_train,Y_test=train_test_split(X , Y , test_size=0.25 , random_state=42 )
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
classifier=LogisticRegression()
classifier.fit(X_train,Y_train) 
y_pred=classifier.predict(X_test)
print("predicted rating for messi is",y_pred[0])


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
df=pd.read_csv(list_files[5])

# creating correlation file
corr_matrix=df.corr()
corr_matrix.to_csv("Correlation.csv")

#plotting---------------------------------------------------------
Spain_overall=[]
France_overall=[]
English_overall=[]
Italy_overall=[]
German_overall=[]
Spain_value=[]
France_value=[]
English_value=[]
Italy_value=[]
German_value=[]
Barca_avg=[]
City_value=[]
United_value=[]
Liver_value=[]
Arsenal_value=[]
Spurs_value=[]
Chelsea_value=[]

BPL=[City_value,United_value,Liver_value,Arsenal_value,Spurs_value,Chelsea_value]


for i in list_files:
    df=pd.read_csv(i)
    france_rating=df.loc[(df['league_name']=='French Ligue 1')]
    English_rating=df.loc[(df['league_name']=='English Premier League')]
    spain_rating=df.loc[(df['league_name']=='Spain Primera Division')]
    german_rating=df.loc[(df['league_name']=='German 1. Bundesliga')]
    italy_rating=df.loc[(df['league_name']=='Italian Serie A')]
    
    Spain_overall.append(spain_rating["overall"].mean())
    France_overall.append(france_rating["overall"].mean())
    English_overall.append(English_rating["overall"].mean())
    German_overall.append(german_rating["overall"].mean())
    Italy_overall.append(italy_rating['overall'].mean())


    Spain_value.append(spain_rating["value_eur"].mean())
    France_value.append(france_rating["value_eur"].mean())
    English_value.append(English_rating["value_eur"].mean())
    German_value.append( german_rating["value_eur"].mean())
    Italy_value.append(italy_rating["value_eur"].mean())

    
    Catalony=df.loc[df["club_name"]=="FC Barcelona"]
    Barca_avg.append(Catalony["overall"].mean())


    City_df=df.loc[df["club_name"]=="Manchester City"]
    United_df=df.loc[df["club_name"]=="Manchester United"]
    Liver_df=df.loc[df["club_name"]=="Liverpool"]
    Arsenal_df=df.loc[df["club_name"]=="Arsenal"]
    Spurs_df=df.loc[df["club_name"]=="Tottenham Hotspur"]
    Chelsea_df=df.loc[df["club_name"]=="Chelsea"]

    City_value.append(City_df["value_eur"].sum())
    United_value.append(United_df["value_eur"].sum())
    Liver_value.append(Liver_df["value_eur"].sum())
    Chelsea_value.append(Chelsea_df["value_eur"].sum())
    Arsenal_value.append(Arsenal_df["value_eur"].sum())
    Spurs_value.append(Spurs_df["value_eur"].sum())
    


plt.style.use("cyberpunk")

figure, axis = plt.subplots(2, 2,figsize=(10,10))
axis[0,0].scatter(df["movement_balance"],df["height_cm"],color="red",s=1,)
axis[0,0].set_title("Negative Correlation between Height and Balance.\n",size=8)
axis[0,0].set_xlabel("Balance",)
axis[0,0].set_ylabel("Height")

axis[0,1].scatter(df["overall"],df["mentality_composure"],s=1)
axis[0,1].set_title("the greater the player is, the more composed he is when playing.\n",size=8)
axis[0,1].set_xlabel("Overall")
axis[0,1].set_ylabel("Composure")

axis[1,0].scatter(df["overall"],df["movement_reactions"],color="yellow",s=1,)
axis[1,0].set_title("the greater the player is, the faster are his movement reactions.\n",size=8)
axis[1,0].set_xlabel("Overall")
axis[1,0].set_ylabel("Reactions")

axis[1,1].scatter(df["dribbling"],df["passing"],color="white",s=1)
axis[1,1].set_title("the better the player with controlling ,the better he is with passing it",size=6)
axis[1,1].set_xlabel("dribbling")
axis[1,1].set_ylabel("passing")
plt.tight_layout()



plt.figure(figsize=(10,10),dpi=100)
plt.plot(list_fifas,Barca_avg,'ro-')
plt.title("Barca's team overall average")
plt.xlabel("Years")
plt.ylabel("Overall Avg")
mplcyberpunk.add_glow_effects()


countries=[Spain_overall,France_overall,English_overall,German_overall,Italy_overall]
labels=["Spain","France","England","Germany","Italy"]
colors=['gold','blue','salmon','firebrick','limegreen']

countries=[Spain_value,France_value,English_value,German_value,Italy_value]
for i,j,k in zip(countries,labels,colors):
    plt.plot(list_fifas,i,marker='o',label=j,color=k)
plt.title("League values")
plt.xlabel("Years")
plt.ylabel("Values, By Billion")
plt.legend()
mplcyberpunk.add_glow_effects()

labels=["City","United","Liverpool","Arsenal","Spurs","Chelsea"]
BPL=[City_value,United_value,Liver_value,Arsenal_value,Spurs_value,Chelsea_value]
colors=['c','lightcoral','red','wheat','ghostwhite','blue']
plt.figure(figsize=(10,10),dpi=100)
for i,j,k in zip(BPL,labels,colors):
    plt.plot(list_fifas,i,marker='o',label=j,color=k)
plt.title("Club values")
plt.xlabel("Years")
plt.ylabel("Values, By 100 Million")
plt.legend()
mplcyberpunk.add_glow_effects()





plt.show()  




    
